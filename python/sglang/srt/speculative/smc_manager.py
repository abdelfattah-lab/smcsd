from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional

import torch

from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    FINISH_ABORT,
    Req,
    ScheduleBatch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.smc_info import (
    SMCDraftInput,
    _release_internal_req,
    clone_req_for_smc_particle,
    get_smc_reserved_kv_len,
    set_smc_reserved_kv_len,
    validate_smc_parent_req,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_SMC_DIAG_PATH_ENV = "SGLANG_SMC_DIAG_PATH"


def _append_smc_diag_record(record: dict) -> None:
    record_path = os.environ.get(_SMC_DIAG_PATH_ENV)
    if not record_path:
        return
    payload = dict(record)
    payload["pid"] = os.getpid()
    payload["timestamp_ns"] = time.perf_counter_ns()
    with open(record_path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class SMCFinishedParticleSnapshot:
    output_ids: List[int]
    finished_reason: Optional[BaseFinishReason]
    finished_len: Optional[int]


@dataclass
class SMCGroupState:
    group_id: str
    parent_req: Req
    particle_reqs: Dict[int, Req]
    log_weights: torch.Tensor
    step_counts: List[int] = field(default_factory=list)
    resampled_at_step: int = 0
    finished_particles: Dict[int, SMCFinishedParticleSnapshot] = field(
        default_factory=dict
    )
    # Deferred log_weight updates: list of (particle_indices, logprob_diffs)
    # Applied lazily in _launch_pending_resamples / _finalize to keep
    # on_batch_done 100% CPU (zero GPU kernel launches).
    pending_diffs: List[tuple] = field(default_factory=list)

    def flush_pending_diffs(self) -> None:
        """Apply all deferred log_weight updates to the GPU tensor.

        Called from _launch_pending_resamples (overlapped) or _finalize_group.
        This is the only place GPU log_weight kernels fire.
        """
        if not self.pending_diffs:
            return
        lw = self.log_weights
        dev = lw.device
        for pidxs, diffs in self.pending_diffs:
            if isinstance(pidxs, int):
                # Single particle: pidxs is a plain int, diffs is 1-elem tensor
                lw[pidxs] += diffs[0].to(dtype=lw.dtype, device=dev)
            else:
                # Multiple particles: pidxs is a Python list
                pidx_t = torch.tensor(pidxs, dtype=torch.int64, device=dev)
                lw[pidx_t] += diffs.to(dtype=lw.dtype, device=dev)
        self.pending_diffs.clear()

    def active_particle_indices(self) -> List[int]:
        return [
            idx
            for idx, req in self.particle_reqs.items()
            if idx not in self.finished_particles and not req.finished()
        ]

    def all_active_aligned(self) -> bool:
        """Check that all active particles have taken the same number of steps
        and have advanced past the last resampling point."""
        active = self.active_particle_indices()
        if not active:
            return True
        first_count = self.step_counts[active[0]]
        if first_count <= self.resampled_at_step:
            return False
        for idx in active[1:]:
            if self.step_counts[idx] != first_count:
                return False
        return True

class SMCManager:
    def __init__(self, server_args):
        self.server_args = server_args
        self.groups: Dict[str, SMCGroupState] = {}
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.device: torch.device | str = "cpu"

    def has_active_groups(self) -> bool:
        return bool(self.groups)

    def smc_held_token_count(self) -> int:
        """Count unique token slots held by SMC particle requests.

        Particles within a group share prefix token slots (via refcount),
        so we collect the union of all referenced slot indices to avoid
        double-counting shared slots.
        """
        if not self.groups or self.req_to_token_pool is None:
            return 0
        held: set = set()
        for group in self.groups.values():
            for req in group.particle_reqs.values():
                if req.req_pool_idx is None:
                    continue
                reserved = get_smc_reserved_kv_len(req)
                if reserved > 0:
                    indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :reserved
                    ]
                    held.update(indices.cpu().tolist())
        return len(held)

    def clear(self) -> None:
        self.groups.clear()

    def get_group(self, group_id: Optional[str]) -> Optional[SMCGroupState]:
        if group_id is None:
            return None
        return self.groups.get(group_id)

    def get_group_for_req(self, req: Req) -> Optional[SMCGroupState]:
        return self.get_group(req.smc_group_id)

    def get_active_particle_reqs(self, group_id: Optional[str]) -> List[Req]:
        group = self.get_group(group_id)
        if group is None:
            return []
        return [
            group.particle_reqs[idx]
            for idx in sorted(group.active_particle_indices())
        ]

    def get_active_particle_reqs_in_collection(
        self,
        group_id: Optional[str],
        reqs: List[Req],
    ) -> List[Req]:
        if group_id is None:
            return []
        req_ids = {id(req) for req in reqs}
        return [
            req
            for req in self.get_active_particle_reqs(group_id)
            if id(req) in req_ids
        ]

    def all_active_members_present(
        self,
        group_id: Optional[str],
        reqs: List[Req],
    ) -> bool:
        active = self.get_active_particle_reqs(group_id)
        if not active:
            return False
        req_ids = {id(req) for req in reqs}
        return all(id(req) in req_ids for req in active)

    def get_particle_lag(self, req: Req) -> int:
        group = self.get_group_for_req(req)
        if group is None:
            return 0
        counts = group.step_counts
        active = group.active_particle_indices()
        if not active:
            return 0
        max_step = max(counts[idx] for idx in active)
        return max_step - counts[req.smc_particle_idx]

    def get_group_lag(self, group_id: Optional[str]) -> int:
        group = self.get_group(group_id)
        if group is None:
            return 0
        active = group.active_particle_indices()
        if not active:
            return 0
        counts = group.step_counts
        max_step = max(counts[idx] for idx in active)
        min_active_step = min(counts[idx] for idx in active)
        return max_step - min_active_step

    def create_group(self, parent_req: Req, scheduler) -> Optional[str]:
        if parent_req.rid in self.groups:
            return None
        self.req_to_token_pool = scheduler.req_to_token_pool
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        self.device = scheduler.device

        error = validate_smc_parent_req(parent_req)
        if error is not None:
            return error

        particle_reqs: List[Req] = []
        for particle_idx in range(self.server_args.smc_n_particles):
            particle_req = clone_req_for_smc_particle(
                parent_req,
                particle_idx=particle_idx,
                role="particle",
                temperature_multiplier=self.server_args.smc_draft_temperature,
                return_logprob=False,
            )
            particle_req.smc_group_id = parent_req.rid
            particle_req.draft_prefix_materialized = False
            particle_reqs.append(particle_req)

        if scheduler.req_to_token_pool.alloc(particle_reqs) is None:
            return "SMC particle allocation failed because req_to_token_pool is full."

        if parent_req.output_ids:
            desired_seq_len = len(parent_req.origin_input_ids) + len(parent_req.output_ids) - 1
            shared_seq_len = min(parent_req.kv_committed_len, desired_seq_len)
        else:
            shared_seq_len = parent_req.kv_committed_len

        for particle_req in particle_reqs:
            scheduler.req_to_token_pool.copy_block_table(
                parent_req.req_pool_idx,
                particle_req.req_pool_idx,
                shared_seq_len,
                scheduler.token_to_kv_pool_allocator,
            )
            particle_req.kv_committed_len = shared_seq_len
            particle_req.kv_allocated_len = shared_seq_len
            set_smc_reserved_kv_len(particle_req, shared_seq_len)
            particle_req.prefix_indices = scheduler.req_to_token_pool.req_to_token[
                particle_req.req_pool_idx, :shared_seq_len
            ].to(dtype=torch.int64, copy=True)
            particle_req.cache_protected_len = shared_seq_len
            particle_req.draft_prefix_materialized = shared_seq_len == 0

        group = SMCGroupState(
            group_id=parent_req.rid,
            parent_req=parent_req,
            particle_reqs={req.smc_particle_idx: req for req in particle_reqs},
            log_weights=torch.zeros(
                self.server_args.smc_n_particles,
                dtype=torch.float64,
                device=self.device,
            ),
            step_counts=[0] * self.server_args.smc_n_particles,
        )
        self.groups[parent_req.rid] = group
        return None

    def _build_particle_batch(
        self,
        particle_reqs: List[Req],
        scheduler,
        use_future_map: bool = True,
    ) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=particle_reqs,
            req_to_token_pool=scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            tree_cache=scheduler.tree_cache,
            model_config=scheduler.model_config,
            enable_overlap=scheduler.enable_overlap,
            spec_algorithm=SpeculativeAlgorithm.SMC,
        )
        batch.forward_mode = ForwardMode.DECODE
        batch.multimodal_inputs = [None] * len(particle_reqs)
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in particle_reqs],
            dtype=torch.int64,
            device=scheduler.device,
        )
        visible_seq_lens = torch.tensor(
            [len(req.origin_input_ids) + len(req.output_ids) for req in particle_reqs],
            dtype=torch.int64,
            device=scheduler.device,
        )
        batch.seq_lens = visible_seq_lens
        batch.seq_lens_cpu = visible_seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
        batch.orig_seq_lens = visible_seq_lens.to(dtype=torch.int32)
        last_token_ids = torch.tensor(
            [
                req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                for req in particle_reqs
            ],
            dtype=torch.int32,
            device=scheduler.device,
        )
        batch.output_ids = last_token_ids
        batch.top_logprobs_nums = [0] * len(particle_reqs)
        batch.token_ids_logprobs = [None] * len(particle_reqs)
        batch.spec_info = SMCDraftInput(
            last_token_ids=last_token_ids,
            new_seq_lens=visible_seq_lens,
            committed_seq_lens_cpu=batch.seq_lens_cpu,
        )
        if use_future_map and scheduler.enable_overlap and scheduler.future_map is not None:
            future_indices = scheduler.future_map.alloc_future_indices(len(particle_reqs))
            scheduler.future_map.store_to_map_for_new_smc_batch(
                future_indices,
                batch.spec_info,
            )
            batch.spec_info.future_indices = future_indices
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            scheduler.model_config.vocab_size,
        )
        return batch

    def on_particle_finished(self, req: Req) -> None:
        group = self.groups.get(req.smc_group_id)
        if group is None:
            return
        particle_idx = req.smc_particle_idx
        if particle_idx in group.finished_particles:
            return
        group.finished_particles[particle_idx] = SMCFinishedParticleSnapshot(
            output_ids=list(req.output_ids),
            finished_reason=copy.copy(req.finished_reason),
            finished_len=req.finished_len,
        )

    def _finalize_group(self, group_id: str) -> Optional[Req]:
        group = self.groups.pop(group_id, None)
        if group is None:
            return None

        # Flush deferred log_weight diffs so best-particle selection is correct
        group.flush_pending_diffs()

        best_idx = None
        best_key = None
        best_output_ids: List[int] = []
        best_finish_reason: Optional[BaseFinishReason] = None
        best_finished_len: Optional[int] = None
        for particle_idx, req in group.particle_reqs.items():
            if particle_idx in group.finished_particles:
                snapshot = group.finished_particles[particle_idx]
                output_ids = snapshot.output_ids
                finish_reason = snapshot.finished_reason
                finished_len = snapshot.finished_len
            else:
                output_ids = list(req.output_ids)
                finish_reason = copy.copy(req.finished_reason)
                finished_len = req.finished_len

            key = (float(group.log_weights[particle_idx].item()), len(output_ids))
            if best_key is None or key > best_key:
                best_idx = particle_idx
                best_key = key
                best_output_ids = output_ids
                best_finish_reason = finish_reason
                best_finished_len = finished_len

        _append_smc_diag_record(
            {
                "type": "finalize_group",
                "group_id": group_id,
                "log_weights": [float(x) for x in group.log_weights.tolist()],
                "best_idx": best_idx,
                "particle_output_ids": {
                    str(particle_idx): (
                        list(group.finished_particles[particle_idx].output_ids)
                        if particle_idx in group.finished_particles
                        else list(req.output_ids)
                    )
                    for particle_idx, req in group.particle_reqs.items()
                },
                "best_output_ids": list(best_output_ids),
            }
        )

        # Release KV cache and req_pool entries for all particle requests.
        # Particles that were already released during decode (finished early)
        # will be skipped by _release_internal_req (req_pool_idx is None).
        for particle_idx, req in group.particle_reqs.items():
            _release_internal_req(
                req,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        parent_req = group.parent_req
        parent_req.output_ids = list(best_output_ids)
        parent_req.finished_reason = (
            best_finish_reason
            if best_finish_reason is not None
            else FINISH_ABORT("SMC group finalized without a finished particle.")
        )
        parent_req.finished_len = (
            best_finished_len
            if best_finished_len is not None
            else len(parent_req.output_ids)
        )
        return parent_req

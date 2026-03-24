from __future__ import annotations

import copy
from dataclasses import dataclass, field
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
    clone_req_for_smc_particle,
    effective_sample_size,
    multinomial_resample,
    normalize_log_weights,
    systematic_resample,
    validate_smc_parent_req,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


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
    step_counts: Dict[int, int] = field(default_factory=dict)
    resampled_at_step: int = 0
    finished_particles: Dict[int, SMCFinishedParticleSnapshot] = field(
        default_factory=dict
    )

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
        counts = [self.step_counts.get(idx, 0) for idx in active]
        # All must be at the same step and ahead of last resample point
        return len(set(counts)) == 1 and counts[0] > self.resampled_at_step


class SMCManager:
    def __init__(self, server_args):
        self.server_args = server_args
        self.groups: Dict[str, SMCGroupState] = {}
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.device: torch.device | str = "cpu"

    def has_active_groups(self) -> bool:
        return bool(self.groups)

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
        max_step = max(counts.get(idx, 0) for idx in active)
        return max_step - counts.get(req.smc_particle_idx, 0)

    def get_group_lag(self, group_id: Optional[str]) -> int:
        group = self.get_group(group_id)
        if group is None:
            return 0
        active = group.active_particle_indices()
        if not active:
            return 0
        counts = group.step_counts
        max_step = max(counts.get(idx, 0) for idx in active)
        min_active_step = min(counts.get(idx, 0) for idx in active)
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
            step_counts={req.smc_particle_idx: 0 for req in particle_reqs},
        )
        self.groups[parent_req.rid] = group

        particle_batch = self._build_particle_batch(particle_reqs, scheduler)
        if scheduler.running_batch.is_empty():
            scheduler.running_batch = particle_batch
        else:
            scheduler.running_batch.merge_batch(particle_batch)
        return None

    def _build_particle_batch(self, particle_reqs: List[Req], scheduler) -> ScheduleBatch:
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
        batch.orig_seq_lens = torch.tensor(
            [len(req.origin_input_ids) for req in particle_reqs],
            dtype=torch.int32,
            device=scheduler.device,
        )
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
        )
        if scheduler.enable_overlap and scheduler.future_map is not None:
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

    def on_particle_step(self, req: Req, logprob_diff: float | torch.Tensor) -> Optional[Req]:
        group = self.groups.get(req.smc_group_id)
        if group is None:
            return None

        particle_idx = req.smc_particle_idx
        group.log_weights[particle_idx].add_(
            torch.as_tensor(
                logprob_diff,
                dtype=group.log_weights.dtype,
                device=group.log_weights.device,
            )
        )
        group.step_counts[particle_idx] = group.step_counts.get(particle_idx, 0) + 1

        if not group.all_active_aligned():
            return None

        self._maybe_resample(group)
        # Record the step at which we resampled so we can detect the next round
        active = group.active_particle_indices()
        if active:
            group.resampled_at_step = group.step_counts[active[0]]

        if not active:
            return self._finalize_group(group.group_id)
        return None

    def on_particle_step_batch(
        self,
        reqs: List[Req],
        logprob_diffs: torch.Tensor,
    ) -> List[Req]:
        if not reqs:
            return []

        if not torch.is_tensor(logprob_diffs):
            logprob_diffs = torch.as_tensor(logprob_diffs, dtype=torch.float32)

        grouped_reqs: Dict[str, List[tuple[int, Req]]] = {}
        for row, req in enumerate(reqs):
            group_id = req.smc_group_id
            if group_id is None or group_id not in self.groups:
                continue
            grouped_reqs.setdefault(group_id, []).append((row, req))

        finalized_reqs: List[Req] = []
        for group_id, entries in grouped_reqs.items():
            group = self.groups.get(group_id)
            if group is None:
                continue

            row_indices = torch.tensor(
                [row for row, _ in entries],
                dtype=torch.int64,
                device=logprob_diffs.device,
            )
            particle_indices = torch.tensor(
                [req.smc_particle_idx for _, req in entries],
                dtype=torch.int64,
                device=group.log_weights.device,
            )
            group.log_weights[particle_indices] += logprob_diffs[row_indices].to(
                dtype=group.log_weights.dtype,
                device=group.log_weights.device,
            )
            for _, req in entries:
                particle_idx = req.smc_particle_idx
                group.step_counts[particle_idx] = group.step_counts.get(particle_idx, 0) + 1

            if not group.all_active_aligned():
                continue

            self._maybe_resample(group)
            active = group.active_particle_indices()
            if active:
                group.resampled_at_step = group.step_counts[active[0]]
            else:
                finalized_req = self._finalize_group(group_id)
                if finalized_req is not None:
                    finalized_reqs.append(finalized_req)

        return finalized_reqs

    def _maybe_resample(self, group: SMCGroupState) -> None:
        active_indices = group.active_particle_indices()
        if len(active_indices) <= 1:
            return

        normalized_weights = normalize_log_weights(
            group.log_weights[active_indices],
            device=self.device,
        )
        ess = effective_sample_size(normalized_weights, device=self.device)
        if ess >= len(active_indices) * self.server_args.smc_resample_threshold:
            return

        if self.server_args.smc_resample_method == "multinomial":
            ancestors = multinomial_resample(normalized_weights, device=self.device)
        else:
            ancestors = systematic_resample(normalized_weights, device=self.device)

        source_indices = [active_indices[idx] for idx in ancestors]
        snapshots = [
            self._snapshot_req_state(group.particle_reqs[idx]) for idx in source_indices
        ]
        for dst_idx, snapshot in zip(active_indices, snapshots, strict=True):
            dst_req = group.particle_reqs[dst_idx]
            self._restore_req_state(dst_req, snapshot)
            group.log_weights[dst_idx] = 0.0
        for snapshot in snapshots:
            self._release_snapshot(snapshot)

    def _snapshot_req_state(self, req: Req) -> dict:
        seq_len = req.kv_committed_len
        if seq_len > 0:
            indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :seq_len
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.inc_ref(indices)
        else:
            indices = torch.empty((0,), dtype=torch.int64)
        return {
            "indices": indices,
            "output_ids": list(req.output_ids),
            "finished_reason": copy.copy(req.finished_reason),
            "finished_len": req.finished_len,
            "finished_output": req.finished_output,
            "to_finish": copy.copy(req.to_finish),
            "kv_committed_len": req.kv_committed_len,
            "cache_protected_len": req.cache_protected_len,
            "logprob_start_len": req.logprob_start_len,
            "draft_prefix_materialized": req.draft_prefix_materialized,
        }

    def _restore_req_state(self, req: Req, snapshot: dict) -> None:
        if req.kv_allocated_len > 0:
            old_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.kv_allocated_len
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.dec_ref_and_free(old_indices)

        indices = snapshot["indices"]
        if indices.numel() > 0:
            self.token_to_kv_pool_allocator.inc_ref(indices)
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, indices.shape[0])),
                indices.to(torch.int32),
            )
            req.prefix_indices = indices.to(dtype=torch.int64, copy=True)
        else:
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)

        req.output_ids = list(snapshot["output_ids"])
        req.finished_reason = copy.copy(snapshot["finished_reason"])
        req.finished_len = snapshot["finished_len"]
        req.finished_output = snapshot["finished_output"]
        req.to_finish = copy.copy(snapshot["to_finish"])
        req.kv_committed_len = snapshot["kv_committed_len"]
        req.kv_allocated_len = snapshot["kv_committed_len"]
        req.cache_protected_len = snapshot["cache_protected_len"]
        req.logprob_start_len = snapshot["logprob_start_len"]
        req.draft_prefix_materialized = snapshot["draft_prefix_materialized"]

    def _release_snapshot(self, snapshot: dict) -> None:
        indices = snapshot["indices"]
        if indices.numel() > 0:
            self.token_to_kv_pool_allocator.dec_ref_and_free(indices)

    def _finalize_group(self, group_id: str) -> Optional[Req]:
        group = self.groups.pop(group_id, None)
        if group is None:
            return None

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

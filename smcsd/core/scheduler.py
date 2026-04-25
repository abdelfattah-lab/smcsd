from __future__ import annotations

import json
import logging
import os
import signal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import psutil
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.server_args import PortArgs, ServerArgs
from smcsd.common.utils import (
    _release_internal_req,
    _release_smc_parent_req,
    clone_req_for_smc_particle,
    compute_smc_shared_prefix_len,
    validate_smc_parent_req,
)
from smcsd.mem_cache.allocator import copy_block_table
from sglang.srt.utils import DynamicGradMode, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class SMCMetricLogger:
    """Lightweight host-side aggregator for SMC proposal diagnostics.

    Metrics are computed once per decode cycle from per-group slot weights.
    This intentionally lives outside the fused resampling kernel so Phase 0
    instrumentation is easy to audit and can be disabled with near-zero cost.
    """

    def __init__(self, *, enabled: bool, log_interval: int = 50, jsonl_path: Optional[str] = None):
        self.enabled = bool(enabled)
        self.log_interval = max(int(log_interval or 1), 1)
        self.jsonl_path = jsonl_path
        self._fh = None
        self.step = 0
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        if self.enabled and jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            self._fh = open(jsonl_path, "a", buffering=1)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def _safe_float(x) -> float:
        return float(x) if x is not None else 0.0

    @torch.no_grad()
    def snapshot_decode_step(self, slot_state: "ScheduleBatchSMC") -> List[Dict]:
        """Capture pre-resampling group metrics.

        Must be called before ``collect_resample_jobs_batch`` because the fused
        collect kernel zeroes interval/cumulative weights for rows that resample.
        """
        if not self.enabled:
            return []
        rows = torch.nonzero(slot_state.row_in_use, as_tuple=False).flatten()
        if rows.numel() == 0:
            return []

        group_metrics = []
        for row_t in rows.cpu().tolist():
            row = int(row_t)
            slots = slot_state.group_to_slots[row].to(torch.int64)
            slots = slots[slots >= 0]
            if slots.numel() == 0:
                continue
            interval_lw = slot_state.interval_weights[slots].detach().double()
            cumulative_lw = slot_state.log_weights[slots].detach().double()
            finite_mask = torch.isfinite(interval_lw)
            if not finite_mask.any():
                continue
            finite_interval = interval_lw[finite_mask]
            finite_cumulative = cumulative_lw[torch.isfinite(cumulative_lw)]
            norm_lw = finite_interval - torch.logsumexp(finite_interval, dim=0)
            weights = norm_lw.exp()
            ess = 1.0 / torch.sum(weights * weights).clamp_min(1e-300)
            logw_var = finite_interval.var(unbiased=False) if finite_interval.numel() > 1 else torch.zeros((), dtype=torch.float64, device=finite_interval.device)
            cum_var = finite_cumulative.var(unbiased=False) if finite_cumulative.numel() > 1 else torch.zeros((), dtype=torch.float64, device=finite_interval.device)
            group_metrics.append({
                "row": row,
                "group_id": slot_state.row_to_group_id.get(row, str(row)),
                "ess": float(ess.item()),
                "ess_frac": float((ess / max(slots.numel(), 1)).item()),
                "logw_mean": float(finite_interval.mean().item()),
                "logw_var": float(logw_var.item()),
                "cum_logw_mean": float(finite_cumulative.mean().item()) if finite_cumulative.numel() else 0.0,
                "cum_logw_var": float(cum_var.item()),
                "min_weight": float(weights.min().item()),
                "max_weight": float(weights.max().item()),
                "resampled": False,
            })
        return group_metrics

    def record_decode_step(self, group_metrics: List[Dict], plan, did_resample: bool) -> None:
        if not self.enabled or not group_metrics:
            return
        self.step += 1
        if plan is not None:
            for g in group_metrics:
                row = g["row"]
                g["resampled"] = bool(plan.resample_mask[row].item())

        n = len(group_metrics)
        aggregate = {
            "step": self.step,
            "n_groups": n,
            "ess_mean": sum(g["ess"] for g in group_metrics) / n,
            "ess_frac_mean": sum(g["ess_frac"] for g in group_metrics) / n,
            "logw_var_mean": sum(g["logw_var"] for g in group_metrics) / n,
            "cum_logw_var_mean": sum(g["cum_logw_var"] for g in group_metrics) / n,
            "max_weight_mean": sum(g["max_weight"] for g in group_metrics) / n,
            "resampled_groups": sum(1 for g in group_metrics if g["resampled"]),
            "did_resample": bool(did_resample),
            "n_resample_jobs": int(getattr(plan, "n_jobs", 0) if plan is not None else 0),
        }

        for k, v in aggregate.items():
            if isinstance(v, (int, float, bool)):
                self.totals[k] += float(v)
        self.counts["steps"] += 1

        if self._fh is not None:
            self._fh.write(json.dumps({"aggregate": aggregate, "groups": group_metrics}, sort_keys=True) + "\n")

        if self.step % self.log_interval == 0:
            logger.info(
                "SMC metrics step=%d groups=%d ESS/N=%.3f logw_var=%.3f "
                "cum_logw_var=%.3f resampled=%d jobs=%d max_w=%.3f",
                aggregate["step"], aggregate["n_groups"], aggregate["ess_frac_mean"],
                aggregate["logw_var_mean"], aggregate["cum_logw_var_mean"],
                aggregate["resampled_groups"], aggregate["n_resample_jobs"],
                aggregate["max_weight_mean"],
            )

    def summary(self) -> Dict[str, float]:
        steps = max(self.counts.get("steps", 0), 1)
        return {
            "steps": self.counts.get("steps", 0),
            "ess_frac_mean": self.totals.get("ess_frac_mean", 0.0) / steps,
            "logw_var_mean": self.totals.get("logw_var_mean", 0.0) / steps,
            "cum_logw_var_mean": self.totals.get("cum_logw_var_mean", 0.0) / steps,
            "resampled_groups_per_step": self.totals.get("resampled_groups", 0.0) / steps,
            "resample_jobs_per_step": self.totals.get("n_resample_jobs", 0.0) / steps,
            "max_weight_mean": self.totals.get("max_weight_mean", 0.0) / steps,
        }


def _prepare_req_for_private_prefill(req: Req) -> None:
    """Prepare a particle for prefill without any prefix-cache participation."""
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req.last_node = None
    req.last_host_node = None
    req.last_host_backup_node = None
    req.host_hit_length = 0
    req.mamba_branching_seqlen = None
    req.cache_protected_len = 0
    req.init_next_round_input(tree_cache=None)


@dataclass
class SequenceGroup:
    """Scheduler-side handle for one SMC group before/during decoding.

    Owns the parent ``Req`` (the user-visible request), the materialised
    particle ``Req`` objects, and basic metadata.  Cumulative weights live
    slot-indexed on ``ScheduleBatchSMC`` (not here) once the group is
    materialised.
    """

    parent_req: Req
    n_particles: int
    particle_temperature: float
    particle_reqs: Dict[int, Req] = field(default_factory=dict)

    @property
    def group_id(self) -> str:
        return self.parent_req.rid

    def has_materialized_particles(self) -> bool:
        return bool(self.particle_reqs)

    def materialize_particles(self) -> None:
        """Clone ``n_particles`` Reqs off the parent Req and register them.

        Weight tensors are allocated slot-indexed on ``ScheduleBatchSMC`` at
        ``allocate_slots`` time — not on the SequenceGroup.
        """
        if self.particle_reqs:
            return
        parent_req = self.parent_req
        particle_reqs: List[Req] = []
        for particle_idx in range(self.n_particles):
            particle_req = clone_req_for_smc_particle(
                parent_req,
                particle_idx=particle_idx,
                temperature=self.particle_temperature,
                return_logprob=False,
            )
            particle_reqs.append(particle_req)

        self.particle_reqs = {req.smc_particle_idx: req for req in particle_reqs}

    def clear_particles(self) -> None:
        self.particle_reqs = {}


class SMCCoordinator:
    """SMC resample coordinator.

    One fused Triton kernel per decode step:

    1. ``collect`` — for every in-use group row, normalise interval weights,
       check ESS against ``threshold * N``, and (if below threshold) run
       systematic resampling, emitting flat ``dst_slots`` / ``src_slots`` /
       ``row_of_job`` tensors.  Zeros the resampled rows' weights in place.

    2. ``dispatch`` — hand those flat tensors to ``batched_resample_kv`` for
       a fused KV block-table copy + refcount update, then vector-copy the
       per-slot state tensors (seq_lens, finished_mask, token_counts,
       all_token_ids, …).  Python-side ``copy_req_metadata`` loops over the
       at-most-``max_slots`` copies — accepted unavoidable cost.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        resample_threshold: float,
    ) -> None:
        if torch.device(device).type != "cuda":
            raise ValueError("SMCCoordinator requires CUDA")
        self.device = device
        self.resample_threshold = resample_threshold
        self._step_counter = 0
        logger.info(
            "SMCCoordinator: resample_threshold=%s (fused systematic kernel)",
            resample_threshold,
        )

    # ── Public API ──────────────────────────────────────────

    def collect_resample_jobs_batch(self, slot_state: "ScheduleBatchSMC"):
        """Run the fused collect kernel over all in-use group rows.

        Returns a ``BatchedResampleResult``.  The ``step_counter`` increments
        on every call so ``tl.rand(step_counter, row)`` draws independent
        stratified uniforms across steps.
        """
        from smcsd.core.kernels.fused_collect import batched_collect_fused

        self._step_counter += 1
        return batched_collect_fused(
            slot_state.log_weights,
            slot_state.interval_weights,
            slot_state.group_to_slots,
            slot_state.row_in_use,
            self.resample_threshold,
            step_counter=self._step_counter,
        )

    def dispatch_resample_batch(
        self,
        plan,
        slot_state: "ScheduleBatchSMC",
        *,
        rebuild_active: bool = True,
    ) -> None:
        """Apply a ``BatchedResampleResult`` plan.

        Fused KV copy + per-slot state copies + Req-metadata loop.  No-op on
        an empty plan.  ``rebuild_active`` may be deferred by the caller if
        other membership changes are about to happen in the same cycle.
        """
        if plan.n_jobs == 0:
            return

        from smcsd.core.kernels.fused_resample_kv import batched_resample_kv

        dst_idx = plan.dst_slots.to(torch.int64)
        src_idx = plan.src_slots.to(torch.int64)

        if __debug__:
            assert not torch.isin(dst_idx, src_idx).any().item(), (
                "Cross-group dst/src slot overlap detected"
            )

        dst_pool_indices = slot_state.req_pool_indices[dst_idx].to(torch.int32)
        src_pool_indices = slot_state.req_pool_indices[src_idx].to(torch.int32)
        dst_alloc_lens = slot_state.kv_allocated_lens[dst_idx].to(torch.int32)
        src_seq_lens = slot_state.seq_lens[src_idx].to(torch.int32)

        to_free = batched_resample_kv(
            slot_state.req_to_token_pool.req_to_token,
            slot_state.token_to_kv_pool_allocator.slot_ref_count,
            dst_pool_indices,
            src_pool_indices,
            dst_alloc_lens,
            src_seq_lens,
        )
        if to_free.numel() > 0:
            slot_state.token_to_kv_pool_allocator.free(to_free)

        # Vector-copy the per-slot tensors dst ← src.
        slot_state.seq_lens[dst_idx] = slot_state.seq_lens[src_idx]
        slot_state.kv_allocated_lens[dst_idx] = slot_state.kv_allocated_lens[src_idx]
        slot_state.verified_ids[dst_idx] = slot_state.verified_ids[src_idx]
        slot_state.finished_mask[dst_idx] = slot_state.finished_mask[src_idx]
        slot_state.token_counts[dst_idx] = slot_state.token_counts[src_idx]
        slot_state.all_token_ids[dst_idx] = slot_state.all_token_ids[src_idx]

        # Req-level metadata is Python-only (output_ids list, finished_reason
        # object, etc.) — unavoidable per-copy loop.
        for dst_slot, src_slot in zip(
            dst_idx.tolist(), src_idx.tolist(), strict=True
        ):
            slot_state.copy_req_metadata(dst_slot, src_slot)

        # Resampling can copy finished ancestors into previously active
        # slots, flipping the live set.
        if rebuild_active:
            slot_state.rebuild_active_slots()


class SMCScheduler(Scheduler):
    """Slot-based SMC scheduler.  The decode loop uses ``ScheduleBatchSMC``;
    prefill still goes through upstream ``ScheduleBatch``."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ) -> None:
        super().__init__(
            server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
            pp_rank, attn_cp_rank, moe_dp_rank, dp_rank,
        )

        from smcsd.core.req_state import ScheduleBatchSMC

        # SMCEngine (or core auto-resolution) has sized the req_to_token_pool
        # for G * (N+1) Reqs; back out G = max concurrent user groups.
        n_particles = server_args.smc_n_particles
        self.max_user_groups = self.max_running_requests // (n_particles + 1)

        self.waiting_groups: Deque[SequenceGroup] = deque()
        self.prefill_groups: List[SequenceGroup] = []
        self.running_groups: List[SequenceGroup] = []
        self.slot_state = ScheduleBatchSMC(
            max_num_reqs=self.max_user_groups * n_particles,
            device=self.device,
            gamma_plus_1=server_args.speculative_num_draft_tokens,
            vocab_size=self.model_config.vocab_size,
            max_output_len=server_args.context_length,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            n_particles=n_particles,
        )
        self.coordinator = SMCCoordinator(
            device=self.device,
            resample_threshold=server_args.smc_resample_threshold,
        )
        self.smc_metrics = SMCMetricLogger(
            enabled=getattr(server_args, "smc_metrics", False),
            log_interval=getattr(server_args, "smc_metrics_log_interval", 50),
            jsonl_path=getattr(server_args, "smc_metrics_jsonl", None),
        )

    def _make_runtime_tracking_batch(
        self,
        batch: Optional[object],
    ) -> Optional[ScheduleBatch]:
        if batch is None:
            return None
        if isinstance(batch, ScheduleBatch):
            return batch

        reqs = list(getattr(batch, "reqs", []) or [])
        return ScheduleBatch(
            reqs=reqs,
            forward_mode=getattr(batch, "forward_mode", None),
            return_logprob=getattr(batch, "return_logprob", False),
            batch_is_full=False,
        )

    # ── Worker overrides: use SMC variants ──

    def init_tp_model_worker(self):
        # Construct SMCTpModelWorker so the target model_runner uses
        # SMCRefCountedTokenAllocator instead of TokenToKVPoolAllocator.
        from smcsd.managers.smc_tp_worker import SMCTpModelWorker

        self.tp_worker = SMCTpModelWorker(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            pp_rank=self.pp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
            dp_rank=self.dp_rank,
            nccl_port=self.nccl_port,
        )

    def maybe_init_draft_worker(self):
        from smcsd.core.worker import SMCWorker

        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=self.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.dp_rank,
            attn_cp_rank=self.attn_cp_rank,
            moe_dp_rank=self.moe_dp_rank,
        )
        self.draft_worker = SMCWorker(**draft_worker_kwargs)

    # ── Event Loop ──

    def run_event_loop(self) -> None:
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None
        with self.device_module.StreamContext(self.schedule_stream):
            self._event_loop()

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            batch, batch_kind = self._get_next_batch()
            tracking_batch = self._make_runtime_tracking_batch(batch)
            self.cur_batch = tracking_batch
            self.running_batch = (
                tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
            )

            if batch is not None:
                result = self.run_batch(batch)
                if batch_kind == "prefill":
                    self._process_prefill_result(batch, result)
                else:
                    self._process_decode_result(result)
            else:
                self.self_check_during_idle()

            self.last_batch = tracking_batch
            if hasattr(self, "waiting_queue"):
                self.waiting_queue = []

    # ── Runtime Memory Checks (override base mixin) ──
    #
    # SMC keeps its decode KV slots inside ScheduleBatchSMC, which the base
    # SchedulerRuntimeCheckerMixin doesn't know about.  We override the two
    # idle-path leak checks so slot-held tokens/reqs are folded into the
    # conservation formulas — without leaking SMC concepts into core scheduler
    # code.  Refcount state is already reflected via available_size (a shared
    # page stays out of free_pages until its last refcount drops).
    #
    # self_check_during_busy is intentionally NOT overridden: our event loop
    # never dispatches it (matching the PP / disagg / multiplex loops, which
    # also omit the busy check).  Re-add it here if this loop is ever wired
    # to call self_check_during_busy.

    def _check_radix_cache_memory(self):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        session_held = self._session_held_tokens()
        slot_held = self.slot_state.held_token_count()
        memory_leak = (available_size + evictable_size) != (
            self.max_total_num_tokens - protected_size - session_held - slot_held
        )
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, "
            f"{protected_size=}, {session_held=}, {slot_held=}\n"
        )
        return memory_leak, token_msg

    def _check_req_pool(self):
        from sglang.srt.environ import envs
        from sglang.srt.utils.common import raise_error_or_warn

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self._session_held_req_count()
        slot_req_count = self.slot_state.held_req_count()
        if (
            len(self.req_to_token_pool.free_slots) + session_req_count + slot_req_count
            != req_total_size
        ):
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"slot_held={slot_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    # ── Request Admission ──

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if is_retracted:
            # SMC has no retraction path: particle groups are atomic and
            # cannot be partially retracted, and there is no group-aware
            # re-admission protocol.  ScheduleBatch.retract_decode is also
            # unreachable here (decode runs through ScheduleBatchSMC).
            raise NotImplementedError(
                "SMCScheduler does not support re-admitting retracted reqs."
            )
        if self.disaggregation_mode != DisaggregationMode.NULL:
            raise RuntimeError("SMCScheduler only supports non-disaggregated generation.")
        if not self._set_or_validate_priority(req):
            return
        if self._abort_on_queue_limit(req):
            return
        error_msg = validate_smc_parent_req(req)
        if error_msg is not None:
            self._emit_abort(req, error_msg)
            return
        group = SequenceGroup(
            parent_req=req,
            n_particles=self.server_args.smc_n_particles,
            particle_temperature=self.server_args.smc_draft_temperature,
        )
        self.waiting_groups.append(group)
        req.time_stats.set_wait_queue_entry_time()

    def _abort_on_queue_limit(self, req: Req) -> bool:
        if (
            self.max_queued_requests is None
            or len(self.waiting_groups) + 1 <= self.max_queued_requests
        ):
            return False
        self._emit_abort(req, "The request queue is full.")
        return True

    def _emit_abort(self, req: Req, error_msg: str) -> None:
        req.set_finish_with_abort(error_msg)
        req.check_finished()
        req.time_stats.set_completion_time()
        self.stream_output([req], False)

    # ── Batch Selection ──

    def _get_next_batch(self) -> Tuple[Optional[ScheduleBatch], Optional[str]]:
        self._drain_finished_groups()

        if self.prefill_groups:
            raise RuntimeError("SMCScheduler has an unprocessed prefill batch.")

        self.prefill_groups = self._admit_prefill_groups()
        if self.prefill_groups:
            batch = self._build_prefill_batch(self.prefill_groups)
            if batch is None:
                self.prefill_groups = []
            else:
                set_schedule_time_batch(batch)
                return batch, "prefill"

        if not self.slot_state.is_empty():
            batch = self._prepare_decode_batch()
            if batch is not None:
                return batch, "decode"

        return None, None

    def _admit_prefill_groups(self) -> List[SequenceGroup]:
        admitted: List[SequenceGroup] = []
        remaining_capacity = self.slot_state.available_slot_count()

        while self.waiting_groups:
            group = self.waiting_groups[0]
            group_size = group.n_particles
            if group_size > remaining_capacity:
                break
            admitted.append(self.waiting_groups.popleft())
            remaining_capacity -= group_size
            if remaining_capacity <= 0:
                break
        return admitted

    # ── Prefill (uses ScheduleBatch) ──

    def _build_prefill_batch(
        self, groups: List[SequenceGroup]
    ) -> Optional[ScheduleBatch]:
        parent_reqs: List[Req] = []
        for group in groups:
            if group.has_materialized_particles():
                raise RuntimeError(
                    f"Group {group.group_id} entered prefill after particle materialization."
                )
            _prepare_req_for_private_prefill(group.parent_req)
            parent_reqs.append(group.parent_req)

        if not parent_reqs:
            return None

        batch = ScheduleBatch.init_new(
            parent_reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        batch.prepare_for_extend()
        return batch

    def _process_prefill_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        groups = self.prefill_groups
        self.prefill_groups = []
        if not groups:
            raise RuntimeError("Prefill result without active prefill group.")

        if result.copy_done is not None:
            result.copy_done.synchronize()

        next_token_ids = result.next_token_ids.tolist()
        assert len(next_token_ids) == len(batch.reqs) == len(groups)

        for group, req, next_token_id in zip(groups, batch.reqs, next_token_ids):
            assert req is group.parent_req

            req.output_ids.append(next_token_id)
            req.check_finished()

            if req.finished():
                release_kv_cache(req, self.tree_cache)
                req.time_stats.set_completion_time()
                self.stream_output([req], False)
                continue

            error_msg = self._materialize_group(group)
            if error_msg is not None:
                self._abort_group(group, error_msg)
                continue

            self.running_groups.append(group)

    def _materialize_group(self, group: SequenceGroup) -> Optional[str]:
        parent_req = group.parent_req
        try:
            self.model_worker.materialize_smc_parent_draft_prefix(parent_req)
        except Exception as exc:
            return f"SMC parent draft prefill failed: {exc}"

        group.materialize_particles()
        particle_reqs = list(group.particle_reqs.values())
        if self.req_to_token_pool.alloc(particle_reqs) is None:
            group.clear_particles()
            return "SMC particle allocation failed: req_to_token_pool full."

        shared_seq_len = compute_smc_shared_prefix_len(parent_req)

        try:
            for particle_req in particle_reqs:
                copy_block_table(
                    self.req_to_token_pool,
                    parent_req.req_pool_idx,
                    particle_req.req_pool_idx,
                    shared_seq_len,
                    self.token_to_kv_pool_allocator,
                )
                particle_req.kv_committed_len = shared_seq_len
                particle_req.kv_allocated_len = shared_seq_len
                particle_req.prefix_indices = self.req_to_token_pool.req_to_token[
                    particle_req.req_pool_idx, :shared_seq_len
                ].to(dtype=torch.int64, copy=True)
                particle_req.cache_protected_len = shared_seq_len
        except Exception as exc:
            for particle_req in particle_reqs:
                _release_internal_req(
                    particle_req,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                )
            group.clear_particles()
            return f"SMC bootstrap KV fanout failed: {exc}"

        _release_smc_parent_req(
            parent_req,
            tree_cache=self.tree_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        # Populate slot state
        try:
            self.slot_state.allocate_slots(
                group_id=group.group_id,
                particle_reqs=particle_reqs,
                shared_seq_len=shared_seq_len,
            )
        except Exception as exc:
            for particle_req in particle_reqs:
                _release_internal_req(
                    particle_req,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                )
            group.clear_particles()
            return f"SMC slot allocation failed: {exc}"
        return None

    def _abort_group(self, group: SequenceGroup, error_msg: str) -> None:
        parent_req = group.parent_req
        parent_req.finished_reason = FINISH_ABORT(error_msg)
        parent_req.finished_len = len(parent_req.output_ids)
        if group.has_materialized_particles():
            for req in group.particle_reqs.values():
                _release_internal_req(
                    req,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                )
            group.clear_particles()
        if parent_req.req_pool_idx is not None:
            release_kv_cache(parent_req, self.tree_cache)
        parent_req.time_stats.set_completion_time()
        self.stream_output([parent_req], False)

    # ── Decode (slot-based, no ScheduleGroupBatch) ──

    def _prepare_decode_batch(self):
        """Prepare decode via slot state. Returns ModelWorkerBatch or None."""
        draft_input = self.slot_state.prepare_for_decode()
        if draft_input.decode_ctx is None:
            return None
        return self.slot_state.build_model_worker_batch(draft_input)

    def _process_decode_result(self, result: GenerationBatchResult) -> None:
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if result.logprob_diff is None:
            raise RuntimeError("SMCScheduler requires batched logprob_diff.")

        logprob_diff = (
            result.logprob_diff
            if torch.is_tensor(result.logprob_diff)
            else torch.as_tensor(
                result.logprob_diff, dtype=torch.float32, device=self.device
            )
        )

        # Extract bonus_ids from the result's next_draft_input
        next_draft = result.next_draft_input
        bonus_ids = next_draft.verified_id if next_draft is not None else None
        if bonus_ids is None:
            raise RuntimeError("SMCScheduler: result missing next_draft_input.verified_id")

        # Write results back to slot state (defer rebuild to end of cycle)
        newly_finished = self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
        )

        # Snapshot diagnostics before fused collect mutates resampled rows'
        # interval/cumulative weights.
        metrics_snapshot = self.smc_metrics.snapshot_decode_step(self.slot_state)

        # One fused collect over every in-use group row, then dispatch the
        # resulting dst/src plan.  The kernel gates on row_in_use, so we
        # don't need to filter the group list on the Python side.
        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        did_resample = plan.n_jobs > 0
        self.smc_metrics.record_decode_step(metrics_snapshot, plan, did_resample)
        if did_resample:
            self.coordinator.dispatch_resample_batch(
                plan, self.slot_state, rebuild_active=False,
            )

        # Single rebuild per decode cycle if membership changed.
        if newly_finished or did_resample:
            self.slot_state.rebuild_active_slots()

        # Drain finished groups
        self._drain_finished_groups()

    def _drain_finished_groups(self) -> None:
        remaining: List[SequenceGroup] = []
        for group in self.running_groups:
            if self.slot_state.group_has_active(group.group_id):
                remaining.append(group)
                continue
            self._finalize_group(group)
        self.running_groups = remaining

    def _finalize_group(self, group: SequenceGroup) -> None:
        if not group.has_materialized_particles():
            # Shouldn't happen in normal operation — handle gracefully
            parent_req = group.parent_req
            release_kv_cache(parent_req, self.tree_cache)
            parent_req.time_stats.set_completion_time()
            self.stream_output([parent_req], False)
            return

        parent_req = self.slot_state.finalize_group(group.group_id, group.parent_req)
        parent_req.time_stats.set_completion_time()
        self.stream_output([parent_req], False)


def run_smc_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    dp_rank = configure_scheduler(
        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    try:
        scheduler = SMCScheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"SMCScheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

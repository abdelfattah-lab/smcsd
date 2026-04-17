from __future__ import annotations

import copy
import logging
import signal
import time
from collections import Counter, deque
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
    multinomial_resample,
    normalize_log_weights,
    should_resample,
    systematic_resample,
    validate_smc_parent_req,
)
from smcsd.mem_cache.allocator import copy_block_table
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import DynamicGradMode, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@dataclass
class ResampleJobSet:
    """Per-group dst/src pairs produced by the slow path."""

    group_id: str
    dst_slots: List[int]
    src_slots: List[int]


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
    parent_req: Req
    n_particles: int
    particle_temperature: float
    particle_reqs: Dict[int, Req] = field(default_factory=dict)
    log_weights: Optional[torch.Tensor] = None
    interval_log_weights: Optional[torch.Tensor] = None

    @property
    def group_id(self) -> str:
        return self.parent_req.rid

    def has_materialized_particles(self) -> bool:
        return bool(self.particle_reqs)

    def active_particle_indices(self) -> List[int]:
        return [
            idx for idx, req in self.particle_reqs.items() if not req.finished()
        ]

    def active_particle_reqs(self) -> List[Req]:
        return [
            self.particle_reqs[idx] for idx in sorted(self.active_particle_indices())
        ]

    def active_slot_keys(self) -> List[Tuple[str, int]]:
        return [(self.group_id, idx) for idx in self.active_particle_indices()]

    def all_particle_indices(self) -> List[int]:
        return sorted(self.particle_reqs)

    def materialize_particles(
        self,
        *,
        device: torch.device | str,
    ) -> None:
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
        self.log_weights = torch.zeros(
            self.n_particles,
            dtype=torch.float64,
            device=device,
        )
        self.interval_log_weights = torch.zeros_like(self.log_weights)

    def clear_particles(self) -> None:
        self.particle_reqs = {}
        self.log_weights = None
        self.interval_log_weights = None


### V1b dedicated scheduler classes (ScheduleGroupBatch, SMCCoordinator, SMCScheduler,
### run_smc_scheduler_process) have been removed. See git history for the old code.
### The V1a path (Engine() with SMCManager in scheduler.py) is unchanged.

class SMCCoordinatorV2:
    """SMC resample coordinator with two paths:

    1. **slow path** (`fast_resample=False`, default): per-group Python
       normalize + ESS + systematic/multinomial resample + Counter-based
       dst/src pairing + per-slot `resample_copy_slot`.  Zero fusion — the
       reference against which the fast path is validated.
    2. **fast path** (`fast_resample=True`): one fused Triton kernel (see
       `fused_collect_kernel.py`) emits flat `dst/src/row_of_job` tensors
       consumed directly by `batched_resample_kv` (fused KV copy) plus a
       per-pair `copy_req_metadata` Python loop (inherent unavoidable cost).
       Requires systematic resampling + CUDA.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        resample_threshold: float,
        resample_method: str,
        fast_resample: bool = False,
    ) -> None:
        self.device = device
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method
        self.fast_resample = fast_resample
        self._fast_step_counter = 0
        if self.fast_resample:
            if resample_method != "systematic":
                raise ValueError(
                    "fast_resample requires --smc-resample-method=systematic"
                )
            if torch.device(device).type != "cuda":
                raise ValueError("fast_resample requires CUDA")
        logger.warning(
            "SMCCoordinatorV2: resample_method=%s fast_resample=%s",
            resample_method,
            fast_resample,
        )

    # ── Public API ──────────────────────────────────────────

    def collect_resample_jobs_batch(
        self,
        group_ids: List[str],
        slot_state: "ScheduleBatchSMC",
    ):
        """Collect resample jobs for all active groups.

        Returns a `BatchedResampleResult` (fast path) or a plain
        `List[ResampleJobSet]` (slow path).  `dispatch_resample_batch`
        accepts both.
        """
        if self.fast_resample:
            return self._collect_fast(slot_state)
        return self._collect_slow(group_ids, slot_state)

    def dispatch_resample_batch(
        self,
        plan,
        slot_state: "ScheduleBatchSMC",
        *,
        rebuild_active: bool = True,
    ) -> None:
        """Dispatch a collect plan.  Fast path → fused KV copy; slow path →
        per-slot `resample_copy_slot`.  No-op on empty plans.
        """
        if isinstance(plan, list):
            if not plan:
                return
            self._dispatch_slow(plan, slot_state)
        else:
            # BatchedResampleResult
            if plan.n_jobs == 0:
                return
            self._dispatch_fast(plan, slot_state)

        # Resampling can copy finished ancestors into previously active slots.
        # Caller may defer rebuild to batch with other membership changes.
        if rebuild_active:
            slot_state.rebuild_active_slots()

    # ── Slow path (golden truth) ────────────────────────────

    def _collect_slow(
        self,
        group_ids: List[str],
        slot_state: "ScheduleBatchSMC",
    ) -> List[ResampleJobSet]:
        """Per-group Python collection.  Preserves legacy semantics
        (includes all particles, relies on finished_mask propagation via
        `resample_copy_slot`).
        """
        jobs: List[ResampleJobSet] = []
        for group_id in group_ids:
            job = self._collect_one(group_id, slot_state)
            if job is not None:
                jobs.append(job)
        return jobs

    def _collect_one(
        self,
        group_id: str,
        slot_state: "ScheduleBatchSMC",
    ) -> Optional[ResampleJobSet]:
        """Single-group Python collect.  Preserves legacy behaviour: all
        particles (including finished) participate; `finished_mask` is
        propagated downstream via `resample_copy_slot`.
        """
        iw = slot_state.group_interval_weights.get(group_id)
        if iw is None:
            return None
        slots = slot_state.group_slot_lists.get(group_id, [])
        if len(slots) <= 1:
            return None

        pidx_gpu = slot_state.particle_indices[slots]
        resample_pidxs = pidx_gpu.tolist()
        pidx_to_slot = dict(zip(resample_pidxs, slots))

        normalized_weights = normalize_log_weights(
            iw[resample_pidxs], device=self.device,
        )
        if not should_resample(
            normalized_weights,
            len(resample_pidxs),
            self.resample_threshold,
            device=self.device,
        ):
            return None

        if self.resample_method == "multinomial":
            ancestors_t = multinomial_resample(normalized_weights, device=self.device)
        else:
            ancestors_t = systematic_resample(normalized_weights, device=self.device)

        # Reset cumulative weights (writes through the group_log_weights view
        # into the stacked storage).
        pidx_t = pidx_gpu.to(torch.int64)
        slot_state.group_log_weights[group_id][pidx_t] = 0.0
        iw[pidx_t] = 0.0

        ancestor_pidxs = [resample_pidxs[idx] for idx in ancestors_t.tolist()]

        keep_counts: Counter[int] = Counter()
        target_counts = Counter(ancestor_pidxs)
        dst_pidxs: List[int] = []
        src_pidxs: List[int] = []
        for pidx in resample_pidxs:
            if keep_counts[pidx] < target_counts[pidx]:
                keep_counts[pidx] += 1
                continue
            dst_pidxs.append(pidx)
        for pidx in resample_pidxs:
            remaining = target_counts[pidx] - keep_counts[pidx]
            if remaining > 0:
                src_pidxs.extend([pidx] * remaining)
        if not dst_pidxs:
            return None

        dst_slots = [pidx_to_slot[p] for p in dst_pidxs]
        src_slots = [pidx_to_slot[p] for p in src_pidxs]
        return ResampleJobSet(
            group_id=group_id, dst_slots=dst_slots, src_slots=src_slots,
        )

    def _dispatch_slow(
        self,
        jobs: List[ResampleJobSet],
        slot_state: "ScheduleBatchSMC",
    ) -> None:
        """No-fusion dispatch: per-slot `resample_copy_slot` inside one
        free_group_begin/end wrapper."""
        if __debug__:
            all_dst: set = set()
            all_src: set = set()
            for job in jobs:
                all_dst.update(job.dst_slots)
                all_src.update(job.src_slots)
            assert all_dst.isdisjoint(all_src), (
                "Cross-group dst/src slot overlap detected (slow path)"
            )
        slot_state.token_to_kv_pool_allocator.free_group_begin()
        try:
            for job in jobs:
                for dst_slot, src_slot in zip(job.dst_slots, job.src_slots):
                    slot_state.resample_copy_slot(dst_slot, src_slot)
        finally:
            slot_state.token_to_kv_pool_allocator.free_group_end()

    # ── Fast path ───────────────────────────────────────────

    def _collect_fast(self, slot_state: "ScheduleBatchSMC"):
        """One fused kernel launch.  Returns `BatchedResampleResult`.

        Seeds are drawn inside the kernel via Philox (`tl.rand`), keyed by a
        monotonic host-side step counter so consecutive calls produce
        independent stratified u ~ U[0,1/n_active) values without any
        per-step host allocation or device sync.
        """
        from smcsd.v2.kernels.fused_collect import batched_collect_fused

        stacked = slot_state.stacked
        self._fast_step_counter += 1
        return batched_collect_fused(
            stacked,
            self.resample_threshold,
            step_counter=self._fast_step_counter,
            scratch_dst=stacked.scratch_dst_flat,
            scratch_src=stacked.scratch_src_flat,
            scratch_row=stacked.scratch_row_of_job,
            scratch_counter=stacked.scratch_counter,
            scratch_mask=stacked.scratch_resample_mask,
        )

    def _dispatch_fast(self, plan, slot_state: "ScheduleBatchSMC") -> None:
        """Fused KV copy + vectorised slot-tensor copies + req metadata loop."""
        from smcsd.v2.kernels.fused_resample_kv import batched_resample_kv

        dst_idx = plan.dst_slots.to(torch.int64)
        src_idx = plan.src_slots.to(torch.int64)

        if __debug__:
            assert not torch.isin(dst_idx, src_idx).any().item(), (
                "Cross-group dst/src slot overlap detected (fast path)"
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

        # Batch-copy slot tensors on device.
        slot_state.seq_lens[dst_idx] = slot_state.seq_lens[src_idx]
        slot_state.kv_allocated_lens[dst_idx] = slot_state.kv_allocated_lens[src_idx]
        slot_state.verified_ids[dst_idx] = slot_state.verified_ids[src_idx]
        slot_state.finished_mask[dst_idx] = slot_state.finished_mask[src_idx]
        slot_state.token_counts[dst_idx] = slot_state.token_counts[src_idx]
        slot_state.all_token_ids[dst_idx] = slot_state.all_token_ids[src_idx]

        # Req metadata loop — accepted unavoidable Python-side cost.
        for dst_slot, src_slot in zip(
            dst_idx.tolist(), src_idx.tolist(), strict=True
        ):
            slot_state.copy_req_metadata(dst_slot, src_slot)


class SMCSchedulerV2(Scheduler):
    """Slot-based SMC scheduler. Decode loop uses ScheduleBatchSMC instead of
    ScheduleGroupBatch. Prefill still uses ScheduleBatch (upstream code).

    Coexists with SMCScheduler — switch via run_smc_scheduler_v2_process.
    """

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

        from smcsd.v2.req_state import ScheduleBatchSMC

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
        self.coordinator = SMCCoordinatorV2(
            device=self.device,
            resample_threshold=server_args.smc_resample_threshold,
            resample_method=server_args.smc_resample_method,
            fast_resample=server_args.smc_fast_resample,
        )
        self._group_idx_counter = 0

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
        from smcsd.v2.worker import SMCWorkerV2

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
        self.draft_worker = SMCWorkerV2(**draft_worker_kwargs)

    # ── Event Loop ──

    def run_event_loop(self) -> None:
        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None
        with self.device_module.StreamContext(self.schedule_stream):
            self._event_loop_v2()

    @DynamicGradMode()
    def _event_loop_v2(self) -> None:
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
    # SMC v2 keeps its decode KV slots inside ScheduleBatchSMC, which the base
    # SchedulerRuntimeCheckerMixin doesn't know about.  We override the two
    # idle-path leak checks so slot-held tokens/reqs are folded into the
    # conservation formulas — without leaking SMC concepts into core scheduler
    # code.  Refcount state is already reflected via available_size (a shared
    # page stays out of free_pages until its last refcount drops).
    #
    # self_check_during_busy is intentionally NOT overridden: _event_loop_v2
    # never dispatches it (matching the PP / disagg / multiplex loops, which
    # also omit the busy check).  Re-add it here if the v2 loop is ever wired
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
            # SMC v2 has no retraction path: particle groups are atomic and
            # cannot be partially retracted, and there is no group-aware
            # re-admission protocol.  ScheduleBatch.retract_decode is also
            # unreachable in v2 (decode runs through ScheduleBatchSMC).
            raise NotImplementedError(
                "SMCSchedulerV2 does not support re-admitting retracted reqs."
            )
        if self.disaggregation_mode != DisaggregationMode.NULL:
            raise RuntimeError("SMCSchedulerV2 only supports non-disaggregated generation.")
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
            raise RuntimeError("SMCSchedulerV2 has an unprocessed prefill batch.")

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

        group.materialize_particles(device=self.device)
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
        group_idx = self._group_idx_counter
        self._group_idx_counter += 1
        try:
            self.slot_state.allocate_slots(
                group_id=group.group_id,
                group_idx=group_idx,
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
            raise RuntimeError("SMCSchedulerV2 requires batched logprob_diff.")

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
            raise RuntimeError("SMCSchedulerV2: result missing next_draft_input.verified_id")

        # Write results back to slot state (defer rebuild to end of cycle)
        newly_finished = self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
        )

        # Resample all groups.  Either the slow path (List[ResampleJobSet],
        # golden truth) or the fast path (BatchedResampleResult, one fused
        # kernel) depending on `server_args.smc_fast_resample`.
        active_group_ids = [
            group.group_id
            for group in self.running_groups
            if self.slot_state.group_has_active(group.group_id)
        ]
        plan = self.coordinator.collect_resample_jobs_batch(
            active_group_ids, self.slot_state
        )
        did_resample = (
            bool(plan) if isinstance(plan, list) else plan.n_jobs > 0
        )
        if did_resample:
            self.coordinator.dispatch_resample_batch(
                plan, self.slot_state, rebuild_active=False,
            )

        # Single rebuild per decode cycle if membership changed
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
            # Shouldn't happen in v2 — but handle gracefully
            parent_req = group.parent_req
            release_kv_cache(parent_req, self.tree_cache)
            parent_req.time_stats.set_completion_time()
            self.stream_output([parent_req], False)
            return

        parent_req = self.slot_state.finalize_group(group.group_id, group.parent_req)
        parent_req.time_stats.set_completion_time()
        self.stream_output([parent_req], False)


def run_smc_scheduler_v2_process(
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
        scheduler = SMCSchedulerV2(
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
        logger.error(f"SMCSchedulerV2 hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

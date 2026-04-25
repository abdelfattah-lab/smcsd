from __future__ import annotations

import logging
import signal
from collections import deque
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
        if slot_state.eagle_hidden_states is not None:
            slot_state.eagle_hidden_states[dst_idx] = (
                slot_state.eagle_hidden_states[src_idx]
            )
        if slot_state.eagle_first_draft_token_ids is not None:
            slot_state.eagle_first_draft_token_ids[dst_idx] = (
                slot_state.eagle_first_draft_token_ids[src_idx]
            )
        if slot_state.eagle_first_draft_logprobs is not None:
            slot_state.eagle_first_draft_logprobs[dst_idx] = (
                slot_state.eagle_first_draft_logprobs[src_idx]
            )
        if slot_state.eagle_topk_p is not None:
            slot_state.eagle_topk_p[dst_idx] = slot_state.eagle_topk_p[src_idx]
        if slot_state.eagle_topk_index is not None:
            slot_state.eagle_topk_index[dst_idx] = slot_state.eagle_topk_index[src_idx]

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
            gamma_plus_1=server_args.smc_gamma + 1,
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

        next_draft = result.next_draft_input
        e3_hidden = getattr(next_draft, "hidden_state", None) if next_draft else None
        e3_x1 = (
            getattr(next_draft, "first_draft_token_id", None) if next_draft else None
        )
        e3_x1_lp = (
            getattr(next_draft, "first_draft_logprob", None) if next_draft else None
        )
        e3_topk_p = getattr(next_draft, "eagle_topk_p", None) if next_draft else None
        e3_topk_i = (
            getattr(next_draft, "eagle_topk_index", None) if next_draft else None
        )

        for i, (group, req, next_token_id) in enumerate(
            zip(groups, batch.reqs, next_token_ids)
        ):
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

            if e3_hidden is not None:
                slots = self.slot_state.group_slot_lists[group.group_id]
                self.slot_state.scatter_prefill_eagle3(
                    slots,
                    hidden_state=e3_hidden[i],
                    first_draft_token_id=e3_x1[i : i + 1] if e3_x1 is not None else None,
                    first_draft_logprob=(
                        e3_x1_lp[i : i + 1] if e3_x1_lp is not None else None
                    ),
                    eagle_topk_p=e3_topk_p[i] if e3_topk_p is not None else None,
                    eagle_topk_index=e3_topk_i[i] if e3_topk_i is not None else None,
                )

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

        next_hidden = getattr(next_draft, "hidden_state", None)
        next_x1 = getattr(next_draft, "first_draft_token_id", None)
        next_x1_lp = getattr(next_draft, "first_draft_logprob", None)
        next_topk_p = getattr(next_draft, "eagle_topk_p", None)
        next_topk_i = getattr(next_draft, "eagle_topk_index", None)

        # Write results back to slot state (defer rebuild to end of cycle)
        newly_finished = self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            next_hidden_state=next_hidden,
            next_first_draft_token_id=next_x1,
            next_first_draft_logprob=next_x1_lp,
            next_eagle_topk_p=next_topk_p,
            next_eagle_topk_index=next_topk_i,
            rebuild_active=False,
        )

        # One fused collect over every in-use group row, then dispatch the
        # resulting dst/src plan.  The kernel gates on row_in_use, so we
        # don't need to filter the group list on the Python side.
        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        did_resample = plan.n_jobs > 0
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

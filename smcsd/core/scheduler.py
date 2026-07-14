from __future__ import annotations

import logging
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import psutil
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler, configure_scheduler_process
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
    copy_smc_resampled_hybrid_state,
    fanout_smc_parent_hybrid_state,
    validate_smc_parent_req,
)
from smcsd.core.info import SMCParticleOutput
from smcsd.mem_cache.allocator import copy_block_table
from sglang.srt.utils import DynamicGradMode
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
    """SMC resample coordinator (fused systematic kernel only).

    One fused Triton kernel per decode step:

    1. ``collect`` — for every in-use group, normalise interval weights, check
       ESS against ``threshold * N``, and (if below threshold) run systematic
       resampling, emitting a device-resident flat ``dst/src/row`` plan.
    2. ``dispatch`` — one device-driven ``batched_resample_kv`` launch
       (worst-case grid, gated on-device by ``plan.counter``) that applies
       the KV block-table copy + refcounts AND all per-slot lineage-tensor
       copies.  All decode-time particle state is tensor-resident, so no
       Req-level metadata is touched and no host sync occurs.

    Only the fused systematic path is supported.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        resample_threshold: float,
        resample_method: str,
    ) -> None:
        if resample_method != "systematic":
            raise ValueError(
                f"smc_resample_method={resample_method!r} is not supported; "
                "only 'systematic' is currently implemented."
            )
        if torch.device(device).type != "cuda":
            raise ValueError("SMCCoordinator requires CUDA")
        self.device = device
        self.resample_threshold = resample_threshold
        self.resample_method = resample_method
        self._fast_step_counter = 0
        logger.info(
            "SMCCoordinator: resample_method=%s (fused systematic kernel)",
            resample_method,
        )

    # ── Public API ──────────────────────────────────────────

    def collect_resample_jobs_batch(self, slot_state: "ScheduleBatchSMC"):
        """One fused-kernel launch over all in-use group rows.

        Returns a ``BatchedResampleResult`` consumed by
        ``dispatch_resample_batch``.
        """
        from smcsd.core.kernels.fused_collect import batched_collect_fused

        self._fast_step_counter += 1
        return batched_collect_fused(
            slot_state.log_weights,
            slot_state.interval_weights,
            slot_state.group_to_slots,
            slot_state.row_in_use,
            self.resample_threshold,
            step_counter=self._fast_step_counter,
        )

    def dispatch_resample_batch(
        self,
        plan,
        slot_state: "ScheduleBatchSMC",
    ) -> None:
        """Apply a resample plan in one device-driven kernel launch — no
        host sync.

        The grid is the host-known worst case (``live_rows × (N-1)``, from
        CPU bookkeeping); each program reads the true job count from
        ``plan.counter`` and exits early, so an empty plan costs one no-op
        launch instead of a blocking ``.item()``.  The kernel performs the
        KV block-table copy + refcounts AND every per-slot lineage-tensor
        copy (finish state included), gathering pool rows / lengths
        in-kernel from the slot tensors.

        KV pages whose refcount hits zero are appended to the current
        snapshot phase's ``slot_state.kv_freed_buf`` row; the scheduler
        frees them in postprocessing (which under overlap runs after the
        NEXT step's dispatch has been enqueued — hence the double buffer).
        Deferral is safe: a refcount-0 page is unreachable from every
        block table but stays out of the allocator's free pool until
        ``free()`` runs, so it cannot be re-allocated in between.
        """
        # Capacity bound: every group row resamples and keeps one survivor
        # (counts sum to N, so dead slots <= N-1 per row).  Deliberately
        # NOT the live-row count: a static grid never changes between
        # steps, which keeps this launch CUDA-graph-capturable.  Free rows
        # are gated off by row_in_use in collect, so they emit no jobs and
        # the extra programs exit on the counter load.
        max_jobs = slot_state.max_groups * (slot_state.n_particles - 1)
        if max_jobs == 0:  # N == 1: resampling is structurally impossible
            return

        from smcsd.core.kernels.fused_resample_kv import batched_resample_kv

        batched_resample_kv(
            slot_state.req_to_token_pool.req_to_token,
            slot_state.token_to_kv_pool_allocator.slot_ref_count,
            plan_dst=plan.dst_flat,
            plan_src=plan.src_flat,
            plan_counter=plan.counter,
            max_jobs=max_jobs,
            req_pool_indices=slot_state.req_pool_indices,
            kv_allocated_lens=slot_state.kv_allocated_lens,
            seq_lens=slot_state.seq_lens,
            verified_ids=slot_state.verified_ids,
            prev_last_draft_ids=slot_state.prev_last_draft_ids,
            finished_mask=slot_state.finished_mask,
            finished_len=slot_state.finished_len,
            finish_reason_code=slot_state.finish_reason_code,
            matched_eos_token=slot_state.matched_eos_token,
            token_counts=slot_state.token_counts,
            all_token_ids=slot_state.all_token_ids,
            freed_buf=slot_state.kv_freed_buf[slot_state._snap_phase],
            freed_counter=slot_state.kv_freed_counter[slot_state._snap_phase],
        )


class SMCScheduler(Scheduler):
    """Slot-based SMC scheduler. Decode loop uses ScheduleBatchSMC instead of
    ScheduleGroupBatch. Prefill still uses ScheduleBatch (upstream code).

    Coexists with SMCScheduler — switch via run_smc_scheduler_process.
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

        from smcsd.core.req_state import ScheduleBatchSMC

        # SMCEngine (or core auto-resolution) has sized the req_to_token_pool
        # for G * (N+1) Reqs; back out G = max concurrent user groups.
        n_particles = server_args.smc_n_particles
        self.max_user_groups = self.max_running_requests // (n_particles + 1)

        self.waiting_groups: Deque[SequenceGroup] = deque()
        self.prefill_groups: List[SequenceGroup] = []
        self.running_groups: List[SequenceGroup] = []
        # Slots reserved by admission but not yet claimed by allocate_slots.
        # allocate_slots runs in prefill POSTPROCESSING, which the overlap
        # loop defers by one iteration — without this reservation the next
        # _admit_prefill_groups reads stale free_slots and over-admits,
        # violating max_running_requests (observed: rr=1 running every
        # queued group concurrently, blowing decode past the captured
        # cuda-graph buckets).
        self._pending_admitted_slots = 0
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
            resample_method=server_args.smc_resample_method,
        )

        # Resolution order: SMC_ENABLE_OVERLAP env (kill switch) >
        # SMCEngine kwarg (server_args attr) > default ON.  The hybrid
        # (Mamba) gate is gone: the recurrent-state resample copy is now a
        # device-driven fused kernel enqueued inside _resample (before the
        # snapshot), so it is stream-ordered ahead of the next forward.
        _ov_env = os.environ.get("SMC_ENABLE_OVERLAP")
        want_overlap = (
            bool(int(_ov_env))
            if _ov_env is not None
            else bool(getattr(server_args, "smc_enable_overlap", True))
        )
        self._use_overlap_loop = want_overlap
        if self._use_overlap_loop:
            logger.info("SMCScheduler: overlapped scheduling enabled.")

        # Debug instrumentation (scheduler process — the one that owns the
        # CUDA context, so torch.cuda.* must be queried here, not from the
        # engine/profiling script process):
        #   SMC_SYNC_DEBUG=1        warn (with stack) on every syncing CUDA op
        #   SMC_LOG_ALLOC_RETRIES=1 log whenever the caching allocator hits
        #                           its synchronize-and-retry slow path
        if bool(int(os.environ.get("SMC_SYNC_DEBUG", "0"))):
            torch.cuda.set_sync_debug_mode("warn")
            logger.info("SMCScheduler: CUDA sync debug mode = warn")
        self._log_alloc_retries = bool(
            int(os.environ.get("SMC_LOG_ALLOC_RETRIES", "0"))
        )
        self._last_alloc_retries = 0

    def _maybe_log_alloc_retries(self) -> None:
        """Log when the CUDA caching allocator hit cudaMalloc failure and
        synchronized to retry — the serialization mechanism that masquerades
        as a sync in whatever op allocated next."""
        if not self._log_alloc_retries:
            return
        retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        if retries != self._last_alloc_retries:
            logger.warning(
                "CUDA caching allocator sync-retry: num_alloc_retries "
                "%d -> %d (reserved=%.0fMB allocated=%.0fMB)",
                self._last_alloc_retries,
                retries,
                torch.cuda.memory_reserved() / 1e6,
                torch.cuda.memory_allocated() / 1e6,
            )
            self._last_alloc_retries = retries

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
        # Upstream's Scheduler.maybe_init_draft_worker initializes
        # external_corpus_manager (used only by ngram speculative decoding).
        # Our override replaces the body, so we must set it ourselves.
        self.external_corpus_manager = None
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
            if self._use_overlap_loop:
                self._event_loop_overlap()
            else:
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
                    self._process_prefill_result(
                        batch, result, self._take_prefill_groups()
                    )
                else:
                    # GPU-side step first (write-back + fused resample,
                    # enqueued behind the decode forward), then host-side
                    # postprocessing (the sync quarantine).
                    plan, snapshot = self._resample(result)
                    self._process_decode_result(result, plan, snapshot)
            else:
                # self_check_during_idle was removed in upstream sglang;
                # only self_check_during_busy remains.
                pass

            self.last_batch = tracking_batch
            if hasattr(self, "waiting_queue"):
                self.waiting_queue = []

    @DynamicGradMode()
    def _event_loop_overlap(self) -> None:
        """Overlapped scheduler loop: postprocessing of step t runs on the
        CPU while the GPU executes step t+1.

        Per iteration: prepare + launch the next batch (sync-free for
        decode — forward, write-back, collect, dispatch, and the host
        snapshot are all pure enqueues), THEN pop and postprocess the
        previous batch's result.  The snapshot event completes during the
        new batch's forward, so postprocessing never blocks on the stream
        tail.

        One-step-late semantics this accepts:
        * A fully-finished group is drained one step late and receives one
          extra decode step — valid (absorbing states, weight increment 0;
          an extra resample of frozen weights is still a proper SMC step),
          but it shifts RNG consumption vs the sequential loop.
        * KV pages freed at step t re-enter the allocator pool at t's
          postprocessing, i.e. after t+1 was prepared — see the headroom
          check below.
        * Admission capacity decisions are one step stale (conservative).
        """
        result_queue: Deque = deque()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self._flush_result_queue(result_queue)
                self.cancel_bubble_timer()
                continue

            # Headroom: the pending step's freed pages are not yet back in
            # the allocator pool.  If the next decode allocation could need
            # them, settle pending postprocessing first (host-metadata
            # check only — no sync).
            if result_queue:
                need = (
                    self.slot_state.active_particle_count()
                    * self.slot_state.gamma_plus_1
                )
                if self.token_to_kv_pool_allocator.available_size() < need:
                    self._flush_result_queue(result_queue)

            batch, batch_kind = self._get_next_batch()
            tracking_batch = self._make_runtime_tracking_batch(batch)
            self.cur_batch = tracking_batch
            self.running_batch = (
                tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
            )

            if batch is not None:
                result = self.run_batch(batch)
                if batch_kind == "decode":
                    plan, snapshot = self._resample(result)
                    result_queue.append(
                        ("decode", batch, result, plan, snapshot)
                    )
                else:
                    result_queue.append(
                        ("prefill", batch, result, self._take_prefill_groups())
                    )

            # Keep at most one in-flight entry: postprocess everything
            # older than the batch launched above (or everything, when
            # idle).  This is where the CPU work overlaps the GPU.
            pending_limit = 1 if batch is not None else 0
            while len(result_queue) > pending_limit:
                self._process_queued_result(result_queue)

            self._maybe_log_alloc_retries()
            self.last_batch = tracking_batch
            if hasattr(self, "waiting_queue"):
                self.waiting_queue = []

    def _process_queued_result(self, result_queue: Deque) -> None:
        entry = result_queue.popleft()
        if entry[0] == "prefill":
            _, q_batch, q_result, q_groups = entry
            self._process_prefill_result(q_batch, q_result, q_groups)
        else:
            _, _, q_result, q_plan, q_snapshot = entry
            self._process_decode_result(q_result, q_plan, q_snapshot)

    def _flush_result_queue(self, result_queue: Deque) -> None:
        """Settle all pending postprocessing (pause, headroom pressure)."""
        while result_queue:
            self._process_queued_result(result_queue)
        self.last_batch = None

    # ── Runtime Memory Checks (override base mixin) ──
    #
    # SMC keeps its decode KV slots inside ScheduleBatchSMC, which the base
    # SchedulerRuntimeCheckerMixin doesn't know about.  We override the two
    # idle-path leak checks so slot-held tokens/reqs are folded into the
    # conservation formulas — without leaking SMC concepts into core scheduler
    # code.  Refcount state is already reflected via available_size (a shared
    # page stays out of free_pages until its last refcount drops).
    #
    # self_check_during_busy is intentionally NOT overridden: _event_loop
    # never dispatches it (matching the PP / disagg / multiplex loops, which
    # also omit the busy check).

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
        # NOTE: no drain here.  Draining reads finish state, which on the
        # prepare path would be a device read at the stream tail; groups
        # are drained in decode postprocessing (from the host snapshot),
        # which always runs before the next _get_next_batch.

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
        remaining_capacity = (
            self.slot_state.available_slot_count() - self._pending_admitted_slots
        )

        while self.waiting_groups:
            group = self.waiting_groups[0]
            group_size = group.n_particles
            if group_size > remaining_capacity:
                break
            admitted.append(self.waiting_groups.popleft())
            self._pending_admitted_slots += group_size
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

    def _take_prefill_groups(self) -> List[SequenceGroup]:
        """Detach the pending prefill groups from scheduler state.

        Called at launch time so the groups can ride the result queue under
        overlapped scheduling — ``self.prefill_groups`` must be empty again
        before the next ``_get_next_batch`` runs.
        """
        groups = self.prefill_groups
        self.prefill_groups = []
        return groups

    def _process_prefill_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        groups: List[SequenceGroup],
    ) -> None:
        if not groups:
            raise RuntimeError("Prefill result without active prefill group.")

        # `result.copy_done` is always None on the SMC path: the SMC server
        # args force disable_overlap_schedule, so the inherited run_batch
        # never takes the overlap branch that creates it.  The .tolist()
        # below is a synchronous device read — safe without an event.
        next_token_ids = result.next_token_ids.tolist()
        assert len(next_token_ids) == len(batch.reqs) == len(groups)

        for i, (group, req, next_token_id) in enumerate(
            zip(groups, batch.reqs, next_token_ids)
        ):
            assert req is group.parent_req
            # Admission resolves here: whether the group materializes,
            # finishes at prefill, or aborts, true slot accounting
            # (allocate_slots / never-claimed) takes over from the
            # reservation made in _admit_prefill_groups.
            self._pending_admitted_slots -= group.n_particles

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

    def _materialize_group(
        self,
        group: SequenceGroup,
    ) -> Optional[str]:
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
            fanout_smc_parent_hybrid_state(
                target_pool=self.req_to_token_pool,
                draft_pool=getattr(
                    self.model_worker,
                    "_dense_draft_hybrid_req_to_token_pool",
                    None,
                ),
                parent_req=parent_req,
                particle_reqs=particle_reqs,
                device=self.device,
            )
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

    def _resample(self, result: GenerationBatchResult):
        """GPU-side post-decode step: write-back + fused collect/resample.

        Runs right after ``run_batch`` and before postprocessing.  Every op
        here is a GPU tensor op enqueued behind the decode forward — no
        event waits, no Req objects, no host syncs — so under a
        future overlapped scheduler the host can move on to preparing the
        next batch while this work executes.

        Returns ``(plan, snapshot)`` for ``_process_decode_result``: the
        device-resident resample plan, and the host-snapshot handle whose
        event gates postprocessing's pinned-buffer reads (freed-page
        cursor, finished mask).  Freed KV pages accumulate in
        ``slot_state.kv_freed_buf`` (freed there, not here).
        """
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
        # This step's last drafted token, deferred into next step's leading
        # 2-token draft forward.  Carried but not yet consumed (Step 2).
        prev_last_draft_ids = (
            next_draft.prev_last_draft_id if next_draft is not None else None
        )
        # Per-particle bonus-token normalizer log Z, accumulated into the weight
        # alongside logprob_diff (0 at power_alpha=1).
        bonus_logz = next_draft.bonus_logz if next_draft is not None else None

        # GPU write-back: token scatter, finish flags, weight accumulation.
        # Sync-free — finish state lands in slot tensors, not Req objects.
        self.slot_state.write_back_gpu(
            next_token_ids=result.next_token_ids,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            prev_last_draft_ids=prev_last_draft_ids,
            bonus_logz=bonus_logz,
        )

        # Snapshot the per-row log Z_hat increment BEFORE the resample kernel
        # zeroes weights, then fold it into group_log_Z_hat for the rows that
        # actually resample (unbiased-estimator product over resample steps).
        logZ_inc = self.slot_state.resample_logZ_increment()

        # Resample all groups via the fused systematic kernel.
        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        self.slot_state.group_log_Z_hat += torch.where(
            plan.resample_mask, logZ_inc, torch.zeros_like(logZ_inc)
        )
        self.coordinator.dispatch_resample_batch(plan, self.slot_state)

        copy_smc_resampled_hybrid_state(
            target_pool=self.req_to_token_pool,
            draft_pool=getattr(
                self.model_worker,
                "_dense_draft_hybrid_req_to_token_pool",
                None,
            ),
            slot_state=self.slot_state,
            plan=plan,
            device=self.device,
        )

        snapshot = self.slot_state.snapshot_to_host()
        return plan, snapshot

    def _process_decode_result(
        self,
        result: GenerationBatchResult,
        plan,
        snapshot,
    ) -> None:
        """Host-side postprocessing — the sync quarantine.

        Everything that needs a GPU→CPU round trip lands here, after the
        GPU-side ``_resample``.  Device state is read through the pinned
        host snapshot (gated by its event, which completes during the next
        step's forward) — never via ``.item()`` on device tensors, which
        would block on the stream tail under overlapped scheduling.  The
        accepted exceptions: the hybrid Mamba plan slice (sequential loop
        only) and finalize_group's device reads (rare, once per group
        lifetime).
        """
        snapshot.wait()

        # Free the KV pages the resample kernel released into this phase's
        # capture buffer.  Deferred from dispatch: refcount-0 pages are
        # unreachable but not yet in the allocator's free pool, so nothing
        # can re-allocate them in between.  The count comes from pinned
        # memory; the free itself is enqueue-only (host-known shape), and
        # the cursor reset is stream-ordered before this phase's next use.
        n_freed = int(self.slot_state.kv_freed_count_host[snapshot.phase].item())
        if n_freed > 0:
            self.token_to_kv_pool_allocator.free(
                self.slot_state.kv_freed_buf[snapshot.phase, :n_freed].to(
                    torch.int64
                )
            )
            self.slot_state.kv_freed_counter[snapshot.phase].zero_()

        # No rebuild here: neither finishing (absorbing-state semantics) nor
        # resampling changes slot membership — only allocate_slots /
        # free_group_slots do, and both rebuild themselves.

        # Drain finished groups, reading finish state from the pinned
        # snapshot (post-resample lineage of this step).
        self._drain_finished_groups(
            self.slot_state.finished_mask_host[snapshot.phase]
        )

    def _drain_finished_groups(self, finished_mask_host) -> None:
        remaining: List[SequenceGroup] = []
        for group in self.running_groups:
            if self.slot_state.group_has_active(
                group.group_id, finished_mask_host
            ):
                remaining.append(group)
                continue
            self._finalize_group(group)
        self.running_groups = remaining

    def _finalize_group(self, group: SequenceGroup) -> None:
        if not group.has_materialized_particles():
            # Shouldn't happen — but handle gracefully
            parent_req = group.parent_req
            release_kv_cache(parent_req, self.tree_cache)
            parent_req.time_stats.set_completion_time()
            self.stream_output([parent_req], False)
            return

        parent_req = self.slot_state.finalize_group(group.group_id, group.parent_req)
        parent_req.time_stats.set_completion_time()
        # Emit the full particle collection + unbiased log Z_hat on the same
        # scheduler->engine socket, BEFORE the token output.  FIFO delivery
        # guarantees the engine sees this while the rid is still pending (the
        # finish signal rides on the BatchTokenIDOutput that follows).
        #
        # Gated on smc_emit_particle_output (a dynamic ServerArgs attribute
        # set only by the offline SMCEngine, like smc_power_alpha): in HTTP
        # mode this socket feeds a real DetokenizerManager, whose
        # TypeBasedDispatcher raises ValueError on unknown message types —
        # an ungated send kills the detokenizer on the first finalized
        # group.  parent_req.smc_* stay populated either way.
        if getattr(self.server_args, "smc_emit_particle_output", False):
            self.send_to_detokenizer.send_output(
                SMCParticleOutput(
                    rid=parent_req.rid,
                    log_Z_hat=parent_req.smc_log_Z_hat,
                    log_w_tilde=parent_req.smc_log_w_tilde,
                    particle_output_ids=parent_req.smc_particle_output_ids,
                )
            )
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
    # upstream renamed configure_scheduler -> configure_scheduler_process,
    # added gpu_id as the second positional arg, and now calls
    # kill_itself_when_parent_died() internally.
    dp_rank = configure_scheduler_process(
        server_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
    )

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

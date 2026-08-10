from __future__ import annotations

import json
import logging
import os
import signal
import time
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
            pending_suffix_lens=slot_state.pending_draft_suffix_lens,
            target_seq_lens=(
                slot_state.target_seq_lens
                if slot_state.cross_tokenizer_enabled
                else None
            ),
            target_verified_ids=(
                slot_state.target_verified_ids
                if slot_state.cross_tokenizer_enabled
                else None
            ),
            target_token_counts=(
                slot_state.target_token_counts
                if slot_state.cross_tokenizer_enabled
                else None
            ),
            target_all_token_ids=(
                slot_state.target_all_token_ids
                if slot_state.cross_tokenizer_enabled
                else None
            ),
        )
        if (
            slot_state.cross_tokenizer_enabled
            and slot_state.draft_req_to_token_pool is not None
            and slot_state.draft_kv_freed_buf is not None
        ):
            draft_allocator = slot_state.draft_token_to_kv_pool_allocator
            if hasattr(draft_allocator, "slot_ref_count"):
                batched_resample_kv(
                    slot_state.draft_req_to_token_pool.req_to_token,
                    draft_allocator.slot_ref_count,
                    plan_dst=plan.dst_flat,
                    plan_src=plan.src_flat,
                    plan_counter=plan.counter,
                    max_jobs=max_jobs,
                    req_pool_indices=slot_state.draft_req_pool_indices,
                    kv_allocated_lens=slot_state.draft_kv_allocated_lens,
                    seq_lens=slot_state.draft_seq_lens,
                    verified_ids=slot_state.draft_verified_ids,
                    prev_last_draft_ids=slot_state.prev_last_draft_ids,
                    finished_mask=slot_state.finished_mask,
                    finished_len=slot_state.finished_len,
                    finish_reason_code=slot_state.finish_reason_code,
                    matched_eos_token=slot_state.matched_eos_token,
                    token_counts=slot_state.draft_token_counts,
                    all_token_ids=slot_state.draft_all_token_ids,
                    freed_buf=slot_state.draft_kv_freed_buf[slot_state._snap_phase],
                    freed_counter=slot_state.draft_kv_freed_counter[
                        slot_state._snap_phase
                    ],
                )
            else:
                self._dispatch_private_draft_resample_host(plan, slot_state)

    def _dispatch_private_draft_resample_host(self, plan, slot_state) -> None:
        """Correct fallback for private non-refcounted draft KV pools.

        Cross-tokenizer draft workers use their own normal SGLang allocator.
        Those KV pages cannot be shared with refcounts, so resampling must make
        an owning physical copy for every dst <- src lineage copy.
        """
        n_jobs = plan.n_jobs_sync()
        if n_jobs == 0:
            return

        allocator = slot_state.draft_token_to_kv_pool_allocator
        req_to_token = slot_state.draft_req_to_token_pool.req_to_token
        dst_slots = plan.dst_slots.to("cpu").tolist()
        src_slots = plan.src_slots.to("cpu").tolist()

        for dst_slot, src_slot in zip(dst_slots, src_slots):
            dst_slot = int(dst_slot)
            src_slot = int(src_slot)
            dst_pool = int(slot_state.draft_req_pool_indices[dst_slot].item())
            src_pool = int(slot_state.draft_req_pool_indices[src_slot].item())
            dst_alloc = int(slot_state.draft_kv_allocated_lens[dst_slot].item())
            src_alloc = int(slot_state.draft_kv_allocated_lens[src_slot].item())

            if dst_alloc > 0:
                old_dst = req_to_token[dst_pool, :dst_alloc].to(
                    dtype=torch.int64, copy=True
                )
                allocator.free(old_dst)

            if src_alloc > 0:
                src_indices = req_to_token[src_pool, :src_alloc].to(
                    dtype=torch.int64, copy=True
                )
                dst_indices = allocator.alloc(src_alloc)
                if dst_indices is None:
                    raise RuntimeError("draft KV pool full during SMC resample.")
                kv_copy = allocator.get_cpu_copy(src_indices)
                allocator.load_cpu_copy(kv_copy, dst_indices.to(torch.int64))
                req_to_token[
                    dst_pool, :src_alloc
                ] = dst_indices.to(dtype=req_to_token.dtype)

            slot_state.draft_seq_lens[dst_slot] = slot_state.draft_seq_lens[src_slot]
            slot_state.draft_kv_allocated_lens[dst_slot] = (
                slot_state.draft_kv_allocated_lens[src_slot]
            )
            slot_state.draft_verified_ids[dst_slot] = (
                slot_state.draft_verified_ids[src_slot]
            )
            src_count = int(slot_state.draft_token_counts[src_slot].item())
            slot_state.draft_token_counts[dst_slot] = (
                slot_state.draft_token_counts[src_slot]
            )
            if src_count > 0:
                slot_state.draft_all_token_ids[dst_slot, :src_count] = (
                    slot_state.draft_all_token_ids[src_slot, :src_count]
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
        self._smc_pending_draft_input_ids: Dict[str, List[int]] = {}
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
            cross_tokenizer_enabled=getattr(
                server_args, "smc_cross_tokenizer", False
            ),
            final_selection=getattr(
                server_args, "smc_final_selection", "posterior_sample"
            ),
        )
        if getattr(server_args, "smc_cross_tokenizer", False):
            self.slot_state.configure_draft_pools(
                req_to_token_pool=self.model_worker.draft_req_to_token_pool,
                token_to_kv_pool_allocator=(
                    self.model_worker.draft_token_to_kv_pool_allocator
                ),
                tree_cache=self.model_worker._draft_tree_cache,
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
        # The cross-tokenizer mapper and boundary-state updates are host-side
        # and must complete before the next cycle consumes their lineages.
        if getattr(server_args, "smc_cross_tokenizer", False):
            want_overlap = False
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

        # SMC_SCHED_PROFILE=1: time each sequential decode-step phase
        # (batch-build, forward, resample, postprocess) with a cuda sync at
        # each boundary, accumulate, and log a rolling average.  Profiling
        # only — the syncs it adds serialize the loop, so never leave it on
        # for a throughput measurement.
        self._sched_profile = bool(int(os.environ.get("SMC_SCHED_PROFILE", "0")))
        self._sched_prof_acc = {"build": 0.0, "fwd": 0.0, "resample": 0.0, "post": 0.0}
        self._resample_sub_acc = {"wb": 0.0, "collect": 0.0, "dispatch": 0.0}
        self._sched_prof_n = 0
        self._sched_prof_every = int(os.environ.get("SMC_SCHED_PROFILE_EVERY", "100"))

        # SMC_LOG_ESS=1: accumulate per-step effective sample size (ESS) and
        # resample rate across in-use groups, log a rolling average.  Adds a
        # host sync per step (the .item() reads) — diagnostic only, never on a
        # throughput run.
        self._log_ess = bool(int(os.environ.get("SMC_LOG_ESS", "0")))
        self._ess_acc = 0.0
        self._ess_min_acc = 0.0
        self._ess_n = 0
        self._ess_resamples = 0
        self._ess_every = int(os.environ.get("SMC_LOG_ESS_EVERY", "200"))
        self._smc_trace_jsonl = os.environ.get("SMC_TRACE_JSONL")
        self._smc_trace_resample_limit = int(
            os.environ.get("SMC_TRACE_RESAMPLE_LIMIT", "0")
        )
        self._smc_trace_resample_rows = 0

    def _sched_clock(self) -> float:
        """Wall clock with a device sync, so each phase delta is true GPU+host
        time for that phase (profiling only)."""
        torch.cuda.synchronize()
        return time.perf_counter()

    def _sched_prof_record(
        self, build: float, fwd: float, resample: float, post: float
    ) -> None:
        a = self._sched_prof_acc
        a["build"] += build
        a["fwd"] += fwd
        a["resample"] += resample
        a["post"] += post
        self._sched_prof_n += 1
        if self._sched_prof_n == 1:
            print("[SMC_SCHED_PROFILE] profiled decode path active", flush=True)
        if self._sched_prof_n % self._sched_prof_every == 0:
            w = self._sched_prof_every  # windowed average (reset below)
            tot = a["build"] + a["fwd"] + a["resample"] + a["post"]
            print(
                f"[SMC_SCHED_PROFILE] n={self._sched_prof_n} win-avg ms/step: "
                f"build={1e3 * a['build'] / w:.3f} fwd={1e3 * a['fwd'] / w:.3f} "
                f"resample={1e3 * a['resample'] / w:.3f} post={1e3 * a['post'] / w:.3f} "
                f"total={1e3 * tot / w:.3f}",
                flush=True,
            )
            s = self._resample_sub_acc
            print(
                f"[SMC_SCHED_PROFILE]   resample split ms/step: "
                f"wb={1e3 * s['wb'] / w:.3f} collect={1e3 * s['collect'] / w:.3f} "
                f"dispatch={1e3 * s['dispatch'] / w:.3f}",
                flush=True,
            )
            for k in a:
                a[k] = 0.0
            for k in s:
                s[k] = 0.0

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

    def _maybe_dump_smc_resample_trace(
        self,
        plan,
        pre_interval_weights: torch.Tensor | None = None,
    ) -> None:
        if not self._smc_trace_jsonl:
            return
        if (
            self._smc_trace_resample_limit > 0
            and self._smc_trace_resample_rows >= self._smc_trace_resample_limit
        ):
            return
        try:
            if pre_interval_weights is None:
                # This fallback is only for externally constructed plans. The
                # production caller supplies a pre-collect clone because
                # fused_collect zeroes every row that resamples.
                pre_interval_weights = self.slot_state.interval_weights
            group_slots = self.slot_state.group_to_slots.detach().cpu()
            in_use = self.slot_state.row_in_use.detach().cpu()
            pre_weights = pre_interval_weights.detach().cpu()
            pre_resample_rows = []
            for row, is_in_use in enumerate(in_use.tolist()):
                if not is_in_use:
                    continue
                slots = group_slots[row, : self.slot_state.n_particles].to(
                    torch.int64
                )
                log_weights = pre_weights[slots]
                normalized = torch.softmax(log_weights, dim=0)
                ess = 1.0 / torch.sum(normalized.square())
                pre_resample_rows.append(
                    {
                        "row": int(row),
                        "slots": [int(slot) for slot in slots.tolist()],
                        "interval_log_weights": [
                            float(value) for value in log_weights.tolist()
                        ],
                        "normalized_weights": [
                            float(value) for value in normalized.tolist()
                        ],
                        "ess": float(ess.item()),
                    }
                )
            active_slots = self.slot_state.active_slots.detach().cpu().tolist()
            active_pool = self.slot_state.req_pool_indices[
                self.slot_state.active_slots
            ].detach().cpu().tolist()
            resample_mask = plan.resample_mask.detach().cpu().tolist()
            row_jobs = plan.row_of_job.detach().cpu().tolist()
            src_jobs = plan.src_slots.detach().cpu().tolist()
            dst_jobs = plan.dst_slots.detach().cpu().tolist()
            record = {
                "trace_kind": "smc_resample_step",
                "tp_rank": self.tp_rank,
                "active_slots": [int(x) for x in active_slots],
                "active_req_pool_indices": [int(x) for x in active_pool],
                "pre_resample_rows": pre_resample_rows,
                "in_use": [bool(value) for value in in_use.tolist()],
                "resample_mask": [bool(x) for x in resample_mask],
                "jobs": [
                    {"row": int(row), "src_slot": int(src), "dst_slot": int(dst)}
                    for row, src, dst in zip(row_jobs, src_jobs, dst_jobs)
                ],
                "threshold": float(self.coordinator.resample_threshold),
                "n_particles": int(self.slot_state.n_particles),
            }
            with open(self._smc_trace_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._smc_trace_resample_rows += 1
        except Exception as exc:
            logger.warning("SMC resample trace dump failed: %s", exc)

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

            _prof = self._sched_profile
            _t = self._sched_clock if _prof else None
            _b0 = _t() if _prof else 0.0
            batch, batch_kind = self._get_next_batch()
            tracking_batch = self._make_runtime_tracking_batch(batch)
            self.cur_batch = tracking_batch
            self.running_batch = (
                tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
            )

            if batch is not None:
                if _prof and batch_kind == "decode":
                    _b1 = _t()
                    result = self.run_batch(batch)
                    _b2 = _t()
                    plan, snapshot = self._resample(result)
                    _b3 = _t()
                    self._process_decode_result(result, plan, snapshot)
                    _b4 = _t()
                    self._sched_prof_record(_b1 - _b0, _b2 - _b1, _b3 - _b2, _b4 - _b3)
                else:
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

    def handle_generate_request(self, recv_req):
        draft_ids = getattr(recv_req, "smc_draft_input_ids", None)
        if draft_ids is not None and recv_req.rid is not None:
            self._smc_pending_draft_input_ids[str(recv_req.rid)] = list(draft_ids)
        return super().handle_generate_request(recv_req)

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
        draft_ids = self._smc_pending_draft_input_ids.pop(str(req.rid), None)
        if draft_ids is not None:
            req.smc_draft_origin_input_ids = draft_ids
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
        pending_admitted_slots = getattr(self, "_pending_admitted_slots", 0)
        remaining_capacity = (
            self.slot_state.available_slot_count() - pending_admitted_slots
        )

        while self.waiting_groups:
            group = self.waiting_groups[0]
            group_size = group.n_particles
            if group_size > remaining_capacity:
                break
            admitted.append(self.waiting_groups.popleft())
            pending_admitted_slots += group_size
            self._pending_admitted_slots = pending_admitted_slots
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

            if bool(int(os.environ.get("SMC_DEBUG_CROSS", "0"))):
                print(
                    "[SMC_CROSS_DBG] "
                    f"prefill rid={req.rid} next={next_token_id} "
                    f"out_len={len(req.output_ids)} "
                    f"max_new={req.sampling_params.max_new_tokens} "
                    f"finished={req.finished()} "
                    f"reason={req.finished_reason}",
                    flush=True,
                )

            if req.finished():
                release_kv_cache(req, self.tree_cache)
                req.time_stats.set_completion_time()
                self.stream_output([req], False)
                continue

            error_msg = self._materialize_group(group)
            if error_msg is not None:
                if bool(int(os.environ.get("SMC_DEBUG_CROSS", "0"))):
                    print(
                        "[SMC_CROSS_DBG] "
                        f"materialize_error rid={req.rid} error={error_msg}",
                        flush=True,
                    )
                self._abort_group(group, error_msg)
                continue

            self.running_groups.append(group)

    def _materialize_group(
        self,
        group: SequenceGroup,
    ) -> Optional[str]:
        parent_req = group.parent_req
        group.materialize_particles()
        particle_reqs = list(group.particle_reqs.values())
        try:
            self.model_worker.materialize_smc_parent_draft_prefix(
                parent_req, particle_reqs
            )
        except Exception as exc:
            group.clear_particles()
            return f"SMC parent draft prefill failed: {exc}"

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

        if self._sched_profile:
            torch.cuda.synchronize()
            _r0 = time.perf_counter()

        if (
            getattr(self.server_args, "smc_cross_tokenizer", False)
            and next_draft is not None
            and next_draft.proxy_valid_mask is not None
        ):
            self.slot_state.set_mapping_boundary_states(
                next_draft.mapping_boundary_states
            )
            self.slot_state.write_back_cross_tokenizer_gpu(
                next_token_ids=result.next_token_ids,
                logprob_diff=logprob_diff,
                bonus_ids=bonus_ids,
                proxy_valid_mask=next_draft.proxy_valid_mask,
                proxy_lens=next_draft.proxy_lens,
                emit_target_bonus=next_draft.emit_target_bonus,
                bonus_logz=next_draft.bonus_logz,
                draft_verified_ids=next_draft.draft_verified_id,
                draft_accepted_ids=next_draft.draft_accepted_ids,
                draft_accepted_lens=next_draft.draft_accepted_lens,
                draft_visible_lens=next_draft.draft_visible_lens,
                prev_last_draft_ids=prev_last_draft_ids,
                identity_const_lens=next_draft.identity_const_lens,
                proxy_lens_host=next_draft.proxy_lens_host,
                draft_visible_lens_host=next_draft.draft_visible_lens_host,
                fast_writeback_lens=next_draft.fast_writeback_lens,
                emit_target_bonus_host=next_draft.emit_target_bonus_host,
                pending_draft_suffix_lens=next_draft.pending_draft_suffix_lens,
                pending_draft_suffix_lens_host=(
                    getattr(next_draft, "pending_draft_suffix_lens_host", None)
                ),
            )
        else:
            # GPU write-back: token scatter, finish flags, weight accumulation.
            # Sync-free — finish state lands in slot tensors, not Req objects.
            self.slot_state.write_back_gpu(
                next_token_ids=result.next_token_ids,
                logprob_diff=logprob_diff,
                bonus_ids=bonus_ids,
                prev_last_draft_ids=prev_last_draft_ids,
                bonus_logz=next_draft.bonus_logz,
            )

        if self._sched_profile:
            torch.cuda.synchronize()
            _r1 = time.perf_counter()

        # Snapshot the per-row log Z_hat increment BEFORE the resample kernel
        # zeroes weights, then fold it into group_log_Z_hat for the rows that
        # actually resample (unbiased-estimator product over resample steps).
        logZ_inc = self.slot_state.resample_logZ_increment()

        if self._log_ess:
            _ess, _in_use = self.slot_state.ess_stats_in_use()
            _n = int(_in_use.sum().item())
            if _n > 0:
                _ess_iu = _ess[_in_use]
                self._ess_acc += float(_ess_iu.mean().item())
                self._ess_min_acc += float(_ess_iu.min().item())
                _thr = self.coordinator.resample_threshold * self.slot_state.n_particles
                self._ess_resamples += int((_ess_iu < _thr).sum().item())
                self._ess_n += 1
                if self._ess_n % self._ess_every == 0:
                    _N = self.slot_state.n_particles
                    print(
                        f"[SMC_ESS] steps={self._ess_n} N={_N} "
                        f"mean_ESS={self._ess_acc / self._ess_n:.3f} "
                        f"min_ESS={self._ess_min_acc / self._ess_n:.3f} "
                        f"resample_rate="
                        f"{self._ess_resamples / (self._ess_n * max(_n, 1)):.3f} "
                        f"(ESS/N mean={(self._ess_acc / self._ess_n) / _N:.3f})",
                        flush=True,
                    )

        # ``fused_collect`` zeroes interval weights for rows that resample.
        # Trace the values which actually drove the ESS decision, not the
        # post-reset state. Cloning is trace-only, keeping production decode
        # free from a max-slot-sized allocation/copy.
        trace_pre_interval_weights = None
        if (
            self._smc_trace_jsonl
            and (
                self._smc_trace_resample_limit <= 0
                or self._smc_trace_resample_rows < self._smc_trace_resample_limit
            )
        ):
            trace_pre_interval_weights = self.slot_state.interval_weights.clone()

        # Resample all groups via the fused systematic kernel.
        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        self._maybe_dump_smc_resample_trace(plan, trace_pre_interval_weights)
        self.slot_state.group_log_Z_hat += torch.where(
            plan.resample_mask, logZ_inc, torch.zeros_like(logZ_inc)
        )
        if self._sched_profile:
            torch.cuda.synchronize()
            _r2 = time.perf_counter()
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

        if self._sched_profile:
            torch.cuda.synchronize()
            _r3 = time.perf_counter()
            s = self._resample_sub_acc
            s["wb"] += _r1 - _r0
            s["collect"] += _r2 - _r1
            s["dispatch"] += _r3 - _r2
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
        self.slot_state.apply_resample_host_shadows(plan)

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
        if self.slot_state.draft_kv_freed_count_host is not None:
            n_draft_freed = int(
                self.slot_state.draft_kv_freed_count_host[snapshot.phase].item()
            )
            if n_draft_freed > 0:
                self.slot_state.draft_token_to_kv_pool_allocator.free(
                    self.slot_state.draft_kv_freed_buf[
                        snapshot.phase, :n_draft_freed
                    ].to(torch.int64)
                )
                self.slot_state.draft_kv_freed_counter[snapshot.phase].zero_()

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
                    particle_slot_ids=getattr(parent_req, "smc_particle_slot_ids", None),
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

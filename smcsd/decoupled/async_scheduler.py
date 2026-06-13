"""Async (prefetch + barrier-resample) decoupled SMC scheduler.

Unlocks within-group draft/verify overlap on top of the decoupled lockstep path,
validated by the Tier-2 de-risk measurements (see docs/smc/async_smc_design.md):

- **Prefetch.** With the no-bonus anchor (drafter-known), the verifier sends the
  *next* window's StepReq immediately after receiving the current response, then
  runs its local target verify while the drafter computes that next window. The
  drafter needs no changes — it stays a pure reactor; only the verifier's decode
  loop reorders.
- **Barrier resampling.** Resampling happens only at K-window barriers, where the
  pipeline is drained (no StepReq in flight), so the verifier and drafter agree on
  the frontier and resampling reduces to the existing `batched_resample_kv` clone —
  no redraft, no rollback. Its only cost is delaying resampling by ≤K windows
  (measured ~free vs the decoupled baseline).
- **Ride-along finishes.** Particles that finish mid-train stay in the batch (their
  weight increments are masked off in `process_batch_result`) and are dropped at the
  barrier rebuild — keeping the slot set fixed across a train so prefetch stays
  consistent.

Each event-loop iteration runs one K-window train; the pipeline is drained at the
barrier so prefill admission happens cleanly between trains.
"""

from __future__ import annotations

import itertools
import logging
import os
import signal
from typing import List, Optional

import psutil
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import configure_scheduler
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import DynamicGradMode, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

from smcsd.decoupled.scheduler import DecoupledSMCScheduler
from smcsd.decoupled.worker import PendingDecodeStep

logger = logging.getLogger(__name__)

RESAMPLE_INTERVAL_ENV = "SMCSD_RESAMPLE_INTERVAL"


class AsyncDecoupledSMCScheduler(DecoupledSMCScheduler):
    """Decoupled SMC scheduler with prefetch overlap + barrier resampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sglang.srt.utils import get_bool_env_var

        if not get_bool_env_var("SMCSD_DROP_BONUS", "false"):
            raise RuntimeError(
                "AsyncDecoupledSMCScheduler requires no-bonus mode "
                "(SMCSD_DROP_BONUS=1): the next-round anchor must be "
                "drafter-known for the prefetch to be valid."
            )
        self.barrier_k = max(int(os.environ.get(RESAMPLE_INTERVAL_ENV, "2")), 1)
        self.gamma = self.server_args.speculative_num_steps
        self._tag = itertools.count(1)
        logger.info(
            "AsyncDecoupledSMCScheduler: prefetch overlap, resample barrier K=%d",
            self.barrier_k,
        )

    # ── Event loop: one K-window decode train per iteration ──

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            self._drain_finished_groups()

            # Prefill admission (the decode train is always drained at its
            # barrier, so no in-flight StepReq to reconcile here).
            if self.waiting_groups:
                self.prefill_groups = self._admit_prefill_groups()
                if self.prefill_groups:
                    batch = self._build_prefill_batch(self.prefill_groups)
                    if batch is None:
                        self.prefill_groups = []
                    else:
                        set_schedule_time_batch(batch)
                        self._run_tracked_prefill(batch)
                        continue

            if not self.slot_state.is_empty():
                self._run_decode_train()
            else:
                self.cur_batch = None
                self.self_check_during_idle()

    def _run_tracked_prefill(self, batch: ScheduleBatch) -> None:
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        result = self.run_batch(batch)
        self._process_prefill_result(batch, result)
        self.last_batch = tracking_batch

    # ── Decode train: K windows, prefetch overlap, barrier resample ──

    def _run_decode_train(self) -> None:
        worker = self.draft_worker
        client = self._draft_client
        K = self.barrier_k

        # Window 0 of the train: prepare (allocate target KV, advance seq_lens),
        # send the StepReq.  The active slot set is fixed for the whole train.
        batch = self._prepare_decode_batch()
        if batch is None:
            return
        active_list: List[int] = list(batch.spec_info.active_slots_cpu)
        active_t = torch.tensor(active_list, dtype=torch.int64, device=self.device)

        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )

        tag = next(self._tag)
        pending = worker.start_decode(batch, tag=tag)

        for w in range(K):
            resp = client.recv_step_resp()
            if resp.tag != pending.tag:
                raise RuntimeError(
                    f"Async step reply tag mismatch: got {resp.tag}, "
                    f"expected {pending.tag}"
                )
            is_last = w == K - 1

            next_tag = None
            if not is_last:
                # Prefetch the next window from raw lists (no slot_state mutation)
                # so the drafter computes it while we verify the current window.
                anchor_next = torch.from_numpy(resp.tokens)[:, self.gamma].tolist()
                seq_lens_next = self.slot_state.seq_lens[active_t].tolist()
                next_tag = next(self._tag)
                worker.send_step_req(
                    active_list, anchor_next, seq_lens_next, tag=next_tag
                )

            # Verify the current window (overlaps the drafter computing the next).
            result = worker.finish_decode(pending, resp)
            self._writeback_window(result, active_t)

            if not is_last:
                # Prepare the verifier side of the next window (target KV +
                # seq_lens advance) — AFTER writeback so it reads this window's
                # seq_lens, matching lockstep.
                next_batch = self._prepare_decode_batch_fixed(active_t, active_list)
                pending = PendingDecodeStep(
                    batch=next_batch,
                    ctx=next_batch.spec_info.decode_ctx,
                    cache_locs=None,
                    tag=next_tag,
                )

        # Barrier: resample on the K-window-accumulated weights at the drained
        # frontier, mirror the plan to the drafter, then rebuild + drain.
        self._barrier_resample()
        self.slot_state.rebuild_active_slots()
        self._drain_finished_groups()
        self.last_batch = tracking_batch

    def _prepare_decode_batch_fixed(self, active_t, active_list):
        """prepare_for_decode over a FIXED active subset (the train's slots),
        so finishes mid-train don't shrink the set until the barrier rebuild."""
        draft_input = self.slot_state.prepare_for_decode(active=active_t)
        draft_input.active_slots_cpu = active_list
        return self.slot_state.build_model_worker_batch(
            draft_input, active=active_t, active_list=active_list
        )

    def _writeback_window(self, result, active_t) -> None:
        if result.copy_done is not None:
            result.copy_done.synchronize()
        if result.logprob_diff is None:
            raise RuntimeError("Async SMC requires batched logprob_diff.")
        logprob_diff = (
            result.logprob_diff
            if torch.is_tensor(result.logprob_diff)
            else torch.as_tensor(
                result.logprob_diff, dtype=torch.float32, device=self.device
            )
        )
        bonus_ids = result.next_draft_input.verified_id
        self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
            active=active_t,
        )

    def _barrier_resample(self) -> None:
        plan = self.coordinator.collect_resample_jobs_batch(self.slot_state)
        if plan.n_jobs > 0:
            self.coordinator.dispatch_resample_batch(
                plan, self.slot_state, rebuild_active=False,
            )
            self._draft_client.send_commit(
                dst_slots=plan.dst_slots.tolist(),
                src_slots=plan.src_slots.tolist(),
            )


def run_async_smc_scheduler_process(
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
        scheduler = AsyncDecoupledSMCScheduler(
            server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
            pp_rank, attn_cp_rank, moe_dp_rank, dp_rank,
        )
        pipe_writer.send(scheduler.get_init_info())
        scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"AsyncDecoupledSMCScheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

"""Cross-group pipelined scheduler for decoupled SMC (experimental).

Lockstep decoupling leaves one GPU idle at a time.  SMC forbids overlapping a
group with itself (the next round's anchor is the target-sampled bonus token
and resampling reshuffles particle KV), but distinct groups are independent —
so this scheduler partitions live groups into ``SMCSD_PIPELINE_COHORTS``
cohorts (default 2) and overlaps one cohort's draft round (GPU1) with another
cohort's verify (GPU0):

    GPU1 (draft):  [draft A,t]   [draft B,t]   [draft A,t+1] ...
    GPU0 (target): ...........   [verify A,t]  [verify B,t]  ...

Safety rules that keep the drafter mirror consistent:
- A verify only writes back / resamples its own cohort's slots (subset
  ``process_batch_result`` + row-masked collect), so a cohort whose draft is
  in flight can never be resampled underneath the drafter.
- All drafter messages stay on the one FIFO channel; per cohort the order
  StepReq(t) … CommitResample(t) … StepReq(t+1) is preserved by construction.
- Prefill is a sync point: in-flight step replies are drained (held READY)
  before the blocking PrefillReq so reply matching stays unambiguous.
"""

from __future__ import annotations

import itertools
import logging
import os
import signal
from collections import deque
from typing import Deque, Dict, List, Optional

import psutil
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import configure_scheduler
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import DynamicGradMode, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

from smcsd.core.scheduler import SequenceGroup
from smcsd.decoupled.scheduler import DecoupledSMCScheduler
from smcsd.decoupled.worker import PendingDecodeStep

logger = logging.getLogger(__name__)

COHORTS_ENV = "SMCSD_PIPELINE_COHORTS"

IDLE, DRAFTING, READY = "idle", "drafting", "ready"


class _Cohort:
    def __init__(self, idx: int):
        self.idx = idx
        self.group_ids: set = set()
        self.state = IDLE
        self.pending: Optional[PendingDecodeStep] = None
        self.resp = None
        self.active_t: Optional[torch.Tensor] = None
        self.active_list: List[int] = []


class PipelinedDecoupledSMCScheduler(DecoupledSMCScheduler):
    """Decoupled SMC scheduler that pipelines draft/verify across cohorts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_cohorts = max(1, int(os.environ.get(COHORTS_ENV, "2")))
        self.cohorts = [_Cohort(i) for i in range(n_cohorts)]
        self._group_cohort: Dict[str, _Cohort] = {}
        self._slot_cohort: Dict[int, _Cohort] = {}
        self._inflight: Deque[_Cohort] = deque()
        self._tag_counter = itertools.count(1)
        logger.info("PipelinedDecoupledSMCScheduler: %d cohorts", n_cohorts)

    # ── Cohort membership tracks group lifecycle ──

    def _materialize_group(self, group: SequenceGroup) -> Optional[str]:
        error_msg = super()._materialize_group(group)
        if error_msg is None:
            cohort = min(
                self.cohorts,
                key=lambda c: sum(
                    len(self.slot_state.group_slot_lists.get(g, []))
                    for g in c.group_ids
                ),
            )
            cohort.group_ids.add(group.group_id)
            self._group_cohort[group.group_id] = cohort
            for slot in self.slot_state.group_slot_lists[group.group_id]:
                self._slot_cohort[slot] = cohort
        return error_msg

    def _forget_group(self, group_id: str) -> None:
        cohort = self._group_cohort.pop(group_id, None)
        if cohort is not None:
            cohort.group_ids.discard(group_id)
        for slot in [s for s, c in self._slot_cohort.items() if cohort is c]:
            if slot not in self.slot_state.slot_to_req:
                self._slot_cohort.pop(slot, None)

    def _finalize_group(self, group: SequenceGroup) -> None:
        super()._finalize_group(group)
        self._forget_group(group.group_id)

    def _abort_group(self, group: SequenceGroup, error_msg: str) -> None:
        super()._abort_group(group, error_msg)
        self._forget_group(group.group_id)

    # ── Pipelined event loop ──

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            progressed = False

            # 1. Prefill admission (sync point on the drafter channel).
            if self.waiting_groups:
                batch = self._try_build_prefill_batch()
                if batch is not None:
                    self._drain_inflight_to_ready()
                    self._run_tracked_prefill(batch)
                    progressed = True

            # 2. Launch a draft round for every idle cohort with active slots.
            for cohort in self.cohorts:
                if cohort.state == IDLE and self._launch_cohort_step(cohort):
                    progressed = True

            # 3. Collect arrived step replies; wait briefly if drafts are the
            #    only outstanding work.
            self._poll_step_resp(timeout_ms=0)
            if (
                not any(c.state == READY for c in self.cohorts)
                and self._inflight
            ):
                self._poll_step_resp(timeout_ms=100)

            # 4. Verify one ready cohort (overlaps with in-flight drafts).
            ready = next((c for c in self.cohorts if c.state == READY), None)
            if ready is not None:
                self._verify_cohort(ready)
                progressed = True

            if not progressed and not self._inflight:
                self.self_check_during_idle()

            if hasattr(self, "waiting_queue"):
                self.waiting_queue = []

    # ── Prefill ──

    def _try_build_prefill_batch(self) -> Optional[ScheduleBatch]:
        if self.prefill_groups:
            raise RuntimeError("Pipelined scheduler has an unprocessed prefill batch.")
        self.prefill_groups = self._admit_prefill_groups()
        if not self.prefill_groups:
            return None
        batch = self._build_prefill_batch(self.prefill_groups)
        if batch is None:
            self.prefill_groups = []
            return None
        set_schedule_time_batch(batch)
        return batch

    def _run_tracked_prefill(self, batch: ScheduleBatch) -> None:
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        result = self.run_batch(batch)
        self._process_prefill_result(batch, result)
        self.last_batch = tracking_batch

    # ── Draft launch / reply / verify ──

    def _cohort_active_list(self, cohort: _Cohort) -> List[int]:
        return [
            s
            for s in self.slot_state._active_slots_list
            if self._slot_cohort.get(s) is cohort
        ]

    def _launch_cohort_step(self, cohort: _Cohort) -> bool:
        active_list = self._cohort_active_list(cohort)
        if not active_list:
            return False
        active_t = torch.tensor(active_list, dtype=torch.int64, device=self.device)
        draft_input = self.slot_state.prepare_for_decode(active=active_t)
        if draft_input.decode_ctx is None:
            return False
        draft_input.active_slots_cpu = active_list
        batch = self.slot_state.build_model_worker_batch(
            draft_input, active=active_t, active_list=active_list
        )
        cohort.pending = self.draft_worker.start_decode(
            batch, tag=next(self._tag_counter)
        )
        cohort.active_t = active_t
        cohort.active_list = active_list
        cohort.state = DRAFTING
        self._inflight.append(cohort)
        return True

    def _poll_step_resp(self, timeout_ms: int) -> None:
        if not self._inflight:
            return
        resp = self._draft_client.recv_step_resp(timeout_ms=timeout_ms)
        if resp is None:
            return
        cohort = self._inflight.popleft()
        if resp.tag != cohort.pending.tag:
            raise RuntimeError(
                f"Step reply tag {resp.tag} does not match oldest in-flight "
                f"cohort tag {cohort.pending.tag} (FIFO violated?)"
            )
        cohort.resp = resp
        cohort.state = READY

    def _drain_inflight_to_ready(self) -> None:
        while self._inflight:
            self._poll_step_resp(timeout_ms=None)

    def _verify_cohort(self, cohort: _Cohort) -> None:
        tracking_batch = self._make_runtime_tracking_batch(cohort.pending.batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )

        result = self.draft_worker.finish_decode(cohort.pending, cohort.resp)
        self._process_cohort_decode_result(result, cohort)

        self.last_batch = tracking_batch
        cohort.pending = None
        cohort.resp = None
        cohort.active_t = None
        cohort.active_list = []
        cohort.state = IDLE

    def _process_cohort_decode_result(self, result, cohort: _Cohort) -> None:
        # Same as DecoupledSMCScheduler._process_decode_result, restricted to
        # the cohort's slots and group rows.
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

        next_draft = result.next_draft_input
        bonus_ids = next_draft.verified_id if next_draft is not None else None
        if bonus_ids is None:
            raise RuntimeError(
                "SMCScheduler: result missing next_draft_input.verified_id"
            )

        newly_finished = self.slot_state.process_batch_result(
            next_token_ids=result.next_token_ids,
            accept_lens=result.accept_lens,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
            active=cohort.active_t,
        )

        row_mask = torch.zeros(
            self.slot_state.max_groups, dtype=torch.bool, device=self.device
        )
        rows = [
            self.slot_state.group_id_to_row[g]
            for g in cohort.group_ids
            if g in self.slot_state.group_id_to_row
        ]
        if rows:
            row_mask[rows] = True

        plan = self.coordinator.collect_resample_jobs_batch(
            self.slot_state, row_mask=row_mask
        )
        did_resample = plan.n_jobs > 0
        if did_resample:
            self.coordinator.dispatch_resample_batch(
                plan, self.slot_state, rebuild_active=False,
            )
            self._draft_client.send_commit(
                dst_slots=plan.dst_slots.tolist(),
                src_slots=plan.src_slots.tolist(),
            )

        if newly_finished or did_resample:
            self.slot_state.rebuild_active_slots()

        self._drain_finished_groups()


def run_pipelined_smc_scheduler_process(
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
        scheduler = PipelinedDecoupledSMCScheduler(
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
        logger.error(f"PipelinedDecoupledSMCScheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

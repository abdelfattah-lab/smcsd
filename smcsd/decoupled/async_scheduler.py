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
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import psutil
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import configure_scheduler
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import DynamicGradMode, kill_itself_when_parent_died
from sglang.utils import get_exception_traceback

from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from smcsd.decoupled.io_struct import DraftStepResp
from smcsd.decoupled.scheduler import DecoupledSMCScheduler
from smcsd.decoupled.worker import PendingDecodeStep

logger = logging.getLogger(__name__)

RESAMPLE_INTERVAL_ENV = "SMCSD_RESAMPLE_INTERVAL"
SPEC_BARRIER_ENV = "SMCSD_SPEC_BARRIER"


@dataclass
class SpecState:
    """A speculative window fired across a resample barrier (SBP).

    Carried from train T's barrier into train T+1, where it is consumed as
    train T+1's window-0 after a verifier-side ancestor remap. See
    docs/smc/design_speculative_barrier_prefetch.md.
    """

    pending: PendingDecodeStep      # spec window snapshot (ctx.orig_seq_lens = S+KG)
    active_list_T: List[int]        # pre-resample active slot ids (drives the remap)
    active_t_T: torch.Tensor        # same as a device tensor (writeback target)
    tag: int                        # FIFO tag of the spec StepReq
    epoch: int                      # train counter (fail-fast fence)
    ancestor: Optional[np.ndarray]  # a(i): identity survivors, src retired; None=no-resample


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
        # Per-train epoch (fail-fast fence). Every StepReq in a train carries the
        # same epoch; the drafter echoes it back and finish_decode asserts it.
        self._epoch = itertools.count(1)
        # Speculative barrier prefetch (SBP): fire the next train's window-0
        # StepReq across the resample barrier so the drafter computes it during
        # verify+resample. Off by default (clean A/B). See the design doc.
        self._spec_barrier = get_bool_env_var(SPEC_BARRIER_ENV, "false")
        self._spec: Optional[SpecState] = None
        # Optional KV-invariant assertions (SMCSD_SPEC_BARRIER_DEBUG=1): verify at
        # each spec consume that the spec-window verify read stays inside each A1
        # slot's allocated KV (no OOB / illegal memory access).
        self._spec_debug = get_bool_env_var("SMCSD_SPEC_BARRIER_DEBUG", "false")
        # Barrier bonus (SMCSD_BARRIER_BONUS=1): reclaim the exact target-sampled
        # anchor on the BARRIER window only.  Interior windows keep the no-bonus
        # drafter-known anchor (so prefetch still overlaps), but the last window of
        # each train — drained, verify done before the next train's window-0 fires
        # — seeds the next anchor from the target bonus (1/K windows exact). Mutually
        # exclusive with SBP (which fires window-0 pre-verify). See the report.
        self._barrier_bonus = get_bool_env_var("SMCSD_BARRIER_BONUS", "false")
        if self._barrier_bonus and self._spec_barrier:
            raise RuntimeError(
                "SMCSD_BARRIER_BONUS and SMCSD_SPEC_BARRIER are mutually exclusive: "
                "SBP fires the barrier window-0 speculatively before verify, so the "
                "post-verify target bonus is not available in time."
            )
        # Optional timing decomposition (SMCSD_TIMING=1): how long the verifier
        # blocks on the drafter (recv wait = drafter-bound signal) vs its own
        # work (verify dispatch + writeback + prepare + barrier).
        self._timing = get_bool_env_var("SMCSD_TIMING", "false")
        self._t = {"recv": 0.0, "verify": 0.0, "prep": 0.0, "barrier": 0.0}
        self._t_windows = 0
        logger.info(
            "AsyncDecoupledSMCScheduler: prefetch overlap, resample barrier K=%d, "
            "spec_barrier=%s",
            self.barrier_k,
            self._spec_barrier,
        )

    # ── Event loop: one K-window decode train per iteration ──

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # SBP drain/commit guard (load-bearing — design R1).  A spec window
            # fired across the previous barrier is in flight on the FIFO draft
            # channel.  If this iteration will NOT consume it as a normal decode
            # train (engine pausing, prefill about to be admitted, or nothing
            # left to decode), commit it standalone NOW — before any other
            # draft-channel send — so its StepResp leaves the FIFO (else a
            # following prefill RPC's recv would pop it and crash) and the
            # frontier advance it already made is written back, not stranded.
            if self._spec is not None and (
                self._engine_paused
                or self.waiting_groups
                or self.slot_state.is_empty()
            ):
                self._commit_spec_standalone()

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
        tm = self._timing

        # ── Window 0: consume a carried SBP spec window, or cold-prepare ──
        if self._spec is not None:
            state = self._consume_spec_window0()
            if state is None:
                return  # active emptied during the consume (group finished)
            epoch, active_list, active_t, pending, w_start, tracking_batch = state
        else:
            batch = self._prepare_decode_batch()
            if batch is None:
                return
            active_list = list(batch.spec_info.active_slots_cpu)
            active_t = torch.tensor(
                active_list, dtype=torch.int64, device=self.device
            )
            tracking_batch = self._make_runtime_tracking_batch(batch)
            self.cur_batch = tracking_batch
            self.running_batch = (
                tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
            )
            epoch = next(self._epoch)
            tag = next(self._tag)
            pending = worker.start_decode(batch, tag=tag, epoch=epoch)
            w_start = 0

        # ── Windows w_start..K-1: prefetch overlap; fire the SBP spec last ──
        for w in range(w_start, K):
            t0 = time.perf_counter() if tm else 0.0
            resp = client.recv_step_resp()
            if tm:
                self._t["recv"] += time.perf_counter() - t0
            if resp.tag != pending.tag or resp.epoch != pending.epoch:
                raise RuntimeError(
                    f"Async step reply tag/epoch mismatch: got "
                    f"({resp.tag},{resp.epoch}), expected "
                    f"({pending.tag},{pending.epoch})"
                )
            is_last = w == K - 1
            fire_spec = is_last and self._spec_barrier

            next_tag = None
            if not is_last:
                # Prefetch the next window from raw lists (no slot_state mutation)
                # so the drafter computes it while we verify the current window.
                anchor_next = torch.from_numpy(resp.tokens)[:, self.gamma].tolist()
                seq_lens_next = self.slot_state.seq_lens[active_t].tolist()
                next_tag = next(self._tag)
                worker.send_step_req(
                    active_list, anchor_next, seq_lens_next,
                    tag=next_tag, epoch=epoch,
                )
            elif fire_spec:
                # SBP: fire the NEXT train's window-0 across the barrier, off the
                # drafter-known (gamma+1)-th token — before verify+resample so the
                # drafter computes it during them.  Fired BEFORE _barrier_resample's
                # send_commit (FIFO), so the frontier-clone copies this spec window
                # into retired slots.  Remapped at consume time by the ancestor map.
                spec_epoch = next(self._epoch)
                spec_tag = next(self._tag)
                spec_anchor = torch.from_numpy(resp.tokens)[:, self.gamma].tolist()
                spec_seq_lens = self.slot_state.seq_lens[active_t].tolist()
                worker.send_step_req(
                    active_list, spec_anchor, spec_seq_lens,
                    tag=spec_tag, epoch=spec_epoch,
                )

            t0 = time.perf_counter() if tm else 0.0
            # Barrier bonus: the last window is drained (no prefetch fired off it),
            # so override to the exact target-sampled anchor; its bonus becomes the
            # next train's window-0 anchor (set by the writeback's verified_ids).
            db = False if (is_last and self._barrier_bonus) else None
            result = worker.finish_decode(pending, resp, drop_bonus=db)
            self._writeback_window(result, active_t)
            if tm:
                self._t["verify"] += time.perf_counter() - t0

            if not is_last:
                t0 = time.perf_counter() if tm else 0.0
                next_batch = self._prepare_decode_batch_fixed(active_t, active_list)
                if tm:
                    self._t["prep"] += time.perf_counter() - t0
                pending = PendingDecodeStep(
                    batch=next_batch,
                    ctx=next_batch.spec_info.decode_ctx,
                    tag=next_tag,
                    epoch=epoch,
                )
            elif fire_spec:
                # Prep the verifier side of the spec window (advance V S+KG ->
                # S+(K+1)G over the FULL train set, alloc verify KV at each slot's
                # own req_pool_index) and stash it.  This snapshot is load-bearing:
                # it freezes ctx.orig_seq_lens = S+KG, used by the consume's verify
                # cache_locs.  Never rebuilt post-resample.
                t0 = time.perf_counter() if tm else 0.0
                spec_batch = self._prepare_decode_batch_fixed(active_t, active_list)
                if tm:
                    self._t["prep"] += time.perf_counter() - t0
                self._spec = SpecState(
                    pending=PendingDecodeStep(
                        batch=spec_batch,
                        ctx=spec_batch.spec_info.decode_ctx,
                        tag=spec_tag,
                        epoch=spec_epoch,
                    ),
                    active_list_T=active_list,
                    active_t_T=active_t,
                    tag=spec_tag,
                    epoch=spec_epoch,
                    ancestor=None,
                )
            self._t_windows += 1

        # ── Barrier: resample (captures the ancestor map onto self._spec when a
        #    spec fired), rebuild the active set, finalize-drain.  A fired spec is
        #    scored next train over the post-rebuild survivors A1 (always still
        #    allocated), so the drain here is safe — it only frees fully-finished
        #    groups, none of which are in A1. ──
        t0 = time.perf_counter() if tm else 0.0
        self._barrier_resample()
        self.slot_state.rebuild_active_slots()
        self._drain_finished_groups()
        if tm:
            self._t["barrier"] += time.perf_counter() - t0
            if self._t_windows >= 200:
                tot = sum(self._t.values()) or 1.0
                n = self._t_windows
                print(
                    f"[ASYNC_TIMING] {n} windows: "
                    f"recv(drafter-wait)={100*self._t['recv']/tot:.0f}% "
                    f"verify={100*self._t['verify']/tot:.0f}% "
                    f"prep={100*self._t['prep']/tot:.0f}% "
                    f"barrier={100*self._t['barrier']/tot:.0f}% | per-window "
                    f"recv={1e3*self._t['recv']/n:.2f}ms "
                    f"verify={1e3*self._t['verify']/n:.2f}ms",
                    flush=True,
                )
                self._t = {k: 0.0 for k in self._t}
                self._t_windows = 0
        self.last_batch = tracking_batch

    def _consume_spec_window0(self):
        """Adopt the carried SBP spec window as this train's window 0.

        The spec window was drafted over A0 (= spec.active_list_T) and the
        resample cloned each retired slot's KV from its survivor.  We score it
        over the *post-rebuild* active set A1 — exactly the slots that ended up at
        the uniform S+(K+1)G frontier with valid spec KV — gathering each A1
        slot's columns from its adopted ancestor's A0 row.  (Scoring the full A0
        instead would read stale KV for slots retired onto a particle that
        finished before this train.)  Returns the loop state to continue windows
        1..K-1 over A1, or None if nothing remains to decode."""
        spec = self._spec
        K = self.barrier_k
        tm = self._timing
        gamma = self.gamma
        worker = self.draft_worker
        client = self._draft_client

        t0 = time.perf_counter() if tm else 0.0
        resp = client.recv_step_resp()
        if tm:
            self._t["recv"] += time.perf_counter() - t0
        if resp.tag != spec.tag or resp.epoch != spec.epoch:
            raise RuntimeError(
                f"SBP spec reply tag/epoch mismatch: got ({resp.tag},{resp.epoch}), "
                f"expected ({spec.tag},{spec.epoch})"
            )
        epoch = spec.epoch

        # A1 = post-rebuild active set (rebuilt at the previous barrier): the
        # consistent-frontier survivors, incl. slots revived by the resample.
        a1_list = list(self.slot_state._active_slots_list)
        tracking_batch = self._make_runtime_tracking_batch(spec.pending.batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )

        if not a1_list:
            # No survivors at the rebuilt frontier (all finished) — the spec
            # window's tokens are past every answer; pop it and decode nothing
            # this train.  The group was already finalized at the previous
            # barrier's drain.
            self._t_windows += 1
            self._spec = None
            self.last_batch = tracking_batch
            return None

        a1_t = torch.tensor(a1_list, dtype=torch.int64, device=self.device)
        pending_a1, resp_a1, tokens_a1 = self._build_spec_a1(spec, resp, a1_list, a1_t)

        # Fire window 1 across A1 BEFORE verifying window 0, so the drafter
        # computes it while we verify window 0 — the overlap that makes SBP pay
        # off at K=2.  Window 1's anchor is the spec window's (gamma+1)-th token.
        next_tag = None
        if K > 1:
            anchor_next = tokens_a1[:, gamma].tolist()
            seq_lens_next = self.slot_state.seq_lens[a1_t].tolist()
            next_tag = next(self._tag)
            worker.send_step_req(
                a1_list, anchor_next, seq_lens_next, tag=next_tag, epoch=epoch,
            )

        # Verify window 0 over A1 (overlaps the drafter computing window 1).
        t0 = time.perf_counter() if tm else 0.0
        result = worker.finish_decode(pending_a1, resp_a1)
        self._writeback_window(result, a1_t)
        if tm:
            self._t["verify"] += time.perf_counter() - t0
        self._t_windows += 1
        self._spec = None

        # Particles that finished in this spec window (EOS) ride along over A1
        # for the rest of the train (masked, like the cold path) and are dropped
        # at this train's barrier — NOT freed mid-train, so the fixed A1 KV the
        # remaining windows prep against stays allocated.
        if K > 1:
            next_batch = self._prepare_decode_batch_fixed(a1_t, a1_list)
            pending = PendingDecodeStep(
                batch=next_batch,
                ctx=next_batch.spec_info.decode_ctx,
                tag=next_tag,
                epoch=epoch,
            )
        else:
            pending = None  # K==1: no window 1; the loop range(1, 1) is empty

        return epoch, a1_list, a1_t, pending, 1, tracking_batch

    def _commit_spec_standalone(self) -> None:
        """Commit an in-flight spec window without continuing into a decode train
        (event-loop guard path: engine pausing, prefill about to be admitted, or
        nothing left to decode).  Pops the spec StepResp off the FIFO, scores it
        over the post-rebuild survivors A1, writes back the frontier advance, then
        runs the deferred finalize-drain.  The next decode train starts cold from
        the committed frontier (verified_ids now hold the spec window's anchor)."""
        spec = self._spec
        worker = self.draft_worker
        client = self._draft_client

        resp = client.recv_step_resp()
        if resp.tag != spec.tag or resp.epoch != spec.epoch:
            raise RuntimeError(
                f"SBP standalone-commit tag/epoch mismatch: got "
                f"({resp.tag},{resp.epoch}), expected ({spec.tag},{spec.epoch})"
            )
        a1_list = list(self.slot_state._active_slots_list)
        if a1_list:
            a1_t = torch.tensor(a1_list, dtype=torch.int64, device=self.device)
            pending_a1, resp_a1, _ = self._build_spec_a1(spec, resp, a1_list, a1_t)
            result = worker.finish_decode(pending_a1, resp_a1)
            self._writeback_window(result, a1_t)
            self._t_windows += 1
        self._spec = None
        # Mini-barrier: refresh the active set + finalize fully-finished groups so
        # the prefill/idle that follows this guard sees a clean, allocated set.
        self.slot_state.rebuild_active_slots()
        self._drain_finished_groups()

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
            if self._spec is not None:
                # SBP ancestry a(i): retired slot i adopts survivor src[i]; every
                # other slot maps to itself. Slot-indexed; consumed in train T+1
                # to gather each post-rebuild slot's adopted spec-window columns.
                max_slots = self.slot_state.seq_lens.shape[0]
                a = np.arange(max_slots, dtype=np.int64)
                a[plan.dst_slots.cpu().numpy()] = plan.src_slots.cpu().numpy()
                self._spec.ancestor = a
        # n_jobs == 0: leave self._spec.ancestor = None (identity gather).

    def _spec_a1_source_rows(self, spec: SpecState, a1_list: List[int]) -> np.ndarray:
        """The row in the spec window's drafted batch (A0 = spec.active_list_T)
        whose KV + draft each post-rebuild slot (A1) holds: the ancestor it
        adopted in the resample, which is ALWAYS in A0 (a survivor of the drafted
        set).  Identity-by-slot when no resample fired (then A1 ⊆ A0).

        This is well-defined for every A1 slot — survivors (a(s)=s), retired-onto-
        survivor (a(s)=the survivor), and revived finished slots (a(s)=the active
        survivor that overwrote them).  Slots that retired onto a particle which
        finished before this train become finished and are NOT in A1, so they are
        never gathered (which is what made scoring the full A0 read stale KV)."""
        pos_in_A0 = {slot: r for r, slot in enumerate(spec.active_list_T)}
        if spec.ancestor is None:
            gen = (pos_in_A0[s] for s in a1_list)
        else:
            gen = (pos_in_A0[int(spec.ancestor[s])] for s in a1_list)
        return np.fromiter(gen, dtype=np.int64, count=len(a1_list))

    def _build_spec_a1(self, spec: SpecState, resp, a1_list: List[int], a1_t):
        """Gather the carried spec window onto the post-rebuild active set A1 and
        build the verify pending for it.  Every A1 slot sits at the uniform
        S+(K+1)G frontier with valid spec KV (its own, or a survivor's clone), so
        orig_seq_lens = current seq_lens - (gamma+1).  Returns (pending, resp_a1,
        tokens_a1) — tokens_a1[:, gamma] is the next window's anchor."""
        gamma = self.gamma
        src_rows = self._spec_a1_source_rows(spec, a1_list)
        tokens_a1 = resp.tokens[src_rows]
        logprobs_a1 = resp.logprobs[src_rows]
        src_t = torch.as_tensor(src_rows, device=self.device)
        anchor_a1 = spec.pending.batch.spec_info.verified_id[src_t]

        seq_lens_a1 = self.slot_state.seq_lens[a1_t]
        orig_a1 = seq_lens_a1 - (gamma + 1)
        if self._spec_debug:
            # The verify reads cache_locs at [orig_a1, orig_a1 + gamma+1) =
            # [seq_lens-G, seq_lens). It is in-bounds iff each A1 slot's allocated
            # KV covers its frontier. These must hold for survivors, retired-onto-
            # active, and revived-from-finished slots alike; a violation would be a
            # KV-maintenance bug (stale/short alloc) → illegal memory access.
            alloc_a1 = self.slot_state.kv_allocated_lens[a1_t]
            bad_orig = int((orig_a1 < 0).sum().item())
            bad_alloc = int((alloc_a1 < seq_lens_a1).sum().item())
            if bad_orig or bad_alloc:
                raise RuntimeError(
                    f"SBP KV invariant violated: {bad_orig} slots with orig<0, "
                    f"{bad_alloc} slots with kv_allocated<seq_len "
                    f"(min orig={int(orig_a1.min().item())}, "
                    f"min slack={int((alloc_a1 - seq_lens_a1).min().item())}) "
                    f"— would read outside allocated KV. a1={a1_list[:8]}"
                )
        ctx_a1 = SMCDecodeContext(
            orig_seq_lens=orig_a1,
            orig_seq_lens_cpu=orig_a1.cpu(),
            orig_seq_lens_sum=int(orig_a1.sum().item()),
            new_seq_lens=seq_lens_a1,
            gamma=gamma,
        )
        draft_input_a1 = SMCDraftInput(
            verified_id=anchor_a1,
            num_tokens_per_req=gamma + 1,
            decode_ctx=ctx_a1,
        )
        draft_input_a1.active_slots_cpu = a1_list
        batch_a1 = self.slot_state.build_model_worker_batch(
            draft_input_a1, active=a1_t, active_list=a1_list
        )
        pending_a1 = PendingDecodeStep(
            batch=batch_a1, ctx=ctx_a1, tag=spec.tag, epoch=spec.epoch
        )
        resp_a1 = DraftStepResp(
            tokens=tokens_a1, logprobs=logprobs_a1, tag=spec.tag, epoch=spec.epoch
        )
        return pending_a1, resp_a1, tokens_a1


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

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

        # SMCSD_ASYNC_BONUS (default off): the async/staggered FULL-bonus decode path
        # (see docs/smc/async_bonus_design.md). It commits the exact target bonus on
        # every anchor, so the verifier does NOT need the no-bonus "drafter-known
        # anchor" invariant below — but the drafter must still emit gamma+1 columns
        # (SMCSD_DROP_BONUS=1 stays set in the env) so the bet x_g1 = tokens[:, gamma]
        # exists for the run-ahead.
        self._async_bonus = get_bool_env_var("SMCSD_ASYNC_BONUS", "false")
        # Optional KV/no-op-proof assertions (SMCSD_ASYNC_BONUS_DEBUG=1): assert
        # the ragged verify ctx matches the baseline frontier and stays inside
        # allocated KV (docs/smc/async_bonus_design.md §S2 test).
        self._async_bonus_debug = get_bool_env_var("SMCSD_ASYNC_BONUS_DEBUG", "false")

        if not get_bool_env_var("SMCSD_DROP_BONUS", "false") and not self._async_bonus:
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
        # Bonus-bet modes: fire the next window on the drafter-known x_g1 (overlap),
        # but COMMIT the exact target bonus b on every window (db=False, the
        # barrier-bonus path applied to interior windows too).  BET_KEEP keeps the
        # speculative window on a miss (1-token drafter seam, reweighted — full
        # overlap, max accuracy IF the seam variance is cheap); BET_DISCARD drains
        # and re-drafts every particle from b on a miss (no drift = lockstep+bonus
        # accuracy, but pays a serial re-draft).
        self._bet_discard = get_bool_env_var("SMCSD_BET_DISCARD", "false")
        self._bet_keep = get_bool_env_var("SMCSD_BET_KEEP", "false")
        self._bet = self._bet_discard or self._bet_keep
        _on = [
            n
            for n, v in (
                ("SMCSD_SPEC_BARRIER", self._spec_barrier),
                ("SMCSD_BARRIER_BONUS", self._barrier_bonus),
                ("SMCSD_BET_DISCARD", self._bet_discard),
                ("SMCSD_BET_KEEP", self._bet_keep),
                ("SMCSD_ASYNC_BONUS", self._async_bonus),
            )
            if v
        ]
        if len(_on) > 1:
            raise RuntimeError(f"Mutually exclusive SMC mode flags set: {_on}")
        self._bet_stats = get_bool_env_var("SMCSD_BET_STATS", "false")
        self._bet_n = 0
        self._bet_miss_n = 0
        self._n_redraft = 0
        # Fused S4 (docs/smc/async_bonus_design.md §S4 + the FUSED control-flow
        # override): per-particle frontier STAGGER.  One in-flight drafted window
        # per active particle (depth D=1), tagged 'runahead' (seeded by the bet
        # x_g1) or 'redraft' (seeded by the bonus b).  Each round fires ONE ragged
        # draft StepReq that advances matched particles to a new window AND
        # re-drafts the mismatched minority in place — one pass advances everyone.
        self._fused = self._async_bonus and get_bool_env_var("SMCSD_FUSED_S4", "true")
        # Depth-2 re-draft folding (docs/smc/async_bonus_design.md §DEPTH-2),
        # SMCSD_ASYNC_BONUS_DEPTH2=1 (off by default — the committed depth-1 fused
        # path stays byte-identical): VERIFY-FIRST single-fire.  Verify each window
        # FIRST (commit the exact bonus b), THEN fire the next window directly off b
        # in ONE merged pass — no bet, no miss, no re-draft, so PASSES/WINDOW -> 1.0
        # (folds away the depth-1 serial re-draft, the +0.31 passes / `prep` bucket).
        # The trade is the lost run-ahead OVERLAP (the next draft cannot start until
        # this verify finishes); at K=2 the lost overlap eats the win (the recv
        # drafter-wait grows 33%->58%; see the report).  Measured ~92 tok/s (≈ Mode A
        # accuracy but NOT Mode A's 106 throughput): depth-2 needs K>2 interior
        # windows to amortise the lag — at K=2 the fold cannot beat the overlap.
        self._depth2 = self._fused and get_bool_env_var(
            "SMCSD_ASYNC_BONUS_DEPTH2", "false"
        )
        self._inflight = 0
        # Throughput proof (docs/smc/async_bonus_design.md §PROFILE): drafter
        # StepReqs sent / windows committed = passes/window (fused≈1.0, ModeA≈1.34).
        self._passes_sent = 0
        self._windows_committed = 0
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

            # Fused S4 drain guard (analog of the SBP guard above): the fused
            # train fully drains its in-flight windows at every resample barrier,
            # so _inflight is 0 between trains.  This guard is the belt-and-braces
            # drain in case a future path leaves a window in flight before a
            # prefill/pause/idle (docs/smc/async_bonus_design.md §3 / §7.4).
            if self._inflight > 0 and (
                self._engine_paused
                or self.waiting_groups
                or self.slot_state.is_empty()
            ):
                self._commit_async_bonus_standalone()

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
                if self._depth2:
                    self._run_fused_bonus_train_depth2()
                elif self._fused:
                    self._run_fused_bonus_train()
                elif self._async_bonus:
                    self._run_async_bonus_train()
                else:
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

    def _run_async_bonus_train(self) -> None:
        """Async/staggered FULL-bonus decode (SMCSD_ASYNC_BONUS).

        S3 (docs/smc/async_bonus_design.md §S3): ALWAYS-bonus commit, synchronous
        (no run-ahead overlap yet — that is S4).  Built on the S2 ragged-verify
        mirror, with the always-bonus accuracy lever added: every window calls
        `finish_decode(..., drop_bonus=False)`, so the committed anchor is the
        EXACT target bonus `b = result.next_draft_input.verified_id` (weighting
        gamma columns), NOT the drafter's bet `tokens[:, gamma]`.

        The train stays synchronous via a full re-draft on any bet miss — exactly
        `SMCSD_BET_DISCARD`'s mechanics (`_run_decode_train:438-457`): the next
        window is still prefetched off the drafter-known bet `x_g1` to overlap the
        verify, but if ANY particle's `x_g1 != b` the speculative window is drained
        and the WHOLE active set is re-drafted from `b` with `rollback=gamma+1`.
        At S3 this is BET_DISCARD behavior routed through the async-bonus loop +
        the `_build_ragged_ctx` verify ctx; the per-particle SUBSET re-draft is S4.

        `prepare_for_decode`'s KV ALLOCATION is KEPT (`_prepare_decode_batch` /
        `_prepare_decode_batch_fixed` still run); only the verify-ctx construction
        is swapped (S2 contract).  Flag-off stays byte-identical.
        """
        worker = self.draft_worker
        client = self._draft_client
        K = self.barrier_k
        tm = self._timing

        # ── Window 0: cold-prepare (KEEP the allocation), ragged verify ctx ──
        batch = self._prepare_decode_batch()
        if batch is None:
            return
        active_list = list(batch.spec_info.active_slots_cpu)
        active_t = torch.tensor(active_list, dtype=torch.int64, device=self.device)
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        epoch = next(self._epoch)
        tag = next(self._tag)
        # Allocation already done by _prepare_decode_batch; swap ONLY the verify
        # ctx.  start_decode fires the StepReq from the ragged ctx's
        # orig_seq_lens_cpu (== baseline on a uniform frontier).
        ragged_batch, _ = self._build_ragged_ctx(active_t, active_list)
        pending = worker.start_decode(ragged_batch, tag=tag, epoch=epoch)

        # ── Windows 0..K-1: prefetch overlap, barrier resample ──
        for w in range(K):
            t0 = time.perf_counter() if tm else 0.0
            resp = client.recv_step_resp()
            if tm:
                self._t["recv"] += time.perf_counter() - t0
            if resp.tag != pending.tag or resp.epoch != pending.epoch:
                raise RuntimeError(
                    f"Async-bonus step reply tag/epoch mismatch: got "
                    f"({resp.tag},{resp.epoch}), expected "
                    f"({pending.tag},{pending.epoch})"
                )
            is_last = w == K - 1

            next_tag = None
            seq_lens_next = None
            if not is_last:
                # Prefetch the next window off the drafter-known x_g1 (the bet) so
                # the drafter computes it while we verify this window — the overlap.
                # We COMMIT the exact target bonus b below; the bet "wins" per
                # particle iff x_g1 == b, else the speculative window is re-drafted.
                anchor_next = torch.from_numpy(resp.tokens)[:, self.gamma].tolist()
                seq_lens_next = self.slot_state.seq_lens[active_t].tolist()
                next_tag = next(self._tag)
                worker.send_step_req(
                    active_list, anchor_next, seq_lens_next,
                    tag=next_tag, epoch=epoch,
                )

            t0 = time.perf_counter() if tm else 0.0
            # S3: ALWAYS-bonus — drop_bonus=False commits the exact target sample
            # b (weighting gamma columns), exactly like SMCSD_BET_DISCARD.
            result = worker.finish_decode(pending, resp, drop_bonus=False)
            self._writeback_window(result, active_t)

            if not is_last:
                # The next window was already fired off the drafter's bet x_g1; we
                # just committed the exact target bonus b.  Per particle the bet
                # "won" iff x_g1 == b.  (resp.tokens carries gamma+1 columns: the
                # drafter runs no-bonus, so column gamma is its x_g1.)  On ANY miss,
                # discard the speculative window and re-draft EVERY particle from b
                # — synchronous BET_DISCARD mechanics (S4 makes this a subset).
                x_g1 = torch.from_numpy(resp.tokens)[:, self.gamma]
                b_cpu = result.next_draft_input.verified_id.cpu()
                if self._async_bonus_debug:
                    # The committed anchor MUST be the bonus sample b, never the
                    # bet tokens[:, gamma] (the S3 accuracy lever).  finish_decode
                    # with drop_bonus=False sets verified_id from the target
                    # multinomial, so b == the writeback's committed anchor
                    # (verified_ids is int32; compare on int64).
                    committed = self.slot_state.verified_ids[active_t].cpu().to(torch.int64)
                    if not torch.equal(committed, b_cpu.to(torch.int64)):
                        raise RuntimeError(
                            "ASYNC_BONUS S3: committed anchor != target bonus "
                            f"(committed[:8]={committed[:8].tolist()}, "
                            f"b[:8]={b_cpu[:8].tolist()})"
                        )
                bet_miss = x_g1 != b_cpu
                if self._bet_stats:
                    self._bet_n += int(bet_miss.numel())
                    self._bet_miss_n += int(bet_miss.sum().item())
                if bool(bet_miss.any()):
                    # Drain the speculative StepResp (FIFO), then re-fire the whole
                    # active set from b reusing seq_lens_next with rollback=gamma+1
                    # so the drafter undoes the discarded window's seq_len advance
                    # (its length assert passes; the discarded KV is overwritten in
                    # place — slack reuse, no free, no leak).
                    stale = client.recv_step_resp()
                    if stale.tag != next_tag or stale.epoch != epoch:
                        raise RuntimeError(
                            f"ASYNC_BONUS drain tag/epoch mismatch: got "
                            f"({stale.tag},{stale.epoch}), expected ({next_tag},{epoch})"
                        )
                    self._n_redraft += 1
                    next_tag = next(self._tag)
                    worker.send_step_req(
                        active_list, b_cpu.tolist(), seq_lens_next,
                        tag=next_tag, epoch=epoch, rollback=self.gamma + 1,
                    )
                if self._bet_stats and self._bet_n >= 500:
                    print(
                        f"[ASYNC_BONUS_BET] {self._bet_n} bets: "
                        f"x_g1!=b={100*self._bet_miss_n/self._bet_n:.1f}% "
                        f"redrafts={self._n_redraft}",
                        flush=True,
                    )
                    self._bet_n = self._bet_miss_n = 0
            if tm:
                self._t["verify"] += time.perf_counter() - t0

            if not is_last:
                t0 = time.perf_counter() if tm else 0.0
                # KEEP prepare_for_decode's allocation (advances seq_lens +
                # allocs KV over the fixed set); swap ONLY the verify ctx.
                self._prepare_decode_batch_fixed(active_t, active_list)
                next_batch, next_ctx = self._build_ragged_ctx(active_t, active_list)
                if tm:
                    self._t["prep"] += time.perf_counter() - t0
                pending = PendingDecodeStep(
                    batch=next_batch,
                    ctx=next_ctx,
                    tag=next_tag,
                    epoch=epoch,
                )
            self._t_windows += 1

        # ── Barrier: resample (+ ragged-now reset-to-zero, flag-gated inside
        #    _barrier_resample), rebuild, finalize-drain. ──
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
                    f"[ASYNC_BONUS_TIMING] {n} windows: "
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

    # ── Fused S4: per-particle frontier STAGGER (the throughput slice) ──

    def _fire_fused_window(self, active_list, active_t, verified_ids_cpu, rollback,
                           tag, epoch):
        """Fire ONE ragged draft StepReq advancing every in-flight window at its
        own per-particle frontier, and snapshot the verify ctx for it.

        `slot_state.seq_lens` already holds each window's IN-FLIGHT (advanced)
        frontier (committed rows bumped +gamma+1 by the caller's
        `prepare_for_decode`; re-drafted rows left in place — the drafter rewinds
        via the per-slot `rollback`).  The StepReq's committed-prefix seq_lens are
        `seq_lens - (gamma+1)` per slot (== the drafter mirror after rollback).
        Returns the PendingDecodeStep covering this StepReq.  `_inflight += 1`."""
        gamma = self.gamma
        seq_lens = self.slot_state.seq_lens[active_t]
        orig_cpu = (seq_lens - (gamma + 1)).cpu().tolist()
        self._draft_client.send_step(
            slots=active_list,
            verified_ids=verified_ids_cpu,
            seq_lens=orig_cpu,
            tag=tag,
            epoch=epoch,
            rollback=rollback,
        )
        self._inflight += 1
        self._passes_sent += 1
        # The verify ctx is the per-particle ragged frontier (each slot scores its
        # own in-flight window at orig = seq_lens-(gamma+1)).
        ragged_batch, ragged_ctx = self._build_ragged_ctx(active_t, active_list)
        return PendingDecodeStep(
            batch=ragged_batch, ctx=ragged_ctx, tag=tag, epoch=epoch
        )

    def _prealloc_train_kv(self, active_t, budget: int) -> None:
        """Grow each active slot's KV allocation to cover the WHOLE train's
        `budget` tokens in ONE pass, so interior windows advance the frontier with
        a cheap `seq_lens` scatter (no per-window `alloc_token_slots` /
        `assign_req_to_token_pool_func`).  Mirrors `from_slot_gather`'s allocation
        arithmetic (info.py:84-100) but bumps only `kv_allocated_lens`, NOT
        `seq_lens` — the windows advance `seq_lens` themselves, exactly `budget`
        over the train, so the last window lands slack-free
        (`kv_allocated_lens == seq_lens`) for `_fused_resample`'s clone."""
        from sglang.srt.mem_cache.common import alloc_token_slots
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        ss = self.slot_state
        bs = active_t.numel()
        seq_lens = ss.seq_lens[active_t]
        kv_alloc = ss.kv_allocated_lens[active_t]
        # Allocate the gap between the current allocation and seq_lens+budget.
        alloc_start = torch.maximum(kv_alloc, seq_lens)
        new_alloc = torch.clamp(seq_lens + budget - alloc_start, min=0)
        num_needed = int(new_alloc.sum().item())  # single GPU→CPU sync per TRAIN
        if num_needed == 0:
            return
        nxt_kv_lens = alloc_start + new_alloc
        out_cache_loc = alloc_token_slots(ss.tree_cache, num_needed)
        assign_req_to_token_pool_func(
            ss.req_pool_indices[active_t],
            ss.req_to_token_pool.req_to_token,
            alloc_start.to(torch.int32),
            nxt_kv_lens.to(torch.int32),
            out_cache_loc,
            bs,
        )
        ss.kv_allocated_lens[active_t] = nxt_kv_lens

    def _run_fused_bonus_train(self) -> None:
        """Fused S4 (docs/smc/async_bonus_design.md §S4 + the FUSED override): the
        run-ahead OVERLAP + per-particle SUBSET re-draft, the throughput slice.

        The win over Mode A: Mode A fires one overlapped run-ahead off the bet x_g1,
        then on ANY of the N particles missing (1-p^N ≈ 88%) re-drafts the WHOLE
        active set — a second blocking drafter pass that dominates.  The FUSED loop
        re-drafts ONLY the mismatched minority (~20%), folded with the matched
        run-ahead rows into ONE merged verify (the "one pass advances everyone").

        PER ROUND (overlap preserved — the run-ahead is fired BEFORE the verify off
        the drafter-known bet x_g1):
          1. recv the in-flight window; FIRE the next round's run-ahead off x_g1 for
             EVERY active particle (one pass; overlaps the verify below).
          2. RAGGED VERIFY this window (db=False -> commit the exact bonus b).
          3. MATCH/SPLIT: x_g1 == b -> matched (the fired run-ahead stands);
             else mismatched -> the fired run-ahead is invalid (seeded by x_g1, not
             the committed anchor b).  Re-draft ONLY the mismatched subset in place
             (rollback gamma+1, seeded by b) — one mixed StepReq — and MERGE the
             matched run-ahead reply + the re-draft reply into the next verify ctx.
             Frontiers stay UNIFORM (both sub-batches at the same advanced frontier)
             — no stagger, no KV slack.
          4. After K committed rounds: DRAIN the last in-flight, ESS resample
             (reset-to-zero + finished-rider exclusion), rebuild, finalize-drain.
        """
        worker = self.draft_worker
        client = self._draft_client
        K = self.barrier_k
        tm = self._timing
        gamma = self.gamma

        # ── Window 0: cold-prepare (advance + alloc), fire off committed anchor. ──
        batch = self._prepare_decode_batch()
        if batch is None:
            return
        active_list = list(batch.spec_info.active_slots_cpu)
        active_t = torch.tensor(active_list, dtype=torch.int64, device=self.device)
        bs = len(active_list)
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        epoch = next(self._epoch)
        tag = next(self._tag)
        pending = self._fire_fused_window(
            active_list, active_t,
            self.slot_state.verified_ids[active_t].to(torch.int64).cpu().tolist(),
            0, tag, epoch,
        )

        # PREP AMORTIZATION (docs/smc/async_bonus_design.md §S2 — keep prep cheap):
        # window-0 cold-prepare allocated 1 window; grow each slot's KV to cover the
        # remaining (K-1) interior windows in ONE pass now, so interior step-1 prep
        # advances `seq_lens` with a cheap scatter + ragged gather instead of a
        # per-window `alloc_token_slots`/`assign_req_to_token_pool_func`.  The train
        # advances `seq_lens` exactly (K-1)*(gamma+1) more times, so the last window
        # lands slack-free (`kv_allocated == seq_lens`) for `_fused_resample`.
        if K > 1:
            self._prealloc_train_kv(active_t, (K - 1) * (gamma + 1))

        # When a miss round drains the run-ahead to re-draft + merge, the resulting
        # window is already recv'd — stash it so the next round skips its recv (and
        # nothing is in flight to overlap that round).  None => recv fresh (overlap).
        stashed_resp = None
        for w in range(K):
            t0 = time.perf_counter() if tm else 0.0
            if stashed_resp is not None:
                resp = stashed_resp
                stashed_resp = None
            else:
                resp = client.recv_step_resp()
                self._inflight -= 1
            if tm:
                self._t["recv"] += time.perf_counter() - t0
            if resp.tag != pending.tag or resp.epoch != pending.epoch:
                raise RuntimeError(
                    f"Fused-bonus step reply tag/epoch mismatch: got "
                    f"({resp.tag},{resp.epoch}), expected "
                    f"({pending.tag},{pending.epoch})"
                )
            is_last = w == K - 1
            x_g1 = torch.from_numpy(resp.tokens)[:, gamma]              # (bs,) int64

            # 1. OVERLAP: fire the next round's run-ahead off the drafter-known bet
            #    x_g1 (no verify needed) for EVERY particle, BEFORE the verify, so
            #    the drafter computes it while we verify locally.  ONE gather
            #    (`prepare_for_decode` via `_prepare_decode_batch_fixed`) advances
            #    the whole set uniformly (+gamma+1), allocs KV, and yields the verify
            #    ctx in one shot (no second ragged gather — the S3-path prep, reused).
            #    (Skipped on the last window — it drains into the barrier.)
            ra_pending = None
            if not is_last:
                t0 = time.perf_counter() if tm else 0.0
                self.slot_state.verified_ids[active_t] = x_g1.to(
                    dtype=self.slot_state.verified_ids.dtype, device=self.device
                )
                # Interior prep is now ALLOCATION-FREE (KV pre-allocated for the
                # whole train above): advance the frontier with a cheap scatter and
                # gather the ragged verify ctx (no alloc).  `_build_ragged_ctx`
                # reads `orig = seq_lens-(gamma+1)` == this window's committed
                # prefix, identical to the old `_prepare_decode_batch_fixed` ctx.
                self.slot_state.seq_lens[active_t] += gamma + 1
                ra_batch, ra_ctx = self._build_ragged_ctx(active_t, active_list)
                ra_tag = next(self._tag)
                self._draft_client.send_step(
                    slots=active_list, verified_ids=x_g1.tolist(),
                    seq_lens=ra_ctx.orig_seq_lens_cpu.tolist(),
                    tag=ra_tag, epoch=epoch, rollback=0,
                )
                self._inflight += 1
                self._passes_sent += 1
                ra_pending = PendingDecodeStep(
                    batch=ra_batch, ctx=ra_ctx, tag=ra_tag, epoch=epoch,
                )
                if tm:
                    self._t["prep"] += time.perf_counter() - t0

            # 2. RAGGED VERIFY this window (db=False -> commit the exact bonus b).
            t0 = time.perf_counter() if tm else 0.0
            result = worker.finish_decode(pending, resp, drop_bonus=False)
            self._writeback_window(result, active_t)
            b_cpu = result.next_draft_input.verified_id.cpu().to(torch.int64)
            self._windows_committed += 1
            if tm:
                self._t["verify"] += time.perf_counter() - t0

            if is_last:
                break

            # 3. MATCH/SPLIT + SUBSET re-draft + MERGE.  Mismatched rows (x_g1 != b):
            #    the run-ahead we fired off x_g1 is invalid (committed anchor is b).
            #    Re-draft ONLY that subset from b in place (rollback gamma+1) — a
            #    second pass over the minority — and MERGE the matched rows (from the
            #    already-fired run-ahead reply) + the re-drafted rows into the next
            #    verify ctx.  The miss subset re-draws its window at the SAME advanced
            #    frontier (uniform, no slack); matched rows keep the run-ahead they
            #    already drafted.  Mode A re-drafts the WHOLE set on ANY miss; the
            #    FUSED loop re-drafts only the ~20% minority -> ~1 pass/window.
            t0 = time.perf_counter() if tm else 0.0
            miss_mask = x_g1 != b_cpu
            n_miss = int(miss_mask.sum().item())
            if self._bet_stats:
                self._bet_n += bs
                self._bet_miss_n += n_miss
            if n_miss:
                # MISS round: the run-ahead's miss rows were drafted off the wrong
                # anchor (x_g1, not b).  DRAIN it (keep matched rows via the merge),
                # re-draft ONLY the miss subset from b in place (rollback gamma+1),
                # and MERGE.  The merged window is already recv'd (no overlap THIS
                # round) — stashed for the next round.  Misses are the minority, so
                # most rounds stay in the overlapped (matched) branch below.
                self._n_redraft += 1
                ra_resp = client.recv_step_resp()
                self._inflight -= 1
                if ra_resp.tag != ra_pending.tag or ra_resp.epoch != ra_pending.epoch:
                    raise RuntimeError(
                        f"Fused-bonus run-ahead drain tag/epoch mismatch: got "
                        f"({ra_resp.tag},{ra_resp.epoch}), expected "
                        f"({ra_pending.tag},{ra_pending.epoch})"
                    )
                miss_idx = miss_mask.nonzero(as_tuple=True)[0]
                miss_list = [active_list[i] for i in miss_idx.tolist()]
                b_miss = b_cpu[miss_idx]
                miss_slots_t = active_t[miss_idx.to(self.device)]
                orig_miss = (self.slot_state.seq_lens[miss_slots_t]
                             - (gamma + 1)).cpu().tolist()
                rd_tag = next(self._tag)
                client.send_step(
                    slots=miss_list, verified_ids=b_miss.tolist(),
                    seq_lens=orig_miss, tag=rd_tag, epoch=epoch,
                    rollback=gamma + 1,
                )
                self._inflight += 1
                self._passes_sent += 1
                rd_resp = client.recv_step_resp()
                self._inflight -= 1
                if rd_resp.tag != rd_tag or rd_resp.epoch != epoch:
                    raise RuntimeError(
                        f"Fused-bonus subset re-draft tag/epoch mismatch: got "
                        f"({rd_resp.tag},{rd_resp.epoch}), expected ({rd_tag},{epoch})"
                    )
                # MERGE: matched rows ← run-ahead reply, miss rows ← re-draft reply.
                merged_tokens = ra_resp.tokens.copy()
                merged_logprobs = ra_resp.logprobs.copy()
                miss_np = miss_idx.numpy()
                merged_tokens[miss_np] = rd_resp.tokens
                merged_logprobs[miss_np] = rd_resp.logprobs
                stashed_resp = DraftStepResp(
                    tokens=merged_tokens, logprobs=merged_logprobs,
                    tag=rd_tag, epoch=epoch,
                )
                # The next verify ctx x0 must equal each row's drafter anchor
                # (x_g1 matched / b missed).
                next_anchor = torch.where(miss_mask, b_cpu, x_g1)
                self.slot_state.verified_ids[active_t] = next_anchor.to(
                    dtype=self.slot_state.verified_ids.dtype, device=self.device
                )
                ragged_batch, ragged_ctx = self._build_ragged_ctx(active_t, active_list)
                pending = PendingDecodeStep(
                    batch=ragged_batch, ctx=ragged_ctx, tag=rd_tag, epoch=epoch,
                )
            else:
                # MATCHED round: the run-ahead we fired off x_g1 is valid and STAYS
                # IN FLIGHT — the next round recvs it (the overlap).  Its verify ctx
                # x0 is x_g1 (already written to verified_ids when fired).
                pending = ra_pending
            if tm:
                self._t["prep"] += time.perf_counter() - t0
            self._t_windows += 1

            if self._bet_stats and self._bet_n >= 500:
                print(
                    f"[FUSED_S4] {self._bet_n} bets: "
                    f"x_g1!=b={100*self._bet_miss_n/max(self._bet_n,1):.1f}% "
                    f"subset_redrafts={self._n_redraft}",
                    flush=True,
                )
                self._bet_n = self._bet_miss_n = 0

        # ── Barrier: the last window already verified+committed above and fired no
        #    run-ahead, so the pipeline is drained (_inflight == 0) and every
        #    particle is at the SAME uniform frontier (slack-free).  ESS resample. ──
        t0 = time.perf_counter() if tm else 0.0
        self._fused_resample()
        self.slot_state.rebuild_active_slots()
        self._drain_finished_groups()
        if tm:
            self._t["barrier"] += time.perf_counter() - t0
            if self._t_windows >= 200:
                tot = sum(self._t.values()) or 1.0
                n = self._t_windows
                pw = (self._passes_sent / self._windows_committed
                      if self._windows_committed else 0.0)
                print(
                    f"[FUSED_S4_TIMING] {n} windows: "
                    f"recv(drafter-wait)={100*self._t['recv']/tot:.0f}% "
                    f"verify={100*self._t['verify']/tot:.0f}% "
                    f"prep={100*self._t['prep']/tot:.0f}% "
                    f"barrier={100*self._t['barrier']/tot:.0f}% | per-window "
                    f"recv={1e3*self._t['recv']/n:.2f}ms "
                    f"verify={1e3*self._t['verify']/n:.2f}ms | "
                    f"PASSES/WINDOW={pw:.3f} "
                    f"(sent={self._passes_sent} committed={self._windows_committed})",
                    flush=True,
                )
                self._t = {k: 0.0 for k in self._t}
                self._t_windows = 0
                self._passes_sent = 0
                self._windows_committed = 0
        self.last_batch = tracking_batch

    def _run_fused_bonus_train_depth2(self) -> None:
        """Fused S4 + DEPTH-2 re-draft folding (docs/smc/async_bonus_design.md
        §DEPTH-2): FOLD the per-round re-draft into ONE merged draft pass by firing
        each round's window off the EXACT committed anchor `b` (verify-first), so the
        re-draft never happens — there is no bet to miss.  PASSES/WINDOW → 1.0.

        The depth-1 cost is the SERIAL re-draft on a miss: depth-1 fires the next
        run-ahead off the drafter-known bet x_g1 BEFORE the verify (overlap), but the
        committed anchor b is only known AFTER the verify, so on a miss (≈84% of
        rounds, any of N) it must DRAIN the wrong run-ahead and fire a BLOCKING
        subset re-draft from b — a second, non-overlapped drafter pass (the +0.31
        passes/window).

        Depth-2 removes the bet entirely: it VERIFIES this round's window first
        (committing the exact bonus b), THEN fires the next window's draft directly
        off b in ONE merged full-width pass.  No bet, no miss, no re-draft, no drain:
        exactly ONE draft pass per committed window.  The cost is the lost run-ahead
        OVERLAP (the next draft cannot start until this verify finishes); the win is
        eliminating the ≈84%-of-rounds serial re-draft.  Net is measured against
        Mode A's 106 tok/s.

        PER ROUND:
          1. recv the in-flight window (fired off the correct anchor last round).
          2. RAGGED VERIFY it (db=False → commit the exact bonus b).
          3. FIRE the next window off b for the WHOLE active set (one pass), unless
             this was the last (Kth) window — then drain into the barrier.
        At the barrier every particle committed exactly K windows at a uniform
        slack-free frontier (every fire advances +gamma+1; no in-place re-draft),
        so `_fused_resample` runs unchanged.
        """
        worker = self.draft_worker
        client = self._draft_client
        K = self.barrier_k
        tm = self._timing
        gamma = self.gamma
        G = gamma + 1

        # ── Window 0: cold-prepare (advance + alloc), fire off committed anchor. ──
        batch = self._prepare_decode_batch()
        if batch is None:
            return
        active_list = list(batch.spec_info.active_slots_cpu)
        active_t = torch.tensor(active_list, dtype=torch.int64, device=self.device)
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        epoch = next(self._epoch)
        tag = next(self._tag)
        pending = self._fire_fused_window(
            active_list, active_t,
            self.slot_state.verified_ids[active_t].to(torch.int64).cpu().tolist(),
            0, tag, epoch,
        )
        # Pre-grow KV to cover the whole train in one pass (interior prep is then an
        # allocation-free scatter, same as depth-1).  Every fire advances seq_lens
        # by exactly gamma+1 (no in-place re-draft), so the train advances seq_lens
        # exactly K times total → budget (K-1)*(gamma+1) beyond window-0's one window.
        if K > 1:
            self._prealloc_train_kv(active_t, (K - 1) * G)

        for w in range(K):
            t0 = time.perf_counter() if tm else 0.0
            resp = client.recv_step_resp()
            self._inflight -= 1
            if tm:
                self._t["recv"] += time.perf_counter() - t0
            if resp.tag != pending.tag or resp.epoch != pending.epoch:
                raise RuntimeError(
                    f"Fused-depth2 step reply tag/epoch mismatch: got "
                    f"({resp.tag},{resp.epoch}), expected "
                    f"({pending.tag},{pending.epoch})"
                )
            is_last = w == K - 1

            # ── 1. VERIFY this window (db=False → commit the exact bonus b) ──
            t0 = time.perf_counter() if tm else 0.0
            result = worker.finish_decode(pending, resp, drop_bonus=False)
            self._writeback_window(result, active_t)
            b_cpu = result.next_draft_input.verified_id.cpu().to(torch.int64)
            self._windows_committed += 1
            if tm:
                self._t["verify"] += time.perf_counter() - t0

            if is_last:
                break

            # ── 2. FIRE the next window off the exact committed anchor b ──
            # ONE merged full-width pass, rollback 0 (a fresh window).  No bet, so no
            # miss and no re-draft — the depth-2 fold.  `process_batch_result` already
            # set verified_ids = b (step b); re-affirm it, advance the frontier, and
            # build the ragged verify ctx for this fire's window.
            t0 = time.perf_counter() if tm else 0.0
            self.slot_state.verified_ids[active_t] = b_cpu.to(
                dtype=self.slot_state.verified_ids.dtype, device=self.device
            )
            self.slot_state.seq_lens[active_t] += G
            ra_batch, ra_ctx = self._build_ragged_ctx(active_t, active_list)
            ra_tag = next(self._tag)
            self._draft_client.send_step(
                slots=active_list, verified_ids=b_cpu.tolist(),
                seq_lens=ra_ctx.orig_seq_lens_cpu.tolist(),
                tag=ra_tag, epoch=epoch, rollback=0,
            )
            self._inflight += 1
            self._passes_sent += 1
            pending = PendingDecodeStep(
                batch=ra_batch, ctx=ra_ctx, tag=ra_tag, epoch=epoch,
            )
            if tm:
                self._t["prep"] += time.perf_counter() - t0
            self._t_windows += 1

        # ── Barrier: every active row committed exactly K windows and the pipeline
        #    is drained (_inflight == 0 — the last window fired no run-ahead).
        #    Frontiers are uniform + slack-free (every fire advanced seq_lens by
        #    exactly gamma+1, K times total → kv_allocated == seq_lens).  ESS
        #    resample (reset-to-zero + finished-rider exclusion). ──
        t0 = time.perf_counter() if tm else 0.0
        self._fused_resample()
        self.slot_state.rebuild_active_slots()
        self._drain_finished_groups()
        if tm:
            self._t["barrier"] += time.perf_counter() - t0
            if self._t_windows >= 200:
                tot = sum(self._t.values()) or 1.0
                n = self._t_windows
                pw = (self._passes_sent / self._windows_committed
                      if self._windows_committed else 0.0)
                print(
                    f"[FUSED_S4_TIMING] {n} windows (DEPTH2): "
                    f"recv(drafter-wait)={100*self._t['recv']/tot:.0f}% "
                    f"verify={100*self._t['verify']/tot:.0f}% "
                    f"prep={100*self._t['prep']/tot:.0f}% "
                    f"barrier={100*self._t['barrier']/tot:.0f}% | per-window "
                    f"recv={1e3*self._t['recv']/n:.2f}ms "
                    f"verify={1e3*self._t['verify']/n:.2f}ms | "
                    f"PASSES/WINDOW={pw:.3f} "
                    f"(sent={self._passes_sent} committed={self._windows_committed})",
                    flush=True,
                )
                self._t = {k: 0.0 for k in self._t}
                self._t_windows = 0
                self._passes_sent = 0
                self._windows_committed = 0
        self.last_batch = tracking_batch

    def _commit_async_bonus_standalone(self) -> None:
        """Event-loop drain guard (docs/smc/async_bonus_design.md §3 / §7.4): pop
        and commit any in-flight fused window WITHOUT continuing into a train, so
        its StepResp leaves the FIFO and its frontier advance is written back
        before a prefill/pause/idle.  Loops on `_inflight` (>=1).  Steady state
        leaves `_inflight == 0` at each barrier, so this is a belt-and-braces
        drain — without a saved pending it can only discard, so it just drains the
        FIFO replies (the frontier was already committed by the barrier)."""
        while self._inflight > 0:
            self._draft_client.recv_step_resp()
            self._inflight -= 1

    def _fused_live_row_mask(self) -> torch.Tensor:
        """A `(max_groups,)` bool: True iff the in-use group row holds NO finished
        particle.  Used to exclude finished-rider groups from a barrier's collect
        so resampling never picks a dead particle as a survivor (§4 3d-i)."""
        ss = self.slot_state
        gts = ss.group_to_slots.to(torch.int64)          # (max_groups, N), -1 = empty
        valid = gts >= 0
        gather_idx = torch.where(valid, gts, torch.zeros_like(gts))
        fin = ss.finished_mask[gather_idx] & valid        # (max_groups, N)
        any_finished = fin.any(dim=1)
        return ss.row_in_use & ~any_finished

    def _fused_resample(self) -> None:
        """Ragged-now ESS resample for the fused path (docs/smc/async_bonus_design.md
        §S5).  Collect / dispatch / commit + reset-to-zero, with finished-rider
        exclusion + the src-slack-free KV guard.  `_inflight` is 0 here (the last
        train window fired no run-ahead and drained naturally), so every particle is
        committed at the SAME uniform frontier and slack-free; the clone carries each
        particle's committed window for free.

        Finished-rider exclusion (§4, review 2 3d-i): a particle that hit EOS mid-
        train rides along in `group_to_slots`, so systematic resampling could pick
        it as a SURVIVOR (src), cloning dead KV onto a live slot that then decodes
        past EOS.  We EXCLUDE any group row that holds a finished rider from this
        barrier's collect (a `row_mask`); such a group simply resamples at a later
        barrier once the rider is dropped by `rebuild_active_slots` (or the group
        finalizes).  Simple + robust: never clones a dead particle."""
        row_mask = self._fused_live_row_mask()
        plan = self.coordinator.collect_resample_jobs_batch(
            self.slot_state, row_mask=row_mask
        )
        if plan.n_jobs == 0:
            return
        if self._async_bonus_debug:
            src = plan.src_slots.to(torch.int64)
            # src-slack-free guard (the CORRECT §4 invariant for the fused stagger):
            # `batched_resample_kv` copies src's block table [0, seq_lens[src]) into
            # dst (in-bounds in the full-width req_to_token row, refcount-shared) and
            # dispatch sets kv_allocated[dst]=kv_allocated[src].  This is consistent
            # IFF src carries NO speculative slack (kv_allocated[src]==seq_lens[src]),
            # so dst's whole claimed allocation is backed by copied entries.  The
            # drain commits every final-round window (misses re-drafted in place), so
            # every particle is slack-free here.  Ragged frontiers across particles
            # (the stagger) are fine — dst_alloc only governs phase-1 dec-ref.
            slack = int((self.slot_state.kv_allocated_lens[src]
                         != self.slot_state.seq_lens[src]).sum().item())
            if slack:
                raise RuntimeError(
                    f"FUSED_S4: {slack} resample src slots carry KV slack "
                    "(kv_allocated != seq_lens) — clone would leave dst's claimed "
                    "allocation backed by uncopied block-table entries."
                )
            # Finished-rider exclusion (§4): no resample job may pick a finished
            # particle as src (cloning a dead particle's KV onto a live slot would
            # decode past EOS).
            if bool(self.slot_state.finished_mask[src].any().item()):
                raise RuntimeError(
                    "FUSED_S4: resample picked a FINISHED src (finished-rider "
                    "survivor) — would clone dead KV onto a live slot."
                )
        self.coordinator.dispatch_resample_batch(
            plan, self.slot_state, rebuild_active=False,
        )
        self._draft_client.send_commit(
            dst_slots=plan.dst_slots.tolist(),
            src_slots=plan.src_slots.tolist(),
        )
        # Reset-to-zero over the post-rebuild active set (extend the kernel's
        # resampled-row interval_weights zero to all active rows).  finalize_score
        # left intact (B1 repair).
        self.slot_state.reset_interval_weights(self.slot_state.active_slots)

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
            if self._bet:
                db = False  # Mode A/B: commit the exact target bonus b every window
            elif is_last and self._barrier_bonus:
                db = False
            else:
                db = None
            result = worker.finish_decode(pending, resp, drop_bonus=db)
            self._writeback_window(result, active_t)

            if self._bet and not is_last:
                # The next window was already fired (above) on the drafter's x_g1;
                # we just committed the exact target bonus b.  Per particle the bet
                # "won" iff x_g1 == b.  (resp.tokens carries gamma+1 columns: the
                # drafter runs no-bonus, so column gamma is its x_g1.)
                x_g1 = torch.from_numpy(resp.tokens)[:, self.gamma]
                b_cpu = result.next_draft_input.verified_id.cpu()
                bet_miss = x_g1 != b_cpu
                if self._bet_stats:
                    self._bet_n += int(bet_miss.numel())
                    self._bet_miss_n += int(bet_miss.sum().item())
                if self._bet_discard and bool(bet_miss.any()):
                    # Discard the speculative next window (fired on x_g1) and
                    # re-draft EVERY particle from the committed bonus b.  Drain the
                    # in-flight StepResp first (FIFO), then re-fire reusing the
                    # original window's seq_lens_next with rollback=gamma+1 so the
                    # drafter undoes the discarded window's seq_len advance and its
                    # length assertion passes (the discarded KV is overwritten in
                    # place — slack reuse, no free, no leak).
                    stale = client.recv_step_resp()
                    if stale.tag != next_tag or stale.epoch != epoch:
                        raise RuntimeError(
                            f"BET_DISCARD drain tag/epoch mismatch: got "
                            f"({stale.tag},{stale.epoch}), expected ({next_tag},{epoch})"
                        )
                    self._n_redraft += 1
                    next_tag = next(self._tag)
                    worker.send_step_req(
                        active_list, b_cpu.tolist(), seq_lens_next,
                        tag=next_tag, epoch=epoch, rollback=self.gamma + 1,
                    )
                if self._bet_stats and self._bet_n >= 500:
                    mode = "DISCARD" if self._bet_discard else "KEEP"
                    print(
                        f"[BET_{mode}] {self._bet_n} bets: "
                        f"x_g1!=b={100*self._bet_miss_n/self._bet_n:.1f}% "
                        f"redrafts={self._n_redraft}",
                        flush=True,
                    )
                    self._bet_n = self._bet_miss_n = 0
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

    def _build_ragged_ctx(self, active_t, active_list):
        """Hand-build a verify PendingDecodeStep over a per-particle-ragged
        frontier (docs/smc/async_bonus_design.md §S2/§4).

        Each slot's own committed frontier is `seq_lens[slot] - (gamma+1)` — the
        prefix BEFORE the in-flight window's advance.  This GATHERS the existing
        slot tensors into a verify ctx; it does NOT allocate KV (`from_slot_gather`
        allocates, a gather does not — info.py:92-100).  For the uniform
        no-run-ahead case the allocation already happened in the train's
        `prepare_for_decode`, so this is allocation-free and, on a uniform active
        set, bit-identical to `from_slot_gather`'s ctx (the S2 no-op proof).

        Clone of the `_build_spec_a1` ctx-construction pattern, minus the
        ancestor remap (the gather here is identity — each slot scores its own
        frontier).
        """
        gamma = self.gamma
        seq_lens = self.slot_state.seq_lens[active_t]
        orig = seq_lens - (gamma + 1)
        if self._async_bonus_debug:
            # No-op proof guard: over a uniform active set, this ctx's
            # orig_seq_lens MUST equal the baseline `prepare_for_decode`/
            # from_slot_gather ctx (= current seq_lens - (gamma+1)), and the
            # frontier must stay inside allocated KV.  A violation means the
            # gather diverged from the allocating prep — a KV-maintenance bug.
            alloc = self.slot_state.kv_allocated_lens[active_t]
            bad_orig = int((orig < 0).sum().item())
            bad_alloc = int((alloc < seq_lens).sum().item())
            if bad_orig or bad_alloc:
                raise RuntimeError(
                    f"ASYNC_BONUS ragged-ctx invariant violated: {bad_orig} slots "
                    f"orig<0, {bad_alloc} slots kv_allocated<seq_len "
                    f"(min orig={int(orig.min().item())}, "
                    f"min slack={int((alloc - seq_lens).min().item())}). "
                    f"active={active_list[:8]}"
                )
        ctx = SMCDecodeContext(
            orig_seq_lens=orig,
            orig_seq_lens_cpu=orig.cpu(),
            orig_seq_lens_sum=int(orig.sum().item()),
            new_seq_lens=seq_lens,
            gamma=gamma,
        )
        draft_input = SMCDraftInput(
            verified_id=self.slot_state.verified_ids[active_t],
            num_tokens_per_req=gamma + 1,
            decode_ctx=ctx,
        )
        draft_input.active_slots_cpu = active_list
        batch = self.slot_state.build_model_worker_batch(
            draft_input, active=active_t, active_list=active_list
        )
        return batch, ctx

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
            if self._async_bonus:
                # Ragged-now reset-to-zero (docs/smc/async_bonus_design.md §2c/§4):
                # extend the kernel's resampled-row interval_weights zero to ALL
                # active rows.  finalize_score is left intact.  Inert in every
                # existing mode (gated on the flag).
                self.slot_state.reset_interval_weights(self.slot_state.active_slots)
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

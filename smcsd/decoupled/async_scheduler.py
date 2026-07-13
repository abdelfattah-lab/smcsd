"""Lag-1 exact-bonus decoupled SMC scheduler.

The retained async path overlaps draft and verify by firing a run-ahead draft
from the drafter's anchor bet, then verifies only rows whose committed target
bonus matches a valid run-ahead frontier. Miss rows catch up from the exact
target bonus while ready rows continue drafting. Resampling is explicit every
cycle so the verifier and drafter stay on the same strict group-min frontier.
"""

from __future__ import annotations

import itertools
import logging
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set

import numpy as np
import psutil
import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import configure_scheduler_process
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import DynamicGradMode
from sglang.utils import get_exception_traceback

from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from smcsd.decoupled.io_struct import DraftStepResp
from smcsd.decoupled.kv_utils import truncate_block_table_allocations
from smcsd.decoupled.scheduler import DecoupledSMCScheduler
from smcsd.decoupled.worker import PendingDecodeStep

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class LagWindowMeta:
    """Per-slot metadata for one drafted window in the lag-1 bonus pipeline."""

    orig_seq_len: int
    new_seq_len: int
    anchor_id: int


@dataclass(slots=True)
class LagReadyWindow:
    """A valid drafted window that can be target-verified later."""

    tokens: np.ndarray
    logprobs: np.ndarray
    meta: LagWindowMeta
    bet_topk: Optional[np.ndarray] = None  # (W,) top-W anchor candidates (width>=2)


@dataclass(slots=True)
class LagPendingWindow:
    """One mixed draft StepReq in flight for the lag-1 bonus pipeline.

    Metadata and validity are aligned with `active_list_T`.  For catch-up/cold
    rows validity is known when the StepReq is fired; for run-ahead rows it is
    filled in after the predecessor window is verified and the target bonus is
    known.
    """

    active_list_T: List[int]
    tag: int
    epoch: int
    metas: List[LagWindowMeta]
    valid_by_pos: List[int]  # -1 unknown, 0 invalid/stale, 1 valid/ready
    pos_by_slot: Dict[int, int]
    ancestor: Optional[np.ndarray] = None
    # Width-2: per-fired-row winning branch (0 = primary/c0, 1 = alt/c1).  Filled
    # in _lag_verify_ready once the bonus b is known; read at receive to pick the
    # primary vs alt run-ahead window.  Row-indexed (same axis as valid_by_pos).
    branch_choice: Optional[List[int]] = None
    # Width-2: per-fired-row committed bonus b for the WINNING branch (= c1 when
    # the alt won), captured in _lag_verify_ready.  The alt run-ahead was seeded by
    # c1, so this is the anchor (x0) that must condition its later verify — the
    # primary meta.anchor_id holds c0 and would score the alt window against the
    # wrong context.  Row-indexed; -1 when no alt win.
    branch_anchor: Optional[List[int]] = None


@dataclass(slots=True)
class LagPreparedStep:
    """Verifier-side metadata needed to launch one mixed lag-1 draft StepReq."""

    ordered: List[int]
    verified_ids: List[int]
    seq_lens: List[int]
    rollback_payload: object
    truncate_kv_payload: object
    metas: List[LagWindowMeta]
    valid_by_pos: List[int]
    pos_by_slot: Dict[int, int]
    # Width-2: per-row alt seed (the c1 top-2 anchor candidate) for verify rows;
    # -1 for stale/cold rows (no branch).  None => width-1 (no alt fired).
    bet_alt: Optional[List[int]]
    advance_slots: List[int]
    advance_orig: List[int]
    advance_new: List[int]
    advance_num_needed: int
    private_slots: List[int]
    private_orig: List[int]
    private_seq: List[int]


class AsyncDecoupledSMCScheduler(DecoupledSMCScheduler):
    """Decoupled SMC scheduler for lag-1 exact-bonus overlap."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sglang.srt.utils import get_bool_env_var

        # Lag-1 full-bonus pipeline (SMCSD_LAG1_BONUS=1): keep the exact target
        # bonus, but make mismatch repair per-particle and non-blocking. Rows
        # whose run-ahead anchor misses spend one cycle catching up from the exact
        # bonus while matched/ready rows continue drafting. Verification is gated
        # to the strict group-min frontier.
        lag1_bonus = get_bool_env_var("SMCSD_LAG1_BONUS", "false")
        if not lag1_bonus:
            raise RuntimeError(
                "AsyncDecoupledSMCScheduler only supports the retained lag-1 "
                "exact-bonus path. Set SMCSD_LAG1_BONUS=1."
            )
        # Optional KV/no-op-proof assertions for lag-1 stale-suffix and ragged-ctx
        # checks.
        self._lag1_debug = get_bool_env_var("SMCSD_LAG1_DEBUG", "false")

        self.barrier_k = 1
        self.gamma = self.server_args.speculative_num_steps
        self._tag = itertools.count(1)
        # Per-train epoch (fail-fast fence). Every StepReq in a train carries the
        # same epoch; the drafter echoes it back and finish_decode asserts it.
        self._epoch = itertools.count(1)
        self._bet_stats = get_bool_env_var("SMCSD_BET_STATS", "false")
        self._bet_n = 0
        self._bet_miss_n = 0
        self._torch_profiler = None
        self._torch_profile_step_n = 0
        self._torch_profile_total_steps = 0
        self._torch_profile_stopped = False
        self._init_torch_profiler("target")
        self._lag_ready: Dict[int, LagReadyWindow] = {}
        self._lag_stale: Set[int] = set()
        self._lag_stale_shared: Set[int] = set()
        self._lag_pending: Optional[LagPendingWindow] = None
        self._lag_just_verified = torch.zeros(
            self.slot_state.max_slots, dtype=torch.bool, device=self.device
        )
        self._lag_window_offsets = torch.arange(
            self.gamma + 1, dtype=torch.int64, device=self.device
        )
        self._lag_profile = get_bool_env_var("SMCSD_LAG1_PROFILE", "false")
        self._lp = {
            "cycles": 0,
            "active_rows": 0,
            "ready_rows": 0,
            "verify_rows": 0,
            "held_ready_rows": 0,
            "catchup_rows": 0,
            "cold_rows": 0,
            "eligible_rows": 0,
            "resample_jobs": 0,
            "max_lag_windows": 0,
        }
        # Width-W anchor tree (SMCSD_LAG1_ANCHOR_WIDTH): hedge the run-ahead seed
        # over the drafter's top-W anchor candidates.  DEFAULT 2 (the keeper): 70B
        # 1000q acc-neutral (-1pp, within noise) at +6.3% tok/s, traces confirm the
        # 2N draft rows stay hidden.  Set =1 for the old single-bet lag-1 (byte-
        # identical baseline).  The committed token is always the verified bonus b
        # -> accuracy-neutral.  _w2_* are oracle hit-rate counters (measurement only).
        self._anchor_width = max(int(os.environ.get("SMCSD_LAG1_ANCHOR_WIDTH", "2")), 1)
        self._lag_pending_bet_alt: Optional[List[int]] = None
        self._w2_n = 0
        self._w2_prod_hit = 0
        self._w2_top1_hit = 0
        self._w2_top2_hit = 0
        self._w2_alt_win = 0  # realized alt-branch wins (committed b == c1 != c0)
        max_slots = self.slot_state.max_slots
        self._lag_seq_lens_cpu: List[int] = [0] * max_slots
        self._lag_kv_allocated_lens_cpu: List[int] = [0] * max_slots
        self._lag_verified_ids_cpu: List[int] = [0] * max_slots
        self._lag_token_counts_cpu: List[int] = [0] * max_slots
        self._lag_finished_mask_cpu: List[bool] = [False] * max_slots
        # CLONE-JITTER and CLONE-DIV were refuted probes and should not be
        # reintroduced as scheduler modes.
        # RE-DRAFT ORACLE (SMCSD_LAG1_REDRAFT_ORACLE, correctness oracle, throughput
        # irrelevant): after resample, a descendant of a BET-HIT survivor does NOT
        # inherit the survivor's in-flight run-ahead window W'; it is forced onto the
        # stale/catch-up path to RE-DRAFT its own window from its own committed bonus b
        # (serial-baseline-grade per-clone diversity, byte-correct KV).  Bet-MISS descendants
        # are untouched (they already re-draft from their own bonus).  Lag-1 now
        # always resamples at a clean-commit boundary, so every src has alloc==seq
        # and the KV-2 slack guard (smcsd/core/scheduler.py) cannot trip.
        self._redraft_oracle = get_bool_env_var("SMCSD_LAG1_REDRAFT_ORACLE", "false")
        self._lag_bet_hit: Dict[int, bool] = {}  # verify slot -> primary bet-hit this cycle
        self._lag_redraft_clones: Set[int] = set()  # bet-hit dsts: skip W' re-adopt next recv
        # Throughput proof: drafter StepReqs sent / windows committed tracks
        # catch-up/redraft pressure in lag-1 profiling.
        self._passes_sent = 0
        self._windows_committed = 0
        # Optional timing decomposition (SMCSD_TIMING=1): how long the verifier
        # blocks on the drafter (recv wait = drafter-bound signal) vs its own
        # work (verify dispatch + writeback + prepare + barrier).
        self._timing = get_bool_env_var("SMCSD_TIMING", "false")
        self._t = {
            "recv": 0.0,
            "select": 0.0,
            "prep": 0.0,
            "ready": 0.0,
            "verify": 0.0,
            "barrier": 0.0,
        }
        self._t_windows = 0
        logger.info(
            "AsyncDecoupledSMCScheduler: lag-1 exact-bonus overlap, resample K=%d",
            self.barrier_k,
        )

    def _materialize_group(self, group):
        error_msg = super()._materialize_group(group)
        if error_msg is None:
            self._lag_sync_slots_from_state(
                list(self.slot_state.group_slot_lists[group.group_id])
            )
        return error_msg

    def _lag_sync_slots_from_state(self, slots: List[int]) -> None:
        if not slots:
            return
        ss = self.slot_state
        cpu_idx = torch.as_tensor(slots, dtype=torch.int64)
        dev_idx = cpu_idx.to(device=self.device)
        seq_vals = [int(x) for x in ss.seq_lens_host[cpu_idx].tolist()]
        kv_vals = [
            int(x) for x in ss.kv_allocated_lens[dev_idx].detach().cpu().tolist()
        ]
        verified_vals = [
            int(x) for x in ss.verified_ids[dev_idx].detach().cpu().tolist()
        ]
        count_vals = [
            int(x) for x in ss.token_counts[dev_idx].detach().cpu().tolist()
        ]
        finished_vals = [
            bool(x) for x in ss.finished_mask[dev_idx].detach().cpu().tolist()
        ]
        for slot, seq, kv, verified, count, finished in zip(
            slots,
            seq_vals,
            kv_vals,
            verified_vals,
            count_vals,
            finished_vals,
            strict=True,
        ):
            self._lag_seq_lens_cpu[slot] = seq
            self._lag_kv_allocated_lens_cpu[slot] = kv
            self._lag_verified_ids_cpu[slot] = verified
            self._lag_token_counts_cpu[slot] = count
            self._lag_finished_mask_cpu[slot] = finished

    def _lag_set_seq_lens_cpu(self, slots: List[int], values: List[int]) -> None:
        if not slots:
            return
        vals = [int(v) for v in values]
        for slot, val in zip(slots, vals, strict=True):
            self._lag_seq_lens_cpu[slot] = val
        self.slot_state.seq_lens_host[torch.as_tensor(slots, dtype=torch.int64)] = (
            torch.as_tensor(vals, dtype=torch.int64)
        )

    def _lag_truncate_kv_suffix(
        self, slots_t: torch.Tensor, new_alloc_lens: torch.Tensor
    ) -> None:
        if int(slots_t.numel()) == 0:
            return
        ss = self.slot_state
        old_alloc_lens = ss.kv_allocated_lens[slots_t].clone()
        truncate_block_table_allocations(
            ss.req_to_token_pool,
            ss.token_to_kv_pool_allocator,
            ss.req_pool_indices[slots_t].to(torch.int32),
            old_alloc_lens.to(torch.int32),
            new_alloc_lens.to(torch.int32),
        )
        ss.kv_allocated_lens[slots_t] = new_alloc_lens
        slots = [int(x) for x in slots_t.detach().cpu().tolist()]
        vals = [int(x) for x in new_alloc_lens.detach().cpu().tolist()]
        for slot, val in zip(slots, vals, strict=True):
            self._lag_kv_allocated_lens_cpu[slot] = val

    def _lag_extend_kv_for_slots(
        self,
        slots_t: torch.Tensor,
        seq_lens: torch.Tensor,
        gamma_plus_1: int,
        *,
        num_needed: int,
    ) -> torch.Tensor:
        from sglang.srt.mem_cache.common import alloc_token_slots
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        ss = self.slot_state
        bs = int(slots_t.numel())
        alloc_start = torch.maximum(ss.kv_allocated_lens[slots_t], seq_lens)
        needed_len = seq_lens + gamma_plus_1
        new_alloc = torch.clamp(needed_len - alloc_start, min=0)
        if self._lag1_debug:
            actual_needed = int(new_alloc.sum().item())
            if actual_needed != int(num_needed):
                raise RuntimeError(
                    "LAG1_BONUS KV allocation count mismatch: "
                    f"planned={num_needed} actual={actual_needed}"
                )
        nxt_kv_lens = alloc_start + new_alloc
        if num_needed > 0:
            out_cache_loc = alloc_token_slots(ss.tree_cache, int(num_needed))
            assign_req_to_token_pool_func(
                ss.req_pool_indices[slots_t],
                ss.req_to_token_pool.req_to_token,
                alloc_start.to(torch.int32),
                nxt_kv_lens.to(torch.int32),
                out_cache_loc,
                bs,
            )
        return nxt_kv_lens

    # ── Event loop: one K-window decode train per iteration ──

    @DynamicGradMode()
    def _event_loop(self) -> None:
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            if self._lag_pending is not None and (
                self._engine_paused
                or self.waiting_groups
                or self.slot_state.is_empty()
            ):
                self._lag_receive_pending()
                self.slot_state.rebuild_active_slots()
                self._lag_prune_metadata()
                self._lag_drain_finished_groups()

            if self._engine_paused:
                self.cancel_bubble_timer()
                continue

            self._lag_drain_finished_groups()

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
                with self._torch_record("target_lag1_cycle"):
                    self._run_lag1_bonus_train()
                self._torch_profile_step()
            else:
                self.cur_batch = None
                pass

    def _run_tracked_prefill(self, batch: ScheduleBatch) -> None:
        tracking_batch = self._make_runtime_tracking_batch(batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )
        result = self.run_batch(batch)
        self._process_prefill_result(batch, result)
        self.last_batch = tracking_batch

    def _init_torch_profiler(self, role: str) -> None:
        out_dir = os.environ.get("SMCSD_TORCH_PROFILE_DIR")
        if not out_dir:
            return
        roles = {
            part.strip()
            for part in os.environ.get("SMCSD_TORCH_PROFILE_ROLES", "target,draft").split(",")
            if part.strip()
        }
        if role not in roles:
            return
        wait = max(int(os.environ.get("SMCSD_TORCH_PROFILE_WAIT", "5")), 0)
        active = max(int(os.environ.get("SMCSD_TORCH_PROFILE_ACTIVE", "20")), 1)
        with_stack = os.environ.get("SMCSD_TORCH_PROFILE_WITH_STACK", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        role_dir = os.path.join(out_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        self._torch_profile_total_steps = wait + active
        self._torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=0, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                role_dir, worker_name=f"{role}-{os.getpid()}"
            ),
            record_shapes=False,
            with_stack=with_stack,
        )
        self._torch_profiler.start()
        print(
            "[TORCH_PROFILE] "
            f"role={role} dir={role_dir} wait={wait} active={active} "
            f"with_stack={with_stack}",
            flush=True,
        )

    def _torch_record(self, name: str):
        if self._torch_profiler is None or self._torch_profile_stopped:
            return nullcontext()
        return torch.profiler.record_function(name)

    def _torch_profile_step(self) -> None:
        if self._torch_profiler is None or self._torch_profile_stopped:
            return
        self._torch_profiler.step()
        self._torch_profile_step_n += 1
        if self._torch_profile_step_n >= self._torch_profile_total_steps:
            self._torch_profiler.stop()
            self._torch_profile_stopped = True
            print("[TORCH_PROFILE] role=target stopped", flush=True)

    # ── Lag-1 full-bonus pipeline (SMCSD_LAG1_BONUS) ──

    def _lag_refresh_membership_masks(self) -> None:
        return None

    def _lag_prune_metadata(self) -> None:
        live = set(self.slot_state._active_slots_list)
        self._lag_ready = {
            slot: ready for slot, ready in self._lag_ready.items() if slot in live
        }
        self._lag_stale.intersection_update(live)
        self._lag_stale_shared.intersection_update(live)
        self._lag_refresh_membership_masks()

    def _lag_drain_finished_groups(self) -> None:
        protected = set()
        if self._lag_pending is not None:
            protected.update(self._lag_pending.active_list_T)

        remaining: List[object] = []
        finalized = False
        for group in self.running_groups:
            if self.slot_state.group_has_active(
                group.group_id, self._lag_finished_mask_cpu
            ):
                remaining.append(group)
                continue
            slots = self.slot_state.group_slot_lists.get(group.group_id, [])
            if protected.intersection(slots):
                remaining.append(group)
                continue
            self._finalize_group(group)
            finalized = True
        self.running_groups = remaining
        if finalized:
            self._lag_prune_metadata()

    def _lag_receive_pending(self) -> None:
        pending = self._lag_pending
        if pending is None:
            return

        tm = self._timing
        t0 = time.perf_counter() if tm else 0.0
        resp = self._draft_client.recv_step_resp()
        if tm:
            self._t["recv"] += time.perf_counter() - t0
        if resp.tag != pending.tag or resp.epoch != pending.epoch:
            raise RuntimeError(
                f"LAG1_BONUS step reply tag/epoch mismatch: got "
                f"({resp.tag},{resp.epoch}), expected "
                f"({pending.tag},{pending.epoch})"
            )

        pos_by_slot = pending.pos_by_slot
        active_now = self.slot_state._active_slots_list
        src_ref_counts: Dict[int, int] = {}
        mapped_rows: List[tuple[int, int, int]] = []
        for slot in active_now:
            if self._redraft_oracle and slot in self._lag_redraft_clones:
                # Bet-hit clone: do NOT re-adopt the survivor's in-flight W'.  Leaving it
                # unmapped preserves its _lag_stale membership -> next cycle it re-drafts
                # its own window from its own committed bonus (the oracle's whole point).
                continue
            src = int(pending.ancestor[slot]) if pending.ancestor is not None else slot
            row = pos_by_slot.get(src)
            if row is not None:
                mapped_rows.append((slot, src, row))
                src_ref_counts[src] = src_ref_counts.get(src, 0) + 1
        for slot, src, row in mapped_rows:
            valid = pending.valid_by_pos[row]
            if valid < 0:
                raise RuntimeError(
                    "LAG1_BONUS pending row has unknown validity at receive: "
                    f"slot={slot} src={src} active_T={pending.active_list_T[:8]}"
                )
            meta = pending.metas[row]
            if valid:
                # Width-2: if this row's c1 alt branch won, take the alt run-ahead
                # window (drafter already promoted the alt KV into the slot/lineage).
                choice = (
                    pending.branch_choice[row]
                    if pending.branch_choice is not None
                    else 0
                )
                if choice == 1 and resp.alt_tokens is not None:
                    # The alt window was drafted off c1; condition its verify on c1
                    # (the committed b captured at decide time), not the primary's c0
                    # carried in `meta.anchor_id` — else the target scores the alt
                    # window against the wrong x0 → corrupted importance weights.
                    if pending.branch_anchor is None or pending.branch_anchor[row] < 0:
                        raise RuntimeError(
                            "Width-2 alt branch selected without its committed "
                            f"anchor: slot={slot} row={row}"
                        )
                    alt_anchor = pending.branch_anchor[row]
                    alt_meta = LagWindowMeta(
                        orig_seq_len=meta.orig_seq_len,
                        new_seq_len=meta.new_seq_len,
                        anchor_id=alt_anchor,
                    )
                    self._lag_ready[slot] = LagReadyWindow(
                        tokens=resp.alt_tokens[row],
                        logprobs=resp.alt_logprobs[row],
                        meta=alt_meta,
                        bet_topk=(
                            resp.alt_bet_topk[row]
                            if resp.alt_bet_topk is not None
                            else None
                        ),
                    )
                else:
                    self._lag_ready[slot] = LagReadyWindow(
                        tokens=resp.tokens[row],
                        logprobs=resp.logprobs[row],
                        meta=meta,
                        bet_topk=(
                            resp.bet_topk[row] if resp.bet_topk is not None else None
                        ),
                    )
                self._lag_stale.discard(slot)
                self._lag_stale_shared.discard(slot)
            else:
                self._lag_ready.pop(slot, None)
                self._lag_stale.add(slot)
                if src_ref_counts.get(src, 0) > 1:
                    self._lag_stale_shared.add(slot)
                else:
                    self._lag_stale_shared.discard(slot)

        self._lag_pending = None
        self._lag_refresh_membership_masks()
        if self._redraft_oracle:
            # The skip above only needed to suppress THIS pending's W' re-adoption;
            # after the descendants re-draft, they re-adopt their own windows normally.
            self._lag_redraft_clones = set()

    def _lag_select_ready_slots(self, active_list: List[int]) -> List[int]:
        if not self._lag_ready:
            return []
        ss = self.slot_state
        # Strict lag-1 rule: only rows at the group's minimum committed frontier
        # may verify.  Rows that ran ahead stay held-ready until lagging rows
        # catch up, so resample compares commensurable weights.
        active = getattr(ss, "_active_slots_set", None)
        if active is None or len(active) != len(active_list):
            active = set(active_list)
        eligible: Set[int] = set()
        counts = self._lag_token_counts_cpu
        finished = self._lag_finished_mask_cpu
        for group_slots in ss.group_slot_lists.values():
            live: List[int] = []
            min_count: Optional[int] = None
            for slot in group_slots:
                if slot not in active or finished[slot]:
                    continue
                count = counts[slot]
                live.append(slot)
                if min_count is None or count < min_count:
                    min_count = count
            if min_count is None:
                continue
            for slot in live:
                if counts[slot] == min_count:
                    eligible.add(slot)
        return [s for s in active_list if s in self._lag_ready and s in eligible]

    def _lag_max_committed_lag_windows(self) -> int:
        counts = self._lag_token_counts_cpu
        finished = self._lag_finished_mask_cpu
        G = self.gamma + 1
        max_lag = 0
        for slots in self.slot_state.group_slot_lists.values():
            live_counts = [counts[s] for s in slots if not finished[s]]
            if len(live_counts) < 2:
                continue
            max_lag = max(max_lag, (max(live_counts) - min(live_counts)) // G)
        return int(max_lag)

    def _lag_profile_add(
        self,
        *,
        active_rows: int,
        ready_rows: int,
        verify_rows: int,
        held_ready_rows: int,
        catchup_rows: int,
        cold_rows: int,
    ) -> None:
        if not self._lag_profile:
            return
        self._lp["cycles"] += 1
        self._lp["active_rows"] += active_rows
        self._lp["ready_rows"] += ready_rows
        self._lp["verify_rows"] += verify_rows
        self._lp["held_ready_rows"] += held_ready_rows
        self._lp["catchup_rows"] += catchup_rows
        self._lp["cold_rows"] += cold_rows
        self._lp["max_lag_windows"] = max(
            int(self._lp["max_lag_windows"]), self._lag_max_committed_lag_windows()
        )

    def _lag_profile_report(self) -> None:
        if not self._lag_profile:
            return
        n = int(self._lp["cycles"])
        if n < 200:
            return
        print(
            "[LAG1_PROFILE] "
            f"cycles={n} "
            f"active_rows={self._lp['active_rows']/n:.1f} "
            f"ready_rows={self._lp['ready_rows']/n:.1f} "
            f"verify_rows={self._lp['verify_rows']/n:.1f} "
            f"held_ready_rows={self._lp['held_ready_rows']/n:.1f} "
            f"catchup_rows={self._lp['catchup_rows']/n:.1f} "
            f"cold_rows={self._lp['cold_rows']/n:.1f} "
            f"eligible_rows={self._lp['eligible_rows']/n:.1f} "
            f"resample_jobs={self._lp['resample_jobs']/n:.2f}/cycle "
            f"max_lag_windows={int(self._lp['max_lag_windows'])}",
            flush=True,
        )
        self._lp = {k: 0 for k in self._lp}

    def _lag_resample_row_mask(
        self, verify_slots: Optional[List[int]] = None
    ) -> torch.Tensor:
        ss = self.slot_state
        gts = ss.group_to_slots.to(torch.int64)
        valid = gts >= 0
        gather_idx = gts.clamp_min(0)
        finished = ss.finished_mask[gather_idx] & valid
        row_mask = ss.row_in_use & ~finished.any(dim=1)

        # Only resample a group at a clean commit boundary: every live row in
        # the group just verified a ready window in this cycle.  This avoids
        # cloning while some particles carry held-ready or catch-up windows.
        just_verified = self._lag_just_verified
        just_verified.zero_()
        if verify_slots:
            just_verified[
                torch.as_tensor(verify_slots, dtype=torch.int64, device=self.device)
            ] = True
        live = valid & ~finished
        all_live_verified = (~live | just_verified[gather_idx]).all(dim=1)
        row_mask = row_mask & all_live_verified

        return row_mask

    def _lag_prepare_advance_slots(
        self, slots: List[int]
    ) -> tuple[List[int], List[int], int]:
        if not slots:
            return [], [], 0
        G = self.gamma + 1
        ss = self.slot_state
        seq_cpu = self._lag_seq_lens_cpu
        alloc_cpu = self._lag_kv_allocated_lens_cpu
        orig: List[int] = []
        new: List[int] = []
        num_needed = 0
        for slot in slots:
            old_seq = seq_cpu[slot]
            old_alloc = alloc_cpu[slot]
            new_seq = old_seq + G
            orig.append(old_seq)
            new.append(new_seq)
            if old_alloc == old_seq:
                num_needed += G
            else:
                num_needed += max(new_seq - max(old_alloc, old_seq), 0)
        return orig, new, num_needed

    def _lag_commit_advance_slots(
        self,
        slots: List[int],
        orig: List[int],
        new: List[int],
        num_needed: int,
    ) -> None:
        if not slots:
            return
        G = self.gamma + 1
        ss = self.slot_state
        active_t = torch.as_tensor(slots, dtype=torch.int64, device=self.device)
        seq = ss.seq_lens[active_t]
        if self._lag1_debug:
            seq_cpu = [int(x) for x in seq.cpu().tolist()]
            if seq_cpu != orig:
                raise RuntimeError(
                    "LAG1_BONUS seq_lens CPU mirror diverged before advance: "
                    f"slots={slots[:8]} mirror={orig[:8]} tensor={seq_cpu[:8]}"
                )
        new_seq = seq + G
        new_kv_alloc = self._lag_extend_kv_for_slots(
            active_t, seq, G, num_needed=num_needed
        )
        ss.kv_allocated_lens[active_t] = new_kv_alloc
        ss.seq_lens[active_t] = new_seq
        seq_cpu = self._lag_seq_lens_cpu
        alloc_cpu = self._lag_kv_allocated_lens_cpu
        self._lag_set_seq_lens_cpu(slots, new)
        for slot, new_seq_len in zip(slots, new, strict=True):
            seq_cpu[slot] = int(new_seq_len)
            alloc_cpu[slot] = int(max(alloc_cpu[slot], new_seq_len))

    def _lag_advance_slots(self, slots: List[int]) -> tuple[List[int], List[int]]:
        orig, new, num_needed = self._lag_prepare_advance_slots(slots)
        self._lag_commit_advance_slots(slots, orig, new, num_needed)
        return orig, new

    def _lag_prepare_stale_suffix(
        self, slots: List[int]
    ) -> tuple[
        List[int],
        List[int],
        List[int],
        List[bool],
        bool,
        bool,
        List[int],
        List[int],
        List[int],
    ]:
        """Make stale catch-up suffix KV private before re-drafting it.

        A stale row may have inherited its future suffix through a resample while
        the StepReq was in flight.  Reusing that shared suffix for catch-up would
        let one clone overwrite another clone's draft/verify KV.  Drop this
        row's ownership of [S, S+G), allocate fresh verifier cells for the same
        range, and ask the drafter to do the same via per-row truncate_kv=True.
        """
        if not slots:
            return [], [], [], [], False, False, [], [], []
        G = self.gamma + 1
        ss = self.slot_state
        seq_mirror = self._lag_seq_lens_cpu
        alloc_mirror = self._lag_kv_allocated_lens_cpu
        verified_mirror = self._lag_verified_ids_cpu
        shared = self._lag_stale_shared
        has_shared = bool(shared)
        seq_cpu: List[int] = []
        orig_cpu: List[int] = []
        anchors: List[int] = []
        truncate_flags: List[bool] = [False] * len(slots)
        private_slots: List[int] = []
        private_orig: List[int] = []
        private_seq: List[int] = []
        any_truncate = False
        all_truncate = has_shared
        for i, slot in enumerate(slots):
            seq_len = seq_mirror[slot]
            alloc_len = alloc_mirror[slot]
            orig_len = seq_len - G
            if orig_len < 0:
                raise RuntimeError(
                    "LAG1_BONUS stale suffix privatize found negative committed "
                    f"frontier: slot={slot} seq={seq_len}"
                )
            if alloc_len < seq_len:
                raise RuntimeError(
                    "LAG1_BONUS cannot privatize stale suffix from short allocation: "
                    f"slot={slot} alloc={alloc_len} seq={seq_len}"
                )
            seq_cpu.append(seq_len)
            orig_cpu.append(orig_len)
            anchors.append(verified_mirror[slot])
            if not has_shared:
                continue
            do_truncate = slot in shared
            if do_truncate:
                truncate_flags[i] = True
            else:
                all_truncate = False
            if do_truncate:
                any_truncate = True
                private_slots.append(slot)
                private_orig.append(orig_len)
                private_seq.append(seq_len)
        self._lag_check_stale_suffix_privacy(slots, truncate_flags)
        return (
            orig_cpu,
            seq_cpu,
            anchors,
            truncate_flags,
            any_truncate,
            all_truncate,
            private_slots,
            private_orig,
            private_seq,
        )

    def _lag_check_stale_suffix_privacy(
        self, slots: List[int], truncate_flags: List[bool]
    ) -> None:
        if not self._lag1_debug or not slots:
            return
        G = self.gamma + 1
        ss = self.slot_state
        active_t = torch.as_tensor(slots, dtype=torch.int64, device=self.device)
        seq = ss.seq_lens[active_t]
        committed = seq - G
        suffix_pos = committed.unsqueeze(1) + self._lag_window_offsets.unsqueeze(0)
        pool_idx = ss.req_pool_indices[active_t].unsqueeze(1)
        suffix_kv = ss.req_to_token_pool.req_to_token[pool_idx, suffix_pos].to(
            torch.int64
        )
        needs_private = (
            ss.token_to_kv_pool_allocator.slot_ref_count[suffix_kv] > 1
        ).any(dim=1)
        actual_flags = [bool(x) for x in needs_private.cpu().tolist()]
        if actual_flags != truncate_flags:
            raise RuntimeError(
                "LAG1_BONUS stale suffix sharing provenance mismatch: "
                f"slots={slots[:8]} expected={truncate_flags[:8]} "
                f"actual={actual_flags[:8]}"
            )

    def _lag_commit_stale_suffix_privatize(
        self,
        private_slots: List[int],
        private_orig: List[int],
        private_seq: List[int],
    ) -> None:
        if not private_slots:
            return
        G = self.gamma + 1
        ss = self.slot_state
        private_t = torch.as_tensor(
            private_slots, dtype=torch.int64, device=self.device
        )
        private_committed = torch.as_tensor(
            private_orig, dtype=ss.seq_lens.dtype, device=self.device
        )
        self._lag_truncate_kv_suffix(private_t, private_committed)
        new_alloc = self._lag_extend_kv_for_slots(
            private_t, private_committed, G, num_needed=len(private_slots) * G
        )
        ss.kv_allocated_lens[private_t] = new_alloc
        for slot, seq_len in zip(private_slots, private_seq, strict=True):
            self._lag_kv_allocated_lens_cpu[slot] = int(seq_len)

    def _lag_privatize_stale_suffix(
        self, slots: List[int]
    ) -> tuple[List[int], List[int], List[int], List[bool]]:
        (
            orig_cpu,
            seq_cpu,
            anchors,
            truncate_flags,
            _any_truncate,
            _all_truncate,
            private_slots,
            private_orig,
            private_seq,
        ) = self._lag_prepare_stale_suffix(slots)
        self._lag_commit_stale_suffix_privatize(
            private_slots, private_orig, private_seq
        )
        return orig_cpu, seq_cpu, anchors, truncate_flags

    def _lag_commit_mixed_allocations(
        self,
        advance_slots: List[int],
        advance_orig: List[int],
        advance_new: List[int],
        advance_num_needed: int,
        private_slots: List[int],
        private_orig: List[int],
        private_seq: List[int],
    ) -> None:
        """Commit verifier KV changes needed before firing one mixed StepReq."""
        if not advance_slots and not private_slots:
            return
        if not private_slots:
            self._lag_commit_advance_slots(
                advance_slots,
                advance_orig,
                advance_new,
                advance_num_needed,
            )
            return

        G = self.gamma + 1
        ss = self.slot_state
        device = self.device

        private_t = torch.as_tensor(private_slots, dtype=torch.int64, device=device)
        private_committed = torch.as_tensor(
            private_orig, dtype=ss.seq_lens.dtype, device=device
        )
        self._lag_truncate_kv_suffix(private_t, private_committed)

        alloc_slots = advance_slots + private_slots
        alloc_seq_cpu = advance_orig + private_orig
        active_t = torch.as_tensor(alloc_slots, dtype=torch.int64, device=device)
        alloc_seq = torch.as_tensor(
            alloc_seq_cpu, dtype=ss.seq_lens.dtype, device=device
        )
        num_needed = advance_num_needed + len(private_slots) * G

        if self._lag1_debug and advance_slots:
            seq = ss.seq_lens[active_t[: len(advance_slots)]]
            seq_cpu = [int(x) for x in seq.cpu().tolist()]
            if seq_cpu != advance_orig:
                raise RuntimeError(
                    "LAG1_BONUS seq_lens CPU mirror diverged before mixed advance: "
                    f"slots={advance_slots[:8]} mirror={advance_orig[:8]} "
                    f"tensor={seq_cpu[:8]}"
                )

        new_alloc = self._lag_extend_kv_for_slots(
            active_t, alloc_seq, G, num_needed=num_needed
        )
        ss.kv_allocated_lens[active_t] = new_alloc
        if advance_slots:
            advance_t = active_t[: len(advance_slots)]
            ss.seq_lens[advance_t] = alloc_seq[: len(advance_slots)] + G

        seq_cpu = self._lag_seq_lens_cpu
        alloc_cpu = self._lag_kv_allocated_lens_cpu
        self._lag_set_seq_lens_cpu(advance_slots, advance_new)
        for slot, new_seq_len in zip(advance_slots, advance_new, strict=True):
            seq_cpu[slot] = int(new_seq_len)
            alloc_cpu[slot] = int(max(alloc_cpu[slot], new_seq_len))
        for slot, seq_len in zip(private_slots, private_seq, strict=True):
            alloc_cpu[slot] = int(seq_len)

    def _lag_build_ready_pending(self, slots: List[int]):
        gamma = self.gamma
        device = self.device
        active_t = torch.as_tensor(slots, dtype=torch.int64, device=device)
        ready = [self._lag_ready[s] for s in slots]
        orig_cpu = torch.tensor([r.meta.orig_seq_len for r in ready], dtype=torch.int64)
        new_cpu = torch.tensor([r.meta.new_seq_len for r in ready], dtype=torch.int64)
        orig = orig_cpu.to(device=device)
        new = new_cpu.to(device=device)
        alloc_cpu = [self._lag_kv_allocated_lens_cpu[s] for s in slots]
        new_cpu_list = [int(x) for x in new_cpu.tolist()]
        if any(a < n for a, n in zip(alloc_cpu, new_cpu_list, strict=True)):
            raise RuntimeError(
                "LAG1_BONUS ready window is outside allocated verifier KV: "
                f"slots={slots[:8]} alloc={alloc_cpu[:8]} "
                f"new={new_cpu_list[:8]}"
            )
        anchors = torch.tensor(
            [r.meta.anchor_id for r in ready], dtype=torch.int32, device=device
        )
        ctx = SMCDecodeContext(
            orig_seq_lens=orig,
            orig_seq_lens_cpu=orig_cpu,
            orig_seq_lens_sum=int(orig_cpu.sum().item()),
            new_seq_lens=new,
            gamma=gamma,
        )
        ctx.new_seq_lens_cpu = new_cpu
        ctx.new_seq_lens_sum = int(new_cpu.sum().item())
        draft_input = SMCDraftInput(
            verified_id=anchors,
            num_tokens_per_req=gamma + 1,
            decode_ctx=ctx,
        )
        draft_input.active_slots_cpu = slots
        batch = self._lag_build_subset_model_worker_batch(
            draft_input, active_t, slots
        )
        pending = PendingDecodeStep(batch=batch, ctx=ctx, tag=0, epoch=0)
        resp = DraftStepResp(
            tokens=np.stack([r.tokens for r in ready], axis=0),
            logprobs=np.stack([r.logprobs for r in ready], axis=0),
            tag=0,
            epoch=0,
        )
        return active_t, pending, resp

    def _lag_build_subset_model_worker_batch(
        self,
        draft_input: SMCDraftInput,
        active: torch.Tensor,
        active_list: List[int],
    ) -> ModelWorkerBatch:
        ss = self.slot_state
        ctx = draft_input.decode_ctx
        seq_lens_cpu = ss.seq_lens_host[torch.as_tensor(active_list, dtype=torch.int64)]
        seq_lens_sum = int(seq_lens_cpu.sum().item())
        bs = len(active_list)

        template = ss._mwb_cache
        if template is None or ss._mwb_version != ss._membership_version:
            placeholder = torch.empty(
                ss.num_active, dtype=torch.int32, device=self.device
            )
            template = ss.build_model_worker_batch(
                SMCDraftInput(
                    verified_id=placeholder,
                    num_tokens_per_req=self.gamma + 1,
                )
            )

        sampling_info = SamplingBatchInfo(
            temperatures=ss._stub_temperatures[:bs],
            top_ps=ss._stub_top_ps[:bs],
            top_ks=ss._stub_top_ks[:bs],
            min_ps=ss._stub_min_ps[:bs],
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=ss.vocab_size,
        )
        return replace(
            template,
            input_ids=draft_input.verified_id,
            req_pool_indices=ss.req_pool_indices[active],
            seq_lens=ctx.new_seq_lens if ctx is not None else ss.seq_lens[active],
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            top_logprobs_nums=[0] * bs,
            multimodal_inputs=[None] * bs,
            sampling_info=sampling_info,
            spec_info=draft_input,
            reqs=[ss.slot_to_req[s] for s in active_list],
        )

    def _lag_prepare_mixed_launch(
        self,
        verify_slots: List[int],
        stale_slots: List[int],
        cold_slots: List[int],
    ) -> Optional[LagPreparedStep]:
        gamma = self.gamma
        G = gamma + 1
        ss = self.slot_state
        seq_cpu_mirror = self._lag_seq_lens_cpu
        alloc_cpu_mirror = self._lag_kv_allocated_lens_cpu
        verified_cpu = self._lag_verified_ids_cpu
        ready = self._lag_ready
        shared = self._lag_stale_shared
        has_shared = bool(shared)

        ordered: List[int] = []
        verified_ids: List[int] = []
        seq_lens: List[int] = []
        metas: List[LagWindowMeta] = []
        valid_by_pos: List[int] = []
        pos_by_slot: Dict[int, int] = {}
        width2 = self._anchor_width >= 2
        bet_alt: List[int] = []  # per-row alt (c1) seed; -1 = no branch

        advance_slots: List[int] = []
        advance_orig: List[int] = []
        advance_new: List[int] = []
        advance_num_needed = 0

        private_slots: List[int] = []
        private_orig: List[int] = []
        private_seq: List[int] = []

        stale_truncate: List[bool] = []
        truncate_rows: Optional[List[bool]] = [] if stale_slots and has_shared else None
        any_truncate = False
        all_truncate = bool(stale_slots) and has_shared

        for slot in verify_slots:
            orig = seq_cpu_mirror[slot]
            old_alloc = alloc_cpu_mirror[slot]
            new = orig + G
            advance_slots.append(slot)
            advance_orig.append(orig)
            advance_new.append(new)
            if old_alloc == orig:
                advance_num_needed += G
            else:
                advance_num_needed += max(new - max(old_alloc, orig), 0)
            ready_window = ready[slot]
            if orig != ready_window.meta.new_seq_len:
                raise RuntimeError(
                    "LAG1_BONUS runahead does not start at ready window end: "
                    f"slot={slot} orig={orig} ready_end={ready_window.meta.new_seq_len}"
                )
            anchor = int(ready_window.tokens[gamma])
            pos_by_slot[slot] = len(ordered)
            ordered.append(slot)
            verified_ids.append(anchor)
            seq_lens.append(orig)
            metas.append(LagWindowMeta(orig, new, anchor))
            valid_by_pos.append(-1)
            if truncate_rows is not None:
                truncate_rows.append(False)
            # Width-2: the alt run-ahead seed = the best top-W anchor candidate that
            # differs from the (sampled) primary bet, so {primary, alt} covers two
            # distinct guesses for the committed bonus b.
            c1 = -1
            if width2 and ready_window.bet_topk is not None:
                for cand in ready_window.bet_topk[: self._anchor_width]:
                    if int(cand) != anchor:
                        c1 = int(cand)
                        break
            bet_alt.append(c1)

        for slot in stale_slots:
            seq_len = seq_cpu_mirror[slot]
            alloc_len = alloc_cpu_mirror[slot]
            orig = seq_len - G
            if orig < 0:
                raise RuntimeError(
                    "LAG1_BONUS stale suffix privatize found negative committed "
                    f"frontier: slot={slot} seq={seq_len}"
                )
            if alloc_len < seq_len:
                raise RuntimeError(
                    "LAG1_BONUS cannot privatize stale suffix from short allocation: "
                    f"slot={slot} alloc={alloc_len} seq={seq_len}"
                )
            do_truncate = has_shared and slot in shared
            stale_truncate.append(do_truncate)
            if truncate_rows is not None:
                truncate_rows.append(do_truncate)
            if do_truncate:
                any_truncate = True
                private_slots.append(slot)
                private_orig.append(orig)
                private_seq.append(seq_len)
            else:
                all_truncate = False

            anchor = int(verified_cpu[slot])
            pos_by_slot[slot] = len(ordered)
            ordered.append(slot)
            verified_ids.append(anchor)
            seq_lens.append(orig)
            metas.append(LagWindowMeta(orig, seq_len, anchor))
            valid_by_pos.append(1)
            bet_alt.append(-1)

        for slot in cold_slots:
            orig = seq_cpu_mirror[slot]
            old_alloc = alloc_cpu_mirror[slot]
            new = orig + G
            advance_slots.append(slot)
            advance_orig.append(orig)
            advance_new.append(new)
            if old_alloc == orig:
                advance_num_needed += G
            else:
                advance_num_needed += max(new - max(old_alloc, orig), 0)
            anchor = int(verified_cpu[slot])
            pos_by_slot[slot] = len(ordered)
            ordered.append(slot)
            verified_ids.append(anchor)
            seq_lens.append(orig)
            metas.append(LagWindowMeta(orig, new, anchor))
            valid_by_pos.append(1)
            bet_alt.append(-1)
            if truncate_rows is not None:
                truncate_rows.append(False)

        if not ordered:
            return None

        if stale_slots:
            self._lag_check_stale_suffix_privacy(stale_slots, stale_truncate)

        if not stale_slots:
            rollback_payload = 0
        else:
            n_verify = len(verify_slots)
            n_stale = len(stale_slots)
            n_cold = len(cold_slots)
            rollback_payload = [0] * n_verify + [G] * n_stale + [0] * n_cold

        if not any_truncate:
            truncate_kv_payload = False
        elif len(stale_slots) == len(ordered) and all_truncate:
            truncate_kv_payload = True
        elif len(stale_slots) == len(ordered):
            truncate_kv_payload = stale_truncate
        else:
            truncate_kv_payload = truncate_rows

        return LagPreparedStep(
            ordered=ordered,
            verified_ids=verified_ids,
            seq_lens=seq_lens,
            rollback_payload=rollback_payload,
            truncate_kv_payload=truncate_kv_payload,
            metas=metas,
            valid_by_pos=valid_by_pos,
            pos_by_slot=pos_by_slot,
            advance_slots=advance_slots,
            advance_orig=advance_orig,
            advance_new=advance_new,
            advance_num_needed=advance_num_needed,
            private_slots=private_slots,
            private_orig=private_orig,
            private_seq=private_seq,
            bet_alt=(bet_alt if width2 and any(c >= 0 for c in bet_alt) else None),
        )

    def _lag_fire_mixed_step(
        self,
        verify_slots: List[int],
        stale_slots: List[int],
        cold_slots: List[int],
    ) -> None:
        if self._lag_pending is not None:
            raise RuntimeError("LAG1_BONUS attempted to fire with a StepReq in flight")
        prepared = self._lag_prepare_mixed_launch(
            verify_slots, stale_slots, cold_slots
        )
        if prepared is None:
            return

        tag = next(self._tag)
        epoch = next(self._epoch)
        self._lag_commit_mixed_allocations(
            prepared.advance_slots,
            prepared.advance_orig,
            prepared.advance_new,
            prepared.advance_num_needed,
            prepared.private_slots,
            prepared.private_orig,
            prepared.private_seq,
        )
        self._draft_client.send_step(
            slots=prepared.ordered,
            verified_ids=prepared.verified_ids,
            seq_lens=prepared.seq_lens,
            tag=tag,
            epoch=epoch,
            rollback=prepared.rollback_payload,
            truncate_kv=prepared.truncate_kv_payload,
            bet_alt=prepared.bet_alt,
        )
        for slot in stale_slots:
            self._lag_stale.discard(slot)
            self._lag_stale_shared.discard(slot)
        self._lag_pending = LagPendingWindow(
            active_list_T=prepared.ordered,
            tag=tag,
            epoch=epoch,
            metas=prepared.metas,
            valid_by_pos=prepared.valid_by_pos,
            pos_by_slot=prepared.pos_by_slot,
            branch_choice=(
                [0] * len(prepared.ordered) if prepared.bet_alt is not None else None
            ),
            branch_anchor=(
                [-1] * len(prepared.ordered) if prepared.bet_alt is not None else None
            ),
        )
        self._lag_pending_bet_alt = prepared.bet_alt
        self._passes_sent += 1

    def _lag_verify_ready(
        self, slots: List[int], *, update_pending: bool = True
    ) -> List[int]:
        if not slots:
            return []
        tm = self._timing
        t0 = time.perf_counter() if tm else 0.0
        active_t, pending, resp = self._lag_build_ready_pending(slots)
        if tm:
            self._t["ready"] += time.perf_counter() - t0
        tracking_batch = self._make_runtime_tracking_batch(pending.batch)
        self.cur_batch = tracking_batch
        self.running_batch = (
            tracking_batch if tracking_batch is not None else ScheduleBatch(reqs=[])
        )

        t0 = time.perf_counter() if tm else 0.0
        result = self.draft_worker.finish_decode(pending, resp)
        newly_finished = self._writeback_window(result, active_t)
        if tm:
            self._t["verify"] += time.perf_counter() - t0
        self._windows_committed += 1

        x_g1 = torch.tensor(
            [int(self._lag_ready[s].tokens[self.gamma]) for s in slots],
            dtype=torch.int64,
        )
        b_cpu = result.next_draft_input.verified_id.cpu().to(torch.int64)
        match = x_g1 == b_cpu
        b_list = [int(x) for x in b_cpu.tolist()]
        match_list = [bool(x) for x in match.tolist()]
        if self._anchor_width >= 2:
            topk = [self._lag_ready[s].bet_topk for s in slots]
            if all(t is not None for t in topk):
                # Oracle: how often would the committed bonus b land in the
                # drafter's top-2 anchor candidates vs the single (production) bet?
                tk = torch.as_tensor(np.stack(topk), dtype=torch.int64)  # (n, W)
                top1 = tk[:, 0] == b_cpu
                top2 = (tk[:, :2] == b_cpu.unsqueeze(1)).any(dim=1)
                self._w2_n += int(b_cpu.numel())
                self._w2_prod_hit += int(match.sum().item())
                self._w2_top1_hit += int(top1.sum().item())
                self._w2_top2_hit += int(top2.sum().item())
                if self._w2_n >= 500:
                    n = self._w2_n
                    print(
                        f"[WIDTH2_ORACLE] {n} bets: "
                        f"prod(bet==b)={100*self._w2_prod_hit/n:.1f}% "
                        f"top1={100*self._w2_top1_hit/n:.1f}% "
                        f"top2={100*self._w2_top2_hit/n:.1f}% "
                        f"realized_alt_win={100*self._w2_alt_win/n:.1f}% "
                        f"(width-2 ceiling vs prod=+"
                        f"{100*(self._w2_top2_hit-self._w2_prod_hit)/n:.1f}pp)",
                        flush=True,
                    )
                    self._w2_n = 0
                    self._w2_prod_hit = self._w2_top1_hit = self._w2_top2_hit = 0
                    self._w2_alt_win = 0
        if self._bet_stats:
            self._bet_n += int(match.numel())
            self._bet_miss_n += int((~match).sum().item())
        # A2: fuse the count-bump, valid_by_pos back-fill, and ready-pop into a
        # single per-slot pass (was 3 separate zips). Order-preserving; hoist the
        # pending guard out of the loop.
        G = self.gamma + 1
        scc = self._lag_token_counts_cpu
        svc = self._lag_verified_ids_cpu
        upd = update_pending and self._lag_pending is not None
        pos_by_slot = self._lag_pending.pos_by_slot if upd else None
        valid_by_pos = self._lag_pending.valid_by_pos if upd else None
        branch_choice = self._lag_pending.branch_choice if upd else None
        branch_anchor = self._lag_pending.branch_anchor if upd else None
        fired_bet_alt = self._lag_pending_bet_alt if upd else None
        promote_slots: List[int] = []
        for slot, bonus_id, ok in zip(slots, b_list, match_list, strict=True):
            scc[slot] += G
            svc[slot] = bonus_id
            if upd:
                row = pos_by_slot.get(slot)
                if row is not None:
                    valid = ok
                    choice = 0
                    # Width-2: the just-fired run-ahead for this slot was seeded by
                    # the primary bet c0 (=x_g1) AND, if branched, an alt c1.  The
                    # alt branch won iff the committed bonus b matched c1 (not c0).
                    if fired_bet_alt is not None:
                        c1 = fired_bet_alt[row]
                        if c1 >= 0 and (not ok) and bonus_id == c1:
                            valid = True
                            choice = 1
                            promote_slots.append(slot)
                            self._w2_alt_win += 1
                            # The alt run-ahead was seeded by c1 (== committed b);
                            # record it so the later verify conditions the window on
                            # the correct x0 (the primary meta holds c0).
                            if branch_anchor is not None:
                                branch_anchor[row] = bonus_id
                    if branch_choice is not None:
                        branch_choice[row] = choice
                    valid_by_pos[row] = 1 if valid else 0
            self._lag_ready.pop(slot, None)
        if promote_slots:
            # Resolve the drafter's lineage BEFORE the resample's send_commit so
            # resampled descendants inherit the winning (c1) run-ahead KV.
            self._draft_client.send_promote_alt(
                promote_slots, [True] * len(promote_slots)
            )
        if self._bet_stats and self._bet_n >= 500:
            print(
                f"[LAG1_BONUS] {self._bet_n} bets: "
                f"x_g1!=b={100*self._bet_miss_n/max(self._bet_n,1):.1f}% "
                f"passes_sent={self._passes_sent} committed={self._windows_committed}",
                flush=True,
            )
            self._bet_n = self._bet_miss_n = 0
        if self._redraft_oracle:
            # Stash this cycle's per-slot PRIMARY bet-hit (c0==b) so the same-cycle
            # resample can route bet-hit descendants onto the re-draft path.  slots and
            # match_list are 1:1 aligned; width-2 alt-wins are excluded by design (a
            # clean signal — alt-win lineages keep the existing promote/inherit path).
            self._lag_bet_hit = dict(zip(slots, match_list, strict=True))
        return newly_finished

    def _lag_apply_resample_plan(self, dsts: List[int], srcs: List[int]) -> None:
        seq_vals = [self._lag_seq_lens_cpu[src] for src in srcs]
        alloc_vals = [self._lag_kv_allocated_lens_cpu[src] for src in srcs]
        verified_vals = [self._lag_verified_ids_cpu[src] for src in srcs]
        count_vals = [self._lag_token_counts_cpu[src] for src in srcs]
        finished_vals = [self._lag_finished_mask_cpu[src] for src in srcs]
        self._lag_set_seq_lens_cpu(dsts, seq_vals)
        for dst, alloc, verified, count, finished in zip(
            dsts,
            alloc_vals,
            verified_vals,
            count_vals,
            finished_vals,
            strict=True,
        ):
            self._lag_kv_allocated_lens_cpu[dst] = alloc
            self._lag_verified_ids_cpu[dst] = verified
            self._lag_token_counts_cpu[dst] = count
            self._lag_finished_mask_cpu[dst] = finished

        if self._lag_pending is not None:
            max_slots = self.slot_state.seq_lens.shape[0]
            a = (
                self._lag_pending.ancestor.copy()
                if self._lag_pending.ancestor is not None
                else np.arange(max_slots, dtype=np.int64)
            )
            a[np.asarray(dsts, dtype=np.int64)] = np.asarray(srcs, dtype=np.int64)
            self._lag_pending.ancestor = a

        if self._redraft_oracle:
            # Holds only THIS resample's bet-hit dsts.
            self._lag_redraft_clones = set()
        for dst, src in zip(dsts, srcs, strict=True):
            src_ready = self._lag_ready.get(src)
            self._lag_ready.pop(dst, None)
            if self._redraft_oracle and self._lag_bet_hit.get(src, False):
                # Oracle: a descendant of a BET-HIT survivor must NOT inherit the
                # survivor's (valid but shared) run-ahead window.  Force it stale so
                # next cycle it re-drafts its own window from its own committed bonus b
                # with privatized KV (mark _stale_shared so do_truncate frees the shared
                # suffix); _redraft_clones suppresses its W' re-adoption at the next recv.
                self._lag_stale.add(dst)
                self._lag_stale_shared.add(dst)
                self._lag_redraft_clones.add(dst)
                continue
            if src_ready is not None:
                self._lag_ready[dst] = LagReadyWindow(
                    tokens=src_ready.tokens,
                    logprobs=src_ready.logprobs,
                    meta=src_ready.meta,
                    bet_topk=src_ready.bet_topk,
                )
            self._lag_stale.discard(dst)
            self._lag_stale_shared.discard(dst)
            if src in self._lag_stale:
                self._lag_stale.add(dst)
                self._lag_stale_shared.add(src)
                self._lag_stale_shared.add(dst)

    def _lag_resample(self, verify_slots: Optional[List[int]] = None) -> bool:
        row_mask = self._lag_resample_row_mask(verify_slots)
        if self._lag_profile:
            self._lp["eligible_rows"] += int(row_mask.sum().item())
        plan = self._lag_collect_resample_jobs(row_mask)
        n_jobs = plan.n_jobs_sync()
        if self._lag_profile:
            self._lp["resample_jobs"] += int(n_jobs)
        if n_jobs == 0:
            return False
        self.coordinator.dispatch_resample_batch(plan, self.slot_state)
        dsts = [int(x) for x in plan.dst_slots.detach().cpu().tolist()]
        srcs = [int(x) for x in plan.src_slots.detach().cpu().tolist()]
        self._draft_client.send_commit(dst_slots=dsts, src_slots=srcs)
        self._lag_apply_resample_plan(dsts, srcs)
        return True

    def _lag_collect_resample_jobs(self, row_mask: torch.Tensor):
        from smcsd.core.kernels.fused_collect import batched_collect_fused

        self.coordinator._fast_step_counter += 1
        ss = self.slot_state
        return batched_collect_fused(
            ss.log_weights,
            ss.interval_weights,
            ss.group_to_slots,
            ss.row_in_use & row_mask,
            self.coordinator.resample_threshold,
            step_counter=self.coordinator._fast_step_counter,
        )

    def _run_lag1_bonus_train(self) -> None:
        """Bounded-lag exact-bonus pipeline.

        Each event-loop pass consumes at most one mixed draft reply, verifies the
        ready rows that are allowed by the per-group one-window lag cap, then
        fires the next mixed draft batch before target verify so draft and verify
        overlap.  Rows whose run-ahead anchor mismatches are not verified from the
        stale reply; they are queued for a catch-up draft from the exact target
        bonus and rejoin on the following cycle.
        """
        tm = self._timing
        self._lag_receive_pending()
        active_list = list(self.slot_state._active_slots_list)
        if not active_list:
            self._lag_drain_finished_groups()
            return

        t0 = time.perf_counter() if tm else 0.0
        verify_slots = self._lag_select_ready_slots(active_list)
        verify_set = set(verify_slots) if verify_slots else None
        stale_slots: List[int] = []
        cold_slots: List[int] = []
        ready_rows = 0
        ready = self._lag_ready
        stale = self._lag_stale
        for slot in active_list:
            is_ready = slot in ready
            if is_ready:
                ready_rows += 1
            if verify_set is not None and slot in verify_set:
                continue
            if slot in stale:
                stale_slots.append(slot)
            elif not is_ready:
                cold_slots.append(slot)
        held_ready_rows = ready_rows - len(verify_slots)
        self._lag_profile_add(
            active_rows=len(active_list),
            ready_rows=ready_rows,
            verify_rows=len(verify_slots),
            held_ready_rows=held_ready_rows,
            catchup_rows=len(stale_slots),
            cold_rows=len(cold_slots),
        )
        if tm:
            self._t["select"] += time.perf_counter() - t0

        t0 = time.perf_counter() if tm else 0.0
        self._lag_fire_mixed_step(verify_slots, stale_slots, cold_slots)
        if tm:
            self._t["prep"] += time.perf_counter() - t0

        if verify_slots:
            newly_finished = self._lag_verify_ready(verify_slots)
            resample_verify_slots = list(verify_slots)
            t0 = time.perf_counter() if tm else 0.0
            did_resample = self._lag_resample(resample_verify_slots)
            if newly_finished or did_resample:
                self.slot_state.rebuild_active_slots()
                self._lag_prune_metadata()
                self._lag_drain_finished_groups()
            if tm:
                self._t["barrier"] += time.perf_counter() - t0
                self._t_windows += 1
                if self._t_windows >= 200:
                    tot = sum(self._t.values()) or 1.0
                    n = self._t_windows
                    pw = (
                        self._passes_sent / self._windows_committed
                        if self._windows_committed
                        else 0.0
                    )
                    print(
                        f"[LAG1_BONUS_TIMING] {n} cycles: "
                        f"recv(drafter-wait)={100*self._t['recv']/tot:.0f}% "
                        f"select={100*self._t['select']/tot:.0f}% "
                        f"prep={100*self._t['prep']/tot:.0f}% "
                        f"ready={100*self._t['ready']/tot:.0f}% "
                        f"verify={100*self._t['verify']/tot:.0f}% "
                        f"barrier={100*self._t['barrier']/tot:.0f}% | per-cycle "
                        f"recv={1e3*self._t['recv']/n:.2f}ms "
                        f"select={1e3*self._t['select']/n:.2f}ms "
                        f"prep={1e3*self._t['prep']/n:.2f}ms "
                        f"ready={1e3*self._t['ready']/n:.2f}ms "
                        f"verify={1e3*self._t['verify']/n:.2f}ms "
                        f"resampled={int(did_resample)} | PASSES/CYCLE={pw:.3f}",
                        flush=True,
                    )
                    self._t = {k: 0.0 for k in self._t}
                    self._t_windows = 0
                    self._passes_sent = 0
                    self._windows_committed = 0
            self._lag_profile_report()

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
        newly_finished = self._process_batch_result(
            next_token_ids=result.next_token_ids,
            logprob_diff=logprob_diff,
            bonus_ids=bonus_ids,
            rebuild_active=False,
            active=active_t,
        )
        for slot in newly_finished:
            self._lag_finished_mask_cpu[slot] = True
        return newly_finished


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
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
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

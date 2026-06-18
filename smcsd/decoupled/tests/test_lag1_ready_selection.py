"""Unit tests for lag-1 ready-window selection.

The leader-selection optimization is safe only when every live row at the group
minimum frontier is ready too.  Then verifying one-window-ahead leaders in the
same target batch keeps the post-verify committed lag bounded by one window.
"""

import unittest
from itertools import count
from types import SimpleNamespace

import numpy as np
import torch

from smcsd.decoupled.async_scheduler import AsyncDecoupledSMCScheduler
from smcsd.decoupled.async_scheduler import LagReadyWindow, LagWindowMeta


def _select(*, counts, ready, verify_leaders, active=None, finished=None, group=None):
    if active is None:
        active = list(range(len(counts)))
    if finished is None:
        finished = [False] * len(counts)
    if group is None:
        group = active
    scheduler = SimpleNamespace(
        gamma=8,
        _lag_ready={slot: object() for slot in ready},
        _lag_verify_leaders=verify_leaders,
        slot_state=SimpleNamespace(
            group_slot_lists={"g": list(group)},
            token_counts_cpu=list(counts),
            finished_mask_cpu=list(finished),
        ),
    )
    return AsyncDecoupledSMCScheduler._lag_select_ready_slots(scheduler, active)


class TestLag1ReadySelection(unittest.TestCase):
    def test_default_selects_only_min_frontier_rows(self):
        self.assertEqual(
            _select(counts=[10, 19, 19], ready=[0, 1, 2], verify_leaders=False),
            [0],
        )

    def test_verify_leaders_selects_one_window_ahead_when_min_ready(self):
        self.assertEqual(
            _select(counts=[10, 19, 19], ready=[0, 1, 2], verify_leaders=True),
            [0, 1, 2],
        )

    def test_verify_leaders_holds_leaders_when_min_missing(self):
        self.assertEqual(
            _select(counts=[10, 19, 19], ready=[1, 2], verify_leaders=True),
            [],
        )

    def test_verify_leaders_requires_all_min_rows_ready(self):
        self.assertEqual(
            _select(counts=[10, 10, 19], ready=[0, 2], verify_leaders=True),
            [0],
        )

    def test_finished_min_row_does_not_block_leaders(self):
        self.assertEqual(
            _select(
                counts=[10, 19, 28],
                ready=[1, 2],
                verify_leaders=True,
                finished=[True, False, False],
            ),
            [1, 2],
        )


class TestLag1PreDraftPrep(unittest.TestCase):
    def test_stale_suffix_flags_are_built_in_one_pass(self):
        scheduler = SimpleNamespace(
            gamma=2,
            _lag_stale_shared={4, 6},
            _async_bonus_debug=False,
            _lag_check_stale_suffix_privacy=lambda slots, flags: None,
            slot_state=SimpleNamespace(
                seq_lens_cpu=[0, 0, 0, 0, 13, 14, 15],
                kv_allocated_lens_cpu=[0, 0, 0, 0, 13, 14, 15],
                verified_ids_cpu=[0, 0, 0, 0, 104, 105, 106],
            ),
        )

        (
            orig,
            seq,
            anchors,
            truncate,
            any_truncate,
            all_truncate,
            private_slots,
            private_orig,
            private_seq,
        ) = AsyncDecoupledSMCScheduler._lag_prepare_stale_suffix(
            scheduler, [4, 5, 6]
        )

        self.assertEqual(orig, [10, 11, 12])
        self.assertEqual(seq, [13, 14, 15])
        self.assertEqual(anchors, [104, 105, 106])
        self.assertEqual(truncate, [True, False, True])
        self.assertTrue(any_truncate)
        self.assertFalse(all_truncate)
        self.assertEqual(private_slots, [4, 6])
        self.assertEqual(private_orig, [10, 12])
        self.assertEqual(private_seq, [13, 15])

    def test_mixed_fire_builds_span_payloads_and_combined_alloc_plan(self):
        sent = {}
        committed = {}
        calls = []

        scheduler = SimpleNamespace(
            gamma=2,
            _lag_pending=None,
            _lag_ready={
                1: LagReadyWindow(
                    tokens=np.array([10, 11, 12], dtype=np.int32),
                    logprobs=np.zeros(3, dtype=np.float32),
                    meta=LagWindowMeta(orig_seq_len=7, new_seq_len=10, anchor_id=9),
                )
            },
            slot_state=SimpleNamespace(
                seq_lens_cpu=[0, 10, 33, 20],
                kv_allocated_lens_cpu=[0, 10, 33, 20],
                verified_ids_cpu=[0, 12, 302, 32],
            ),
            _tag=count(100),
            _epoch=count(200),
            _lag_stale={2},
            _lag_stale_shared={2},
            _async_bonus_debug=False,
            _passes_sent=0,
            _lag_commit_mixed_allocations=lambda *args: (
                calls.append("commit"),
                committed.setdefault("args", args),
            ),
            _draft_client=SimpleNamespace(
                send_step=lambda **kwargs: (
                    calls.append("send"),
                    sent.setdefault("kwargs", kwargs),
                )
            ),
        )
        scheduler._lag_prepare_mixed_launch = (
            lambda verify, stale, cold: AsyncDecoupledSMCScheduler._lag_prepare_mixed_launch(
                scheduler, verify, stale, cold
            )
        )
        scheduler._lag_check_stale_suffix_privacy = lambda slots, flags: None

        AsyncDecoupledSMCScheduler._lag_fire_mixed_step(
            scheduler, [1], [2], [3]
        )

        self.assertEqual(sent["kwargs"]["slots"], [1, 2, 3])
        self.assertEqual(sent["kwargs"]["verified_ids"], [12, 302, 32])
        self.assertEqual(sent["kwargs"]["seq_lens"], [10, 30, 20])
        self.assertEqual(sent["kwargs"]["rollback"], [0, 3, 0])
        self.assertEqual(sent["kwargs"]["truncate_kv"], [False, True, False])
        self.assertEqual(
            committed["args"],
            ([1, 3], [10, 20], [13, 23], 6, [2], [30], [33]),
        )
        self.assertNotIn(2, scheduler._lag_stale)
        self.assertNotIn(2, scheduler._lag_stale_shared)
        self.assertEqual(scheduler._lag_pending.active_list_T, [1, 2, 3])
        self.assertEqual(scheduler._lag_pending.pos_by_slot, {1: 0, 2: 1, 3: 2})
        self.assertEqual(scheduler._lag_pending.valid_by_pos, [-1, 1, 1])
        self.assertEqual(
            [
                (m.orig_seq_len, m.new_seq_len, m.anchor_id)
                for m in scheduler._lag_pending.metas
            ],
            [(10, 13, 12), (30, 33, 302), (20, 23, 32)],
        )
        self.assertEqual(scheduler._passes_sent, 1)
        self.assertEqual(calls, ["commit", "send"])

    def test_mixed_fire_no_stale_skips_privatize_payloads(self):
        sent = {}
        committed = {}
        calls = []

        scheduler = SimpleNamespace(
            gamma=2,
            _lag_pending=None,
            _lag_ready={
                1: LagReadyWindow(
                    tokens=np.array([10, 11, 12], dtype=np.int32),
                    logprobs=np.zeros(3, dtype=np.float32),
                    meta=LagWindowMeta(orig_seq_len=7, new_seq_len=10, anchor_id=9),
                )
            },
            slot_state=SimpleNamespace(
                seq_lens_cpu=[0, 10, 0, 20],
                kv_allocated_lens_cpu=[0, 10, 0, 20],
                verified_ids_cpu=[0, 12, 22, 32],
            ),
            _tag=count(100),
            _epoch=count(200),
            _lag_stale=set(),
            _lag_stale_shared=set(),
            _async_bonus_debug=False,
            _passes_sent=0,
            _lag_commit_mixed_allocations=lambda *args: (
                calls.append("commit"),
                committed.setdefault("args", args),
            ),
            _draft_client=SimpleNamespace(
                send_step=lambda **kwargs: (
                    calls.append("send"),
                    sent.setdefault("kwargs", kwargs),
                )
            ),
        )
        scheduler._lag_prepare_mixed_launch = (
            lambda verify, stale, cold: AsyncDecoupledSMCScheduler._lag_prepare_mixed_launch(
                scheduler, verify, stale, cold
            )
        )
        scheduler._lag_check_stale_suffix_privacy = lambda slots, flags: None

        AsyncDecoupledSMCScheduler._lag_fire_mixed_step(
            scheduler, [1], [], [3]
        )

        self.assertEqual(sent["kwargs"]["slots"], [1, 3])
        self.assertEqual(sent["kwargs"]["verified_ids"], [12, 32])
        self.assertEqual(sent["kwargs"]["seq_lens"], [10, 20])
        self.assertEqual(sent["kwargs"]["rollback"], 0)
        self.assertFalse(sent["kwargs"]["truncate_kv"])
        self.assertEqual(
            committed["args"],
            ([1, 3], [10, 20], [13, 23], 6, [], [], []),
        )
        self.assertEqual(scheduler._lag_pending.active_list_T, [1, 3])
        self.assertEqual(scheduler._lag_pending.pos_by_slot, {1: 0, 3: 1})
        self.assertEqual(scheduler._lag_pending.valid_by_pos, [-1, 1])
        self.assertEqual(scheduler._passes_sent, 1)
        self.assertEqual(calls, ["commit", "send"])

    def test_lag1_resample_does_not_reset_all_active_interval_weights(self):
        calls = []
        plan = SimpleNamespace(
            n_jobs=1,
            dst_slots=torch.tensor([1], dtype=torch.int32),
            src_slots=torch.tensor([0], dtype=torch.int32),
        )
        scheduler = SimpleNamespace(
            _lag_profile=False,
            _lag_resample_row_mask=lambda verify_slots: "row-mask",
            _lag_apply_resample_plan=lambda dsts, srcs: calls.append(
                ("apply", dsts, srcs)
            ),
            slot_state=SimpleNamespace(
                active_slots=torch.tensor([0, 1], dtype=torch.int64),
                reset_interval_weights=lambda active: calls.append(("reset", active)),
            ),
            coordinator=SimpleNamespace(
                collect_resample_jobs_batch=lambda slot_state, row_mask: plan,
                dispatch_resample_batch=lambda p, slot_state, rebuild_active: calls.append(
                    ("dispatch", p, rebuild_active)
                ) or ([1], [0]),
            ),
            _draft_client=SimpleNamespace(
                send_commit=lambda dst_slots, src_slots: calls.append(
                    ("commit", dst_slots, src_slots)
                )
            ),
        )

        did_resample = AsyncDecoupledSMCScheduler._lag_resample(
            scheduler, verify_slots=[0]
        )

        self.assertTrue(did_resample)
        self.assertIn(("dispatch", plan, False), calls)
        self.assertIn(("commit", [1], [0]), calls)
        self.assertIn(("apply", [1], [0]), calls)
        self.assertFalse(any(call[0] == "reset" for call in calls))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for lag-1 ready-window selection.

Strict lag-1 rule: a ready row may commit only at its group's minimum committed
frontier.
"""

import unittest
from itertools import count
from types import SimpleNamespace

import numpy as np
import torch

from smcsd.decoupled.async_scheduler import AsyncDecoupledSMCScheduler
from smcsd.decoupled.async_scheduler import LagReadyWindow, LagWindowMeta


def _select(*, counts, ready, active=None, finished=None, group=None):
    if active is None:
        active = list(range(len(counts)))
    if finished is None:
        finished = [False] * len(counts)
    if group is None:
        group = active
    scheduler = SimpleNamespace(
        gamma=8,
        _lag_ready={slot: object() for slot in ready},
        _lag_token_counts_cpu=list(counts),
        _lag_finished_mask_cpu=list(finished),
        slot_state=SimpleNamespace(
            group_slot_lists={"g": list(group)},
        ),
    )
    return AsyncDecoupledSMCScheduler._lag_select_ready_slots(scheduler, active)


class TestLag1ReadySelection(unittest.TestCase):
    def test_default_selects_only_min_frontier_rows(self):
        self.assertEqual(
            _select(counts=[10, 19, 19], ready=[0, 1, 2]),
            [0],
        )

    def test_finished_min_row_does_not_unblock_non_min_rows(self):
        self.assertEqual(
            _select(
                counts=[10, 19, 28],
                ready=[1, 2],
                finished=[True, False, False],
            ),
            [1],
        )

    def test_ready_rows_ahead_of_group_minimum_are_held(self):
        self.assertEqual(
            _select(counts=[10, 19, 28], ready=[0, 1, 2]),
            [0],
        )


class TestLag1PreDraftPrep(unittest.TestCase):
    def test_stale_suffix_flags_are_built_in_one_pass(self):
        scheduler = SimpleNamespace(
            gamma=2,
            _lag_stale_shared={4, 6},
            _lag1_debug=False,
            _lag_check_stale_suffix_privacy=lambda slots, flags: None,
            _lag_seq_lens_cpu=[0, 0, 0, 0, 13, 14, 15],
            _lag_kv_allocated_lens_cpu=[0, 0, 0, 0, 13, 14, 15],
            _lag_verified_ids_cpu=[0, 0, 0, 0, 104, 105, 106],
            slot_state=SimpleNamespace(),
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
            _anchor_width=1,
            _lag_pending=None,
            _lag_ready={
                1: LagReadyWindow(
                    tokens=np.array([10, 11, 12], dtype=np.int32),
                    logprobs=np.zeros(3, dtype=np.float32),
                    meta=LagWindowMeta(orig_seq_len=7, new_seq_len=10, anchor_id=9),
                )
            },
            _lag_seq_lens_cpu=[0, 10, 33, 20],
            _lag_kv_allocated_lens_cpu=[0, 10, 33, 20],
            _lag_verified_ids_cpu=[0, 12, 302, 32],
            slot_state=SimpleNamespace(),
            _tag=count(100),
            _epoch=count(200),
            _lag_stale={2},
            _lag_stale_shared={2},
            _lag1_debug=False,
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
            _anchor_width=1,
            _lag_pending=None,
            _lag_ready={
                1: LagReadyWindow(
                    tokens=np.array([10, 11, 12], dtype=np.int32),
                    logprobs=np.zeros(3, dtype=np.float32),
                    meta=LagWindowMeta(orig_seq_len=7, new_seq_len=10, anchor_id=9),
                )
            },
            _lag_seq_lens_cpu=[0, 10, 0, 20],
            _lag_kv_allocated_lens_cpu=[0, 10, 0, 20],
            _lag_verified_ids_cpu=[0, 12, 22, 32],
            slot_state=SimpleNamespace(),
            _tag=count(100),
            _epoch=count(200),
            _lag_stale=set(),
            _lag_stale_shared=set(),
            _lag1_debug=False,
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
            n_jobs_sync=lambda: 1,
            dst_slots=torch.tensor([1], dtype=torch.int32),
            src_slots=torch.tensor([0], dtype=torch.int32),
            resample_mask=torch.tensor([True]),
        )
        snapshot = SimpleNamespace(phase=0, wait=lambda: calls.append(("wait", 0)))
        scheduler = SimpleNamespace(
            _lag_profile=False,
            _lag_resample_row_mask=lambda verify_slots: "row-mask",
            _lag_apply_resample_plan=lambda dsts, srcs: calls.append(
                ("apply", dsts, srcs)
            ),
            _lag_collect_resample_jobs=lambda row_mask: calls.append(
                ("collect", row_mask)
            )
            or plan,
            slot_state=SimpleNamespace(
                active_slots=torch.tensor([0, 1], dtype=torch.int64),
                reset_interval_weights=lambda active: calls.append(("reset", active)),
                resample_logZ_increment=lambda: torch.tensor([2.0]),
                group_log_Z_hat=torch.tensor([3.0]),
                snapshot_to_host=lambda: snapshot,
                kv_freed_count_host=torch.tensor([[2]], dtype=torch.int32),
                kv_freed_buf=torch.tensor([[10, 11, 0]], dtype=torch.int32),
                kv_freed_counter=torch.ones((1, 1), dtype=torch.int32),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                free=lambda pages: calls.append(("free", pages.tolist()))
            ),
            coordinator=SimpleNamespace(
                dispatch_resample_batch=lambda p, slot_state: calls.append(
                    ("dispatch", p)
                ),
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
        self.assertIn(("collect", "row-mask"), calls)
        self.assertIn(("dispatch", plan), calls)
        self.assertIn(("wait", 0), calls)
        self.assertIn(("free", [10, 11]), calls)
        self.assertIn(("commit", [1], [0]), calls)
        self.assertIn(("apply", [1], [0]), calls)
        self.assertFalse(any(call[0] == "reset" for call in calls))
        self.assertEqual(float(scheduler.slot_state.group_log_Z_hat.item()), 5.0)
        self.assertEqual(int(scheduler.slot_state.kv_freed_counter[0, 0].item()), 0)


if __name__ == "__main__":
    unittest.main()

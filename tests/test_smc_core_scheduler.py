"""Unit tests for SMC scheduler glue and ``ScheduleBatchSMC``.

Covers:

* ``SMCScheduler._admit_prefill_groups`` admission gating against
  ``slot_state.available_slot_count()``.
* ``ScheduleBatchSMC.finalize_group`` — picks the highest-scoring particle
  and respects each particle's ``finished_len`` watermark.

Pure CPU.  Mocks the KV pools / allocator so the test isolates the
scheduler logic.  The fused resample path (Triton) is covered by
``test_smc_core_kernels.py`` on CUDA.
"""

import unittest
from collections import deque
from types import SimpleNamespace

import torch

from smcsd.core import scheduler as core_scheduler_mod
from smcsd.core.req_state import ScheduleBatchSMC
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


SMCScheduler = core_scheduler_mod.SMCScheduler
SequenceGroup = core_scheduler_mod.SequenceGroup


# ---------------------------------------------------------------------------
# Minimal fakes — just what the methods under test touch.
# ---------------------------------------------------------------------------


class _FakeReq:
    """Stand-in for sglang's Req object, with only the fields SMC reads."""

    def __init__(
        self,
        *,
        rid,
        particle_idx,
        req_pool_idx,
        output_ids,
        kv_indices,
        finished_reason=None,
        finished_len=None,
    ):
        self.rid = rid
        self.smc_particle_idx = particle_idx
        self.smc_group_id = rid.split("_")[0]
        self.req_pool_idx = req_pool_idx
        self.output_ids = list(output_ids)
        self.kv_committed_len = len(kv_indices)
        self.kv_allocated_len = len(kv_indices)
        self.cache_protected_len = len(kv_indices)
        self.finished_reason = finished_reason
        self.finished_len = finished_len
        self.finished_output = False
        self.to_finish = None
        self.prefix_indices = torch.tensor(kv_indices, dtype=torch.int64)
        self.decoded_text = ""
        self.surr_offset = None
        self.read_offset = None
        self.origin_input_ids = [1, 2, 3]

    def finished(self):
        return self.finished_reason is not None


class _FakeRuntimeReq:
    def __init__(self, *, group_id, particle_idx, req_pool_idx):
        self.rid = f"{group_id}_p{particle_idx}"
        self.smc_group_id = group_id
        self.smc_particle_idx = particle_idx
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = []
        self.kv_committed_len = 1
        self.kv_allocated_len = 1
        self.finished_reason = None


class _FakeReqToTokenPool:
    """Just enough surface for ``ScheduleBatchSMC`` KV bookkeeping."""

    def __init__(self, rows):
        self.req_to_token = torch.tensor(rows, dtype=torch.int32)
        self.device = "cpu"

    def write(self, key, value):
        row, cols = key
        self.req_to_token[row, cols] = value


class _FakeAllocator:
    """Records inc/dec/free invocations and maintains a slot_ref_count."""

    def __init__(self, size=256):
        self.inc_calls = []
        self.dec_calls = []
        self.free_calls = []
        self.page_size = 1
        self.slot_ref_count = torch.zeros(size, dtype=torch.int32)

    def inc_ref(self, indices):
        self.inc_calls.append(indices.clone())

    def dec_ref_and_free(self, indices):
        self.dec_calls.append(indices.clone())

    def free(self, indices):
        self.free_calls.append(indices.clone())


def _make_runtime_group(group_id, n_particles, *, pool_idx_base=0):
    reqs = {
        i: _FakeRuntimeReq(
            group_id=group_id,
            particle_idx=i,
            req_pool_idx=pool_idx_base + i,
        )
        for i in range(n_particles)
    }
    return SequenceGroup(
        parent_req=SimpleNamespace(rid=group_id),
        n_particles=n_particles,
        particle_temperature=0.7,
        particle_reqs=reqs,
    )


def _build_slot_state(*, max_num_reqs, rows, allocator_size=256, n_particles=1):
    return ScheduleBatchSMC(
        max_num_reqs=max_num_reqs,
        device="cpu",
        gamma_plus_1=2,
        vocab_size=32000,
        max_output_len=8,
        req_to_token_pool=_FakeReqToTokenPool(rows),
        token_to_kv_pool_allocator=_FakeAllocator(size=allocator_size),
        tree_cache=SimpleNamespace(),
        model_config=SimpleNamespace(),
        n_particles=n_particles,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSMCSchedulerAdmission(CustomTestCase):
    """``_admit_prefill_groups`` gates on available slot capacity."""

    def test_admit_prefill_groups_respects_free_slot_capacity(self):
        """When the slot pool is empty, queued groups stay queued."""
        queued_group = _make_runtime_group("g0", n_particles=2, pool_idx_base=10)
        scheduler = SimpleNamespace(
            waiting_groups=deque([queued_group]),
            max_running_requests=4,
            slot_state=SimpleNamespace(available_slot_count=lambda: 0),
        )
        scheduler._emit_abort = lambda req, error_msg: self.fail(
            f"unexpected abort for {req.rid}: {error_msg}"
        )

        admitted = SMCScheduler._admit_prefill_groups(scheduler)

        self.assertEqual(admitted, [])
        self.assertEqual(
            [group.group_id for group in scheduler.waiting_groups], ["g0"]
        )

    def test_admit_prefill_groups_drains_when_capacity_available(self):
        """When room is available, groups are admitted FIFO and dequeued."""
        g0 = _make_runtime_group("g0", n_particles=2, pool_idx_base=10)
        g1 = _make_runtime_group("g1", n_particles=2, pool_idx_base=20)
        scheduler = SimpleNamespace(
            waiting_groups=deque([g0, g1]),
            max_running_requests=4,
            slot_state=SimpleNamespace(available_slot_count=lambda: 4),
        )
        scheduler._emit_abort = lambda req, error_msg: self.fail(
            f"unexpected abort for {req.rid}: {error_msg}"
        )

        admitted = SMCScheduler._admit_prefill_groups(scheduler)

        self.assertEqual([g.group_id for g in admitted], ["g0", "g1"])
        self.assertEqual(len(scheduler.waiting_groups), 0)

    def test_admit_prefill_groups_blocks_on_insufficient_capacity(self):
        """A group whose particle count exceeds available slot capacity is
        not admitted and stays queued; no abort is emitted.  (The scheduler
        does not today have a policy for aborting oversized groups — that
        would be a separate feature.)"""
        oversized = _make_runtime_group("g0", n_particles=8, pool_idx_base=10)
        scheduler = SimpleNamespace(
            waiting_groups=deque([oversized]),
            max_running_requests=4,
            slot_state=SimpleNamespace(available_slot_count=lambda: 4),
        )
        scheduler._emit_abort = lambda req, error_msg: self.fail(
            f"unexpected abort for {req.rid}: {error_msg}"
        )

        admitted = SMCScheduler._admit_prefill_groups(scheduler)

        self.assertEqual(admitted, [])
        self.assertEqual(
            [g.group_id for g in scheduler.waiting_groups], ["g0"]
        )


class TestSMCFinalizeGroup(CustomTestCase):
    """``ScheduleBatchSMC.finalize_group`` — argmax on slot-indexed log_weights."""

    def test_finalize_picks_best_visible_finished_length(self):
        """Two particles tie on log_weight: pick the one with greater visible
        output (token_count clipped to finished_len), copy its
        output_ids / finished_reason / finished_len into parent_req."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 2, 3, 4], [5, 6, 7, 0]],
            n_particles=2,
        )
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[1, 2],
            kv_indices=[1, 2, 3, 4],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=2,
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[1, 2, 3],
            kv_indices=[5, 6, 7],
            finished_reason=SimpleNamespace(type="length"),
            finished_len=3,
        )
        slot_state.slot_to_req = {0: req0, 1: req1}
        slot_state.group_slot_lists = {"g": [0, 1]}
        # Equal finalize_score → tiebreak on visible output length.  Finalize
        # selects on the never-reset lifetime `finalize_score` (B1 repair).
        slot_state.finalize_score[0] = 0.0
        slot_state.finalize_score[1] = 0.0
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.kv_allocated_lens[0] = 4
        slot_state.kv_allocated_lens[1] = 3
        slot_state.token_counts[0] = 4  # finished_len=2 → visible=2
        slot_state.token_counts[1] = 3  # finished_len=3 → visible=3 ✓ winner

        freed = []
        slot_state.free_group_slots = lambda group_id: freed.append(group_id)

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        finalized = slot_state.finalize_group("g", parent_req)

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [1, 2, 3])
        self.assertEqual(parent_req.finished_len, 3)
        self.assertEqual(parent_req.finished_reason.type, "length")
        self.assertEqual(freed, ["g"])

    def test_finalize_picks_highest_log_weight(self):
        """When finalize_scores differ, the heavier particle wins regardless
        of output length."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        heavy = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[1],
            kv_indices=[1],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        light = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[2, 3, 4],
            kv_indices=[2],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=3,
        )
        slot_state.slot_to_req = {0: heavy, 1: light}
        slot_state.group_slot_lists = {"g": [0, 1]}
        slot_state.finalize_score[0] = 1.0    # heavier
        slot_state.finalize_score[1] = 0.0
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.kv_allocated_lens[0] = 1
        slot_state.kv_allocated_lens[1] = 1
        slot_state.token_counts[0] = 1
        slot_state.token_counts[1] = 3

        slot_state.free_group_slots = lambda group_id: None

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        finalized = slot_state.finalize_group("g", parent_req)

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [1])
        self.assertEqual(parent_req.finished_len, 1)


class TestSMCResetAndFinalizeScore(CustomTestCase):
    """S1 (docs/smc/async_bonus_design.md §2c + B1 repair): `finalize_score` is
    a net-new lifetime accumulator never touched by reset/collect, and
    `reset_interval_weights` zeros only `interval_weights`."""

    def _accumulate(self, slot_state, active, d):
        """Mirror process_batch_result step (e): all three accumulators += d."""
        slot_state.log_weights[active] += d
        slot_state.interval_weights[active] += d
        slot_state.finalize_score[active] += d

    def test_reset_zeros_interval_leaves_finalize_score(self):
        slot_state = _build_slot_state(max_num_reqs=4, rows=[[0]] * 4, n_particles=2)
        active = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        # Accumulate known weights across two windows (like two decode steps).
        self._accumulate(slot_state, active, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self._accumulate(slot_state, active, torch.tensor([0.5, 0.5, 0.5, 0.5]))

        lifetime = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(slot_state.interval_weights[active], lifetime))
        self.assertTrue(torch.allclose(slot_state.finalize_score[active], lifetime))

        # Simulate a resample: the fused kernel zeros interval_weights AND
        # log_weights on the resampled rows (slots 0,1).  reset_interval_weights
        # then extends the interval zero to ALL active rows.
        resampled = torch.tensor([0, 1], dtype=torch.int64)
        slot_state.interval_weights[resampled] = 0.0
        slot_state.log_weights[resampled] = 0.0

        slot_state.reset_interval_weights(active)

        # interval_weights zeroed everywhere; finalize_score untouched.
        self.assertTrue(torch.all(slot_state.interval_weights[active] == 0.0))
        self.assertTrue(torch.allclose(slot_state.finalize_score[active], lifetime))
        # log_weights zeroed only on resampled rows (kernel behavior); reset
        # does NOT touch log_weights on non-resampled rows.
        self.assertTrue(torch.all(slot_state.log_weights[resampled] == 0.0))
        self.assertTrue(
            torch.allclose(
                slot_state.log_weights[torch.tensor([2, 3])],
                torch.tensor([3.5, 4.5], dtype=torch.float64),
            )
        )

    def test_finalize_selects_on_true_lifetime_score(self):
        """After a reset zeros log_weights, finalize must pick the particle with
        the highest TRUE LIFETIME score — NOT the (kernel-zeroed) log_weights.
        Reference: argmax over the lifetime sum accumulated across all windows."""
        slot_state = _build_slot_state(max_num_reqs=2, rows=[[0]] * 2, n_particles=2)
        active = torch.tensor([0, 1], dtype=torch.int64)

        # Slot 0 is the true lifetime winner (3.0 vs 1.0).
        self._accumulate(slot_state, active, torch.tensor([3.0, 1.0]))

        # A resample zeros slot 0's interval AND log weight (it resampled);
        # slot 1 did not.  Now log_weights would WRONGLY favor slot 1.
        slot_state.interval_weights[torch.tensor([0])] = 0.0
        slot_state.log_weights[torch.tensor([0])] = 0.0
        slot_state.reset_interval_weights(active)

        # Sanity: the void selector (log_weights) would pick the wrong slot.
        self.assertGreater(
            float(slot_state.log_weights[1]), float(slot_state.log_weights[0])
        )

        req0 = _FakeReq(
            rid="g_p0", particle_idx=0, req_pool_idx=0,
            output_ids=[1], kv_indices=[1],
            finished_reason=SimpleNamespace(type="stop"), finished_len=1,
        )
        req1 = _FakeReq(
            rid="g_p1", particle_idx=1, req_pool_idx=1,
            output_ids=[2], kv_indices=[2],
            finished_reason=SimpleNamespace(type="stop"), finished_len=1,
        )
        slot_state.slot_to_req = {0: req0, 1: req1}
        slot_state.group_slot_lists = {"g": [0, 1]}
        slot_state.token_counts[0] = 1
        slot_state.token_counts[1] = 1
        slot_state.free_group_slots = lambda group_id: None

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        slot_state.finalize_group("g", parent_req)

        # Reference: the slot maximizing the true lifetime score is slot 0.
        ref_best = max([0, 1], key=lambda s: float(slot_state.finalize_score[s]))
        self.assertEqual(ref_best, 0)
        self.assertEqual(parent_req.output_ids, [1])  # slot 0's output


if __name__ == "__main__":
    unittest.main()

"""Unit tests for SMC v2 data structures and scheduler glue.

Covers:
  * `SMCSchedulerV2._admit_prefill_groups` admission gating against
    `slot_state.available_slot_count()`.
  * `SMCCoordinatorV2` slow-path resample (`collect_resample_jobs_batch`
    + `dispatch_resample_batch`) over `ScheduleBatchSMC`: dead slots
    take the survivor's KV block table, finished-state propagates,
    refcounts move correctly.
  * `ScheduleBatchSMC.finalize_group` picks the highest-scoring particle
    and respects each particle's `finished_len` watermark.

Pure CPU — no Triton, no GPU.  Mocks `req_to_token_pool` and
`token_to_kv_pool_allocator` so the test isolates v2's own logic.
"""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import torch

from smcsd.v2 import scheduler as v2_scheduler_mod
from smcsd.v2.req_state import ScheduleBatchSMC
from smcsd.v2.info import SMCDecodeContext, SMCEagleDraftInputV2, SMCDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")


SMCCoordinatorV2 = v2_scheduler_mod.SMCCoordinatorV2
SMCSchedulerV2 = v2_scheduler_mod.SMCSchedulerV2
SequenceGroup = v2_scheduler_mod.SequenceGroup


# ---------------------------------------------------------------------------
# Fakes (kept tiny — only what the methods under test actually touch)
# ---------------------------------------------------------------------------


class _FakeReq:
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
    """Just enough surface for ScheduleBatchSMC.resample_copy_slot."""

    def __init__(self, rows):
        self.req_to_token = torch.tensor(rows, dtype=torch.int32)
        self.device = "cpu"

    def write(self, key, value):
        row, cols = key
        self.req_to_token[row, cols] = value


class _FakeAllocator:
    """Records inc/dec/free invocations + maintains a slot_ref_count tensor."""

    def __init__(self, size=256):
        self.inc_calls = []
        self.dec_calls = []
        self.free_calls = []
        self.free_group_depth = 0
        self.page_size = 1
        self.slot_ref_count = torch.zeros(size, dtype=torch.int32)

    def free_group_begin(self):
        self.free_group_depth += 1

    def free_group_end(self):
        self.free_group_depth -= 1

    def inc_ref(self, indices):
        self.inc_calls.append(indices.clone())

    def dec_ref_and_free(self, indices):
        self.dec_calls.append(indices.clone())

    def free(self, indices):
        self.free_calls.append(indices.clone())


class _FakeTreeTokenAllocator:
    def __init__(self, size=4096):
        self.page_size = 1
        self._free = torch.arange(size, dtype=torch.int64)
        self._cursor = 0

    def available_size(self):
        return len(self._free) - self._cursor

    def alloc(self, num_tokens):
        start = self._cursor
        end = start + num_tokens
        if end > len(self._free):
            return None
        self._cursor = end
        return self._free[start:end].clone()


class _FakeTreeCache:
    def __init__(self, size=4096):
        self.token_to_kv_pool_allocator = _FakeTreeTokenAllocator(size=size)

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        return None

    def pretty_print(self):
        return None


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
        log_weights=torch.zeros(n_particles, dtype=torch.float64),
    )


def _build_slot_state(
    *, max_num_reqs, rows, allocator_size=256, smc_draft_kind="lm", smc_eagle_topk=4
):
    return ScheduleBatchSMC(
        max_num_reqs=max_num_reqs,
        device="cpu",
        gamma_plus_1=2,
        vocab_size=32000,
        max_output_len=8,
        req_to_token_pool=_FakeReqToTokenPool(rows),
        token_to_kv_pool_allocator=_FakeAllocator(size=allocator_size),
        tree_cache=_FakeTreeCache(),
        model_config=SimpleNamespace(),
        smc_draft_kind=smc_draft_kind,
        smc_eagle_topk=smc_eagle_topk,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSMCSchedulerAdmission(CustomTestCase):
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

        admitted = SMCSchedulerV2._admit_prefill_groups(scheduler)

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

        admitted = SMCSchedulerV2._admit_prefill_groups(scheduler)

        self.assertEqual([g.group_id for g in admitted], ["g0", "g1"])
        self.assertEqual(len(scheduler.waiting_groups), 0)

    def test_admit_prefill_groups_aborts_oversized_group(self):
        """A group whose particle count exceeds max_running_requests is aborted."""
        oversized = _make_runtime_group("g0", n_particles=8, pool_idx_base=10)
        scheduler = SimpleNamespace(
            waiting_groups=deque([oversized]),
            max_running_requests=4,
            slot_state=SimpleNamespace(available_slot_count=lambda: 4),
        )
        aborted = []
        scheduler._emit_abort = lambda req, error_msg: aborted.append(
            (req.rid, error_msg)
        )

        admitted = SMCSchedulerV2._admit_prefill_groups(scheduler)

        self.assertEqual(admitted, [])
        self.assertEqual(len(scheduler.waiting_groups), 0)
        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0][0], "g0")


class TestSMCDraftCarrierSelection(CustomTestCase):
    def test_prepare_for_decode_returns_lm_carrier_by_default(self):
        slot_state = _build_slot_state(max_num_reqs=2, rows=[[0, 0, 0], [0, 0, 0]])
        slot_state.active_slots = torch.tensor([0], dtype=torch.int64)
        slot_state.num_active = 1
        slot_state.req_pool_indices[0] = 0
        slot_state.seq_lens[0] = 1
        slot_state.kv_allocated_lens[0] = 1
        slot_state.verified_ids[0] = 42

        fake_ctx = SMCDecodeContext(
            orig_seq_lens=torch.tensor([1], dtype=torch.int64),
            orig_seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
            orig_seq_lens_sum=1,
            new_seq_lens=torch.tensor([3], dtype=torch.int64),
            gamma=1,
        )
        with patch(
            "smcsd.v2.req_state.SMCDecodeContext.from_slot_gather",
            return_value=(fake_ctx, torch.tensor([3], dtype=torch.int64)),
        ):
            draft_input = slot_state.prepare_for_decode()

        self.assertIsInstance(draft_input, SMCDraftInputV2)
        self.assertEqual(draft_input.verified_id.tolist(), [42])
        self.assertIs(draft_input.decode_ctx, fake_ctx)

    def test_prepare_for_decode_returns_eagle_carrier_when_requested(self):
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[0, 0, 0], [0, 0, 0]],
            smc_draft_kind="eagle",
            smc_eagle_topk=4,
        )
        slot_state.active_slots = torch.tensor([0], dtype=torch.int64)
        slot_state.num_active = 1
        slot_state.req_pool_indices[0] = 0
        slot_state.seq_lens[0] = 1
        slot_state.kv_allocated_lens[0] = 1
        slot_state.verified_ids[0] = 77

        fake_ctx = SMCDecodeContext(
            orig_seq_lens=torch.tensor([1], dtype=torch.int64),
            orig_seq_lens_cpu=torch.tensor([1], dtype=torch.int64),
            orig_seq_lens_sum=1,
            new_seq_lens=torch.tensor([3], dtype=torch.int64),
            gamma=1,
        )
        with patch(
            "smcsd.v2.req_state.SMCDecodeContext.from_slot_gather",
            return_value=(fake_ctx, torch.tensor([3], dtype=torch.int64)),
        ):
            draft_input = slot_state.prepare_for_decode()

        self.assertIsInstance(draft_input, SMCEagleDraftInputV2)
        self.assertEqual(draft_input.verified_id.tolist(), [77])
        self.assertIs(draft_input.decode_ctx, fake_ctx)
        self.assertIsNone(draft_input.hidden_states)
        self.assertIsNone(draft_input.topk_p)
        self.assertIsNone(draft_input.topk_index)


class TestSMCResampleSlowPath(CustomTestCase):
    def test_resample_copies_survivor_state_and_refcounts(self):
        """Skewed weights → all particles end up as copies of the lone survivor.

        Setup: 3 particles in group "g".  Particle 1 has weight ≈ 1.0
        (log_weight 0), particle 0 / particle 2 have effectively zero
        weight (log_weight = -1e10) so systematic resample is forced to
        pick particle 1 for every ancestor draw — deterministic regardless
        of the torch RNG.

        Particle 1 is also marked finished (`finished_reason` set).  After
        resampling, both other particles must inherit its output_ids,
        finished_reason, finished_len, and finished_mask.

        KV bookkeeping: dst slots dec_ref their old indices, inc_ref the
        survivor's indices, and the dst block-table rows now match the
        survivor's row.
        """
        slot_state = _build_slot_state(
            max_num_reqs=3,
            rows=[[1, 0, 0], [7, 0, 0], [3, 4, 0]],
        )
        req0 = _FakeReq(
            rid="g_p0",
            particle_idx=0,
            req_pool_idx=0,
            output_ids=[10],
            kv_indices=[1],
        )
        req1 = _FakeReq(
            rid="g_p1",
            particle_idx=1,
            req_pool_idx=1,
            output_ids=[99],
            kv_indices=[7],
            finished_reason=SimpleNamespace(type="stop"),
            finished_len=1,
        )
        req2 = _FakeReq(
            rid="g_p2",
            particle_idx=2,
            req_pool_idx=2,
            output_ids=[20, 21],
            kv_indices=[3, 4],
        )
        slot_state.slot_to_req = {0: req0, 1: req1, 2: req2}
        slot_state.group_slot_lists = {"g": [0, 1, 2]}
        # Override the stacked-storage views with plain tensors — the
        # collect path only reads & writes via `__getitem__` on the dict.
        slot_state.group_log_weights = {
            "g": torch.tensor([-1e10, 0.0, -1e10], dtype=torch.float64)
        }
        slot_state.group_interval_weights = {
            "g": torch.tensor([-1e10, 0.0, -1e10], dtype=torch.float64)
        }
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.req_pool_indices[2] = 2
        slot_state.seq_lens[0] = 1
        slot_state.seq_lens[1] = 1
        slot_state.seq_lens[2] = 2
        slot_state.kv_allocated_lens[0] = 1
        slot_state.kv_allocated_lens[1] = 1
        slot_state.kv_allocated_lens[2] = 2
        slot_state.token_counts[0] = 1
        slot_state.token_counts[1] = 1
        slot_state.token_counts[2] = 2
        slot_state.particle_indices[0] = 0
        slot_state.particle_indices[1] = 1
        slot_state.particle_indices[2] = 2
        slot_state.finished_mask[1] = True
        slot_state.rebuild_active_slots()

        coordinator = SMCCoordinatorV2(
            device="cpu",
            resample_threshold=0.75,
            resample_method="systematic",
        )

        plan = coordinator.collect_resample_jobs_batch(["g"], slot_state)
        coordinator.dispatch_resample_batch(plan, slot_state)

        # Survivor state propagated to dead particles
        self.assertEqual(req0.output_ids, [99])
        self.assertEqual(req2.output_ids, [99])
        self.assertEqual(req0.finished_reason.type, "stop")
        self.assertEqual(req2.finished_reason.type, "stop")
        self.assertEqual(req0.finished_len, 1)
        self.assertEqual(req2.finished_len, 1)
        self.assertTrue(slot_state.finished_mask[0].item())
        self.assertTrue(slot_state.finished_mask[1].item())
        self.assertTrue(slot_state.finished_mask[2].item())
        self.assertFalse(slot_state.group_has_active("g"))
        self.assertEqual(slot_state.active_particle_count(), 0)

        # Interval weights zeroed (group reset for the next decode step)
        self.assertEqual(
            slot_state.group_interval_weights["g"].tolist(), [0.0, 0.0, 0.0]
        )

        # Block table: dst rows now hold the survivor's KV index (7)
        self.assertEqual(int(slot_state.req_to_token_pool.req_to_token[0, 0].item()), 7)
        self.assertEqual(int(slot_state.req_to_token_pool.req_to_token[2, 0].item()), 7)

        # Refcount: 2 inc_ref calls (one per dst inheriting src's KV)
        # and 2 dec_ref_and_free calls (one per dst dropping its old KV).
        # The dropped allocations are sized 1 (slot 0) and 2 (slot 2).
        allocator = slot_state.token_to_kv_pool_allocator
        self.assertEqual(len(allocator.inc_calls), 2)
        self.assertEqual(len(allocator.dec_calls), 2)
        self.assertEqual(
            sorted(t.numel() for t in allocator.dec_calls), [1, 2],
        )

    def test_no_resample_when_weights_are_equal(self):
        """Equal weights → ESS = N → coordinator emits an empty plan and
        does NOT touch req state or refcounts."""
        slot_state = _build_slot_state(
            max_num_reqs=3,
            rows=[[1, 0, 0], [7, 0, 0], [3, 4, 0]],
        )
        req0 = _FakeReq(rid="g_p0", particle_idx=0, req_pool_idx=0,
                        output_ids=[1], kv_indices=[1])
        req1 = _FakeReq(rid="g_p1", particle_idx=1, req_pool_idx=1,
                        output_ids=[7], kv_indices=[7])
        req2 = _FakeReq(rid="g_p2", particle_idx=2, req_pool_idx=2,
                        output_ids=[3, 4], kv_indices=[3, 4])
        slot_state.slot_to_req = {0: req0, 1: req1, 2: req2}
        slot_state.group_slot_lists = {"g": [0, 1, 2]}
        slot_state.group_log_weights = {
            "g": torch.zeros(3, dtype=torch.float64)
        }
        slot_state.group_interval_weights = {
            "g": torch.zeros(3, dtype=torch.float64)
        }
        slot_state.req_pool_indices[:3] = torch.tensor([0, 1, 2])
        slot_state.particle_indices[:3] = torch.tensor([0, 1, 2])
        slot_state.seq_lens[:3] = torch.tensor([1, 1, 2])
        slot_state.kv_allocated_lens[:3] = torch.tensor([1, 1, 2])
        slot_state.token_counts[:3] = torch.tensor([1, 1, 2])
        slot_state.rebuild_active_slots()

        coordinator = SMCCoordinatorV2(
            device="cpu",
            resample_threshold=0.5,  # ESS=3 ≥ 0.5*3=1.5 → no resample
            resample_method="systematic",
        )

        plan = coordinator.collect_resample_jobs_batch(["g"], slot_state)
        coordinator.dispatch_resample_batch(plan, slot_state)

        allocator = slot_state.token_to_kv_pool_allocator
        self.assertEqual(allocator.inc_calls, [])
        self.assertEqual(allocator.dec_calls, [])
        self.assertEqual(req0.output_ids, [1])
        self.assertEqual(req1.output_ids, [7])
        self.assertEqual(req2.output_ids, [3, 4])


class TestSMCFinalizeGroup(CustomTestCase):
    def test_finalize_picks_best_visible_finished_length(self):
        """When two particles tie on log_weight, finalize_group must pick
        the one with greater visible output (token_count clipped to
        finished_len), then copy its output_ids / finished_reason /
        finished_len into the parent_req.
        """
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 2, 3, 4], [5, 6, 7, 0]],
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
        slot_state.group_log_weights = {
            "g": torch.tensor([0.0, 0.0], dtype=torch.float64)
        }
        slot_state.req_pool_indices[0] = 0
        slot_state.req_pool_indices[1] = 1
        slot_state.kv_allocated_lens[0] = 4
        slot_state.kv_allocated_lens[1] = 3
        slot_state.particle_indices[0] = 0
        slot_state.particle_indices[1] = 1
        slot_state.token_counts[0] = 4  # but finished_len=2 → visible=2
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


if __name__ == "__main__":
    unittest.main()

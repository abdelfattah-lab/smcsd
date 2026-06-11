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
        self.size = size  # read by ScheduleBatchSMC for kv_freed_buf sizing
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
    """``ScheduleBatchSMC.finalize_group`` — posterior sample over
    slot-indexed log_weights + tensor-resident finish state + the
    particle-collection / log-Ẑ attachments."""

    def _arm_group(self, slot_state, *, log_weights, finished_lens,
                   reason_codes, matched_toks, outputs):
        """Populate the tensor-resident finish state for a 2-slot group."""
        slot_state.group_slot_lists = {"g": [0, 1]}
        for s in (0, 1):
            slot_state.log_weights[s] = log_weights[s]
            slot_state.interval_weights[s] = log_weights[s]
            slot_state.finished_len[s] = finished_lens[s]
            slot_state.finish_reason_code[s] = reason_codes[s]
            slot_state.matched_eos_token[s] = matched_toks[s]
            slot_state.token_counts[s] = len(outputs[s])
            slot_state.all_token_ids[s, : len(outputs[s])] = torch.tensor(
                outputs[s], dtype=torch.int32
            )

    def test_finalize_samples_dominant_weight(self):
        """With one particle overwhelmingly heavier, the posterior sample
        deterministically picks it; its tensor-resident finish state is
        materialised onto parent_req (EOS → FINISH_MATCHED_TOKEN)."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        # softmax([0, -1e9]) == [1, 0] exactly in float64 → deterministic.
        self._arm_group(
            slot_state,
            log_weights=[0.0, -1e9],
            finished_lens=[2, 3],
            reason_codes=[2, 1],          # winner finished via EOS
            matched_toks=[7, 0],
            outputs=[[5, 7], [8, 9, 10]],
        )
        freed = []
        slot_state.free_group_slots = lambda group_id: freed.append(group_id)

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        finalized = slot_state.finalize_group("g", parent_req)

        self.assertIs(finalized, parent_req)
        self.assertEqual(parent_req.output_ids, [5, 7])
        self.assertEqual(parent_req.finished_len, 2)
        from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
        self.assertIsInstance(parent_req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(freed, ["g"])

    def test_finalize_attaches_particle_collection_and_log_Z(self):
        """parent_req carries the full collection: per-particle outputs
        (sliced to finished_len), final log-weights, and the unbiased
        log-Ẑ tail ``logsumexp(interval_weights) - log(N)``."""
        import math

        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        self._arm_group(
            slot_state,
            log_weights=[0.0, 0.0],
            finished_lens=[1, 3],
            reason_codes=[1, 1],          # both length-finished
            matched_toks=[0, 0],
            outputs=[[5], [8, 9, 10]],
        )
        slot_state.free_group_slots = lambda group_id: None

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        slot_state.finalize_group("g", parent_req)

        self.assertEqual(
            parent_req.smc_particle_output_ids, [[5], [8, 9, 10]]
        )
        self.assertEqual(parent_req.smc_log_w_tilde, [0.0, 0.0])
        # No resample boundary accrued (no group row) → log Ẑ is just the
        # tail: logsumexp([0, 0]) - log(2) = 0.
        self.assertAlmostEqual(parent_req.smc_log_Z_hat, 0.0, places=12)
        # Length finish materialises FINISH_LENGTH.
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH
        self.assertIsInstance(parent_req.finished_reason, FINISH_LENGTH)
        # math import is used implicitly above via the documented formula.
        del math


class TestSMCCycleDiagnostics(CustomTestCase):
    """Per-group decode-cycle diagnostics: ``accumulate_cycle_counters``,
    ``accumulate_ess`` (SMC_TRACK_ESS gate), and their materialisation as
    ``smc_n_cycles`` / ``smc_n_resamples`` / ``smc_mean_ess`` at finalize."""

    def _arm_row(self, slot_state, *, log_weights, outputs):
        """Arm a 2-particle group at row 0 with full row bookkeeping (the
        finalize-group tests above skip the row; diagnostics need it)."""
        slot_state.group_slot_lists = {"g": [0, 1]}
        slot_state.group_id_to_row = {"g": 0}
        slot_state.group_to_slots[0, :2] = torch.tensor([0, 1], dtype=torch.int32)
        slot_state.row_in_use[0] = True
        for s in (0, 1):
            slot_state.log_weights[s] = log_weights[s]
            slot_state.interval_weights[s] = log_weights[s]
            slot_state.finished_len[s] = len(outputs[s])
            slot_state.finish_reason_code[s] = 1
            slot_state.token_counts[s] = len(outputs[s])
            slot_state.all_token_ids[s, : len(outputs[s])] = torch.tensor(
                outputs[s], dtype=torch.int32
            )
        slot_state.free_group_slots = lambda group_id: None

    def test_cycle_counters_accumulate_and_attach(self):
        """Three cycles, one of which resamples → n_cycles=3, n_resamples=1
        on the finalized parent.  Free rows stay untouched."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        self._arm_row(slot_state, log_weights=[0.0, 0.0], outputs=[[5], [6]])

        mask_hit = torch.zeros(slot_state.max_groups, dtype=torch.bool)
        mask_hit[0] = True
        mask_miss = torch.zeros(slot_state.max_groups, dtype=torch.bool)
        slot_state.accumulate_cycle_counters(mask_miss)
        slot_state.accumulate_cycle_counters(mask_hit)
        slot_state.accumulate_cycle_counters(mask_miss)

        self.assertEqual(int(slot_state.group_n_cycles[0]), 3)
        self.assertEqual(int(slot_state.group_n_resamples[0]), 1)
        self.assertEqual(int(slot_state.group_n_cycles[1:].sum()), 0)

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        slot_state.finalize_group("g", parent_req)
        self.assertEqual(parent_req.smc_n_cycles, 3)
        self.assertEqual(parent_req.smc_n_resamples, 1)
        self.assertIsNone(parent_req.smc_mean_ess)  # track_ess off

    def test_mean_ess_tracks_interval_weights(self):
        """With SMC_TRACK_ESS on: equal interval weights give ESS = N per
        cycle; a degenerate pair gives ESS ≈ 1.  finalize reports the mean."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        slot_state.track_ess = True
        self._arm_row(slot_state, log_weights=[0.0, 0.0], outputs=[[5], [6]])
        mask_miss = torch.zeros(slot_state.max_groups, dtype=torch.bool)

        # Cycle 1: equal weights → ESS = 2.
        slot_state.accumulate_ess()
        slot_state.accumulate_cycle_counters(mask_miss)
        # Cycle 2: degenerate weights → ESS → 1.
        slot_state.interval_weights[1] = -1e9
        slot_state.accumulate_ess()
        slot_state.accumulate_cycle_counters(mask_miss)

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        slot_state.finalize_group("g", parent_req)
        self.assertEqual(parent_req.smc_n_cycles, 2)
        self.assertAlmostEqual(parent_req.smc_mean_ess, 1.5, places=9)

    def test_finalize_without_row_defaults_diagnostics(self):
        """Groups armed without row bookkeeping (the legacy test path)
        finalize with zeroed counters and mean_ess=None."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 0, 0, 0], [2, 0, 0, 0]],
            n_particles=2,
        )
        slot_state.group_slot_lists = {"g": [0, 1]}
        for s in (0, 1):
            slot_state.finished_len[s] = 1
            slot_state.finish_reason_code[s] = 1
            slot_state.token_counts[s] = 1
            slot_state.all_token_ids[s, 0] = 5
        slot_state.free_group_slots = lambda group_id: None

        parent_req = SimpleNamespace(
            output_ids=[], finished_reason=None, finished_len=None
        )
        slot_state.finalize_group("g", parent_req)
        self.assertEqual(parent_req.smc_n_cycles, 0)
        self.assertEqual(parent_req.smc_n_resamples, 0)
        self.assertIsNone(parent_req.smc_mean_ess)


class _FakeSamplingParams(SimpleNamespace):
    pass


class TestSMCAllocateSlots(CustomTestCase):
    """``allocate_slots`` seeding — regression for the deferred-bonus
    mixed-batch crash."""

    def _make_particle(self, *, pool_idx, origin_input_ids, output_ids):
        return SimpleNamespace(
            req_pool_idx=pool_idx,
            origin_input_ids=list(origin_input_ids),
            output_ids=list(output_ids),
            eos_token_ids=[2],
            sampling_params=_FakeSamplingParams(
                ignore_eos=False,
                max_new_tokens=8,
                stop_token_ids=None,
            ),
        )

    def test_prev_last_draft_seeded_with_last_committed_token(self):
        """The 2-token deferred-bonus head consumes prev_last_draft_ids on a
        group's FIRST decode step, so allocate_slots must seed it with the
        last committed token (whose S-1 draft KV the head rewrites
        idempotently) — never a sentinel, which would reach the embedding
        as a token id in mixed-step batches."""
        slot_state = _build_slot_state(
            max_num_reqs=2,
            rows=[[1, 2, 3, 0], [4, 5, 6, 0]],
            n_particles=2,
        )
        # Committed prefix = prompt [11, 12, 13]; x0=99 sampled at prefill
        # (its KV not yet written) → shared_seq_len = 3, last committed
        # token = 13.
        particles = [
            self._make_particle(
                pool_idx=i, origin_input_ids=[11, 12, 13], output_ids=[99]
            )
            for i in range(2)
        ]

        slots = slot_state.allocate_slots(
            group_id="g", particle_reqs=particles, shared_seq_len=3
        )

        for s in slots:
            self.assertEqual(int(slot_state.prev_last_draft_ids[s].item()), 13)
            self.assertEqual(int(slot_state.verified_ids[s].item()), 99)
            self.assertEqual(int(slot_state.seq_lens[s].item()), 3)
        # All slots active (absorbing-state semantics keep membership
        # static between allocate and free).
        self.assertEqual(slot_state.num_active, 2)


if __name__ == "__main__":
    unittest.main()

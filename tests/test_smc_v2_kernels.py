"""Triton kernel correctness tests for SMC v2.

Covers the two fused kernels under `python/sglang/srt/smc/v2/kernels/`:

  * `fused_collect.batched_collect_fused`
        normalize → ESS check → systematic resample → dead/excess
        compaction, emitting flat (dst, src, row) tensors.

  * `fused_resample_kv.batched_resample_kv`
        per-job dec_ref(old dst) → block-table copy(src → dst) →
        inc_ref(src), atomically.

Tests focus on observable invariants that hold regardless of the kernel's
internal Philox seed:

  * resample_mask reflects ESS < threshold * n_active
  * dst_slots are unique and disjoint from src_slots (per-row contract)
  * job count == sum of dead cells == sum of excess copies
  * weights are zeroed only on rows that resampled
  * KV refcounts move correctly and to_free reports slots that hit zero
"""

import unittest

import torch

from smcsd.v2.kernels.fused_collect import batched_collect_fused
from smcsd.v2.kernels.fused_resample_kv import batched_resample_kv
from smcsd.v2.stacked_state import StackedGroupState
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedCollectKernel(CustomTestCase):
    """`batched_collect_fused` — fused per-row ESS + systematic resample."""

    DEVICE = "cuda"

    def _make_stacked(self, max_groups=4, n_particles=4):
        return StackedGroupState(
            max_groups=max_groups,
            n_particles=n_particles,
            device=torch.device(self.DEVICE),
        )

    def _set_iw(self, stacked, group_id, values):
        """Override the interval_weights for `group_id`'s row (whichever index
        register_group happened to allocate)."""
        row = stacked.group_id_to_row[group_id]
        stacked.interval_weights[row] = torch.tensor(
            values, dtype=torch.float64, device=self.DEVICE,
        )
        return row

    def test_no_resample_when_weights_are_equal(self):
        """ESS == N → resample_mask is False everywhere, no jobs emitted,
        weights untouched."""
        stacked = self._make_stacked(max_groups=2, n_particles=4)
        stacked.register_group("g0", slots=[10, 11, 12, 13])

        before_iw = stacked.interval_weights.clone()
        before_lw = stacked.log_weights.clone()

        result = batched_collect_fused(stacked, threshold=0.5, step_counter=1)

        self.assertEqual(result.n_jobs, 0)
        # All rows report False (the registered one because ESS=N is high
        # enough; the unused one because row_in_use=False).
        self.assertFalse(bool(result.resample_mask.any().item()))
        # No mutation when no resample.
        self.assertTrue(torch.equal(stacked.interval_weights, before_iw))
        self.assertTrue(torch.equal(stacked.log_weights, before_lw))

    def test_dominant_weight_forces_full_resample(self):
        """When one particle has weight ≈ 1, all dead cells are the survivor's slot."""
        stacked = self._make_stacked(max_groups=2, n_particles=4)
        stacked.register_group("g0", slots=[20, 21, 22, 23])
        # Particle 1 dominates; the other three are essentially zero weight.
        row = self._set_iw(stacked, "g0", [-1e10, 0.0, -1e10, -1e10])

        result = batched_collect_fused(stacked, threshold=0.5, step_counter=2)

        self.assertTrue(bool(result.resample_mask[row].item()))
        # 3 dead cells (all except particle 1), so 3 jobs.
        self.assertEqual(result.n_jobs, 3)

        dst = result.dst_slots.tolist()
        src = result.src_slots.tolist()
        rows = result.row_of_job.tolist()
        # Per-row contract: dst unique, dst ∩ src = ∅.
        self.assertEqual(sorted(dst), sorted(set(dst)))
        self.assertTrue(set(dst).isdisjoint(set(src)))
        # All dst are the dead slots {20, 22, 23}; all src are the survivor's slot 21.
        self.assertEqual(set(dst), {20, 22, 23})
        self.assertEqual(set(src), {21})
        # All jobs are tagged with this row.
        self.assertEqual(set(rows), {row})
        # Weights zeroed on this row only.
        zero4 = torch.zeros(4, dtype=torch.float64, device=self.DEVICE)
        self.assertTrue(torch.equal(stacked.interval_weights[row], zero4))
        self.assertTrue(torch.equal(stacked.log_weights[row], zero4))

    def test_multi_row_isolates_resample_to_skewed_row(self):
        """Two groups: only the skewed one resamples; the uniform one is untouched.
        Specifically tests per-row independence of the kernel."""
        stacked = self._make_stacked(max_groups=2, n_particles=4)
        stacked.register_group("g_uniform", slots=[100, 101, 102, 103])  # uniform → no resample
        stacked.register_group("g_skewed", slots=[200, 201, 202, 203])   # skewed  → resample

        skewed_row = self._set_iw(stacked, "g_skewed", [-1e10, -1e10, 0.0, -1e10])
        uniform_row = stacked.group_id_to_row["g_uniform"]

        before_uniform_iw = stacked.interval_weights[uniform_row].clone()
        before_uniform_lw = stacked.log_weights[uniform_row].clone()

        result = batched_collect_fused(stacked, threshold=0.5, step_counter=3)

        self.assertFalse(bool(result.resample_mask[uniform_row].item()))
        self.assertTrue(bool(result.resample_mask[skewed_row].item()))
        self.assertEqual(result.n_jobs, 3)
        # All emitted jobs are tagged with the skewed row.
        self.assertEqual(set(result.row_of_job.tolist()), {skewed_row})
        # All dst slots are from the skewed group.
        self.assertEqual(set(result.dst_slots.tolist()), {200, 201, 203})
        # All src slots are the lone survivor in the skewed group.
        self.assertEqual(set(result.src_slots.tolist()), {202})
        # Uniform row weights untouched.
        self.assertTrue(torch.equal(stacked.interval_weights[uniform_row], before_uniform_iw))
        self.assertTrue(torch.equal(stacked.log_weights[uniform_row], before_uniform_lw))
        # Skewed row weights zeroed.
        zero4 = torch.zeros(4, dtype=torch.float64, device=self.DEVICE)
        self.assertTrue(torch.equal(stacked.interval_weights[skewed_row], zero4))
        self.assertTrue(torch.equal(stacked.log_weights[skewed_row], zero4))

    def test_dead_and_src_partition_matches_target_counts(self):
        """For an arbitrary skewed distribution, the kernel's flat (dst, src)
        emission must satisfy:  |dst| == |src|  (conservation —
        Σ target_counts == n_active so #dead == sum of excess copies)."""
        stacked = self._make_stacked(max_groups=1, n_particles=4)
        stacked.register_group("g0", slots=[300, 301, 302, 303])
        # Two survivors (particles 1 & 2), two dead (particles 0 & 3).
        # ess = 1 / (0.5² + 0.5²) = 2; need threshold * n_active > ess to fire,
        # so use threshold=0.75 → 0.75 * 4 = 3 > 2.
        self._set_iw(stacked, "g0", [-1e10, 0.0, 0.0, -1e10])

        result = batched_collect_fused(stacked, threshold=0.75, step_counter=4)

        self.assertTrue(bool(result.resample_mask.any().item()))
        # Dead cells are particles 0 and 3 (weight ≈ 0). Their slots: {300, 303}.
        self.assertEqual(set(result.dst_slots.tolist()), {300, 303})
        # Survivor slots are particles 1 and 2 ({301, 302}); src must be a
        # subset (specific membership depends on the systematic offset).
        self.assertTrue(set(result.src_slots.tolist()).issubset({301, 302}))
        # Conservation: #dst == #src.
        self.assertEqual(len(result.dst_slots), len(result.src_slots))


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedResampleKVKernel(CustomTestCase):
    """`batched_resample_kv` — fused dec_ref + block-table copy + inc_ref."""

    DEVICE = "cuda"

    def _build_pool(self, rows, refcount_size=64):
        """Build (req_to_token, refcount) tensors and pre-arm refcounts.

        Each KV index that appears anywhere in `rows` starts at refcount=1.
        """
        req_to_token = torch.tensor(
            rows, dtype=torch.int32, device=self.DEVICE
        )
        refcount = torch.zeros(refcount_size, dtype=torch.int32, device=self.DEVICE)
        # Mark each used KV slot as having one owner.
        for row in rows:
            for idx in row:
                if idx > 0:
                    refcount[idx] = 1
        return req_to_token, refcount

    def test_single_pair_copies_block_table_and_updates_refcounts(self):
        """One eviction job: dst row was [11, 12], src row [21, 22, 23].
        After the kernel: dst row's first 3 entries == src row, refcount on
        {21,22,23} bumped by 1, refcount on {11,12} dropped by 1 (and since
        they had only one owner, they end up in to_free)."""
        # Pool layout (only entries up to alloc_len matter):
        #   pool 0 → [11, 12, 0, 0]   (dst, alloc=2)
        #   pool 1 → [21, 22, 23, 0]  (src, len=3)
        rows = [[11, 12, 0, 0], [21, 22, 23, 0]]
        req_to_token, refcount = self._build_pool(rows)

        dst_pool = torch.tensor([0], dtype=torch.int32, device=self.DEVICE)
        src_pool = torch.tensor([1], dtype=torch.int32, device=self.DEVICE)
        dst_alloc = torch.tensor([2], dtype=torch.int32, device=self.DEVICE)
        src_seq = torch.tensor([3], dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token, refcount, dst_pool, src_pool, dst_alloc, src_seq,
        )

        # Block table was copied
        self.assertEqual(req_to_token[0, :3].tolist(), [21, 22, 23])
        # Refcount: src indices bumped to 2; dst's old indices dropped to 0.
        self.assertEqual(refcount[21].item(), 2)
        self.assertEqual(refcount[22].item(), 2)
        self.assertEqual(refcount[23].item(), 2)
        self.assertEqual(refcount[11].item(), 0)
        self.assertEqual(refcount[12].item(), 0)
        # to_free contains exactly the freed dst indices (sorted, unique).
        self.assertEqual(sorted(to_free.tolist()), [11, 12])

    def test_multiple_pairs_run_in_parallel(self):
        """Two eviction jobs at once. Each dst inherits its own src; refcounts
        and to_free reflect both jobs."""
        # Pool layout (alloc lens shown):
        #   pool 0 → [11, 12]      (dst0, alloc=2)
        #   pool 1 → [13]          (dst1, alloc=1)
        #   pool 2 → [21, 22, 23]  (src0, len=3)
        #   pool 3 → [24, 25]      (src1, len=2)
        rows = [
            [11, 12, 0, 0],
            [13, 0, 0, 0],
            [21, 22, 23, 0],
            [24, 25, 0, 0],
        ]
        req_to_token, refcount = self._build_pool(rows)

        dst_pool = torch.tensor([0, 1], dtype=torch.int32, device=self.DEVICE)
        src_pool = torch.tensor([2, 3], dtype=torch.int32, device=self.DEVICE)
        dst_alloc = torch.tensor([2, 1], dtype=torch.int32, device=self.DEVICE)
        src_seq = torch.tensor([3, 2], dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token, refcount, dst_pool, src_pool, dst_alloc, src_seq,
        )

        # Block tables copied in place.
        self.assertEqual(req_to_token[0, :3].tolist(), [21, 22, 23])
        self.assertEqual(req_to_token[1, :2].tolist(), [24, 25])
        # Refcounts: src indices ↑, dst's old indices ↓.
        for idx in (21, 22, 23, 24, 25):
            self.assertEqual(refcount[idx].item(), 2)
        for idx in (11, 12, 13):
            self.assertEqual(refcount[idx].item(), 0)
        self.assertEqual(sorted(to_free.tolist()), [11, 12, 13])

    def test_shared_old_kv_only_freed_when_refcount_hits_zero(self):
        """If a freed dst index still has another owner (refcount started at 2),
        it should NOT appear in to_free — only the genuinely-free slot does."""
        rows = [[11, 12, 0, 0], [21, 22, 23, 0]]
        req_to_token, refcount = self._build_pool(rows)
        # Pretend slot 12 is shared with another req (refcount already 2).
        refcount[12] = 2

        dst_pool = torch.tensor([0], dtype=torch.int32, device=self.DEVICE)
        src_pool = torch.tensor([1], dtype=torch.int32, device=self.DEVICE)
        dst_alloc = torch.tensor([2], dtype=torch.int32, device=self.DEVICE)
        src_seq = torch.tensor([3], dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token, refcount, dst_pool, src_pool, dst_alloc, src_seq,
        )

        # 12 still has the other owner → refcount = 1, not in to_free.
        self.assertEqual(refcount[11].item(), 0)
        self.assertEqual(refcount[12].item(), 1)
        self.assertEqual(to_free.tolist(), [11])

    def test_empty_input_returns_empty_to_free(self):
        """Zero jobs → kernel is skipped, returns an empty int64 tensor."""
        rows = [[11, 12, 0, 0]]
        req_to_token, refcount = self._build_pool(rows)
        empty_int32 = torch.empty(0, dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token,
            refcount,
            empty_int32,
            empty_int32,
            empty_int32,
            empty_int32,
        )

        self.assertEqual(to_free.numel(), 0)
        self.assertEqual(to_free.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()

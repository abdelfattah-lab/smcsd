"""Triton kernel correctness tests for SMC.

Covers the two fused kernels under ``smcsd/core/kernels/``:

* ``fused_collect.batched_collect_fused`` — normalise → ESS check →
  systematic resample → dead/excess compaction → flat emission.
* ``fused_resample_kv.batched_resample_kv`` — per-job
  ``dec_ref(old dst) → block-table copy(src → dst) → inc_ref(src)``,
  atomically.

Tests focus on observable invariants that hold regardless of the kernel's
internal Philox seed:

* resample_mask reflects ``ESS < threshold × N``
* dst_slots are unique and disjoint from src_slots
* ``|dst| == |src|`` (conservation: Σ target_counts = N)
* weights are zeroed only on rows that resampled
* KV refcounts move correctly and ``to_free`` reports slots that hit zero
"""

import unittest
from dataclasses import dataclass
from typing import List, Optional

import torch

from smcsd.core.kernels.fused_collect import batched_collect_fused
from smcsd.core.kernels.fused_resample_kv import batched_resample_kv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")


@dataclass
class CollectFixture:
    """Minimal standalone harness matching ``ScheduleBatchSMC``'s tensor
    layout for the fused_collect kernel — without instantiating the full
    scheduler state object.  Used only in these tests.
    """

    log_weights: torch.Tensor         # (max_slots,) float64
    interval_weights: torch.Tensor    # (max_slots,) float64
    group_to_slots: torch.Tensor      # (max_groups, N) int32
    row_in_use: torch.Tensor          # (max_groups,) bool
    slots_by_row: List[Optional[List[int]]]

    @classmethod
    def build(
        cls,
        *,
        max_groups: int,
        n_particles: int,
        max_slots: int,
        device: torch.device | str,
    ) -> "CollectFixture":
        dev = torch.device(device)
        return cls(
            log_weights=torch.zeros(max_slots, dtype=torch.float64, device=dev),
            interval_weights=torch.zeros(max_slots, dtype=torch.float64, device=dev),
            group_to_slots=torch.full(
                (max_groups, n_particles), -1, dtype=torch.int32, device=dev
            ),
            row_in_use=torch.zeros(max_groups, dtype=torch.bool, device=dev),
            slots_by_row=[None] * max_groups,
        )

    def register(self, row: int, slots: List[int]) -> None:
        """Claim ``row`` with the given slot ids.  Mirrors the production
        ``allocate_slots`` lookup update."""
        if self.slots_by_row[row] is not None:
            raise ValueError(f"row {row} already in use")
        self.slots_by_row[row] = list(slots)
        slots_t = torch.as_tensor(
            slots, dtype=torch.int32, device=self.group_to_slots.device,
        )
        self.group_to_slots[row, : slots_t.numel()] = slots_t
        self.row_in_use[row] = True
        idx = slots_t.to(torch.int64)
        self.log_weights[idx] = 0.0
        self.interval_weights[idx] = 0.0

    def set_iw(self, row: int, values: List[float]) -> None:
        """Override the interval-weight entries for ``row``'s slots."""
        slots = self.slots_by_row[row]
        assert slots is not None, f"row {row} is free"
        slot_t = torch.as_tensor(
            slots, dtype=torch.int64, device=self.interval_weights.device
        )
        self.interval_weights[slot_t] = torch.as_tensor(
            values, dtype=torch.float64, device=self.interval_weights.device,
        )

    def iw_of_row(self, row: int) -> torch.Tensor:
        slots = self.slots_by_row[row]
        assert slots is not None
        slot_t = torch.as_tensor(
            slots, dtype=torch.int64, device=self.interval_weights.device
        )
        return self.interval_weights[slot_t].clone()

    def lw_of_row(self, row: int) -> torch.Tensor:
        slots = self.slots_by_row[row]
        assert slots is not None
        slot_t = torch.as_tensor(
            slots, dtype=torch.int64, device=self.log_weights.device
        )
        return self.log_weights[slot_t].clone()

    def run(self, threshold: float, step_counter: int):
        return batched_collect_fused(
            self.log_weights,
            self.interval_weights,
            self.group_to_slots,
            self.row_in_use,
            threshold,
            step_counter=step_counter,
        )


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedCollectKernel(CustomTestCase):
    """``batched_collect_fused`` — fused per-row ESS + systematic resample."""

    DEVICE = "cuda"

    def _make(self, max_groups=4, n_particles=4, max_slots=512) -> CollectFixture:
        return CollectFixture.build(
            max_groups=max_groups,
            n_particles=n_particles,
            max_slots=max_slots,
            device=self.DEVICE,
        )

    def test_no_resample_when_weights_are_equal(self):
        """ESS == N → resample_mask False everywhere, no jobs emitted,
        weights untouched."""
        fx = self._make(max_groups=2, n_particles=4)
        fx.register(row=0, slots=[10, 11, 12, 13])

        before_iw = fx.iw_of_row(0)
        before_lw = fx.lw_of_row(0)

        result = fx.run(threshold=0.5, step_counter=1)

        self.assertEqual(result.n_jobs, 0)
        self.assertFalse(bool(result.resample_mask.any().item()))
        self.assertTrue(torch.equal(fx.iw_of_row(0), before_iw))
        self.assertTrue(torch.equal(fx.lw_of_row(0), before_lw))

    def test_dominant_weight_forces_full_resample(self):
        """When one particle has weight ≈ 1, dst ⊇ the other slots; src is
        the survivor's slot."""
        fx = self._make(max_groups=2, n_particles=4)
        fx.register(row=0, slots=[20, 21, 22, 23])
        fx.set_iw(row=0, values=[-1e10, 0.0, -1e10, -1e10])  # particle 1 dominates

        result = fx.run(threshold=0.5, step_counter=2)

        self.assertTrue(bool(result.resample_mask[0].item()))
        self.assertEqual(result.n_jobs, 3)

        dst = result.dst_slots.tolist()
        src = result.src_slots.tolist()
        rows = result.row_of_job.tolist()
        self.assertEqual(sorted(dst), sorted(set(dst)))  # unique
        self.assertTrue(set(dst).isdisjoint(set(src)))
        self.assertEqual(set(dst), {20, 22, 23})
        self.assertEqual(set(src), {21})
        self.assertEqual(set(rows), {0})

        # Weights zeroed on the resampled row.
        zero4 = torch.zeros(4, dtype=torch.float64, device=self.DEVICE)
        self.assertTrue(torch.equal(fx.iw_of_row(0), zero4))
        self.assertTrue(torch.equal(fx.lw_of_row(0), zero4))

    def test_multi_row_isolates_resample_to_skewed_row(self):
        """Two groups; only the skewed one resamples.  The uniform row's
        weights and status are untouched."""
        fx = self._make(max_groups=2, n_particles=4)
        fx.register(row=0, slots=[100, 101, 102, 103])  # uniform → no resample
        fx.register(row=1, slots=[200, 201, 202, 203])  # skewed  → resample
        fx.set_iw(row=1, values=[-1e10, -1e10, 0.0, -1e10])

        before_uniform_iw = fx.iw_of_row(0)
        before_uniform_lw = fx.lw_of_row(0)

        result = fx.run(threshold=0.5, step_counter=3)

        self.assertFalse(bool(result.resample_mask[0].item()))
        self.assertTrue(bool(result.resample_mask[1].item()))
        self.assertEqual(result.n_jobs, 3)
        self.assertEqual(set(result.row_of_job.tolist()), {1})
        self.assertEqual(set(result.dst_slots.tolist()), {200, 201, 203})
        self.assertEqual(set(result.src_slots.tolist()), {202})

        # Uniform row weights untouched; skewed row zeroed.
        self.assertTrue(torch.equal(fx.iw_of_row(0), before_uniform_iw))
        self.assertTrue(torch.equal(fx.lw_of_row(0), before_uniform_lw))
        zero4 = torch.zeros(4, dtype=torch.float64, device=self.DEVICE)
        self.assertTrue(torch.equal(fx.iw_of_row(1), zero4))
        self.assertTrue(torch.equal(fx.lw_of_row(1), zero4))

    def test_dead_and_src_partition_matches_target_counts(self):
        """Conservation: ``|dst| == |src|`` — Σ target_counts == N, so #dead
        matches sum of excess copies for an arbitrary skewed distribution."""
        fx = self._make(max_groups=1, n_particles=4)
        fx.register(row=0, slots=[300, 301, 302, 303])
        # Two survivors (particles 1 & 2), two dead (particles 0 & 3).
        # ess = 1 / (0.5² + 0.5²) = 2; threshold × N = 0.75 × 4 = 3 > 2 → fires.
        fx.set_iw(row=0, values=[-1e10, 0.0, 0.0, -1e10])

        result = fx.run(threshold=0.75, step_counter=4)

        self.assertTrue(bool(result.resample_mask.any().item()))
        self.assertEqual(set(result.dst_slots.tolist()), {300, 303})
        self.assertTrue(set(result.src_slots.tolist()).issubset({301, 302}))
        self.assertEqual(len(result.dst_slots), len(result.src_slots))

    def test_free_row_is_inert(self):
        """A row with ``row_in_use=False`` contributes no jobs and its
        resample_mask entry stays False, even when neighbouring rows fire.
        """
        fx = self._make(max_groups=3, n_particles=4)
        fx.register(row=1, slots=[400, 401, 402, 403])
        fx.set_iw(row=1, values=[0.0, -1e10, -1e10, -1e10])  # row 1 will resample

        result = fx.run(threshold=0.5, step_counter=5)

        # Only row 1 fires.
        self.assertFalse(bool(result.resample_mask[0].item()))
        self.assertTrue(bool(result.resample_mask[1].item()))
        self.assertFalse(bool(result.resample_mask[2].item()))
        self.assertEqual(set(result.row_of_job.tolist()), {1})

    def test_disjoint_slots_across_rows_emit_disjoint_jobs(self):
        """Slot ids are unique across groups by construction; dst / src
        tensors must therefore never cross row boundaries."""
        fx = self._make(max_groups=2, n_particles=4)
        fx.register(row=0, slots=[10, 11, 12, 13])
        fx.register(row=1, slots=[20, 21, 22, 23])
        fx.set_iw(row=0, values=[-1e10, -1e10, 0.0, -1e10])
        fx.set_iw(row=1, values=[0.0, -1e10, -1e10, -1e10])

        result = fx.run(threshold=0.5, step_counter=6)
        self.assertTrue(bool(result.resample_mask[0].item()))
        self.assertTrue(bool(result.resample_mask[1].item()))

        row0_slots = set(range(10, 14))
        row1_slots = set(range(20, 24))
        rows = result.row_of_job.tolist()
        for i, r in enumerate(rows):
            dst = int(result.dst_slots[i].item())
            src = int(result.src_slots[i].item())
            expected = row0_slots if r == 0 else row1_slots
            self.assertIn(dst, expected)
            self.assertIn(src, expected)

    def test_batch_many_groups_mixed_regimes(self):
        """8 groups × N=8: alternating uniform (no resample) and skewed
        (full resample) rows.  Exercises the kernel's batched launch grid
        and the atomic counter's partitioning across rows.

        Expected:
          * resample_mask = [F, T, F, T, F, T, F, T]
          * n_jobs       = 4 × (N-1)  (one survivor per skewed row)
          * unresampled rows' weights untouched; resampled rows' zeroed
        """
        G, N = 8, 8
        fx = self._make(max_groups=G, n_particles=N, max_slots=G * N + 16)

        for r in range(G):
            fx.register(row=r, slots=[r * N + c for c in range(N)])

        uniform = [0.0] * N
        skewed = [-1e10] * (N - 1) + [0.0]  # sole survivor is the last col
        for r in range(G):
            fx.set_iw(row=r, values=uniform if r % 2 == 0 else skewed)

        pre = [fx.iw_of_row(r) for r in range(G)]

        result = fx.run(threshold=0.5, step_counter=42)

        expected_mask = [bool(r % 2 == 1) for r in range(G)]
        self.assertEqual(result.resample_mask.tolist(), expected_mask)
        self.assertEqual(result.n_jobs, (G // 2) * (N - 1))

        # Per-row invariants across the whole batch.
        jobs_by_row: dict = {}
        for i, r in enumerate(result.row_of_job.tolist()):
            dst = int(result.dst_slots[i].item())
            src = int(result.src_slots[i].item())
            jobs_by_row.setdefault(r, []).append((dst, src))

        self.assertEqual(set(jobs_by_row.keys()), {1, 3, 5, 7})
        for r, pairs in jobs_by_row.items():
            dsts = [d for d, _ in pairs]
            srcs = [s for _, s in pairs]
            self.assertEqual(len(dsts), N - 1)
            self.assertEqual(len(set(dsts)), N - 1, f"row {r}: dst not unique")
            self.assertTrue(
                set(dsts).isdisjoint(set(srcs)),
                f"row {r}: dst/src overlap",
            )

        # Global: dst is unique across ALL rows (slots unique by construction).
        all_dst = result.dst_slots.tolist()
        self.assertEqual(len(all_dst), len(set(all_dst)),
                         "dst duplicated across rows")

        # Uniform rows untouched; skewed rows zeroed.
        zero_N = torch.zeros(N, dtype=torch.float64, device=self.DEVICE)
        for r in range(G):
            if r % 2 == 0:
                self.assertTrue(
                    torch.equal(fx.iw_of_row(r), pre[r]),
                    f"uniform row {r} weights were modified",
                )
            else:
                self.assertTrue(
                    torch.equal(fx.iw_of_row(r), zero_N),
                    f"skewed row {r} interval_weights not zeroed",
                )
                self.assertTrue(
                    torch.equal(fx.lw_of_row(r), zero_N),
                    f"skewed row {r} log_weights not zeroed",
                )

    def test_batch_large_N_all_resample(self):
        """Larger N=16 with every row resampling — stresses the compile-
        time unrolled per-draw loop and the `BLOCK = next_pow2(N)`
        padding behaviour.
        """
        G, N = 4, 16
        fx = self._make(max_groups=G, n_particles=N, max_slots=G * N + 8)

        for r in range(G):
            fx.register(row=r, slots=[r * N + c for c in range(N)])
            # One dominant particle per row, different column each time.
            vals = [-1e10] * N
            vals[r % N] = 0.0
            fx.set_iw(row=r, values=vals)

        result = fx.run(threshold=0.5, step_counter=77)

        self.assertTrue(result.resample_mask.all().item())
        self.assertEqual(result.n_jobs, G * (N - 1))

        # Per-row, src must all be the lone survivor's slot.
        for r in range(G):
            survivor_slot = r * N + (r % N)
            mask = result.row_of_job == r
            self.assertEqual(int(mask.sum().item()), N - 1)
            self.assertEqual(
                set(result.src_slots[mask].tolist()),
                {survivor_slot},
                f"row {r}: src should all be {survivor_slot}",
            )

    def test_batch_invariants_hold_across_seeds(self):
        """Vary ``step_counter``: Philox draws differ, so the specific
        ancestors can shift, but the kernel's structural invariants
        (|dst|==|src|, dst unique, dst∩src=∅, weights zeroed exactly on
        resampled rows) must hold for every seed.
        """
        G, N = 4, 8

        # Weights that force resample with two equal-weight survivors, so
        # the systematic offset genuinely affects the ancestor set.
        vals = [-1e10, 0.0, -1e10, 0.0, -1e10, 0.0, -1e10, 0.0]

        for seed in (1, 7, 99, 12345):
            fx = self._make(max_groups=G, n_particles=N, max_slots=G * N + 8)
            for r in range(G):
                fx.register(row=r, slots=[r * N + c for c in range(N)])
                fx.set_iw(row=r, values=vals)

            result = fx.run(threshold=0.75, step_counter=seed)

            self.assertTrue(
                result.resample_mask.all().item(),
                f"seed={seed}: every row should resample",
            )
            self.assertEqual(
                len(result.dst_slots), len(result.src_slots),
                f"seed={seed}: |dst| != |src|",
            )

            all_dst = result.dst_slots.tolist()
            all_src = result.src_slots.tolist()
            self.assertEqual(
                len(all_dst), len(set(all_dst)),
                f"seed={seed}: dst has duplicates",
            )
            self.assertTrue(
                set(all_dst).isdisjoint(set(all_src)),
                f"seed={seed}: dst/src overlap",
            )

            # Weights zeroed on every row.
            zero_N = torch.zeros(N, dtype=torch.float64, device=self.DEVICE)
            for r in range(G):
                self.assertTrue(
                    torch.equal(fx.iw_of_row(r), zero_N),
                    f"seed={seed} row={r} iw not zeroed",
                )
                self.assertTrue(
                    torch.equal(fx.lw_of_row(r), zero_N),
                    f"seed={seed} row={r} lw not zeroed",
                )


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedResampleKVKernel(CustomTestCase):
    """``batched_resample_kv`` — fused dec_ref + block-table copy + inc_ref."""

    DEVICE = "cuda"

    def _build_pool(self, rows, refcount_size=64):
        """Build ``(req_to_token, refcount)`` tensors and pre-arm refcounts.

        Every non-zero KV index appearing in ``rows`` starts at refcount 1.
        """
        req_to_token = torch.tensor(
            rows, dtype=torch.int32, device=self.DEVICE
        )
        refcount = torch.zeros(refcount_size, dtype=torch.int32, device=self.DEVICE)
        for row in rows:
            for idx in row:
                if idx > 0:
                    refcount[idx] = 1
        return req_to_token, refcount

    def test_single_pair_copies_block_table_and_updates_refcounts(self):
        """One eviction job: dst row [11, 12] ← src row [21, 22, 23].
        The dst row is overwritten, src refcounts bump, old dst refcounts
        drop — and since they had a single owner, they show up in to_free."""
        rows = [[11, 12, 0, 0], [21, 22, 23, 0]]
        req_to_token, refcount = self._build_pool(rows)

        dst_pool = torch.tensor([0], dtype=torch.int32, device=self.DEVICE)
        src_pool = torch.tensor([1], dtype=torch.int32, device=self.DEVICE)
        dst_alloc = torch.tensor([2], dtype=torch.int32, device=self.DEVICE)
        src_seq = torch.tensor([3], dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token, refcount, dst_pool, src_pool, dst_alloc, src_seq,
        )

        self.assertEqual(req_to_token[0, :3].tolist(), [21, 22, 23])
        self.assertEqual(refcount[21].item(), 2)
        self.assertEqual(refcount[22].item(), 2)
        self.assertEqual(refcount[23].item(), 2)
        self.assertEqual(refcount[11].item(), 0)
        self.assertEqual(refcount[12].item(), 0)
        self.assertEqual(sorted(to_free.tolist()), [11, 12])

    def test_multiple_pairs_run_in_parallel(self):
        """Two eviction jobs at once.  Each dst inherits its own src;
        refcounts and to_free reflect both jobs."""
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

        self.assertEqual(req_to_token[0, :3].tolist(), [21, 22, 23])
        self.assertEqual(req_to_token[1, :2].tolist(), [24, 25])
        for idx in (21, 22, 23, 24, 25):
            self.assertEqual(refcount[idx].item(), 2)
        for idx in (11, 12, 13):
            self.assertEqual(refcount[idx].item(), 0)
        self.assertEqual(sorted(to_free.tolist()), [11, 12, 13])

    def test_shared_old_kv_only_freed_when_refcount_hits_zero(self):
        """If a freed dst index still has another owner (refcount ≥ 2
        initially), it must NOT appear in to_free — only the genuinely-
        free slot does."""
        rows = [[11, 12, 0, 0], [21, 22, 23, 0]]
        req_to_token, refcount = self._build_pool(rows)
        refcount[12] = 2  # slot 12 is shared

        dst_pool = torch.tensor([0], dtype=torch.int32, device=self.DEVICE)
        src_pool = torch.tensor([1], dtype=torch.int32, device=self.DEVICE)
        dst_alloc = torch.tensor([2], dtype=torch.int32, device=self.DEVICE)
        src_seq = torch.tensor([3], dtype=torch.int32, device=self.DEVICE)

        to_free = batched_resample_kv(
            req_to_token, refcount, dst_pool, src_pool, dst_alloc, src_seq,
        )

        self.assertEqual(refcount[11].item(), 0)
        self.assertEqual(refcount[12].item(), 1)
        self.assertEqual(to_free.tolist(), [11])

    def test_empty_input_returns_empty_to_free(self):
        """Zero jobs → kernel skipped, empty int64 tensor returned."""
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

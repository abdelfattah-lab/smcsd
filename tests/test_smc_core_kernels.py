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
* KV refcounts move correctly and pages hitting zero land in the freed buffer
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

        self.assertEqual(result.n_jobs_sync(), 0)
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
        self.assertEqual(result.n_jobs_sync(), 3)

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
        self.assertEqual(result.n_jobs_sync(), 3)
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
        self.assertEqual(result.n_jobs_sync(), (G // 2) * (N - 1))

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
        self.assertEqual(result.n_jobs_sync(), G * (N - 1))

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


@dataclass
class ResampleFixture:
    """Minimal slot-state harness for the device-driven resample kernel.

    Mirrors the tensors ``dispatch_resample_batch`` hands to
    ``batched_resample_kv``: slot-indexed lineage state, the KV pool, and
    the freed-page capture buffer.  Slot ``i`` owns pool row ``i``.
    """

    req_to_token: torch.Tensor
    refcount: torch.Tensor
    req_pool_indices: torch.Tensor
    kv_allocated_lens: torch.Tensor
    seq_lens: torch.Tensor
    verified_ids: torch.Tensor
    prev_last_draft_ids: torch.Tensor
    finished_mask: torch.Tensor
    finished_len: torch.Tensor
    finish_reason_code: torch.Tensor
    matched_eos_token: torch.Tensor
    token_counts: torch.Tensor
    all_token_ids: torch.Tensor
    freed_buf: torch.Tensor
    freed_counter: torch.Tensor

    @classmethod
    def build(cls, rows, *, device, refcount_size=64) -> "ResampleFixture":
        """``rows[i]`` is slot/pool-row ``i``'s block table.  Every non-zero
        KV index starts at refcount 1.  Lineage tensors get distinct
        per-slot values so copies are observable."""
        dev = torch.device(device)
        n = len(rows)
        req_to_token = torch.tensor(rows, dtype=torch.int32, device=dev)
        refcount = torch.zeros(refcount_size, dtype=torch.int32, device=dev)
        for row in rows:
            for idx in row:
                if idx > 0:
                    refcount[idx] = 1
        used = [sum(1 for v in row if v > 0) for row in rows]
        ids = torch.arange(n, dtype=torch.int64, device=dev)
        return cls(
            req_to_token=req_to_token,
            refcount=refcount,
            req_pool_indices=ids.clone(),
            kv_allocated_lens=torch.tensor(used, dtype=torch.int64, device=dev),
            seq_lens=torch.tensor(used, dtype=torch.int64, device=dev),
            verified_ids=(100 + ids).to(torch.int32),
            prev_last_draft_ids=(200 + ids).to(torch.int32),
            finished_mask=(ids % 2 == 1),
            finished_len=(10 + ids).to(torch.int32),
            finish_reason_code=(ids % 3).to(torch.int8),
            matched_eos_token=(300 + ids).to(torch.int32),
            token_counts=torch.tensor(used, dtype=torch.int32, device=dev),
            all_token_ids=(
                1000 * (1 + ids.to(torch.int32)).unsqueeze(1)
                + torch.arange(8, dtype=torch.int32, device=dev)
            ),
            freed_buf=torch.zeros(refcount_size, dtype=torch.int32, device=dev),
            freed_counter=torch.zeros(1, dtype=torch.int32, device=dev),
        )

    def run(self, dst_slots, src_slots, *, max_jobs=None) -> None:
        dev = self.req_to_token.device
        n_jobs = len(dst_slots)
        if max_jobs is None:
            # Mimic dispatch: a worst-case grid larger than the true job
            # count, gated on-device by the counter.
            max_jobs = max(n_jobs + 2, 4)
        cap = max(max_jobs, 1)
        plan_dst = torch.zeros(cap, dtype=torch.int32, device=dev)
        plan_src = torch.zeros(cap, dtype=torch.int32, device=dev)
        if n_jobs:
            plan_dst[:n_jobs] = torch.tensor(
                dst_slots, dtype=torch.int32, device=dev
            )
            plan_src[:n_jobs] = torch.tensor(
                src_slots, dtype=torch.int32, device=dev
            )
        plan_counter = torch.tensor([n_jobs], dtype=torch.int32, device=dev)
        batched_resample_kv(
            self.req_to_token,
            self.refcount,
            plan_dst=plan_dst,
            plan_src=plan_src,
            plan_counter=plan_counter,
            max_jobs=max_jobs,
            req_pool_indices=self.req_pool_indices,
            kv_allocated_lens=self.kv_allocated_lens,
            seq_lens=self.seq_lens,
            verified_ids=self.verified_ids,
            prev_last_draft_ids=self.prev_last_draft_ids,
            finished_mask=self.finished_mask,
            finished_len=self.finished_len,
            finish_reason_code=self.finish_reason_code,
            matched_eos_token=self.matched_eos_token,
            token_counts=self.token_counts,
            all_token_ids=self.all_token_ids,
            freed_buf=self.freed_buf,
            freed_counter=self.freed_counter,
        )

    def freed(self) -> list:
        n = int(self.freed_counter.item())
        return sorted(self.freed_buf[:n].tolist())

    def assert_lineage_copied(self, test, dst: int, src: int) -> None:
        for name in (
            "seq_lens", "kv_allocated_lens", "verified_ids",
            "prev_last_draft_ids", "finished_mask", "finished_len",
            "finish_reason_code", "matched_eos_token", "token_counts",
        ):
            t = getattr(self, name)
            test.assertEqual(
                t[dst].item(), t[src].item(),
                f"{name}[{dst}] not copied from slot {src}",
            )
        count = int(self.token_counts[src].item())
        test.assertEqual(
            self.all_token_ids[dst, :count].tolist(),
            self.all_token_ids[src, :count].tolist(),
            f"all_token_ids[{dst}, :{count}] not copied from slot {src}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedResampleKernel(CustomTestCase):
    """``batched_resample_kv`` — device-plan-driven dec_ref + block-table
    copy + inc_ref + lineage-tensor copy + freed-page capture."""

    DEVICE = "cuda"

    def test_single_pair_copies_block_table_and_updates_refcounts(self):
        """One job: slot 0 (pages [11, 12]) ← slot 1 (pages [21, 22, 23]).
        The dst row is overwritten, src refcounts bump, old dst refcounts
        drop — single-owner pages land in the freed buffer — and every
        lineage tensor follows the copy."""
        fx = ResampleFixture.build(
            [[11, 12, 0, 0], [21, 22, 23, 0]], device=self.DEVICE
        )

        fx.run(dst_slots=[0], src_slots=[1])

        self.assertEqual(fx.req_to_token[0, :3].tolist(), [21, 22, 23])
        for idx in (21, 22, 23):
            self.assertEqual(fx.refcount[idx].item(), 2)
        for idx in (11, 12):
            self.assertEqual(fx.refcount[idx].item(), 0)
        self.assertEqual(fx.freed(), [11, 12])
        fx.assert_lineage_copied(self, dst=0, src=1)

    def test_multiple_pairs_run_in_parallel(self):
        """Two jobs at once, launched on an oversized (worst-case) grid.
        Each dst inherits its own src; refcounts and the freed buffer
        reflect both jobs."""
        fx = ResampleFixture.build(
            [
                [11, 12, 0, 0],
                [13, 0, 0, 0],
                [21, 22, 23, 0],
                [24, 25, 0, 0],
            ],
            device=self.DEVICE,
        )

        fx.run(dst_slots=[0, 1], src_slots=[2, 3], max_jobs=8)

        self.assertEqual(fx.req_to_token[0, :3].tolist(), [21, 22, 23])
        self.assertEqual(fx.req_to_token[1, :2].tolist(), [24, 25])
        for idx in (21, 22, 23, 24, 25):
            self.assertEqual(fx.refcount[idx].item(), 2)
        for idx in (11, 12, 13):
            self.assertEqual(fx.refcount[idx].item(), 0)
        self.assertEqual(fx.freed(), [11, 12, 13])
        fx.assert_lineage_copied(self, dst=0, src=2)
        fx.assert_lineage_copied(self, dst=1, src=3)

    def test_shared_old_kv_only_freed_when_refcount_hits_zero(self):
        """If a dec_ref'd dst page still has another owner (refcount ≥ 2
        initially), it must NOT be captured as freed — only the genuinely
        free page is."""
        fx = ResampleFixture.build(
            [[11, 12, 0, 0], [21, 22, 23, 0]], device=self.DEVICE
        )
        fx.refcount[12] = 2  # page 12 is shared with a live owner

        fx.run(dst_slots=[0], src_slots=[1])

        self.assertEqual(fx.refcount[11].item(), 0)
        self.assertEqual(fx.refcount[12].item(), 1)
        self.assertEqual(fx.freed(), [11])

    def test_empty_plan_is_noop(self):
        """counter == 0 with a non-zero grid: every program exits on the
        counter load — no tensor moves, no pages freed."""
        fx = ResampleFixture.build(
            [[11, 12, 0, 0], [21, 22, 23, 0]], device=self.DEVICE
        )
        before_r2t = fx.req_to_token.clone()
        before_rc = fx.refcount.clone()
        before_seq = fx.seq_lens.clone()

        fx.run(dst_slots=[], src_slots=[], max_jobs=4)

        self.assertTrue(torch.equal(fx.req_to_token, before_r2t))
        self.assertTrue(torch.equal(fx.refcount, before_rc))
        self.assertTrue(torch.equal(fx.seq_lens, before_seq))
        self.assertEqual(fx.freed(), [])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require CUDA")
class TestFusedWriteBack(CustomTestCase):
    """``fused_write_back`` matches the torch reference on randomized state
    (tokens, EOS placement, prior-finished rows, weight cutoffs)."""

    DEVICE = "cuda"

    def _make_state(self, max_slots, gamma_p1, max_eos=8, max_out=64, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        dev = self.DEVICE

        def randi(shape, hi, dtype=torch.int32):
            return torch.randint(0, hi, shape, generator=g).to(dtype).to(dev)

        state = dict(
            all_token_ids=torch.zeros(
                (max_slots, max_out), dtype=torch.int32, device=dev
            ),
            token_counts=randi((max_slots,), 8),
            verified_ids=randi((max_slots,), 100),
            prev_ids=randi((max_slots,), 100),
            finished_mask=(torch.rand(max_slots, generator=g) < 0.3).to(dev),
            finished_len=randi((max_slots,), 20),
            finish_reason_code=randi((max_slots,), 3, torch.int8),
            matched_eos_token=randi((max_slots,), 100),
            ignore_eos=(torch.rand(max_slots, generator=g) < 0.2).to(dev),
            max_new_tokens=randi((max_slots,), 6) + 6,
            eos_token_ids=torch.where(
                torch.rand(max_slots, max_eos, generator=g) < 0.3,
                torch.randint(
                    0, 50, (max_slots, max_eos), generator=g, dtype=torch.int64
                ),
                torch.full((max_slots, max_eos), -1, dtype=torch.int64),
            ).to(dev),
            log_weights=torch.randn(
                max_slots, generator=g, dtype=torch.float64
            ).to(dev),
            interval_weights=torch.randn(
                max_slots, generator=g, dtype=torch.float64
            ).to(dev),
        )
        return state

    @staticmethod
    def _torch_reference(state, active, next_token_ids, logprob_diff,
                         bonus_ids, prev, stride):
        dev = active.device
        bs = active.shape[0]
        accepted_2d = next_token_ids.reshape(bs, stride)
        offsets = state["token_counts"][active].to(torch.int64)
        row_idx = active.unsqueeze(1).expand(-1, stride)
        col_idx = offsets.unsqueeze(1) + torch.arange(
            stride, dtype=torch.int64, device=dev
        )
        state["all_token_ids"][row_idx, col_idx] = accepted_2d.to(torch.int32)
        state["token_counts"][active] += stride
        state["verified_ids"][active] = bonus_ids.to(torch.int32)
        state["prev_ids"][active] = prev.to(torch.int32)

        updated_counts = state["token_counts"][active]
        max_tokens = state["max_new_tokens"][active]
        length_hit = updated_counts >= max_tokens
        eos_ids = state["eos_token_ids"][active]
        eos_match = (
            accepted_2d.unsqueeze(2).to(torch.int64) == eos_ids.unsqueeze(1)
        ).any(dim=2)
        eos_hit = eos_match.any(dim=1) & ~state["ignore_eos"][active]
        prev_fin = state["finished_mask"][active]
        newly = (length_hit | eos_hit) & ~prev_fin
        state["finished_mask"][active] = prev_fin | newly

        positions = torch.arange(stride, dtype=torch.int64, device=dev)
        first_eos = torch.where(eos_match, positions, stride).min(dim=1).values
        matched_tok = accepted_2d.gather(
            1, first_eos.clamp(max=stride - 1).unsqueeze(1)
        ).squeeze(1)
        eos_branch = newly & ~length_hit
        fin_len = torch.where(
            length_hit,
            max_tokens,
            (updated_counts.to(torch.int64) - stride + first_eos + 1).to(
                max_tokens.dtype
            ),
        )
        fin_code = torch.where(length_hit, 1, 2).to(torch.int8)
        state["finished_len"][active] = torch.where(
            newly, fin_len, state["finished_len"][active]
        )
        state["finish_reason_code"][active] = torch.where(
            newly, fin_code, state["finish_reason_code"][active]
        )
        state["matched_eos_token"][active] = torch.where(
            eos_branch, matched_tok.to(torch.int32),
            state["matched_eos_token"][active],
        )

        gamma = logprob_diff.shape[1]
        cutoff = torch.full((bs,), gamma - 1, dtype=torch.int64, device=dev)
        eos_cut = newly & eos_hit
        cutoff = torch.where(
            eos_cut, first_eos.clamp(max=gamma - 1), cutoff
        )
        cutoff = torch.where(prev_fin, torch.full_like(cutoff, -1), cutoff)
        cols = torch.arange(gamma, device=dev).unsqueeze(0)
        keep = cols <= cutoff.unsqueeze(1)
        d = (logprob_diff.to(torch.float64) * keep).sum(dim=1)
        state["log_weights"][active] += d
        state["interval_weights"][active] += d

    def test_matches_torch_reference(self):
        from smcsd.core.kernels.fused_write_back import fused_write_back

        max_slots, stride = 24, 5
        gamma = stride - 1
        for seed in (0, 1, 7):
            torch.manual_seed(seed)
            bs = 16
            active = torch.randperm(max_slots, device=self.DEVICE)[:bs].to(
                torch.int64
            )
            # Token range overlapping the EOS-id range so EOS hits occur.
            next_tokens = torch.randint(
                0, 60, (bs * stride,), device=self.DEVICE, dtype=torch.int64
            )
            logprob_diff = torch.randn(
                bs, gamma, device=self.DEVICE, dtype=torch.float32
            )
            bonus = torch.randint(
                0, 60, (bs,), device=self.DEVICE, dtype=torch.int64
            )
            prev = torch.randint(
                0, 60, (bs,), device=self.DEVICE, dtype=torch.int64
            )

            ref = self._make_state(max_slots, stride, seed=seed)
            fused = {
                k: v.clone() for k, v in
                self._make_state(max_slots, stride, seed=seed).items()
            }

            self._torch_reference(
                ref, active, next_tokens, logprob_diff, bonus, prev, stride
            )
            fused_write_back(
                active, next_tokens, logprob_diff, bonus, prev,
                all_token_ids=fused["all_token_ids"],
                token_counts=fused["token_counts"],
                verified_ids=fused["verified_ids"],
                prev_ids=fused["prev_ids"],
                finished_mask=fused["finished_mask"],
                finished_len=fused["finished_len"],
                finish_reason_code=fused["finish_reason_code"],
                matched_eos_token=fused["matched_eos_token"],
                ignore_eos=fused["ignore_eos"],
                max_new_tokens=fused["max_new_tokens"],
                eos_token_ids=fused["eos_token_ids"],
                log_weights=fused["log_weights"],
                interval_weights=fused["interval_weights"],
                gamma_plus_1=stride,
            )

            for name in ref:
                if ref[name].dtype == torch.float64:
                    self.assertTrue(
                        torch.allclose(ref[name], fused[name], atol=1e-12),
                        f"seed={seed}: {name} mismatch",
                    )
                else:
                    self.assertTrue(
                        torch.equal(ref[name], fused[name]),
                        f"seed={seed}: {name} mismatch",
                    )

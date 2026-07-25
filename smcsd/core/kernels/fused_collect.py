"""Fused SMC collect kernel.

One Triton program per group row: normalise → ESS check → systematic
resample → dead/excess compaction → atomic flat emission.  The output
``(dst_slots, src_slots, row_of_job)`` tensors feed directly into
``batched_resample_kv`` without any ``.tolist()`` on the hot path.

The kernel also maintains ``group_shared_len`` — a per-row lower bound on
the block-table prefix every particle of the row holds in common, which
group-shared ("cascade") attention reads once per group instead of once per
particle.  A row whose ``counts`` has exactly one non-zero column collapses
to a single lineage, so the bound advances to that survivor's ``seq_len``;
otherwise it is left alone.  See ``ScheduleBatchSMC.group_shared_len`` for
why that is always a valid bound.

Layout (post-refactor)
----------------------

Particle state is slot-indexed.  ``log_weights`` and ``interval_weights``
are flat ``(max_slots,) float64`` tensors — one entry per particle slot.
Group membership is a compact lookup: ``group_to_slots[row, :N]`` holds
the slot ids of row ``row``'s N particles, and ``row_in_use[row]`` gates
whether the row is currently claimed by a group.  Under the global-N
invariant (every SMC group has exactly N particles for its lifetime),
in-use rows are always fully populated.

Per-row data flow (worked example, N=4)
---------------------------------------

    group_to_slots[row, :]   = [ 37   8   51   12 ]      (arbitrary slot ids)
    iw[group_to_slots[row]]  = [ 2.1  1.3  5.7  -0.2 ]

    (logsumexp normalise)
    weights                   = [ 0.15 0.09 0.68  0.08 ]
    ess                       = 1 / Σw²  ≈ 1.78
    threshold × N             = 0.5 × 4 = 2.0
    should_resample           = ess < thr·N → True

    CDF                       = cumsum(weights) = [ 0.15 0.24 0.92 1.00 ]
    u    = tl.rand(step_counter, row);    step  = 1/N = 0.25
    pos_k = u·step + step·k  for k ∈ [0, N)

    For each draw k: ancestor_k = |{ j : cdf[j] < pos_k }|  (scalar)
                     counts[ancestor_k] += 1
    counts                    = [ 1 0 2 1 ]    (col 1 dead, col 2 has surplus)
                              → 3 survivors, so group_shared_len[row] holds
                                (a collapse to [ 0 0 4 0 ] would advance it)

    dead_flag = (counts == 0)            → 1 dst
    excess    = max(counts - 1, 0)       → 1 src

    offset = atomic_add(global_counter, 1)   # reserves one flat slot
    dst[offset]        = group_to_slots[row, 1]   # slot 8
    src[offset]        = group_to_slots[row, 2]   # slot 51
    row_of_job[offset] = row

    iw[group_to_slots[row, :]]  ← zeroed in place
    lw[group_to_slots[row, :]]  ← zeroed in place

Contract for ``batched_resample_kv``
-------------------------------------

* ``len(dst_slots) == len(src_slots) == len(row_of_job) == n_jobs``
* ``set(dst_slots) ∩ set(src_slots) == Ø``  (global disjointness across rows;
  slots are unique within a group and rows' slot sets are disjoint)
* ``dst_slots`` unique (every dead slot is written once)
* ``row_of_job[i]`` always has ``resample_mask[row_of_job[i]] == True``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass
class BatchedResampleResult:
    """Output of one fused-collect launch — a fully device-resident plan.

    The job count lives in ``counter`` (a (1,) int32 GPU tensor); the flat
    buffers are full-capacity with the valid prefix ``[:counter]``.  Nothing
    syncs at construction, so the collect launch is enqueued behind the
    decode forward and consumed by the device-driven dispatch kernel
    without any host round trip.

    Host-side consumers — postprocessing (Mamba hybrid-state slicing) and
    tests — must call ``n_jobs_sync()``, the ONE place a host sync can
    happen.  It is an explicit method, not a property, so hot-path syncs
    stay greppable in review.  The ``dst_slots`` / ``src_slots`` /
    ``row_of_job`` views slice via ``n_jobs_sync()`` and therefore also
    sync; never touch them on the GPU-only path — use the ``*_flat``
    buffers plus ``counter`` there.

    * ``dst_flat``, ``src_flat`` are aligned 1:1 — job ``i`` copies
      ``src_flat[i] → dst_flat[i]``.
    * Intra-row order is deterministic (cumsum-based compaction); inter-row
      order is atomic-completion order.  Neither matters to the downstream
      KV-copy kernel.
    """

    dst_flat: torch.Tensor        # (flat_cap,) int32, valid prefix [:counter]
    src_flat: torch.Tensor        # (flat_cap,) int32, valid prefix [:counter]
    rows_flat: torch.Tensor       # (flat_cap,) int32, valid prefix [:counter]
    counter: torch.Tensor         # (1,) int32 — device-side job count
    resample_mask: torch.Tensor   # (max_groups,) bool
    _n_jobs: Optional[int] = None  # host count cache; None = never synced

    def n_jobs_sync(self) -> int:
        """Job count on the host — the one boundary sync, lazy and cached."""
        if self._n_jobs is None:
            self._n_jobs = int(self.counter.item())
        return self._n_jobs

    @property
    def dst_slots(self) -> torch.Tensor:
        """Host-sliced view of the valid jobs.  Forces ``n_jobs_sync()``."""
        return self.dst_flat[: self.n_jobs_sync()]

    @property
    def src_slots(self) -> torch.Tensor:
        """Host-sliced view of the valid jobs.  Forces ``n_jobs_sync()``."""
        return self.src_flat[: self.n_jobs_sync()]

    @property
    def row_of_job(self) -> torch.Tensor:
        """Host-sliced view of the valid jobs.  Forces ``n_jobs_sync()``."""
        return self.rows_flat[: self.n_jobs_sync()]


@triton.jit
def _fused_collect_kernel(
    # flat slot-major weights (MUTATED: zeroed at resampled rows' slots)
    iw_ptr,                   # (max_slots,) float64
    lw_ptr,                   # (max_slots,) float64
    # per-group lookup and gate
    group_to_slots_ptr,       # (max_groups, N) int32
    row_in_use_ptr,           # (max_groups,)   int8 (bool)
    # shared-prefix tracking (read-only seq_lens; group_shared_len MUTATED
    # at rows that collapse to a single surviving lineage this step)
    seq_lens_ptr,             # (max_slots,)    int64
    group_shared_len_ptr,     # (max_groups,)   int32
    # monotonic host counter: combined with row via tl.rand(step_counter, row)
    # to produce a per-row Philox uniform without any host-side allocation
    # or device sync.
    step_counter,             # int32 scalar
    # outputs
    dst_flat_ptr,             # (max_slots,) int32
    src_flat_ptr,             # (max_slots,) int32
    row_of_job_ptr,           # (max_slots,) int32
    global_counter_ptr,       # (1,)         int32   atomic
    resample_mask_ptr,        # (max_groups,) int32
    THRESHOLD,                # float64
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Skip free rows cheaply — no lookup / no math.
    in_use = tl.load(row_in_use_ptr + row)
    if in_use == 0:
        tl.store(resample_mask_ptr + row, 0)
        return

    n_f = tl.full([], N, dtype=tl.float64)

    # Gather this row's slot ids, then the slots' interval log-weights.
    # Padded cols (when BLOCK > N) load slot 0 and weight -inf; they drop
    # out of LSE and cumsum naturally, and every downstream store is
    # guarded by `mask`.
    slots = tl.load(group_to_slots_ptr + row * N + cols, mask=mask, other=0)
    lw_raw = tl.load(iw_ptr + slots, mask=mask, other=-float("inf"))

    # Normalise via logsumexp.
    max_lw = tl.max(lw_raw, axis=0)
    shifted = tl.exp(lw_raw - max_lw)
    sum_exp = tl.sum(shifted, axis=0)
    lse = max_lw + tl.log(sum_exp)
    weights = tl.exp(lw_raw - lse)  # padded cols contribute 0

    # ESS vs threshold × N.
    sum_w2 = tl.sum(weights * weights, axis=0)
    ess = 1.0 / sum_w2
    should_resample = (N >= 2) & (ess < THRESHOLD * n_f)
    tl.store(
        resample_mask_ptr + row,
        tl.where(should_resample, 1, 0).to(tl.int32),
    )

    if should_resample:
        cdf = tl.cumsum(weights, axis=0)
        step = 1.0 / n_f
        u = tl.rand(step_counter, row).to(tl.float64)
        start_u = u * step

        # counts[col] = number of draws whose ancestor lands in col.
        # Compile-time unrolled over N (the global particle count) — every
        # draw is valid under global-N so there's no k_valid mask.
        counts = tl.zeros([BLOCK], dtype=tl.int32)
        for k in range(N):
            pos_k = start_u + step * k.to(tl.float64)
            ancestor_k = tl.sum((cdf < pos_k).to(tl.int32), axis=0)
            counts = tl.where(cols == ancestor_k, counts + 1, counts)

        # Shared-prefix bound.  Exactly one column with counts > 0 means
        # every particle of this row becomes a copy of that one survivor,
        # so all N block tables are identical up to the survivor's seq_len
        # once `batched_resample_kv` runs.  Publish that as the row's new
        # shared-prefix length.  With >1 survivor the row keeps distinct
        # lineages and the previous bound still holds, so leave it alone —
        # never lower it, or a consumer could read KV that only some
        # particles own.  (Padded cols carry counts == 0 and seq 0, so they
        # affect neither the survivor count nor the sum.)
        n_survivors = tl.sum((counts > 0).to(tl.int32), axis=0)
        if n_survivors == 1:
            seq = tl.load(seq_lens_ptr + slots, mask=mask, other=0)
            surv_seq = tl.sum(tl.where(counts > 0, seq, 0), axis=0)
            tl.store(group_shared_len_ptr + row, surv_seq.to(tl.int32))

        # dead/excess compaction.  dst and src emission are both in
        # col-ascending order; `offset` reserves a contiguous slice of
        # the flat output buffers atomically.
        dead_flag = (counts == 0) & mask
        excess = counts - 1
        excess = tl.where((excess > 0) & mask, excess, 0)

        n_copies = tl.sum(dead_flag.to(tl.int32), axis=0)
        offset = tl.atomic_add(global_counter_ptr, n_copies)

        dead_prefix = tl.cumsum(dead_flag.to(tl.int32), axis=0)  # inclusive
        dst_pos = offset + dead_prefix - 1
        tl.store(dst_flat_ptr + dst_pos, slots, mask=dead_flag)
        tl.store(
            row_of_job_ptr + dst_pos,
            tl.full([BLOCK], row, dtype=tl.int32),
            mask=dead_flag,
        )

        excess_prefix = tl.cumsum(excess, axis=0)
        excess_start = excess_prefix - excess
        for k in range(N):
            write_mask = k < excess
            out_pos = offset + excess_start + k
            tl.store(src_flat_ptr + out_pos, slots, mask=write_mask)

        # Zero this row's slot weights in the flat tensors (next
        # accumulate starts fresh for resampled rows; untouched for
        # non-resampled rows).
        zero = tl.zeros([BLOCK], dtype=tl.float64)
        tl.store(iw_ptr + slots, zero, mask=mask)
        tl.store(lw_ptr + slots, zero, mask=mask)


def batched_collect_fused(
    log_weights: torch.Tensor,
    interval_weights: torch.Tensor,
    group_to_slots: torch.Tensor,
    row_in_use: torch.Tensor,
    threshold: float,
    *,
    step_counter: int,
    seq_lens: Optional[torch.Tensor] = None,
    group_shared_len: Optional[torch.Tensor] = None,
) -> BatchedResampleResult:
    """Launch the fused collect kernel against slot-major weights.

    Parameters
    ----------
    log_weights, interval_weights : (max_slots,) float64, MUTATED
        Flat per-slot cumulative log-weights.  ``interval_weights`` is the
        since-last-resample accumulator; both are zeroed at the resampled
        rows' slot positions on kernel exit.
    group_to_slots : (max_groups, N) int32
        Row → slot-id lookup.  Row ``r`` is in use iff ``row_in_use[r]``;
        in-use rows have all N cells populated.
    row_in_use : (max_groups,) bool
        Gates which rows the kernel processes.
    threshold : float
        ESS threshold.  A row resamples iff ``ess < threshold × N``.
    step_counter : int
        Monotonic host counter.  Must strictly increase across calls to
        avoid re-using the same Philox sequence.  Combined with the row
        id via ``tl.rand(step_counter, row)`` to seed each row.
    seq_lens : (max_slots,) int64, optional
        Per-slot sequence lengths, read only.  Required together with
        ``group_shared_len``.
    group_shared_len : (max_groups,) int32, optional, MUTATED
        Per-group shared-prefix lower bound.  Rows whose resample collapses
        to a single surviving lineage are advanced to that survivor's
        ``seq_lens``; all other rows are left untouched.  Omit both this and
        ``seq_lens`` to skip the tracking entirely (the kernel then writes
        to a scratch row and the bound is simply never advanced).

    Returns
    -------
    BatchedResampleResult
        Device-resident resample plan for ``dispatch_resample_batch``: for
        each of the ``counter`` jobs, copy ``src_flat[i] → dst_flat[i]``
        (tagged with its source row in ``rows_flat[i]``).
        ``resample_mask[r]`` flags rows that actually resampled this step.
        No host sync happens here; see ``BatchedResampleResult``.

    Notes
    -----
    The kernel's output buffers are allocated locally on each call.
    ``5 × torch.empty((max_slots,), int32)`` is microseconds against a
    ~10 ms decode step — not worth pre-allocating.  The buffers outlive
    the call only via the slice views carried by the returned
    ``BatchedResampleResult``.
    """
    device = log_weights.device
    max_groups, N = group_to_slots.shape
    flat_cap = max_groups * N

    plan_dst = torch.empty(flat_cap, dtype=torch.int32, device=device)
    plan_src = torch.empty(flat_cap, dtype=torch.int32, device=device)
    plan_rows = torch.empty(flat_cap, dtype=torch.int32, device=device)
    plan_counter = torch.zeros(1, dtype=torch.int32, device=device)
    plan_mask = torch.zeros(max_groups, dtype=torch.int32, device=device)

    if (seq_lens is None) != (group_shared_len is None):
        raise ValueError(
            "batched_collect_fused: pass seq_lens and group_shared_len "
            "together or not at all"
        )
    if seq_lens is None:
        # Tracking disabled: give the kernel valid-but-throwaway targets so
        # the store is harmless rather than branching inside the hot loop.
        seq_lens = torch.zeros(
            log_weights.numel(), dtype=torch.int64, device=device
        )
        group_shared_len = torch.zeros(
            max_groups, dtype=torch.int32, device=device
        )

    BLOCK = max(triton.next_power_of_2(N), 16)
    _fused_collect_kernel[(max_groups,)](
        interval_weights,
        log_weights,
        group_to_slots,
        row_in_use,
        seq_lens,
        group_shared_len,
        int(step_counter),
        plan_dst,
        plan_src,
        plan_rows,
        plan_counter,
        plan_mask,
        float(threshold),
        N=N,
        BLOCK=BLOCK,
    )

    return BatchedResampleResult(
        dst_flat=plan_dst,
        src_flat=plan_src,
        rows_flat=plan_rows,
        counter=plan_counter,
        resample_mask=plan_mask.to(torch.bool),
    )

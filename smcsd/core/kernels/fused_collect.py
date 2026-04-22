"""Fused SMC collect kernel.

One Triton program per group row: normalise → ESS check → systematic
resample → dead/excess compaction → atomic flat emission.  The output
``(dst_slots, src_slots, row_of_job)`` tensors feed directly into
``batched_resample_kv`` without any ``.tolist()`` on the hot path.

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
    """Output of one fused-collect launch.

    All tensors are GPU-resident except ``n_jobs``, which costs one
    ``.item()`` sync at the kernel boundary (needed so downstream callers
    can slice the flat output tensors).

    * ``dst_slots``, ``src_slots`` are aligned 1:1 — job ``i`` copies
      ``src_slots[i] → dst_slots[i]``.
    * Intra-row order is deterministic (cumsum-based compaction); inter-row
      order is atomic-completion order.  Neither matters to the downstream
      KV-copy kernel.
    """

    dst_slots: torch.Tensor       # (n_jobs,) int32
    src_slots: torch.Tensor       # (n_jobs,) int32
    row_of_job: torch.Tensor      # (n_jobs,) int32
    resample_mask: torch.Tensor   # (max_groups,) bool
    n_jobs: int


@triton.jit
def _fused_collect_kernel(
    # flat slot-major weights (MUTATED: zeroed at resampled rows' slots)
    iw_ptr,                   # (max_slots,) float64
    lw_ptr,                   # (max_slots,) float64
    # per-group lookup and gate
    group_to_slots_ptr,       # (max_groups, N) int32
    row_in_use_ptr,           # (max_groups,)   int8 (bool)
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

    Returns
    -------
    BatchedResampleResult
        Encodes the resample plan for ``dispatch_resample_batch``: for
        each of the ``n_jobs`` jobs, copy ``src_slots[i] → dst_slots[i]``
        (tagged with its source row in ``row_of_job[i]``).
        ``resample_mask[r]`` flags rows that actually resampled this step.

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

    BLOCK = max(triton.next_power_of_2(N), 16)
    _fused_collect_kernel[(max_groups,)](
        interval_weights,
        log_weights,
        group_to_slots,
        row_in_use,
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

    n_jobs = int(plan_counter.item())   # the one boundary sync
    return BatchedResampleResult(
        dst_slots=plan_dst[:n_jobs],
        src_slots=plan_src[:n_jobs],
        row_of_job=plan_rows[:n_jobs],
        resample_mask=plan_mask.to(torch.bool),
        n_jobs=n_jobs,
    )

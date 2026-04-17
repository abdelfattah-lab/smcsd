"""Fused SMC collect kernel.

One Triton program per row of `StackedGroupState` performs normalize → ESS
check → systematic resample → dead/excess counting → atomic-counter-based
compaction, emitting flat `(dst_slots, src_slots, row_of_job)` tensors.

Consumed directly by `batched_resample_kv` in `fused_resample_kv_kernel.py`
with no `.tolist()` on the critical path.

Contract for `batched_resample_kv` (enforced by construction):

  * len(dst_slots) == len(src_slots) == len(row_of_job) == n_jobs
  * set(dst_slots) ∩ set(src_slots) == Ø                (global disjointness)
  * dst_slots is unique (no duplicates)                 (each dead written once)
  * every row_of_job[i] has resample_mask[row_of_job[i]] = True

Per-row data flow (worked example, N=4, n_active=4)

    interval_weights[row, :]  = [  2.1   1.3   5.7  -0.2 ]
    active_cell_mask[row, :]  = [  T     T     T     T   ]

    masked_lw = where(mask, iw, -inf) = same (all active)
    weights   = exp(masked_lw - lse)  = [ 0.15  0.09  0.68  0.08 ]
    ess       = 1 / Σw²              ≈ 1.78
    threshold × n_active             = 0.5 × 4 = 2.0
    should_resample                   = ess < thr·n_active → True

    CDF      = cumsum(weights)        = [ 0.15  0.24  0.92  1.00 ]
    u = tl.rand(step_counter, row); step = 1/n_active = 0.25
    pos_k    = start + step · k  (for k ∈ [0, n_active))

    For each draw k:
        ancestor_k = |{ j : cdf[j] < pos_k }|   (scalar)
        counts[ancestor_k] += 1
    counts    = [ 1  0  2  1 ]   (col 1 dead, col 2 surplus)

    dead_flag = (counts==0) & active_cell_mask → 1 dst
    excess    = max(counts-1, 0)                → 1 src

    n_copies  = 1
    offset    = atomic_add(global_counter, 1)    ← reserves one flat slot
    dst_flat [offset] = particle_to_slot[row, 1]
    src_flat [offset] = particle_to_slot[row, 2]
    row_of_job[offset] = row

    interval_weights[row, :], log_weights[row, :] ← zeroed in place
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass
class BatchedResampleResult:
    """Output of one fused-collect kernel launch.  All tensors are GPU-resident
    except `n_jobs` (a single `.item()` sync at the kernel boundary).

    Intra-row order is deterministic (cumsum-based compaction); inter-row
    order is atomic-completion order.  `batched_resample_kv` does not care
    about order.
    """
    dst_slots:     torch.Tensor   # (n_jobs,) int32  flat
    src_slots:     torch.Tensor   # (n_jobs,) int32  aligned 1:1 with dst
    row_of_job:    torch.Tensor   # (n_jobs,) int32
    resample_mask: torch.Tensor   # (max_G,)  bool
    n_jobs:        int


@triton.jit
def _fused_collect_kernel(
    # stacked state (MUTATED in place: weights zeroed on resampled rows)
    iw_ptr,                       # (max_G, N) float64
    lw_ptr,                       # (max_G, N) float64
    active_cell_mask_ptr,         # (max_G, N) int8
    particle_to_slot_ptr,         # (max_G, N) int32
    n_active_ptr,                 # (max_G,)   int32
    row_in_use_ptr,               # (max_G,)   int8
    # per-call: monotonic step counter; combined with row_id via Philox
    # (`tl.rand`) to draw a fresh U[0,1) seed per row per step without any
    # host-side allocation or device sync.
    step_counter,                 # int32 scalar
    # outputs
    dst_flat_ptr,                 # (max_G * N,) int32
    src_flat_ptr,                 # (max_G * N,) int32
    row_of_job_ptr,               # (max_G * N,) int32
    global_counter_ptr,           # (1,)         int32   atomic
    resample_mask_ptr,            # (max_G,)     int32
    THRESHOLD,                    # float64
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Skip inactive rows immediately.
    in_use = tl.load(row_in_use_ptr + row)
    if in_use == 0:
        tl.store(resample_mask_ptr + row, 0)
        return

    n_active_i = tl.load(n_active_ptr + row)
    n_active_f = n_active_i.to(tl.float64)
    is_candidate = n_active_i >= 2

    # Load cell-level activity and log-weights.  Inactive cells are forced
    # to -inf so the logsumexp / cumsum naturally ignore them.
    cell_i8 = tl.load(active_cell_mask_ptr + row * N + cols, mask=mask, other=0)
    cell_active = cell_i8 != 0
    lw_raw = tl.load(iw_ptr + row * N + cols, mask=mask, other=0.0)
    lw = tl.where(cell_active, lw_raw, -float("inf"))

    # Normalize (safe even when !is_candidate — we just skip emitting below).
    max_lw = tl.max(lw, axis=0)
    shifted = tl.exp(lw - max_lw)
    sum_exp = tl.sum(shifted, axis=0)
    lse = max_lw + tl.log(sum_exp)
    weights = tl.exp(lw - lse)  # 0 at inactive cells

    # ESS vs threshold × n_active
    sum_w2 = tl.sum(weights * weights, axis=0)
    ess = 1.0 / sum_w2
    should_resample = is_candidate & (ess < THRESHOLD * n_active_f)
    tl.store(
        resample_mask_ptr + row,
        tl.where(should_resample, 1, 0).to(tl.int32),
    )

    if should_resample:
        cdf = tl.cumsum(weights, axis=0)

        # Per-draw searchsorted.  For each draw k ∈ [0, n_active), compute
        # ancestor_k = |{ j : cdf[j] < pos_k }| (scalar), then increment
        # counts[ancestor_k] by 1.  Loop is compile-time unrolled to N —
        # draws beyond n_active are masked out by k_valid.
        step = 1.0 / n_active_f
        # Philox-based per-row uniform draw: tl.rand takes (seed, offset) and
        # returns fp32 in [0,1).  We cast to fp64 for the stratified math.
        seed = tl.rand(step_counter, row).to(tl.float64)
        start = seed * step
        counts = tl.zeros([BLOCK], dtype=tl.int32)
        for k in range(N):
            k_valid = k < n_active_i
            pos_k = start + step * k.to(tl.float64)
            ancestor_k = tl.sum((cdf < pos_k).to(tl.int32), axis=0)
            counts = tl.where(
                (cols == ancestor_k) & k_valid,
                counts + 1,
                counts,
            )

        # Dead / excess compaction
        dead_flag = (counts == 0) & cell_active
        excess = counts - 1
        excess = tl.where((excess > 0) & cell_active, excess, 0)

        n_copies = tl.sum(dead_flag.to(tl.int32), axis=0)
        offset = tl.atomic_add(global_counter_ptr, n_copies)

        slots = tl.load(
            particle_to_slot_ptr + row * N + cols, mask=mask, other=-1,
        )

        # Scatter dst: dead cells compacted via dead_flag cumsum.
        dead_prefix = tl.cumsum(dead_flag.to(tl.int32), axis=0)   # inclusive
        dst_pos = offset + dead_prefix - 1
        tl.store(dst_flat_ptr + dst_pos, slots, mask=dead_flag)
        tl.store(
            row_of_job_ptr + dst_pos,
            tl.full([BLOCK], row, dtype=tl.int32),
            mask=dead_flag,
        )

        # Scatter src: each cell with excess>0 emits `excess[p]` copies
        # into [excess_start[p], excess_start[p]+excess[p]).
        excess_prefix = tl.cumsum(excess, axis=0)
        excess_start = excess_prefix - excess
        for k in range(N):
            write_mask = k < excess
            out_pos = offset + excess_start + k
            tl.store(src_flat_ptr + out_pos, slots, mask=write_mask)

        # Zero weights for this row (next accumulate starts fresh).
        zero = tl.zeros([BLOCK], dtype=tl.float64)
        tl.store(iw_ptr + row * N + cols, zero, mask=mask)
        tl.store(lw_ptr + row * N + cols, zero, mask=mask)


def batched_collect_fused(
    stacked,                        # StackedGroupState
    threshold: float,
    *,
    step_counter: int = 0,
    scratch_dst: Optional[torch.Tensor] = None,
    scratch_src: Optional[torch.Tensor] = None,
    scratch_row: Optional[torch.Tensor] = None,
    scratch_counter: Optional[torch.Tensor] = None,
    scratch_mask: Optional[torch.Tensor] = None,
) -> BatchedResampleResult:
    """Launch the fused collect kernel and return flat GPU tensors.

    `step_counter` must strictly increase across calls to avoid reusing
    the same stratified-resampling seed sequence (the kernel feeds it to
    Philox via `tl.rand(step_counter, row)`).  Scratch tensors may be
    supplied for in-place reuse; otherwise they are allocated per call.
    """
    device = stacked.device
    max_G = stacked.max_groups
    N = stacked.N
    flat_cap = max_G * N

    if scratch_dst is None:
        scratch_dst = torch.empty(flat_cap, dtype=torch.int32, device=device)
    if scratch_src is None:
        scratch_src = torch.empty(flat_cap, dtype=torch.int32, device=device)
    if scratch_row is None:
        scratch_row = torch.empty(flat_cap, dtype=torch.int32, device=device)
    if scratch_counter is None:
        scratch_counter = torch.zeros(1, dtype=torch.int32, device=device)
    else:
        scratch_counter.zero_()
    if scratch_mask is None:
        scratch_mask = torch.zeros(max_G, dtype=torch.int32, device=device)
    else:
        scratch_mask.zero_()

    BLOCK = max(triton.next_power_of_2(N), 16)
    _fused_collect_kernel[(max_G,)](
        stacked.interval_weights,
        stacked.log_weights,
        stacked.active_cell_mask,
        stacked.particle_to_slot,
        stacked.n_active,
        stacked.row_in_use,
        int(step_counter),
        scratch_dst,
        scratch_src,
        scratch_row,
        scratch_counter,
        scratch_mask,
        float(threshold),
        N=N,
        BLOCK=BLOCK,
    )

    n_jobs = int(scratch_counter.item())   # the one boundary sync
    return BatchedResampleResult(
        dst_slots=scratch_dst[:n_jobs],
        src_slots=scratch_src[:n_jobs],
        row_of_job=scratch_row[:n_jobs],
        resample_mask=scratch_mask.to(torch.bool),
        n_jobs=n_jobs,
    )

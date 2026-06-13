"""Triton kernel for Karhunen-Loève Expansion (KLE) variance estimation.

Computes the variance of log-weights across particles in each group to
drive the adaptive resample threshold. High variance indicates a complex
semantic junction requiring more aggressive resampling.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _kle_variance_kernel(
    iw_ptr,                   # (max_slots,) float64
    group_to_slots_ptr,       # (max_groups, N) int32
    row_in_use_ptr,           # (max_groups,)   int8 (bool)
    # outputs
    variance_ptr,             # (max_groups,) float64
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    in_use = tl.load(row_in_use_ptr + row)
    if in_use == 0:
        tl.store(variance_ptr + row, 0.0)
        return

    # Gather interval log-weights for this group
    slots = tl.load(group_to_slots_ptr + row * N + cols, mask=mask, other=0)
    # Use -1e15 for stability in mean calculation (approx -inf but finite)
    lw_raw = tl.load(iw_ptr + slots, mask=mask, other=-1e15)

    # Online variance calculation (Welford's algorithm simplified for block)
    # 1. Compute Mean
    sum_lw = tl.sum(tl.where(mask, lw_raw, 0.0), axis=0)
    mean_lw = sum_lw / N

    # 2. Compute Variance: 1/N * sum((x - mean)^2)
    diff = tl.where(mask, lw_raw - mean_lw, 0.0)
    sq_diff = diff * diff
    sum_sq_diff = tl.sum(sq_diff, axis=0)
    variance = sum_sq_diff / N

    tl.store(variance_ptr + row, variance)

def compute_kle_variance(
    interval_weights: torch.Tensor,
    group_to_slots: torch.Tensor,
    row_in_use: torch.Tensor,
) -> torch.Tensor:
    """
    Launches the KLE variance kernel.
    Returns a (max_groups,) float64 tensor containing variance per group.
    """
    device = interval_weights.device
    max_groups, N = group_to_slots.shape
    
    variance = torch.empty(max_groups, dtype=torch.float64, device=device)
    
    BLOCK = max(triton.next_power_of_2(N), 16)
    
    # Grid is (max_groups,)
    _kle_variance_kernel[(max_groups,)](
        interval_weights,
        group_to_slots,
        row_in_use,
        variance,
        N=N,
        BLOCK=BLOCK,
    )
    
    return variance

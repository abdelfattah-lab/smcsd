"""Fused decode write-back kernel (issue #14, scheduler host-op slimming).

One program per active row replaces the ~15 separate indexed torch ops of
``ScheduleBatchSMC.write_back_gpu``: the token scatter into the history
buffer, the verified/prev-draft seed updates, the length/EOS finish checks
with tensor-resident finish state, the EOS-precedence weight cutoff, and the
cutoff-masked float64 weight accumulation.

Semantics are byte-identical to the torch implementation (which remains as
the reference / CPU fallback):

* token_counts advances by STRIDE; tokens land at all_token_ids[s, off:off+S]
* finish: length takes precedence for the *reason* (code 1) but EOS always
  wins for the *weight cutoff*; already-finished rows contribute weight 0
* matched_eos_token written only on the EOS branch (newly & ~length_hit)
* log/interval weights accumulate sum(logprob_diff[i, 0..cutoff]) in float64
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_write_back_kernel(
    active_ptr,            # (bs,) int64
    next_tokens_ptr,       # (bs*STRIDE,) int — this cycle's accepted tokens
    logprob_diff_ptr,      # (bs, GAMMA) float32, contiguous
    bonus_ptr,             # (bs,) int
    prev_ptr,              # (bs,) int
    # slot-indexed state (MUTATED)
    all_token_ids_ptr,     # (max_slots, max_out) int32
    all_token_ids_stride,
    token_counts_ptr,      # (max_slots,) int32
    verified_ptr,          # (max_slots,) int32
    prev_ids_ptr,          # (max_slots,) int32
    finished_mask_ptr,     # (max_slots,) int8 (bool view)
    finished_len_ptr,      # (max_slots,) int32
    finish_reason_ptr,     # (max_slots,) int8
    matched_eos_ptr,       # (max_slots,) int32
    ignore_eos_ptr,        # (max_slots,) int8 (bool view)
    max_new_tokens_ptr,    # (max_slots,) int32
    eos_token_ids_ptr,     # (max_slots, MAX_EOS) int64
    log_weights_ptr,       # (max_slots,) float64
    interval_weights_ptr,  # (max_slots,) float64
    HAS_PREV: tl.constexpr,
    STRIDE: tl.constexpr,   # gamma + 1
    GAMMA: tl.constexpr,    # weight columns
    MAX_EOS: tl.constexpr,
    BLOCK: tl.constexpr,    # >= STRIDE, pow2
):
    i = tl.program_id(0)
    s = tl.load(active_ptr + i)

    offs = tl.arange(0, BLOCK)
    m = offs < STRIDE

    # a. Token scatter + count advance.
    toks = tl.load(next_tokens_ptr + i * STRIDE + offs, mask=m, other=0)
    off = tl.load(token_counts_ptr + s)
    tl.store(
        all_token_ids_ptr + s.to(tl.int64) * all_token_ids_stride + off + offs,
        toks.to(tl.int32),
        mask=m,
    )
    new_count = off + STRIDE
    tl.store(token_counts_ptr + s, new_count)

    # b. Next-step seeds.
    tl.store(verified_ptr + s, tl.load(bonus_ptr + i).to(tl.int32))
    if HAS_PREV:
        tl.store(prev_ids_ptr + s, tl.load(prev_ptr + i).to(tl.int32))

    # c. Finish checks.
    max_new = tl.load(max_new_tokens_ptr + s)
    length_hit = new_count >= max_new

    # First EOS column in this block (== STRIDE when none).  EOS padding is
    # -1, which no token id can match.
    first_eos = tl.full([], STRIDE, dtype=tl.int32)
    for j in range(MAX_EOS):
        e = tl.load(eos_token_ids_ptr + s * MAX_EOS + j)
        match = m & (toks.to(tl.int64) == e)
        pos_j = tl.min(tl.where(match, offs, STRIDE), axis=0)
        first_eos = tl.minimum(first_eos, pos_j.to(tl.int32))
    ignore = tl.load(ignore_eos_ptr + s)
    eos_hit = (first_eos < STRIDE) & (ignore == 0)

    prev_fin = tl.load(finished_mask_ptr + s)
    newly = (length_hit | eos_hit) & (prev_fin == 0)
    tl.store(finished_mask_ptr + s, prev_fin | newly.to(tl.int8))

    matched_tok = tl.load(
        next_tokens_ptr + i * STRIDE + tl.minimum(first_eos, STRIDE - 1)
    )
    # Length takes precedence for the reason (historical behaviour).
    fin_len = tl.where(
        length_hit, max_new, new_count - STRIDE + first_eos + 1
    )
    fin_code = tl.where(length_hit, 1, 2).to(tl.int8)
    eos_branch = newly & (length_hit == 0)

    old_len = tl.load(finished_len_ptr + s)
    tl.store(finished_len_ptr + s, tl.where(newly, fin_len, old_len))
    old_code = tl.load(finish_reason_ptr + s)
    tl.store(finish_reason_ptr + s, tl.where(newly, fin_code, old_code))
    old_match = tl.load(matched_eos_ptr + s)
    tl.store(
        matched_eos_ptr + s,
        tl.where(eos_branch, matched_tok.to(tl.int32), old_match),
    )

    # d. Weight cutoff + accumulation.  EOS wins the cutoff even when the
    # length cap hits in the same block; already-finished rows keep nothing.
    cutoff = tl.full([], GAMMA - 1, dtype=tl.int32)
    eos_cut = newly & eos_hit
    cutoff = tl.where(
        eos_cut, tl.minimum(first_eos, GAMMA - 1), cutoff
    )
    cutoff = tl.where(prev_fin != 0, -1, cutoff)

    offs_g = tl.arange(0, BLOCK)
    mg = offs_g < GAMMA
    lpd = tl.load(
        logprob_diff_ptr + i * GAMMA + offs_g, mask=mg, other=0.0
    ).to(tl.float64)
    keep = mg & (offs_g <= cutoff)
    d = tl.sum(tl.where(keep, lpd, 0.0), axis=0)

    tl.store(log_weights_ptr + s, tl.load(log_weights_ptr + s) + d)
    tl.store(
        interval_weights_ptr + s, tl.load(interval_weights_ptr + s) + d
    )


def fused_write_back(
    active_slots: torch.Tensor,
    next_token_ids: torch.Tensor,
    logprob_diff: torch.Tensor,
    bonus_ids: torch.Tensor,
    prev_last_draft_ids,
    *,
    all_token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    verified_ids: torch.Tensor,
    prev_ids: torch.Tensor,
    finished_mask: torch.Tensor,
    finished_len: torch.Tensor,
    finish_reason_code: torch.Tensor,
    matched_eos_token: torch.Tensor,
    ignore_eos: torch.Tensor,
    max_new_tokens: torch.Tensor,
    eos_token_ids: torch.Tensor,
    log_weights: torch.Tensor,
    interval_weights: torch.Tensor,
    gamma_plus_1: int,
) -> None:
    """Launch the fused write-back kernel (one program per active row)."""
    bs = active_slots.shape[0]
    gamma = logprob_diff.shape[1]
    has_prev = prev_last_draft_ids is not None
    _fused_write_back_kernel[(bs,)](
        active_slots,
        next_token_ids.contiguous(),
        logprob_diff.contiguous(),
        bonus_ids.contiguous(),
        prev_last_draft_ids.contiguous() if has_prev else bonus_ids,
        all_token_ids,
        all_token_ids.stride(0),
        token_counts,
        verified_ids,
        prev_ids,
        finished_mask.view(torch.int8),
        finished_len,
        finish_reason_code,
        matched_eos_token,
        ignore_eos.view(torch.int8),
        max_new_tokens,
        eos_token_ids,
        log_weights,
        interval_weights,
        HAS_PREV=has_prev,
        STRIDE=gamma_plus_1,
        GAMMA=gamma,
        MAX_EOS=eos_token_ids.shape[1],
        BLOCK=max(triton.next_power_of_2(gamma_plus_1), 16),
    )

"""Fused decode-prepare kernel (issue #14, scheduler host-op slimming).

One launch replaces the per-cycle prepare sequence that used to be ~12
separate ops (5 slot gathers, max/clamp arithmetic, the block-table assign
kernel, and 2 scatters):

    per active row i, slot s = active_slots[i]:
        seq            = seq_lens[s]                       (gather)
        req_to_token[pool[s], seq : seq+G] = pages[i*G:]   (block-table write)
        seq_lens[s] = kv_allocated_lens[s] = seq + G       (state advance)
        orig_seq_out[i]  = seq                             (contiguous outputs
        verified_out[i]  = verified_ids[s]                  for the worker)
        prev_out[i]      = prev_last_draft_ids[s]

Relies on the maintained invariant ``kv_allocated_lens == seq_lens`` at
prepare time (set at allocate, advanced together here, copied together on
resample), so every row needs exactly ``G = gamma+1`` fresh pages and the
freshly-allocated page tensor IS the per-row cache-locs table — callers use
``pages.view(bs, G)`` directly instead of re-reading the block table.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_prepare_kernel(
    active_ptr,             # (bs,) int64 — active slot ids
    seq_lens_ptr,           # (max_slots,) int64   MUTATED: += G
    kv_alloc_ptr,           # (max_slots,) int64   MUTATED: += G
    pool_idx_ptr,           # (max_slots,) int64
    verified_ptr,           # (max_slots,) int32
    prev_ptr,               # (max_slots,) int32
    pages_ptr,              # (bs*G,) int64 — freshly allocated KV pages
    req_to_token_ptr,       # (pool_size, max_ctx) int32  MUTATED
    req_to_token_stride,
    orig_seq_out_ptr,       # (bs,) int64
    verified_out_ptr,       # (bs,) int32
    prev_out_ptr,           # (bs,) int32
    G: tl.constexpr,        # gamma + 1
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    s = tl.load(active_ptr + i)
    seq = tl.load(seq_lens_ptr + s)
    pool = tl.load(pool_idx_ptr + s)

    offs = tl.arange(0, BLOCK)
    mask = offs < G
    pages = tl.load(pages_ptr + i * G + offs, mask=mask)
    tl.store(
        req_to_token_ptr + pool * req_to_token_stride + seq + offs,
        pages.to(tl.int32),
        mask=mask,
    )

    tl.store(seq_lens_ptr + s, seq + G)
    tl.store(kv_alloc_ptr + s, seq + G)
    tl.store(orig_seq_out_ptr + i, seq)
    tl.store(verified_out_ptr + i, tl.load(verified_ptr + s))
    tl.store(prev_out_ptr + i, tl.load(prev_ptr + s))


def fused_prepare_decode(
    active_slots: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_allocated_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    verified_ids: torch.Tensor,
    prev_last_draft_ids: torch.Tensor,
    pages: torch.Tensor,
    req_to_token: torch.Tensor,
    gamma_plus_1: int,
):
    """Launch the fused prepare kernel.

    Returns (orig_seq_lens, verified, prev) contiguous (bs,) tensors.
    ``seq_lens`` / ``kv_allocated_lens`` / ``req_to_token`` are mutated
    in place.
    """
    bs = active_slots.shape[0]
    device = active_slots.device
    orig_seq_out = torch.empty(bs, dtype=torch.int64, device=device)
    verified_out = torch.empty(bs, dtype=torch.int32, device=device)
    prev_out = torch.empty(bs, dtype=torch.int32, device=device)

    _fused_prepare_kernel[(bs,)](
        active_slots,
        seq_lens,
        kv_allocated_lens,
        req_pool_indices,
        verified_ids,
        prev_last_draft_ids,
        pages,
        req_to_token,
        req_to_token.stride(0),
        orig_seq_out,
        verified_out,
        prev_out,
        G=gamma_plus_1,
        BLOCK=max(triton.next_power_of_2(gamma_plus_1), 16),
    )
    return orig_seq_out, verified_out, prev_out

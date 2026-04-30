"""Fused Triton kernel for SMC resampling: KV block table copy + refcount update.

Single kernel launch replaces per-eviction Python loop in resample_copy_slot
for the KV block table portion. Fuses dec_ref, block table copy, and inc_ref
across all evictions using CSR offsets and atomic refcount updates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_resample_kv_kernel(
    req_to_token_ptr,  # [pool_size, max_ctx_len] int32
    req_to_token_stride,  # row stride
    refcount_ptr,  # [kv_pool_size] int32
    dst_pool_indices_ptr,  # [n_jobs] int32
    src_pool_indices_ptr,  # [n_jobs] int32
    dst_alloc_lens_ptr,  # [n_jobs] int32
    src_seq_lens_ptr,  # [n_jobs] int32
    dec_out_ptr,  # [total_dec] int32 — captured old dst KV indices
    dec_offsets_ptr,  # [n_jobs + 1] int32 — CSR offsets (starts with 0)
    BLOCK_SIZE: tl.constexpr,
):
    """One thread block per eviction job.

    Phase 1: read req_to_token[dst, :dst_alloc]
             -> capture to dec_out[offset:offset+dst_alloc]
             -> atomic_add(refcount, -1)
    Phase 2: read req_to_token[src, :src_len]
             -> write to req_to_token[dst, :src_len]
             -> atomic_add(refcount, +1)
    """
    job = tl.program_id(0)

    dst_pool = tl.load(dst_pool_indices_ptr + job)
    src_pool = tl.load(src_pool_indices_ptr + job)
    dst_alloc = tl.load(dst_alloc_lens_ptr + job)
    src_len = tl.load(src_seq_lens_ptr + job)

    dst_row = req_to_token_ptr + dst_pool * req_to_token_stride
    src_row = req_to_token_ptr + src_pool * req_to_token_stride

    dec_start = tl.load(dec_offsets_ptr + job)

    # Phase 1: read old dst -> capture + dec_ref
    num_dec_iters = tl.cdiv(dst_alloc, BLOCK_SIZE)
    for i in range(num_dec_iters):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < dst_alloc
        old_kv_idx = tl.load(dst_row + offset, mask=mask)
        tl.store(dec_out_ptr + dec_start + offset, old_kv_idx, mask=mask)
        tl.atomic_add(refcount_ptr + old_kv_idx.to(tl.int64), -1, mask=mask)

    # Phase 2: read src -> copy to dst + inc_ref
    num_src_iters = tl.cdiv(src_len, BLOCK_SIZE)
    for i in range(num_src_iters):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < src_len
        src_kv_idx = tl.load(src_row + offset, mask=mask)
        tl.store(dst_row + offset, src_kv_idx, mask=mask)
        tl.atomic_add(refcount_ptr + src_kv_idx.to(tl.int64), 1, mask=mask)


def batched_resample_kv(
    req_to_token: torch.Tensor,
    refcount: torch.Tensor,
    dst_pool_indices: torch.Tensor,
    src_pool_indices: torch.Tensor,
    dst_alloc_lens: torch.Tensor,
    src_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Fused KV block table copy + refcount update for SMC resampling.

    Returns KV slot indices whose refcount hit 0 (to pass to allocator.free()).
    """
    n_jobs = dst_pool_indices.numel()
    device = req_to_token.device
    if n_jobs == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    dec_cu = torch.cumsum(dst_alloc_lens, dim=0)
    total_dec = int(dec_cu[-1].item())
    dec_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            dec_cu,
        ]
    )
    dec_out = torch.empty(total_dec, dtype=torch.int32, device=device)

    _fused_resample_kv_kernel[(n_jobs,)](
        req_to_token,
        req_to_token.stride(0),
        refcount,
        dst_pool_indices,
        src_pool_indices,
        dst_alloc_lens,
        src_seq_lens,
        dec_out,
        dec_offsets,
        BLOCK_SIZE=128,
    )

    if total_dec == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    dec_i64 = dec_out.to(torch.int64)
    freed_mask = refcount[dec_i64] == 0
    to_free = torch.unique(dec_i64[freed_mask])
    return to_free

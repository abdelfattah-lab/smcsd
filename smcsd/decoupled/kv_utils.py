"""Decoupled SMC KV helpers."""

from __future__ import annotations

import torch

from smcsd.mem_cache.allocator import SMCRefCountedTokenAllocator


def truncate_block_table_allocations(
    req_to_token_pool,
    token_to_kv_pool_allocator: SMCRefCountedTokenAllocator,
    req_pool_indices: torch.Tensor,
    old_alloc_lens: torch.Tensor,
    new_alloc_lens: torch.Tensor,
) -> None:
    """Drop this req's ownership of block-table entries after ``new_alloc_len``.

    The helper intentionally walks one row at a time. Resample clones can share
    the same physical suffix cells; separate ``dec_ref_and_free`` calls preserve
    one decrement per owning row even when rows contain duplicate KV ids.
    """
    if req_pool_indices.numel() == 0:
        return
    if not (
        req_pool_indices.numel()
        == old_alloc_lens.numel()
        == new_alloc_lens.numel()
    ):
        raise ValueError("truncate_block_table_allocations: mismatched tensor sizes")

    pools = req_pool_indices.detach().cpu().tolist()
    old_lens = old_alloc_lens.detach().cpu().tolist()
    new_lens = new_alloc_lens.detach().cpu().tolist()
    for pool_idx, old_len, new_len in zip(pools, old_lens, new_lens, strict=True):
        old_len = int(old_len)
        new_len = int(new_len)
        if new_len < 0 or new_len > old_len:
            raise ValueError(
                "truncate_block_table_allocations requires 0 <= new <= old, got "
                f"new={new_len}, old={old_len}"
            )
        if new_len == old_len:
            continue
        if int(pool_idx) < 0:
            raise RuntimeError(
                "truncate_block_table_allocations cannot truncate an empty req row"
            )
        indices = req_to_token_pool.req_to_token[
            int(pool_idx), new_len:old_len
        ].to(dtype=torch.int64, copy=True)
        token_to_kv_pool_allocator.dec_ref_and_free(indices)

"""SMC-specific KV pool allocator.

Adds per-slot refcount tracking on top of the base ``TokenToKVPoolAllocator``
so that multiple SMC particles can share KV slots safely (parent-prefix
fan-out and resample copy both create multi-owner slots).

Refcount endpoints (set to 1 on allocation, 0 on free) are maintained by
overriding ``alloc`` / ``free``; ``inc_ref`` / ``dec_ref_and_free`` and
``copy_block_table`` are the only public refcount API. The base allocator
is left untouched.
"""

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


class SMCRefCountedTokenAllocator(TokenToKVPoolAllocator):
    """Token-level KV allocator with per-slot refcounts for SMC."""

    def clear(self):
        super().clear()
        self.slot_ref_count = torch.zeros(
            self.size + 1, dtype=torch.int32, device=self.device
        )

    def alloc(self, need_size: int):
        select_index = super().alloc(need_size)
        if select_index is not None and select_index.numel() > 0:
            self.slot_ref_count[select_index] = 1
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() > 0:
            self.slot_ref_count[free_index] = 0
        super().free(free_index)

    def inc_ref(self, indices: torch.Tensor):
        if indices.numel() == 0:
            return
        self.slot_ref_count[indices] += 1

    def dec_ref_and_free(self, indices: torch.Tensor):
        if indices.numel() == 0:
            return
        self.slot_ref_count[indices] -= 1
        to_free = indices[self.slot_ref_count[indices] == 0]
        if to_free.numel() > 0:
            self.free(to_free)


def copy_block_table(
    req_to_token_pool,
    src_req_pool_idx: int,
    dst_req_pool_idx: int,
    seq_len: int,
    token_to_kv_pool_allocator: SMCRefCountedTokenAllocator,
):
    """Copy ``seq_len`` block-table entries from ``src`` to ``dst`` and bump
    the refcount on each copied slot so the donor and recipient both retain
    ownership.

    Used by SMC parent->particle fan-out and any code path that hands an
    existing prefix to a fresh request.
    """
    if seq_len <= 0:
        return
    copied = req_to_token_pool.req_to_token[src_req_pool_idx, :seq_len].clone()
    token_to_kv_pool_allocator.inc_ref(copied.to(torch.int64))
    req_to_token_pool.write((dst_req_pool_idx, slice(0, seq_len)), copied)

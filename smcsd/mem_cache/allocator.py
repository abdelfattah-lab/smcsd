"""SMC-specific KV pool allocator.

Adds per-slot refcount tracking on top of the base ``TokenToKVPoolAllocator``
so that multiple SMC particles can share KV slots safely (parent-prefix
fan-out and resample copy both create multi-owner slots).

Refcount endpoints (set to 1 on allocation, 0 on free) are maintained by
overriding ``alloc`` / ``free``; ``inc_ref`` / ``dec_ref_and_free`` and
``copy_block_table`` are the only public refcount API. The base allocator
is left untouched.

Hybrid models (Mamba/SSM + attention, e.g. Qwen3-Next, Qwen3.5) carry a
recurrent state per request in a separate ``MambaPool`` indexed by req-pool
index, not by token slot. The flat KV refcount machinery doesn't apply to
that state — particles receive their own mamba slot at alloc time, and we
*copy* (not refcount-share) the parent's state on fan-out and the
ancestor's state on resample. The helpers ``copy_mamba_state_one`` /
``copy_mamba_state_batched`` below encapsulate that, and are no-ops on
dense (non-hybrid) request pools.
"""

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool


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


def is_hybrid_req_to_token_pool(req_to_token_pool) -> bool:
    """True iff the pool carries a per-request Mamba/SSM state alongside the
    flat token-level KV block table (Qwen3-Next, Qwen3.5, Falcon-H1, ...)."""
    return isinstance(req_to_token_pool, HybridReqToTokenPool)


def _copy_one(pool, src_req_pool_idx: int, dst_req_pool_idx: int) -> None:
    src_idx = pool.req_index_to_mamba_index_mapping[src_req_pool_idx]
    dst_idx = pool.req_index_to_mamba_index_mapping[dst_req_pool_idx]
    pool.mamba_pool.copy_from(
        src_idx.unsqueeze(0).to(torch.int64),
        dst_idx.unsqueeze(0).to(torch.int64),
    )


def _copy_batched(
    pool,
    src_req_pool_indices: torch.Tensor,
    dst_req_pool_indices: torch.Tensor,
) -> None:
    if src_req_pool_indices.numel() == 0:
        return
    src_mamba = pool.req_index_to_mamba_index_mapping[
        src_req_pool_indices.to(torch.int64)
    ].to(torch.int64)
    dst_mamba = pool.req_index_to_mamba_index_mapping[
        dst_req_pool_indices.to(torch.int64)
    ].to(torch.int64)
    pool.mamba_pool.copy_from(src_mamba, dst_mamba)


def copy_mamba_state_one(
    req_to_token_pool,
    src_req_pool_idx: int,
    dst_req_pool_idx: int,
    *,
    draft_req_to_token_pool=None,
) -> None:
    """Copy the Mamba/SSM recurrent state from src's slot to dst's slot.

    Used during parent->particle fan-out: each particle inherits the parent's
    full prefix mamba state so the first decode step starts from the correct
    recurrent context (zero-state would be silently wrong on hybrid models).

    On hybrid+hybrid pairs, ``draft_req_to_token_pool`` (a
    ``DraftHybridReqToTokenPool``) is passed in so the draft-side state is
    copied in lockstep — otherwise the draft's recurrent context for the
    new particle would also start at zero. ``draft_req_to_token_pool`` is
    None / aliases the target on hybrid+dense.

    No-op on dense targets (the pool has no mamba_pool).
    """
    if is_hybrid_req_to_token_pool(req_to_token_pool):
        _copy_one(req_to_token_pool, src_req_pool_idx, dst_req_pool_idx)
    if (
        draft_req_to_token_pool is not None
        and draft_req_to_token_pool is not req_to_token_pool
        and is_hybrid_req_to_token_pool(draft_req_to_token_pool)
    ):
        _copy_one(draft_req_to_token_pool, src_req_pool_idx, dst_req_pool_idx)


def copy_mamba_state_batched(
    req_to_token_pool,
    src_req_pool_indices: torch.Tensor,
    dst_req_pool_indices: torch.Tensor,
    *,
    draft_req_to_token_pool=None,
) -> None:
    """Batched variant: copy mamba state from src[i] to dst[i] for every i.

    Used during SMC resampling: each resampled slot receives the ancestor
    slot's full mamba state. Operates on both target and draft pools when
    the pair is hybrid+hybrid. No-op on dense targets.
    """
    if is_hybrid_req_to_token_pool(req_to_token_pool):
        _copy_batched(req_to_token_pool, src_req_pool_indices, dst_req_pool_indices)
    if (
        draft_req_to_token_pool is not None
        and draft_req_to_token_pool is not req_to_token_pool
        and is_hybrid_req_to_token_pool(draft_req_to_token_pool)
    ):
        _copy_batched(
            draft_req_to_token_pool, src_req_pool_indices, dst_req_pool_indices
        )

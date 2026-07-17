"""Fused Triton kernel for SMC resampling, driven by a device-resident plan.

One launch applies the entire resample plan emitted by ``fused_collect``
with NO host synchronisation:

* The grid is sized to the host-known worst case (``live_rows × (N-1)``);
  each program loads the true job count from ``plan_counter`` (GPU memory)
  and exits early — the launch never needs the count on the CPU.
* Per-job inputs (pool rows, lengths) are gathered in-kernel from the
  slot-indexed tensors, so no host-side slicing of the plan is required.
* KV pages whose refcount hits zero are appended to a preallocated
  ``freed_buf`` via an atomic cursor; the scheduler frees them later in
  postprocessing.  The ``prev = atomic_add(ref, -1); prev == 1`` transition
  fires exactly once per page, so no dedup pass is needed.

Phases per job (copy ``src_slot → dst_slot``):

1. dec_ref every page in dst's old block table; capture pages hitting 0.
2. copy src's block table over dst + inc_ref each copied page.
3. copy the per-slot lineage scalars (seq/alloc lens, verified ids,
   finish state, token count).
4. copy src's token history row (``all_token_ids``) up to its count.

Safety relies on the fused-collect contract: dst slots are globally
unique and disjoint from src slots, so jobs never write state another job
reads.  A page shared between a dying dst table and any live table cannot
transiently hit refcount 0 — the live table's reference is never
decremented this step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_resample_kernel(
    # device-resident plan (valid prefix gated by plan_counter)
    plan_counter_ptr,  # (1,) int32 — true job count
    plan_dst_ptr,  # (max_jobs+,) int32 slot ids
    plan_src_ptr,  # (max_jobs+,) int32 slot ids
    # slot-indexed state (gathered in-kernel)
    req_pool_indices_ptr,  # (max_slots,) int64
    kv_allocated_lens_ptr,  # (max_slots,) int64
    seq_lens_ptr,  # (max_slots,) int64
    verified_ids_ptr,  # (max_slots,) int32
    prev_last_draft_ids_ptr,  # (max_slots,) int32
    finished_mask_ptr,  # (max_slots,) int8 (bool view)
    finished_len_ptr,  # (max_slots,) int32
    finish_reason_code_ptr,  # (max_slots,) int8
    matched_eos_token_ptr,  # (max_slots,) int32
    token_counts_ptr,  # (max_slots,) int32
    all_token_ids_ptr,  # (max_slots, max_output_len) int32
    all_token_ids_stride,  # row stride
    # KV pool
    req_to_token_ptr,  # (pool_size, max_ctx_len) int32
    req_to_token_stride,  # row stride
    refcount_ptr,  # (kv_pool_size,) int32
    # freed-page capture (persistent, cursor reset by postprocessing)
    freed_buf_ptr,  # (kv_pool_size+,) int32
    freed_counter_ptr,  # (1,) int32 atomic cursor
    BLOCK_SIZE: tl.constexpr,
):
    job = tl.program_id(0)
    n_jobs = tl.load(plan_counter_ptr)
    if job >= n_jobs:
        return

    dst_slot = tl.load(plan_dst_ptr + job)
    src_slot = tl.load(plan_src_ptr + job)

    dst_pool = tl.load(req_pool_indices_ptr + dst_slot)
    src_pool = tl.load(req_pool_indices_ptr + src_slot)
    dst_alloc = tl.load(kv_allocated_lens_ptr + dst_slot)  # OLD dst length
    src_seq = tl.load(seq_lens_ptr + src_slot)

    dst_row = req_to_token_ptr + dst_pool * req_to_token_stride
    src_row = req_to_token_ptr + src_pool * req_to_token_stride

    # Phase 1: dec_ref old dst pages; pages transitioning 1 -> 0 are
    # appended to freed_buf through the atomic cursor.
    num_dec_iters = tl.cdiv(dst_alloc, BLOCK_SIZE)
    for i in range(num_dec_iters):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < dst_alloc
        old_kv_idx = tl.load(dst_row + offset, mask=mask)
        prev = tl.atomic_add(
            refcount_ptr + old_kv_idx.to(tl.int64), -1, mask=mask
        )
        is_freed = mask & (prev == 1)
        n_freed = tl.sum(is_freed.to(tl.int32), axis=0)
        base = tl.atomic_add(freed_counter_ptr, n_freed)
        excl_prefix = tl.cumsum(is_freed.to(tl.int32), axis=0) - is_freed.to(
            tl.int32
        )
        tl.store(freed_buf_ptr + base + excl_prefix, old_kv_idx, mask=is_freed)

    # Phase 2: copy src block table -> dst + inc_ref.
    num_src_iters = tl.cdiv(src_seq, BLOCK_SIZE)
    for i in range(num_src_iters):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < src_seq
        src_kv_idx = tl.load(src_row + offset, mask=mask)
        tl.store(dst_row + offset, src_kv_idx, mask=mask)
        tl.atomic_add(refcount_ptr + src_kv_idx.to(tl.int64), 1, mask=mask)

    # Phase 3: per-slot lineage scalars.  src slots are never dst slots
    # (collect contract), so these reads race with nothing.
    tl.store(seq_lens_ptr + dst_slot, src_seq)
    tl.store(
        kv_allocated_lens_ptr + dst_slot,
        tl.load(kv_allocated_lens_ptr + src_slot),
    )
    tl.store(
        verified_ids_ptr + dst_slot, tl.load(verified_ids_ptr + src_slot)
    )
    tl.store(
        prev_last_draft_ids_ptr + dst_slot,
        tl.load(prev_last_draft_ids_ptr + src_slot),
    )
    tl.store(
        finished_mask_ptr + dst_slot, tl.load(finished_mask_ptr + src_slot)
    )
    tl.store(
        finished_len_ptr + dst_slot, tl.load(finished_len_ptr + src_slot)
    )
    tl.store(
        finish_reason_code_ptr + dst_slot,
        tl.load(finish_reason_code_ptr + src_slot),
    )
    tl.store(
        matched_eos_token_ptr + dst_slot,
        tl.load(matched_eos_token_ptr + src_slot),
    )
    src_count = tl.load(token_counts_ptr + src_slot)
    tl.store(token_counts_ptr + dst_slot, src_count)

    # Phase 4: token history row copy, bounded by src's count (entries
    # beyond it are never read for dst — reads are count/finished_len
    # bounded, and later appends write at dst's own offsets).
    dst_tok = all_token_ids_ptr + dst_slot.to(tl.int64) * all_token_ids_stride
    src_tok = all_token_ids_ptr + src_slot.to(tl.int64) * all_token_ids_stride
    num_tok_iters = tl.cdiv(src_count, BLOCK_SIZE)
    for i in range(num_tok_iters):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < src_count
        tok = tl.load(src_tok + offset, mask=mask)
        tl.store(dst_tok + offset, tok, mask=mask)


@triton.jit
def _dec_ref_tail_kernel(
    slots_ptr,            # (G,) int64 — slot ids whose tail to release
    req_pool_indices_ptr, # (max_slots,) int64
    tail_start_ptr,       # (G,) int64 — first page position to release
    tail_end_ptr,         # (G,) int64 — one past the last position
    req_to_token_ptr,     # (pool_size, max_ctx_len) int32
    req_to_token_stride,
    refcount_ptr,         # (kv_pool_size,) int32
    freed_buf_ptr,        # (kv_pool_size+,) int32
    freed_counter_ptr,    # (1,) int32 atomic cursor
    BLOCK_SIZE: tl.constexpr,
):
    g = tl.program_id(0)
    slot = tl.load(slots_ptr + g)
    pool = tl.load(req_pool_indices_ptr + slot)
    start = tl.load(tail_start_ptr + g)
    end = tl.load(tail_end_ptr + g)
    row = req_to_token_ptr + pool * req_to_token_stride

    length = end - start
    num_iters = tl.cdiv(length, BLOCK_SIZE)
    for i in range(num_iters):
        offset = start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < end
        kv_idx = tl.load(row + offset, mask=mask)
        prev = tl.atomic_add(refcount_ptr + kv_idx.to(tl.int64), -1, mask=mask)
        is_freed = mask & (prev == 1)
        n_freed = tl.sum(is_freed.to(tl.int32), axis=0)
        base = tl.atomic_add(freed_counter_ptr, n_freed)
        excl = tl.cumsum(is_freed.to(tl.int32), axis=0) - is_freed.to(tl.int32)
        tl.store(freed_buf_ptr + base + excl, kv_idx, mask=is_freed)


def dec_ref_tail_pages(
    req_to_token: torch.Tensor,
    refcount: torch.Tensor,
    *,
    slots: torch.Tensor,
    req_pool_indices: torch.Tensor,
    tail_start: torch.Tensor,
    tail_end: torch.Tensor,
    freed_buf: torch.Tensor,
    freed_counter: torch.Tensor,
) -> None:
    """dec_ref a per-slot block-table span ``[tail_start, tail_end)``,
    capturing refcount-0 pages into ``freed_buf`` — one launch, no host
    sync.  Used by the exact-mode collapse to release the winner chain's
    rejected-tail pages (the losers' pages go through the resample
    kernel's Phase 1); the scheduler frees the captured pages in
    postprocessing like any resample-freed page.
    """
    g = slots.numel()
    if g == 0:
        return
    _dec_ref_tail_kernel[(g,)](
        slots,
        req_pool_indices,
        tail_start,
        tail_end,
        req_to_token,
        req_to_token.stride(0),
        refcount,
        freed_buf,
        freed_counter,
        BLOCK_SIZE=64,
    )


def batched_resample_kv(
    req_to_token: torch.Tensor,
    refcount: torch.Tensor,
    *,
    plan_dst: torch.Tensor,
    plan_src: torch.Tensor,
    plan_counter: torch.Tensor,
    max_jobs: int,
    req_pool_indices: torch.Tensor,
    kv_allocated_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    verified_ids: torch.Tensor,
    prev_last_draft_ids: torch.Tensor,
    finished_mask: torch.Tensor,
    finished_len: torch.Tensor,
    finish_reason_code: torch.Tensor,
    matched_eos_token: torch.Tensor,
    token_counts: torch.Tensor,
    all_token_ids: torch.Tensor,
    freed_buf: torch.Tensor,
    freed_counter: torch.Tensor,
) -> None:
    """Apply a device-resident resample plan in one launch — no host sync.

    ``max_jobs`` is the host-known worst-case job count (grid size); the
    true count is read from ``plan_counter`` on-device, so an empty plan
    costs one no-op launch (microseconds) instead of a blocking ``.item()``.

    Pages whose refcount hits zero land in ``freed_buf[:freed_counter]``;
    the caller reads the cursor and frees them in postprocessing, then
    resets the cursor.
    """
    if max_jobs == 0:
        return

    _fused_resample_kernel[(max_jobs,)](
        plan_counter,
        plan_dst,
        plan_src,
        req_pool_indices,
        kv_allocated_lens,
        seq_lens,
        verified_ids,
        prev_last_draft_ids,
        finished_mask.view(torch.int8),
        finished_len,
        finish_reason_code,
        matched_eos_token,
        token_counts,
        all_token_ids,
        all_token_ids.stride(0),
        req_to_token,
        req_to_token.stride(0),
        refcount,
        freed_buf,
        freed_counter,
        BLOCK_SIZE=128,
    )

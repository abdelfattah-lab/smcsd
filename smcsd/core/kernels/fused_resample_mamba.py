"""Device-driven Mamba recurrent-state copy for SMC resampling.

Replaces the trash-padded ``MambaPool.copy_from`` path in
``copy_smc_resampled_hybrid_state``.  That path had two costs:

* **O(flat_cap) work every step** — the job count lives in a device-side
  counter (reading it is a host sync, banned on the hot path), so the torch
  advanced-indexing copy processed every plan slot each step, mostly
  null-slot self-copies on the (majority of) steps with no resample.
* **Advanced-indexing tax** — ``t[:, dst] = t[:, src]`` gathers into an
  intermediate then index_puts (double traffic) with generic-stride int64
  element addressing, measured at ~25-35% of peak bandwidth.

This kernel mirrors ``fused_resample_kv``'s dispatch contract instead: the
grid is the host-known worst case ``(max_jobs, num_layers, row_blocks)`` and
every program first loads the true job count from ``plan_counter`` and exits
if its job is beyond it — an empty plan costs a few no-op launches instead of
a bulk copy, with no host sync.  Valid jobs resolve their indices in-kernel
(slot -> req_pool row -> mamba row) and move contiguous state rows directly
src -> dst: one coalesced read + one write, no intermediates.

Correctness contract (same as the padded path it replaces):
* dst and src slot sets are disjoint (resample invariant: survivors are only
  sources, dead slots only destinations), so there is no read/write aliasing
  and the copy is idempotent.
* Enqueued on the caller's (scheduler) stream inside ``_resample``, i.e.
  stream-ordered ahead of the next forward — overlap-safe.
* Static grid + device counter keeps it CUDA-graph-capturable.
* Negative / out-of-range indices (e.g. EMPTY_SLOT == -1 on a transiently
  stale row) are guarded per-job and skipped.
"""

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024


@triton.jit
def _fused_mamba_resample_copy_kernel(
    plan_counter_ptr,      # (1,) int32 — true job count, device-resident
    dst_slots_ptr,         # (max_jobs,) int32 — slot-state indices (valid prefix)
    src_slots_ptr,         # (max_jobs,) int32
    req_pool_indices_ptr,  # (max_slots,) int64 — slot -> req_pool row
    mapping_ptr,           # (mapping_len,) int32 — req_pool row -> mamba row
    state_ptr,             # (num_layers, num_rows, <payload>) contiguous
    stride_layer,
    stride_row,
    mapping_len,
    num_rows,
    row_elems,
    BLOCK_SIZE: tl.constexpr,
):
    job = tl.program_id(0)
    n_jobs = tl.load(plan_counter_ptr)
    if job >= n_jobs:
        return

    layer = tl.program_id(1).to(tl.int64)
    blk = tl.program_id(2).to(tl.int64)

    dst_slot = tl.load(dst_slots_ptr + job).to(tl.int64)
    src_slot = tl.load(src_slots_ptr + job).to(tl.int64)
    dst_req = tl.load(req_pool_indices_ptr + dst_slot).to(tl.int64)
    src_req = tl.load(req_pool_indices_ptr + src_slot).to(tl.int64)
    if (dst_req < 0) | (dst_req >= mapping_len) | (src_req < 0) | (
        src_req >= mapping_len
    ):
        return
    dst_row = tl.load(mapping_ptr + dst_req).to(tl.int64)
    src_row = tl.load(mapping_ptr + src_req).to(tl.int64)
    if (dst_row < 0) | (dst_row >= num_rows) | (src_row < 0) | (
        src_row >= num_rows
    ):
        return

    offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < row_elems
    src_off = layer * stride_layer + src_row * stride_row + offs
    dst_off = layer * stride_layer + dst_row * stride_row + offs
    data = tl.load(state_ptr + src_off, mask=mask)
    tl.store(state_ptr + dst_off, data, mask=mask)


def fused_mamba_resample_copy(
    pool,
    req_pool_indices: torch.Tensor,
    dst_slots: torch.Tensor,
    src_slots: torch.Tensor,
    plan_counter: torch.Tensor,
    max_jobs: int,
) -> None:
    """Copy Mamba recurrent state src->dst for every valid plan job on ``pool``.

    No-op (zero launches) when the pool has no mamba state or ``max_jobs`` is
    0; a few counter-load-and-exit launches when the plan is empty.
    """
    if pool is None or not hasattr(pool, "mamba_pool") or max_jobs == 0:
        return
    mapping = pool.req_index_to_mamba_index_mapping
    cache = pool.mamba_pool.mamba_cache
    state_tensors = list(cache.conv) + [cache.temporal]
    for t in state_tensors:
        # Contiguity makes the per-row payload a flat block; the pool
        # allocates these with torch.zeros so this holds by construction.
        assert t.is_contiguous(), "mamba state tensor must be contiguous"
        num_layers, num_rows = t.shape[0], t.shape[1]
        row_elems = t[0, 0].numel()
        grid = (max_jobs, num_layers, triton.cdiv(row_elems, _BLOCK_SIZE))
        _fused_mamba_resample_copy_kernel[grid](
            plan_counter,
            dst_slots,
            src_slots,
            req_pool_indices,
            mapping,
            t,
            t.stride(0),
            t.stride(1),
            mapping.numel(),
            num_rows,
            row_elems,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

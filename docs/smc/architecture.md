# SMC v2 Architecture

## Workflow

```
PREFILL (ScheduleBatch) -> parent prefill on score + draft
     |
     v
MATERIALIZE GROUP
  clone parent -> N particle Reqs
  copy_block_table (parent prefix -> particles, inc_ref)
  release parent (dec_ref)
  StackedGroupState.register_group (claim row)
  ScheduleBatchSMC.allocate_slots (fill [max_slots] tensors)
     |
     v
DECODE LOOP (ScheduleBatchSMC, slot-based) ------+
  prepare_for_decode: gather[active] -> ctx      |
    SMCDecodeContext.from_slot_gather            |
      vectorised alloc gamma+1 KV                 |
      seq_lens += gamma+1                         |
    scatter back                                  |
  build_model_worker_batch (sparse -> dense)     |
  SMCWorkerV2._forward_decode                    |
    ctx.prepare_for_draft  -> AR gamma+1 steps   |
    ctx.prepare_for_verify -> TARGET_VERIFY      |
    sample bonus                                 |
  process_batch_result (dense -> sparse scatter) |
    accumulate log/interval weights on stacked   |
  SMCCoordinatorV2                               |
    slow: per-group Python (golden truth)        |
    fast: one fused Triton kernel over stacked   |
  dispatch: batched_resample_kv + slot copies    |
  rebuild_active_slots (at most once / cycle)    |
  --------(until group has no active slot)-------+
     |
     v
FINALIZE (argmax log_weight -> parent Req, free_group_slots)
```

## Key Files (`smcsd/`)

| File | Role |
|------|------|
| `v2/scheduler.py` | `SMCSchedulerV2`, `SMCCoordinatorV2`, `SequenceGroup` |
| `v2/req_state.py` | `ScheduleBatchSMC` — persistent slot-based GPU state |
| `v2/stacked_state.py` | `StackedGroupState` — `(max_G, N)` primary storage |
| `v2/info.py` | `SMCDecodeContext`, `SMCDraftInputV2` |
| `v2/worker.py` | `SMCWorkerV2` (standalone, not inheriting v1) |
| `v2/kernels/fused_collect.py` | Fused normalize+ESS+resample+compaction |
| `v2/kernels/fused_resample_kv.py` | Fused block-table copy + refcount |
| `mem_cache/allocator.py` | `SMCRefCountedTokenAllocator` |

## Two-Tier State

```
ScheduleBatchSMC — slot-major [max_slots] (sparse)
  req_pool_indices, seq_lens, kv_allocated_lens, verified_ids,
  token_counts, finished_mask, group_indices, particle_indices,
  all_token_ids [max_slots, max_output_len]

  active_slots: idx_mapping (batch_idx -> slot_idx),
                sorted by group, rebuilt only on membership change

StackedGroupState — group-major (max_G, N) (dense)
  log_weights, interval_weights  (float64)
  particle_to_slot, active_cell_mask, n_active, row_in_use
  persistent scratch (dst/src/row flat, atomic counter, mask)

  legacy group_log_weights[gid] is a VIEW into log_weights[row, :n]
  -> stacked tensors are single source of truth for fused kernel
```

## Resampling: Slow vs Fast

**Slow path (`fast_resample=False`)** — per-group Python: normalize → ESS → systematic → `Counter` pair → per-pair `resample_copy_slot`. Golden truth.

**Fast path (`fast_resample=True`, systematic + CUDA)** — one kernel per group row:

```
_fused_collect_kernel (one program per row of stacked state)
  mask inactive -> -inf
  lse-normalize -> weights
  ess = 1/Σw²; resample if ess < threshold * n_active
  cdf = cumsum(weights)
  seed = tl.rand(step_counter, row)   # Philox, no host sync
  systematic draws -> counts[ancestor] via scalar searchsorted
  dead_flag = (counts==0) & active
  excess    = max(counts-1, 0)
  offset = atomic_add(global_counter, n_copies)
  scatter flat dst/src/row; zero weights
  -> BatchedResampleResult (only .item() sync at n_jobs)

batched_resample_kv (one program per (dst,src) pair)
  Phase 1: capture req_to_token[dst, :dst_alloc] + dec_ref
  Phase 2: copy req_to_token[src, :src_len] -> dst + inc_ref
  -> to_free for allocator

vectorised slot-tensor copies: seq_lens, kv_allocated_lens,
  verified_ids, finished_mask, token_counts, all_token_ids [dst]=[src]

per-pair Python: Req-side metadata (output_ids, finished_reason,
  offsets) — the only unavoidable host cost
```

## v2 vs v1: What Changed

| | v1 | v2 |
|---|---|---|
| Decode batch | `ScheduleGroupBatch`, rebuilt each iter via `sync_from_groups` | `ScheduleBatchSMC`, persistent slots + `active_slots` gather |
| Per-group state | Python dicts (`group_log_weights`, etc.) | `StackedGroupState` `(max_G, N)` tensors; dicts are views |
| KV alloc/decode | Per-req Python loop | One vectorised `assign_req_to_token_pool_func` |
| Spec carrier | `SMCDraftInput` holds prepare methods + hidden `_orig_seq_lens` | `SMCDraftInputV2` = pure data; prep on `SMCDecodeContext` |
| Worker | `SMCWorker` | `SMCWorkerV2` standalone |
| Resample collect | Per-group Python | One fused Triton launch over whole stacked state |
| Resample dispatch | Per-pair `resample_copy_slot` | Fused `batched_resample_kv` + vectorised slot copies |
| `rebuild_active` | Implicit via rebuild | Deferred, at most once per cycle |

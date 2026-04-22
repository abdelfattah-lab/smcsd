# SMC Pipeline — End-to-End Data Flow

This doc walks through what *happens* in the SMC scheduler, stage by stage.
The state objects each stage touches are cataloged in
[`state.md`](./state.md); the top-level map is [`architecture.md`](./architecture.md).

```
  [admit]  →  [parent prefill]  →  [materialize]  →  [decode cycle] ╶╮
                                                         ▲           │
                                                         └───────────┘
                                                         │
                                                    [finalize]
```

Stages 0–3 happen once per request. Stage 4 is the hot loop. Stage 5 runs
once when the group drains.

---

## 0. Outer event loop

`SMCScheduler.run_event_loop` runs forever in the scheduler subprocess,
driving one batch per iteration on the schedule CUDA stream:

```
while True:
    recv_requests()                      ZMQ pull from SMCEngine
    process_input_requests()             → enqueue as SequenceGroup

    batch, kind = _get_next_batch()      one of:
                                           • prefill ScheduleBatch
                                             (admitted parents only)
                                           • decode ModelWorkerBatch
                                             (from slot_state)
                                           • None (idle)

    if batch is None:
        self_check_during_idle()         leak detection over slots
        continue

    result = run_batch(batch)

    if kind == "prefill":
        _process_prefill_result(...)     sample x₀, materialize groups
    else:
        _process_decode_result(...)      scatter, accumulate, resample,
                                         drain, (maybe) rebuild_active
```

`_get_next_batch` prefers any admitted prefill over decode, so a freshly
arrived group completes its one-shot parent prefill before rejoining the
decode loop.

---

## 1. Admission

Each incoming `Req` is wrapped as `SequenceGroup(parent_req, n_particles)`
and pushed onto `waiting_groups`. Before admission, `validate_smc_parent_req`
rejects incompatible configurations (multimodal, grammars, logprob returns,
stop strings / regex). `_abort_on_queue_limit` protects against an
unbounded backlog.

There is **no retraction path** in SMC. A group is atomic — partial
retraction is not supported, and `ScheduleBatch.retract_decode` is
unreachable here because decode runs through `ScheduleBatchSMC`.

`_admit_prefill_groups` moves groups from waiting → prefill while the slot
pool has room for `N` new particles per group:

```
  available = slot_state.available_slot_count()
  while waiting_groups and groups[0].n_particles <= available:
      move groups[0] waiting → prefill_groups
      available -= n_particles
```

---

## 2. Parent prefill

A standard `ScheduleBatch.init_new(parents, ...).prepare_for_extend()` is
built and run through the score model worker. This is upstream sglang code
— SMC reuses it because prefill runs once per group and isn't a hot path.
The only SMC touch is `_prepare_req_for_private_prefill`, which resets the
parent's prefix-cache bookkeeping so the particles that will clone from it
can't accidentally reuse a tree-cache node.

On completion `_process_prefill_result` walks parent-by-parent:

```
for each parent, with its sampled next_token_id:
    parent.output_ids.append(next_token_id)
    if parent.finished():           # prompt was EOS, max_new_tokens=0, etc.
        release_kv_cache(parent)
        stream_output([parent])
        continue
    err = _materialize_group(group)
    if err:
        _abort_group(group, err)
        continue
    running_groups.append(group)
```

---

## 3. Materialization

The one-time fan-out from 1 parent to N particles. This is the only
stage that bridges the base allocator and SMC's refcount API, and it's
where the ownership invariant of [`state.md` §5](./state.md#5-kv-memory-smcrefcountedtokenallocator)
gets established.

```
          parent Req                           (kv_committed_len = L,
               │                                x₀ already sampled)
               │
  ┌────────────┴────────────────────────────────────────────────┐
  │                                                             │
  │  a) SMCWorker.materialize_smc_parent_draft_prefix(parent) │
  │       draft model materializes its own KV for the same L    │
  │       tokens (target and draft have separate KV caches).    │
  │                                                             │
  │  b) group.materialize_particles(device)                     │
  │       clone_req_for_smc_particle × N                        │
  │       → particle_reqs = {pidx → Req}                        │
  │       initialise log_weights / interval_weights to 0        │
  │                                                             │
  │  c) req_to_token_pool.alloc(particle_reqs)                  │
  │       each particle claims a new row in req_to_token        │
  │                                                             │
  │  d) for each particle: copy_block_table(parent → pᵢ, L)     │
  │       → clones L block-table entries                        │
  │       → inc_ref on each of the L KV slots                   │
  │       particle.kv_committed_len   = L                       │
  │       particle.kv_allocated_len   = L                       │
  │       particle.prefix_indices     = req_to_token[pᵢ, :L]    │
  │       particle.cache_protected_len= L                       │
  │                                                             │
  │  e) _release_smc_parent_req(parent)                         │
  │       dec_ref_and_free the parent's L prefix slots          │
  │       (refcount drops from N+1 to N — still alive)          │
  │       free parent Req slot                                  │
  │                                                             │
  │  f) slot_state.allocate_slots(gid, group_idx, particles, L) │
  │       claim N slots from free_slots                         │
  │       fill per-slot tensors                                 │
  │       claim row from _free_rows                             │
  │       group_to_slots[row, :N] = slots; row_in_use[row] = T  │
  │       log_weights[slots]=0; interval_weights[slots]=0       │
  │       rebuild_active_slots()                                │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
               │
               ▼
          running_groups.append(group)
```

Any failure between (a) and (f) rolls back: particle Reqs are freed via
`_release_internal_req`, the group's particles are cleared, and the parent
is streamed out with a `FINISH_ABORT`.

---

## 4. The decode cycle

Hot loop. Runs as long as any group has at least one unfinished particle.
One cycle does seven things, in order:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  4.1  prepare_for_decode      sparse→gather→KV alloc→scatter    │
  │  4.2  build_model_worker_batch dense ModelWorkerBatch           │
  │  4.3  draft AR × (γ+1)        on the draft model                │
  │  4.4  TARGET_VERIFY            on the score model               │
  │  4.5  process_batch_result    dense→scatter; weight accum;      │
  │                               finish check                      │
  │  4.6  coordinator.collect +   slow: per-group Python            │
  │       dispatch resample       fast: fused Triton                │
  │  4.7  rebuild_active_slots    at most once per cycle            │
  └─────────────────────────────────────────────────────────────────┘
```

### 4.1 `prepare_for_decode`

Input: current `active_slots` (see [`state.md` §2.3](./state.md#23-active_slots--the-sparse--dense-idx-mapping)).
Output: `SMCDraftInput` with an attached `SMCDecodeContext`.

```
  active = slot_state.active_slots                     # [bs] int64

  # gather sparse → contiguous
  seq_lens_g        = slot_state.seq_lens[active]
  kv_alloc_g        = slot_state.kv_allocated_lens[active]
  pool_idx_g        = slot_state.req_pool_indices[active]
  verified_g        = slot_state.verified_ids[active]

  # SMCDecodeContext.from_slot_gather:
  #   alloc_start  = max(kv_alloc_g, seq_lens_g)
  #   needed       = seq_lens_g + γ+1
  #   new_alloc    = clamp(needed - alloc_start, min=0)
  #   total_needed = int(new_alloc.sum())              ← one GPU↔CPU sync
  #
  #   out_cache_loc = alloc_token_slots(tree_cache, total_needed)
  #   assign_req_to_token_pool_func(
  #       pool_idx_g, req_to_token,
  #       alloc_start, alloc_start + new_alloc, out_cache_loc, bs
  #   )                                                 ← one vectorised
  #                                                       block-table write
  #   new_seq_lens = seq_lens_g + γ+1

  # scatter contiguous → sparse
  slot_state.kv_allocated_lens[active] = alloc_start + new_alloc
  slot_state.seq_lens[active]          = new_seq_lens
```

The `SMCDecodeContext` carries `orig_seq_lens`, `new_seq_lens`, and `γ` so
the draft loop can set per-step positions without another CPU sync.

### 4.2 `build_model_worker_batch`

A straight gather of slot tensors into a dense `ModelWorkerBatch`. No SMC
math — just layout. The `SamplingBatchInfo` is minimal (greedy flags False,
temp/top-p/top-k tensors gathered from slots) because the SMC worker does
its own temperature adjustment. `spec_info = SMCDraftInput`.

### 4.3 Draft AR × (γ+1)

In `SMCWorker._forward_decode`:

```
  ctx       = draft_input.decode_ctx
  x₀        = draft_input.verified_id                  # [bs]
  all_tokens = [x₀]
  draft_logprobs = []

  # ctx.prepare_for_draft builds a draft ForwardBatch with per-step
  # cache_locs [bs, γ+1] and per-step seq_lens [bs, γ+1] so the AR loop
  # can slice rather than recompute.

  for step in range(γ + 1):
      draft_fb.input_ids   = current_ids
      draft_fb.positions   = all_positions[:, step]
      draft_fb.out_cache_loc = cache_locs[:, step]
      draft_out = draft_runner.forward(draft_fb)

      logits = draft_out.next_token_logits
      log_probs = log_softmax(logits / smc_draft_temperature)

      next_token = multinomial(log_probs.exp())
      if step < γ:                                     # no logprob for x_{γ+1}
          draft_logprobs.append( log_probs[..., next_token] )

      all_tokens.append(next_token)
      current_ids = next_token
```

Output: `all_tokens = [x₀, x₁, …, x_γ, x_{γ+1}]` and
`draft_logprobs_stacked` of shape `[bs, γ]`.

### 4.4 `TARGET_VERIFY`

One extended forward pass on the score model over the `γ+1` drafted
tokens per particle (`bs·(γ+1)` rows total):

```
  score_input = stack(all_tokens[0:γ+1], dim=1).reshape(-1)   # [bs·(γ+1)]

  verify_batch = clone(batch)
  verify_batch.input_ids     = score_input
  verify_batch.out_cache_loc = cache_locs.reshape(-1)
  verify_batch.seq_lens      = ctx.orig_seq_lens
  verify_batch.forward_mode  = TARGET_VERIFY
  verify_batch.spec_info     = SMCVerifyInput(γ+1 tokens, positions, ...)

  score_out = target_worker.forward(verify_batch)           # bs·(γ+1) logits
  score_log_probs = log_softmax(score_out).reshape(bs, γ+1, vocab)

  # score logprob of the draft's xₖ (for k = 1..γ):
  target_tokens = stack(all_tokens[1:γ+1], dim=1)           # [bs, γ]
  score_logprobs = score_log_probs[:, :γ, :].gather(-1, target_tokens)

  # per-particle log-weight delta (γ terms summed):
  logprob_diff = (score_logprobs - draft_logprobs_stacked).sum(dim=1)     # [bs]

  # bonus token sampled from the score model at position γ:
  bonus = multinomial( softmax( score_logits[:, γ] / smc_target_temperature ) )
```

Returned as `GenerationBatchResult` with:

```
  next_token_ids  = cat(all_tokens[1..γ+1] + [bonus]).reshape(-1)
  accept_lens     = full(bs, γ+1)
  next_draft_input= SMCDraftInput(verified_id=bonus, logprob_diff=…)
  logprob_diff    = [bs]
```

### 4.5 `process_batch_result` — scatter + weight accumulation + finish

Writes the forward result back into sparse slot state. The order below
matches the implementation.

```
  a) history write: per-slot all_token_ids
     ─────────────────────────────────────
     accepted_2d = next_token_ids.reshape(bs, γ+1)
     row_idx  = active_slots.unsqueeze(1).expand(-1, γ+1)
     col_idx  = token_counts[active].unsqueeze(1) + arange(γ+1)
     all_token_ids[row_idx, col_idx] = accepted_2d
     token_counts[active] += γ+1

  b) verified_ids[active] = bonus_ids
     (feeds x₀ for next cycle's draft AR)

  c) finish check (batched on GPU)
     length_hit = token_counts[active] >= max_new_tokens[active]
     eos_hit    = any(accepted_2d == eos_token_ids[active])
                  & ~ignore_eos_t[active]
     newly_finished_mask = (length_hit | eos_hit) & ~finished_mask[active]
     finished_mask[active] |= newly_finished_mask

  d) sync newly finished Reqs (CPU-side output_ids, finished_reason)
     finished_mask stays True forever for that slot; it drops out of
     active_slots on the next rebuild.

  e) weight accumulation, vectorised:
     d = logprob_diff.to(float64)
     log_weights[active_slots]      += d
     interval_weights[active_slots] += d
     # Two in-place index_puts over the full active set. No per-group
     # loop, no `.item()` syncs.

  f) optionally rebuild_active_slots
     (scheduler passes rebuild_active=False here — batched with resample)
```

### 4.6 Resample

There is one resample path — two fused Triton kernels in sequence, no
Python fallback, no flag.  `SMCCoordinator.collect_resample_jobs_batch`
returns a `BatchedResampleResult` (dst_slots, src_slots, row_of_job,
resample_mask, n_jobs) produced by the fused collect kernel;
`dispatch_resample_batch` consumes it via `batched_resample_kv` plus a
short Python loop for Req-level metadata copies.

The kernel sees **all** allocated particles of a group (including finished
ones) as candidates — finish state is handled by copy-propagation, not by
exclusion.

#### Collect + dispatch

```
  batched_collect_fused(
      log_weights, interval_weights, group_to_slots, row_in_use,
      threshold, step_counter,
  )
    ─────────────────────────────────────────────────────
    one Triton program per row, gated on row_in_use[row]:
      • slots  = group_to_slots[row, :N]
      • lw_raw = interval_weights[slots]            ← gather
      • lse-normalise → weights
      • ess = 1 / Σ w²;   resample iff ess < threshold · N
      • cdf = cumsum(weights)
      • u   = tl.rand(step_counter, row)            ← Philox, no host sync
      • systematic draws via scalar searchsorted
      • counts[ancestor] via per-draw scatter-add
      • dead_flag = (counts == 0)   (all N cells allocated under global N)
      • excess    = max(counts - 1, 0)
      • offset = atomic_add(global_counter, n_copies)
      • scatter flat (dst_slot, src_slot, row_of_job) triples
      • scatter-zero log_weights[slots] and interval_weights[slots]
    returns BatchedResampleResult  (one .item() sync at boundary)

  batched_resample_kv( req_to_token, slot_ref_count,
                       dst_pool, src_pool, dst_alloc, src_seq_len )
    ─────────────────────────────────────────────────────
    one Triton program per (dst, src) pair:
      Phase 1:  read req_to_token[dst, :dst_alloc]
                → capture into dec_out
                → atomic_add(refcount, -1)
      Phase 2:  read req_to_token[src, :src_len]
                → write  req_to_token[dst, :src_len]
                → atomic_add(refcount, +1)
    returns to_free = unique( dec_out where refcount == 0 )
      → allocator.free(to_free)

  # vectorised slot-tensor copies (device, no loops):
  seq_lens[dst]          = seq_lens[src]
  kv_allocated_lens[dst] = kv_allocated_lens[src]
  verified_ids[dst]      = verified_ids[src]
  finished_mask[dst]     = finished_mask[src]
  token_counts[dst]      = token_counts[src]
  all_token_ids[dst]     = all_token_ids[src]

  # per-pair Python (the only unavoidable host cost):
  for dst, src in zip(plan.dst_slots.tolist(), plan.src_slots.tolist()):
      slot_state.copy_req_metadata(dst, src)
```

Visually:

```
[ fused_collect kernel ] ─────────────►  one launch, all in-use rows
         │
         ▼
flat (dst, src, row) triples on GPU
         │
         ▼
[ batched_resample_kv ] ─────────────►  one launch, all pairs
         │
         ▼
vectorised slot-tensor copies (one line each)
         │
         ▼
per-pair copy_req_metadata  (the only Python-side loop)
```

The fused collect kernel uses systematic resampling on CUDA
(`SMCCoordinator.__init__` enforces CUDA).  Its Philox seed is driven
by a monotonic `step_counter`, so consecutive decode cycles use
independent stratifications without any host-side allocation or sync.

### 4.7 `rebuild_active_slots`

Runs at the very end of the decode cycle, **and only if** either
`process_batch_result` produced a newly-finished slot or
`dispatch_resample_batch` actually ran. Both of those pass
`rebuild_active=False` during the cycle so we rebuild exactly once:

```
  if newly_finished or did_resample:
      slot_state.rebuild_active_slots()
  _drain_finished_groups()
```

Drain walks `running_groups`, finalising any group whose every particle
is now marked finished.

---

## 5. Finalization

Triggered from `_drain_finished_groups` when `slot_state.group_has_active(gid)`
is False. Calls `slot_state.finalize_group(gid, parent_req)`:

```
  slots = group_slot_lists[gid]

  # Pick best by (log_weight, visible_output_len).
  # visible_output_len = min(req.finished_len or token_count, token_count)
  best_slot = argmax over slots of (log_weights[s], visible_output_len(s))
  best_req  = slot_to_req[best_slot]

  parent_req.output_ids      = list(best_req.output_ids)
  parent_req.finished_reason = copy(best_req.finished_reason)
  parent_req.finished_len    = best_req.finished_len
  (fallback to FINISH_ABORT if best_req.finished_reason is None)

  free_group_slots(gid)
    ├── row_in_use[row] = False; group_to_slots[row] = -1
    │   row returned to _free_rows
    └── for each slot:
          dec_ref_and_free on this slot's KV block-table slice
          req_to_token_pool.free(Req)
          reset slot tensors to EMPTY_SLOT / 0 / False
          log_weights[slot] = 0; interval_weights[slot] = 0
          push slot back onto free_slots

  rebuild_active_slots()

  stream_output([parent_req])
```

Because the fused kernel includes finished particles in the resample
candidate set, a group that finalises here can have any of its slots
marked finished and still hold the best weight — the `argmax` picks it
regardless. The freed row goes back on `_free_rows` for a future group.

---

## Cycle cheat-sheet

One glance, one decode cycle:

```
  active_slots  ─┬─► gather slots                 ┌── scatter to slots
                 │    ↓                           │     all_token_ids,
                 │   SMCDecodeContext             │     verified_ids,
                 │    ↓ (γ+1 KV alloc)            │     finished_mask
                 │   ModelWorkerBatch             │
                 │    ↓                           │
                 │   draft AR × (γ+1)             │
                 │    ↓                           │
                 │   TARGET_VERIFY                │
                 │    ↓ logprob_diff, bonus       │
                 │   process_batch_result ───────►│ (weights → log_weights[active])
                 │    ↓                           │
                 │   coordinator.collect          │
                 │    ↓                           │
                 │   coordinator.dispatch ───────►│ (resample KV + tensors)
                 │    ↓                           │
                 └─► rebuild_active_slots (≤ 1×) ◄┘
```

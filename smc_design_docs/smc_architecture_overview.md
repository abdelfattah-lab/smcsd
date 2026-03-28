# SMC Integration Architecture in SGLang

> **Branch:** `smc_v0`
> **Last verified date:** 2026-03-27

## Overview

SMC is a speculative decoding algorithm that runs **N particles** (parallel generation paths) per request. A lightweight **draft model** proposes tokens, and the **target model** scores them. Particles are weighted by how well the target agrees with the draft, and periodically **resampled** so compute focuses on the most promising paths. All tokens are accepted (no rejection sampling) — quality control is via importance weights.

## Key Files

All paths relative to `python/sglang/srt/`.

| File | Role |
|---|---|
| `speculative/smc_info.py` | Core data structures (`SMCDraftInput`, `SMCScoreInput`, `SMCDraftInputV2Mixin`), resampling algorithms (`systematic_resample`, `multinomial_resample`), particle cloning, mask/position builders |
| `speculative/smc_worker_v2.py` | `SMCWorkerV2` (extends `EAGLEWorkerV2`) — outer draft/score/draft-extend orchestration; `SMCDraftWorker` (extends `StandaloneDraftWorker`) — inner multi-step draft forward |
| `speculative/smc_scheduler.py` | `SMCScheduler` + `PendingResample` — two-bucket resample stream, admission control, weight updates |
| `speculative/smc_manager.py` | `SMCManager` + `SMCGroupState` — group lifecycle: creation, tracking, finalization |
| `speculative/smc_draft_cuda_graph_runner.py` | `SMCDraftCudaGraphRunner` + `SMCDraftInputBuffers` + `SMCDraftSamplingSignature` — fused γ-step draft CUDA graph |
| `managers/scheduler.py` | Main scheduler hooks (`smc_scheduler.step()`, `step_before_forward()`, `step_after_forward()`) |
| `managers/scheduler_output_processor_mixin.py` | Post-prefill init, decode result processing |
| `managers/schedule_batch.py` | `SMCGroupSpan`, `build_smc_group_spans()`, per-request SMC fields on `Req` |
| `managers/overlap_utils.py` | `FutureMap` — CPU-GPU overlap circular buffer (SMC: `last_token_ids_buf`, `new_seq_lens_buf`) |

## Data Structures

```
SMCGroupState             Manager-level: particle_reqs{}, log_weights tensor,
        │                 step_counts[], finished_particles{}, pending_diffs[]
SMCGroupSpan              Batch-level: contiguous [start, end) range in batch
SMCFinishedParticleSnapshot  Snapshot: output_ids, finished_reason, finished_len
```

**Request-level SMC fields** (on `Req` — `schedule_batch.py:789-790`):
- `smc_group_id`, `smc_particle_idx`

**Phase inputs**:
- `SMCDraftInput` (`SpecInput`, `SMCDraftInputV2Mixin`): `last_token_ids`, `new_seq_lens`, `future_indices`, `verify_done`, `positions`
- `SMCScoreInput` (`SpecInput`): `draft_token`, `draft_lengths`, `draft_logprobs`, `positions`, `custom_mask`, `draft_token_num`, `spec_steps`, `target_temperature`, `linear_target_verify`, `capture_hidden_mode`, `smc_logprob_diffs`

## Configuration

Activated via `--speculative-algorithm SMC` with:

| Flag | Default | Purpose |
|---|---|---|
| `--smc-n-particles` | 4 | Particles per request |
| `--smc-gamma` | 4 | Max draft tokens per step |
| `--smc-draft-temperature` | 0.7 | Draft model sampling temperature |
| `--smc-target-temperature` | 1.0 | Temperature for target model scoring during verification |
| `--smc-resample-threshold` | 0.5 | ESS ratio that triggers resampling |
| `--smc-resample-method` | systematic | `systematic` or `multinomial` |

Requires `--speculative-draft-model-path` and `--page-size 1`.

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. PREFILL  (parent request, normal path)                       │
│     - Target model processes prompt                              │
│     - KV cache populated for prefix                              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. GROUP CREATION  (smc_manager.create_group)                   │
│     - Spawn N particle Reqs (each is a single Req)               │
│     - Copy parent KV cache → all particles (copy_block_table)    │
│     - Initialize log_weights = [0, 0, ..., 0]                   │
│     - Release parent req's KV (it's now in particles)            │
│     - Enqueue group for running                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
          ┌────────────────────────────────────┐
          │         SMC ITERATION LOOP         │◄──────────────┐
          └────────────────┬───────────────────┘               │
                           ▼                                   │
┌─────────────────────────────────────────────────────────────┐│
│  3. DRAFT PHASE  (SpecInputType.SMC_DRAFT)                   ││
│     SMCDraftWorker.draft() → SMCWorkerV2._run_eagle_style_  ││
│     draft_reqs() generates γ tokens per particle.            ││
│     Fused path: SMCDraftCudaGraphRunner.replay()             ││
│     Stepwise path: SMCDraftWorker.draft_forward() loops γ    ││
│     Accumulates draft_logprobs = Σ log P_draft(t_i)          ││
│     Output: draft_tokens[bs, γ], draft_logprobs[bs]          ││
└──────────────────────────┬──────────────────────────────────┘│
                           ▼                                   │
┌─────────────────────────────────────────────────────────────┐│
│  4. SCORE PHASE  (SpecInputType.SMC_SCORE)                   ││
│     SMCWorkerV2.verify():                                    ││
│     Build score_tokens = [anchor_token | draft_tokens]       ││
│     Build causal mask, target model forward → logits         ││
│     SMCScoreInput.sample(): accept ALL γ tokens, compute     ││
│     logprob_diff = target_logprob - draft_logprob            ││
│     Sample bonus token from target distribution              ││
│     Output: accept_lens, logprob_diffs, next_last_token_ids  ││
│                                                               ││
│  4b. DRAFT-EXTEND  (inherited from EAGLE)                    ││
│     draft_worker._draft_extend_for_decode():                 ││
│     Fills draft-model KV for newly verified tokens           ││
└──────────────────────────┬──────────────────────────────────┘│
                           ▼                                   │
┌─────────────────────────────────────────────────────────────┐│
│  5. WEIGHT UPDATE  (smc_scheduler.on_batch_done)             ││
│     _update_group() [100% CPU, no GPU ops]:                  ││
│       pending_diffs.append(logprob_diff)  [deferred]         ││
│       step_counts[particle_idx] += 1                         ││
│     If all active particles aligned (same step_count):       ││
│       mark group for resample (unconditionally)              ││
└──────────────────────────┬──────────────────────────────────┘│
                           ▼                                   │
┌─────────────────────────────────────────────────────────────┐│
│  6. RESAMPLE CHECK  (_launch_pending_resamples, deferred)     ││
│     flush_pending_diffs() -> GPU log_weight update           ││
│     ESS = 1 / Σ(softmax(log_weights)²)                      ││
│     If ESS < N × threshold → stall group, resample async    ││
│     Else → no-op (weights already flushed)                   ││
└──────────────────────────┬──────────────────────────────────┘│
                           ▼                                   │
┌─────────────────────────────────────────────────────────────┐│
│  7. TERMINATION CHECK                                        ││
│     Particle hits EOS/max_tokens → snapshot to finished      ││
│     (on_particle_finished → SMCFinishedParticleSnapshot)     ││
│     All finished → FINALIZATION                              ││
│     Else → loop back to DRAFT PHASE ────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  8. FINALIZATION  (smc_manager._finalize_group)              │
│     flush_pending_diffs() (ensure weights are current)       │
│     Select best particle: max(log_weight, len(output_ids))   │
│     Copy best_output_ids → parent_req.output_ids             │
│     Release all particle KV cache & pool slots               │
│     Return parent req to normal output path                  │
└─────────────────────────────────────────────────────────────┘
```

## Two-Bucket Resample Stream

Resampling mutates `req_to_token` rows and shared allocator state (`slot_ref_count`, `free_pages`) on the GPU. To avoid race conditions with the forward stream, `SMCScheduler` uses a **two-bucket model** with a dedicated resample CUDA stream.

```
┌─────────────────────┐      resample completes       ┌─────────────────────┐
│  Bucket A            │  ──────────────────────────►  │  Bucket B            │
│  Waiting for Resample│                               │  Ready for Advance   │
│                      │  ◄──────────────────────────  │                      │
│  resampling_reqs:    │      ESS drops, needs         │  Lives inside        │
│  Dict[group_id,      │      resample                 │  scheduler.running_  │
│       List[Req]]     │                               │  batch               │
└─────────────────────┘                                └─────────────────────┘
        │                                                       │
        │  resample_stream:                                     │  forward_stream:
        │  req_to_token row copies                              │  forward pass
        │  (concurrent with forward)                            │
```

**Event loop integration** — three call sites in `scheduler.py`:

1. **`smc_scheduler.step()`** — called after batch completion: syncs finished resamples, launches new ones
2. **`smc_scheduler.step_before_forward()`** — called before `get_next_batch_to_run()`: ensures completed resamples are merged back before batch selection
3. **`smc_scheduler.step_after_forward()`** — called after `run_batch()`: currently a **no-op** placeholder. All resample work (sync, launch, drain) runs in `step_before_forward()` to guarantee stalled groups leave `running_batch` before batch selection

**Inside the step methods**:
- **`_sync_completed_resamples()`** — polls `PendingResample.done_event.query()`. If ready: apply deferred `inc_ref`/`dec_ref` on schedule stream, copy CPU state (output_ids, kv_committed_len), move reqs Bucket A → B via `merge_batch`
- **`_launch_pending_resamples()`** — for groups flagged by `on_batch_done()`: compute ancestor indices (CPU), stall group B → A via `filter_batch`, copy `req_to_token` rows on `resample_stream`, record done event, defer refcount ops

**`PendingResample`** tracks in-flight resamples:
```
group_id, dst_reqs, src_snapshots, inc_ref (tensors), dec_ref (tensors), done_event
```

**Stream safety**: stalled rows are not in `running_batch`, so `forward_stream` never reads them. Refcount ops happen on `schedule_stream` after event wait. `forward_stream.wait_stream(schedule_stream)` ensures visibility before next forward.

## Scheduler Integration

- **SMC early pop_and_process** — in the overlap event loop, if `last_batch` was an SMC decode batch and the result queue is non-empty, `pop_and_process` runs *before* `step_before_forward()` to ensure weight diffs are available for `_launch_pending_resamples`
- **Admission control** — `SMCScheduler.should_delay_admission(running_req_count, group_size)` prevents new groups when too many particles are stalled
- **Queue reordering** — `_apply_smc_waiting_queue_policy()` moves lagged groups forward to maintain balanced stepping
- **Batch formation** — `SMCGroupSpan` tracks contiguous particle ranges; `build_smc_group_spans(reqs)` identifies them for group-level operations (weight updates, resample checks)
- **Idle detection** — scheduler is only idle when `smc_manager.has_active_groups()` is false
- **Group alignment** — `SMCGroupState.all_active_aligned()` checks all active particles share the same step count before resampling

## KV Cache Management

SGLang maps `req_to_token[req_pool_idx, position] → slot_index`. Slots are reference-counted for sharing.

1. **Initialization**: `copy_block_table()` clones parent's slot indices into each particle's row + `inc_ref`. All N particles share the **same physical KV** for the prompt prefix (no data copy). Each particle's `prefix_indices` and `cache_protected_len` are set to the shared prefix length. Parent row released via `dec_ref`.
2. **Draft phase**: Pre-allocates γ slots per particle via Triton kernel `assign_draft_cache_locs_page_size_1`
3. **Score phase**: Allocates score-token KV slots via `assign_extend_cache_locs_func`
4. **Resampling**: Same as init — `copy_block_table()` from ancestor to evicted particle + `inc_ref`/`dec_ref` (runs on `resample_stream`)
5. **Memory budget**: `max_num_reqs *= (2 * smc_n_particles + 1)` to accommodate all internal particles (`model_runner_kv_cache_mixin.py:378`)

## Weight Mathematics

```
Per particle per step:
  draft_logprob  = Σᵢ log P_draft(tᵢ | t<ᵢ)       [accumulated over γ tokens]
  target_logprob = Σᵢ log_softmax(logits_i / T)[tᵢ]  [T applied inside log_softmax]
  logprob_diff = target_logprob - draft_logprob       [importance weight update]
  log_weight += logprob_diff                           [accumulated across steps]

Resampling:
  weights = softmax(log_weights)
  ESS = 1 / Σ(wᵢ²)
  Resample when ESS < N × threshold
```

**Deferred weight updates**: `SMCGroupState.pending_diffs` accumulates `(particle_idx, diff)` tuples. `flush_pending_diffs()` applies them atomically when the group is aligned.

**Partially-finished particles**: finished particles are excluded from draft/score batches and snapshotted via `SMCFinishedParticleSnapshot`. Their weights freeze (0 - 0 = 0 importance each step), so active particles that score well naturally dominate. Finished particles can still be duplicated or eliminated during resampling.

**Resampling algorithms**:
- **Systematic** (default): evenly-spaced CDF positions with random offset — low variance
- **Multinomial**: `torch.multinomial()` — higher variance, unbiased

Both produce ancestor indices. When `ancestor[i] != i`, particle `i` is evicted and replaced with a copy of `ancestor[i]` (KV + CPU state). Weights reset to 0 after resampling.

## CUDA Graph Optimization

`SMCDraftCudaGraphRunner` captures all γ decode steps into a single CUDA graph:
- **Input buffers** (`SMCDraftInputBuffers`): `input_ids[bs]`, `req_pool_indices[bs]`, `seq_lens[bs]`, `out_cache_loc[bs*γ]`, `positions[bs]`, sampling params (temperatures, top_ps, top_ks, min_ps)
- **Output buffers**: `sampled_token_ids[γ, bs]`, `sampled_token_logprobs[γ, bs]`
- **Replay**: validates batch against `SMCDraftSamplingSignature` (tracks greedy/top_p/top_k/min_p), re-captures on mismatch
- **Draft-extend CUDA graph**: `EAGLEDraftExtendCudaGraphRunner` captures the post-verify KV fill step for the draft model
- **Constraints**: `page_size == 1`, no LoRA, no hybrid SWA, no MRope, no pipeline parallelism

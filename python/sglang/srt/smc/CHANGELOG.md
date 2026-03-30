# SMC Refactor: Clean Package with Bonus Tokens

## Motivation

The original SMC implementation lived inside `speculative/` and inherited heavily
from the EAGLE speculative decoding framework. This created:

- **Unnecessary complexity**: `accept_index`, `fill_new_verified_id` Triton kernels,
  multi-step attention backends, `SMCScoreInput` with EAGLE-style `capture_hidden_mode=FULL`
- **Cross-stream dependencies**: `verify_done` events, forward-stream tensors leaking
  to the schedule stream
- **Metadata maintenance bugs**: `kv_committed_len` vs `seq_lens` offset dance,
  `compute_smc_shared_prefix_len` with `min(committed, visible - 1)`
- **CUDA graph recapture**: `SMCScoreInput` captured with `capture_hidden_mode=FULL`
  but runtime `SMCVerifyInput` used `NULL`, causing recapture on every decode step

## What Changed

### New `smc/` Package (replaces `speculative/smc_*`)

```
smc/
  __init__.py
  smc_info.py          SMCDraftInput (verified_id field), SMCVerifyInput
  smc_workers.py       SMCWorker (draft AR + target verify + bonus sampling)
  smc_manager.py       SMCManager (particle groups, finalization)
  smc_scheduler.py     SMCScheduler (resampling, ESS)
  smc_utils.py         Shared utilities (release, clone, resample, validate)
  smc_debug_utils.py   Diagnostic tracing
```

### Deleted Legacy Files

```
speculative/smc_info.py               (~650 lines, replaced by smc_info.py + smc_utils.py)
speculative/smc_worker_v2.py          (EAGLE-inherited worker, replaced by smc_workers.py)
speculative/smc_draft_cuda_graph_runner.py  (EAGLE multi-step draft graphs, not needed)
speculative/smc_manager.py            (moved to smc/)
speculative/smc_scheduler.py          (moved to smc/)
speculative/smc_debug_utils.py        (moved to smc/)
test/registered/unit/speculative/test_smc_worker_v2_current.py  (dead tests)
```

### Key Design Decisions

#### 1. Bonus Token via Extra Draft Step

```
x0 (anchor) --> draft r+1 AR steps --> x1...x_{r+1}  (x_{r+1} only for KV)
             --> verify [x0, x1, ..., x_r] on target  (r+1 tokens)
             --> sample bonus from tempered target logits at x_r position
             --> commit [x1, ..., x_r, bonus] = r+1 tokens
             --> verified_id = bonus --> next x0
```

The extra draft AR step (r+1 instead of r) ensures x_r's KV exists in the draft
cache, so the bonus token can anchor the next round without `_draft_extend_for_decode`.
This eliminates the EAGLE dependency entirely.

#### 2. Deterministic `seq_lens` Advancement

`SMCDraftInput.prepare_for_decode()` advances `batch.seq_lens` by `gamma+1` on the
schedule stream BEFORE the forward pass. Since SMC always accepts `gamma+1` tokens
(no rejection), this is safe and eliminates cross-stream `verify_done` dependencies.

Original seq_lens are saved as `_orig_seq_lens` for draft/verify to use.

#### 3. `kv_committed_len = seq_lens` (No Offset)

With bonus tokens, `kv_committed_len = visible_seq_len - 1` is a maintained invariant.
The anchor is always at a NEW uncommitted position. So:

- `compute_smc_shared_prefix_len()` simply returns `kv_committed_len`
- No `min(committed, visible - 1)` dance needed
- `kv_committed_len += r+1` = actual KV coverage per step

#### 4. Simple AR Draft Loop

Instead of EAGLE's multi-step attention backends and `SMCDraftCudaGraphRunner`,
the draft model uses standard decode forwards in a for-loop. CUDA graphs work
via `model_runner.graph_runner` (standard decode path). No custom graph capture.

#### 5. `SMCVerifyInput` with `capture_hidden_mode=NULL`

The new `SMCVerifyInput` uses `capture_hidden_mode=NULL` (SMC doesn't need hidden
states). The CUDA graph `get_spec_info()` in both `cuda_graph_runner.py` and
`model_runner.py` was updated to create `SMCVerifyInput` instead of the old
`SMCScoreInput(capture_hidden_mode=FULL)`, eliminating the recapture bug.

### Scheduler Wiring Changes

| File | Change |
|------|--------|
| `speculative/spec_info.py` | `is_smc()` routes to `smc.smc_workers.SMCWorker` |
| `schedule_batch.py` | `prepare_for_decode` routes `is_smc()` through `SMCDraftInput.prepare_for_decode()` |
| `scheduler.py` | SMC overlap path doesn't overwrite `seq_lens` (already advanced); imports from `smc/` |
| `scheduler_output_processor_mixin.py` | Uses `result.logprob_diff`; refreshes `seq_lens` from `kv_committed_len`; uses `verified_id` |
| `overlap_utils.py` | `verified_id` field access instead of `last_token_ids` |
| `utils.py` | `logprob_diff` field on `GenerationBatchResult` (replaces `smc_logprob_diffs`) |
| `cuda_graph_runner.py` | `get_spec_info()` creates `SMCVerifyInput` for SMC |
| `model_runner.py` | Same `SMCVerifyInput` fix for piecewise graph capture |

### Test Changes

- Removed 6 dead test classes (~1200 lines) for deleted code
- Deleted `test_smc_worker_v2_current.py`
- Updated imports: `speculative.smc_*` -> `smc.*`
- Updated field names: `last_token_ids` -> `verified_id`, `smc_logprob_diffs` -> `logprob_diff`
- 35 tests pass

### Verified Working

- 1 particle, normal scheduler: correct output matching vanilla baseline
- 4 particles, normal scheduler: correct output, 5 prompts
- 4 particles, overlap scheduler: correct output
- 8 particles, overlap scheduler (gsm8k): no CUDA graph recaptures
- CUDA graph TARGET_VERIFY: correct logits shape `(bs * (gamma+1), vocab)`

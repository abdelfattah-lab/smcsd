# SMC Speculative Decoding Engine — Core Logic

Status: reference
Owner: cc2869
Last verified commit: fdd4f3b5c
Last verified date: 2026-03-24
Related code:
- [smc_manager.py](/home/cc2869/repositories/sglang/python/sglang/srt/speculative/smc_manager.py)
- [smc_worker_v2.py](/home/cc2869/repositories/sglang/python/sglang/srt/speculative/smc_worker_v2.py)
- [smc_info.py](/home/cc2869/repositories/sglang/python/sglang/srt/speculative/smc_info.py)
Reality check:
- This file captures algorithmic SMC concepts and mental models.
- It does not define the exact current scheduler/runtime contract.

## Overview

The engine runs Sequential Monte Carlo (SMC) with importance resampling on top of
speculative decoding. A **draft model** proposes token continuations for multiple
**particles** (candidate sequences), and a **target model** scores them. Importance
weights accumulate over steps; when particle diversity collapses (low ESS), we
resample to refocus compute on promising sequences.

---

## Per-Request State (`SMCRequest`)

Each request maintains `n_particles` independent candidate sequences:

| Field | Shape / Type | Description |
|---|---|---|
| `particle_ids` | `List[List[int]]` (n_particles x variable) | Full token sequences per particle |
| `log_weights` | `np.ndarray` (n_particles,) | Cumulative log importance weights |
| `finished` | `List[bool]` (n_particles,) | Whether each particle has hit EOS / stop |
| `full_draft_logprobs` | `np.ndarray` (n_particles,) | **Scalar** — summed draft logprobs for current step |
| `target_logprobs` | `np.ndarray` (n_particles,) | **Scalar** — summed target logprobs for current step |
| `full_output_ids` | `List[List[int]]` (n_particles x variable) | Draft tokens generated this step |
| `tokens_generated` | `int` | Total tokens generated so far (across all steps) |

Key: both `full_draft_logprobs[i]` and `target_logprobs[i]` are **scalars** — the
sum of per-token logprobs over the continuation drafted in that step.

---

## Step Phases

Each SMC step consists of four phases. Phase D is deferred to overlap with the
next step's GPU work.

### Phase A — Draft Generation (GPU)

1. Collect **active** (non-finished) particles across all requests.
2. Batch all active particles into a single draft engine call.
3. Draft engine generates up to `gamma` tokens per particle with `return_logprob=True`.
4. Scatter results back per-request:
   - `full_output_ids[i]` = drafted token IDs for particle `i`
   - `full_draft_logprobs[i]` = `sum_logprobs(output_token_logprobs)` — a scalar
   - Check EOS: mark `finished[i] = True` if stop signal, EOS token, or fewer tokens than `step_max`

**Finished particles are excluded from the draft batch** — they produce no new
tokens, so their `full_output_ids` and `full_draft_logprobs` remain at default (empty / 0.0).

### Phase B — Prepare Scoring & Extend Particles (CPU)

1. **Before extending**, record `logprob_start_lens` from current particle lengths
   (so the target model knows where new tokens begin).
2. Extend all particles in-place: `particle_ids[i].extend(full_output_ids[i])`.
   - Finished particles extend with `[]` (no-op).

### Phase C — Target Scoring (GPU)

1. Batch all **active** particles into a single target engine call with
   `max_new_tokens=0` (prefill-only) and `return_logprob=True`.
2. Target model returns `input_token_logprobs` — logprobs for each token position.
3. Extract continuation logprobs (last `n_cont_tokens` entries) and sum them:
   - `target_logprobs[i]` = `sum_logprobs(cont_lps)` / `target_lhts_temperature` — a scalar

**Finished particles are excluded from the score batch** — their `target_logprobs`
remain 0.0.

### Phase D — Weight Update, Resample, Advance (CPU, deferred)

Deferred to run during the **next** step's Phase A GPU call, hiding CPU latency
behind GPU compute.

#### D.1 Weight Update
```
log_importance[i] = target_logprobs[i] - full_draft_logprobs[i]   # scalar
log_weights[i]   += log_importance[i]                              # accumulate
```

For finished particles: both sides are 0.0, so weights are **frozen** at their
last active value.

#### D.2 Resample (conditional)
- Compute `ESS = 1 / sum(normalized_weights^2)`.
- If `ESS < n_particles * resample_threshold` and request is not terminal → resample.
- Resampling methods: **systematic** (default) or **multinomial**.
- After resample: `log_weights` reset to zeros, `finished` status propagated via
  resampled indices (a finished particle can be duplicated or eliminated).

#### D.3 Advance Step
- `tokens_generated += step_generated`
- Check terminal: `all(finished)` or `tokens_generated >= max_tokens` → mark `done`.

---

## Handling Partially-Finished Particles

When some (but not all) particles in a request are finished:

1. **Active particles continue** — finished ones are excluded from both draft and
   target batches via `get_active_indices()`.
2. **Finished particle weights freeze** — `0.0 - 0.0 = 0.0` importance each step,
   so `log_weights` stays at its last value.
3. **Resampling still runs on all particles** — finished particles participate:
   - High-weight finished particle → may be duplicated (copies inherit `finished=True`),
     which can **accelerate** request completion.
   - Low-weight finished particle → may be eliminated, replaced by a copy of an
     active particle, which **delays** completion.
4. **Request terminates** when `all(finished)` or `tokens_generated >= max_tokens`.

### Weight staleness concern

Finished particles' weights are frozen while active particles' weights keep
accumulating. Over many steps, active particles that score well under the target
model will naturally dominate — finished particles become relatively downweighted.
This is generally correct behavior (prefer particles still generating quality text),
but means early-finishing particles rarely survive to be selected unless they had
very high weight when they stopped.

---

## Terminal Conditions

| Condition | Trigger | `finish_reason` |
|---|---|---|
| All particles finished | `all(finished) == True` | `"stop"` |
| Token budget exhausted | `tokens_generated >= max_tokens` | `"length"` |

## Final Output Selection

```python
best_idx = argmax(log_weights)
output = particle_ids[best_idx][prompt_len:]
```

The particle with the highest cumulative log importance weight is selected,
regardless of whether it is finished or still active.

---

## Key Config Parameters (`SMCConfig`)

| Parameter | Default | Description |
|---|---|---|
| `n_particles` | 4 | Number of candidate sequences per request |
| `gamma` | 8 | Max draft tokens per step |
| `draft_temperature` | 0.7 | Sampling temperature for draft model |
| `target_lhts_temperature` | 1.0 | Temperature for target model scoring during verification |
| `resample_threshold` | 0.5 | Resample when `ESS < n_particles * threshold` |
| `resample_method` | `"systematic"` | `"systematic"` or `"multinomial"` |

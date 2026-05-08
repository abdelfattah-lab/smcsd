# SMC vLLM Backend — Architecture

## 1. Overview

The vLLM backend runs SMC inference entirely in-process. The engine, scheduler,
worker, and model runner all live in the same Python process and communicate by
direct method calls.

SMCVLLMEngine talks to EngineCore and injects its own scheduler and gpu model runner.
This backend does not use vLLM's built-in speculative decoding pipeline. 

## 2. Main Components

### 2.1 High-Level Architecture Graph

```text
SMCVLLMEngine.generate()
    |
    v
EngineCore
    |
    v
SMCVLLMScheduler.schedule()
    |
    +--> Normal vLLM prefill path
    |       |
    |       v
    |   Base Scheduler Output
    |       |
    |       v
    |   GPUModelRunner via super().execute_model()
    |       |
    |       v
    |   Upstream sample_tokens()
    |
    +--> SMC draft path after fork
            |
            v
        NewParticleGroupData / Ongoing SMCGroupBatch
            |
            v
        UniProcExecutor / SMCGPUWorker
            |
            v
        SMCGPUModelRunner.execute_model()
            |
            +--> Draft prefill
            |
            +--> Draft decode cycles
            |       |
            |       v
            |   Upstream sampler component
            |
            +--> Target model + target KV cache
            |
            +--> Draft model + draft KV cache
            |
            +--> Runner-side particle rows / block tables / sampler state
            |
            v
        SMCModelRunnerOutput
            |
            v
SMCVLLMScheduler.update_from_output()
    |
    |
    v
EngineCore
```

- `SMCVLLMEngine` drives the normal `EngineCore.step()` loop.
- `SMCVLLMScheduler` lets ordinary prefill requests flow through the normal
  vLLM path.
- once a parent request is forked into particles, SMC-specific work is carried
  through `SMCSchedulerOutput` into `SMCGPUModelRunner`.
- the scheduler and runner then stay in sync through
  `schedule() -> execute_model() -> update_from_output()`.


## 3. Files

| File | Class | Role |
|---|---|---|
| `smcsd/vllm_backend/engine.py` | `SMCVLLMEngine` | Public API; tokenization, request creation, `EngineCore.step()` loop |
| `smcsd/vllm_backend/scheduler.py` | `SMCVLLMScheduler` | Forks parent requests into particle groups, tracks SMC state, consumes custom runner output |
| `smcsd/vllm_backend/worker.py` | `SMCGPUWorker` | Swaps in `SMCGPUModelRunner` |
| `smcsd/vllm_backend/model_runner.py` | `SMCGPUModelRunner` | Draft model loading, draft KV cache, batched draft prefill, batched draft decode |

## 4. End-to-End Control Flow

### 4.1 Parent request path

Each prompt starts as an ordinary vLLM request:

1. `SMCVLLMEngine.generate()` builds `Request(...)`
2. `EngineCore.step()` runs normal vLLM schedule/execute/sample for prefill

At this point:

- prompt KV is already materialized
- the first sampled token ID exists
- the sampled token's KV has not yet been written into cache

### 4.2 Fork into particles

On the next scheduler step, `SMCVLLMScheduler.schedule()` detects a request whose
`output_token_ids` has length 1 and which has not yet been converted into an SMC
group.

It then:

1. removes the parent request from `self.running`
2. allocates particle request IDs
3. calls `kv_cache_manager.fork_blocks(...)`
4. shares only full prefix blocks across particles
5. gives each particle private decode blocks
6. stores `SMCParticleGroupState`
7. returns `NewParticleGroupData` inside `SMCSchedulerOutput`

## 5. GPU Runner Flow

### 5.1 `SMCGPUModelRunner.execute_model()`

`SMCGPUModelRunner.execute_model()` first calls:

```python
base_output = super().execute_model(...)
```
This preserves the normal vLLM path for prefill requests.

Then it handles SMC work:

- for each `new_particle_group`:
  - `add_particle_group(...)`
  - `_run_draft_and_verify(new_group)`
- for each `ongoing_smc_group`:
  - `_run_draft_and_verify(batch)`

### 5.2 `add_particle_group(...)`

This method:

1. adds each particle request to `req_states`
2. writes each particle's block table
3. applies staged writes
4. registers each particle with the upstream sampler state
5. runs batched draft prefill

## 6. Batched Draft Prefill

Draft prefill is batched across all particles and uses upstream vLLM machinery:

1. gather particle block tables with `BlockTables.gather_block_tables(...)`
2. compute slot mappings with `BlockTables.compute_slot_mappings(...)`
3. build attention metadata with `model_state.prepare_attn(...)`
4. run one batched draft-model forward pass over the prompt

## 7. Batched Draft Decode

`smc_draft_cycle()` runs `gamma + 1` autoregressive draft steps over all particles.

For each draft step:

1. build one real `InputBatch` for the particle batch
2. compute slot mappings with `BlockTables.compute_slot_mappings(...)`
3. build attention metadata from the same `InputBatch`
4. run the draft model forward pass
5. compute draft logits
6. reuse the upstream sampler component:
   - `self.sampler(logits, input_batch)`
7. call upstream `post_update(...)` so sampler penalty state and request token
   state stay in sync across draft steps


## 8. Sampling 

There are two different sampling paths in the current backend:

### Normal vLLM sampling

Used for prefill requests before fork:

- driven by upstream `sample_tokens()`
- owned by the normal `EngineCore.step()` lifecycle

### SMC draft sampling

Used after the parent request is forked into particles.

- driven inside `SMCGPUModelRunner.smc_draft_cycle()`
- reuses upstream sampler components (`self.sampler(...)`)
- does not use the full normal `sample_tokens()` lifecycle

## 9. `update_from_output()`

`SMCVLLMScheduler.update_from_output(...)` does two things:

1. calls `super().update_from_output(...)` to let normal vLLM update ordinary
   requests
2. processes `model_runner_output.smc_draft_results`

For each SMC group it:

- appends newly committed draft tokens to each particle's accumulated output
- updates `seed_token_ids`
- advances `num_computed_tokens`
- or finishes the group and emits a final `EngineCoreOutput` for the parent request

## 10. State Synchronization

The backend intentionally keeps some related state in both the scheduler and the
GPU model runner. They stay in sync through the normal engine step protocol:

1. `scheduler.schedule()`
2. `model_runner.execute_model(scheduler_output)`
3. `scheduler.update_from_output(scheduler_output, model_runner_output)`

### 10.1 What lives on the scheduler side

The scheduler owns the high-level logical SMC state:

- which parent requests have become SMC groups
- `SMCParticleGroupState`
  - `parent_req`
  - `n_particles`
  - `num_computed_tokens`
  - `seed_token_ids`
  - `shared_prefix_block_ids`
  - `decode_block_ids`
  - `particle_req_ids`
  - `accumulated_tokens`
- KV allocation / ownership through `kv_cache_manager`
- final per-group completion bookkeeping in `_completed_groups`

This is the source of truth for:

- logical sequence progress
- logical block ownership
- per-particle accumulated output tokens

### 10.2 What lives on the runner side

The runner owns the GPU-execution-facing state:

- `req_states`
  - request rows
  - `all_token_ids`
  - `num_computed_tokens`
  - `last_sampled_tokens`
  - etc.
- `block_tables`
  - dense block-table rows indexed by runner request row
- sampler state
  - temperature / top-k / top-p
  - penalty state
  - bad-words / logit-bias state
- `particle_groups`
  - mapping from `group_id` to runner-local particle rows and req ids
- physical draft KV cache tensors
- physical target KV cache tensors

This is the source of truth for:

- GPU row indices
- actual gathered block tables used by kernels
- per-request sampler state on device
- physical KV contents

### 10.3 What is duplicated logically

The same logical request/group is represented in two places:

- Scheduler knows a group by `group_id` and `particle_req_ids`
- Runner knows the same group by `group_id`, `particle_req_ids`, and
  `particle_rows`

The same logical decode progress also appears in two forms:

- Scheduler tracks `num_computed_tokens` and `seed_token_ids`
- Runner tracks request-row state in `req_states.num_computed_tokens`,
  `req_states.all_token_ids`, and sampler penalty history

The same logical KV layout also appears in two forms:

- Scheduler tracks block ids as Python lists/tuples in
  `shared_prefix_block_ids` and `decode_block_ids`
- Runner tracks the same layout in GPU `block_tables` rows

# SMC vLLM Backend — Architecture

## 1. Overview

The vLLM backend runs SMC inference entirely in-process — no subprocesses, no ZMQ.
All components (engine, scheduler, model runner) live in the same Python process and
communicate via direct method calls.

This contrasts with the sglang backend, where `SMCEngine` launches the scheduler in a
separate subprocess and communicates over ZMQ sockets.

---

## 2. Component Hierarchy

```
SMCVLLMEngine.generate()
  │
  │  builds Request objects, calls self._engine.add_request(request)
  │  then loops self._engine.step() until all requests finish
  ▼
EngineCore                              ← same process, same thread
  │
  ├── self.scheduler                    ← SMCVLLMScheduler instance, same process
  │     schedule() → SMCSchedulerOutput
  │
  └── self.model_executor               ← UniProcExecutor, same process
        execute_model(scheduler_output)
          └── WorkerWrapperBase
                └── SMCGPUWorker        ← same process (UniProc = no subprocess)
                      └── SMCGPUModelRunner
                            execute_model() → runs draft cycle on GPU
```

Each `EngineCore.step()` call does one full iteration:
1. `scheduler.schedule()` → produces `SMCSchedulerOutput`
2. `model_executor.execute_model(scheduler_output)` → GPU forward passes
3. `scheduler.update_from_output(scheduler_output, model_output)` → returns
   `EngineCoreOutputs` with `new_token_ids` per request

---

## 3. Files

| File | Class | Role |
|---|---|---|
| `smcsd/vllm_backend/engine.py` | `SMCVLLMEngine` | Public entry point; tokenization, request construction, step loop |
| `smcsd/vllm_backend/scheduler.py` | `SMCVLLMScheduler` | Extends vllm `Scheduler`; KV block allocation, particle group lifecycle |
| `smcsd/vllm_backend/worker.py` | `SMCGPUWorker` | Extends `GPUWorker`; swaps in `SMCGPUModelRunner` at `init_device` |
| `smcsd/vllm_backend/model_runner.py` | `SMCGPUModelRunner` | Extends v2 `GPUModelRunner`; particle row management, draft cycle |

---

## 4. Injection Points

vLLM selects the scheduler and worker class via two fields on `VllmConfig`.
`SMCVLLMEngine.__init__` sets both after `EngineArgs.create_engine_config()`:

```python
# Scheduler: EngineCore reads this via scheduler_config.get_scheduler_cls()
vllm_config.scheduler_config.scheduler_cls = SMCVLLMScheduler

# Worker: WorkerWrapperBase.init_worker() resolves this string to a class
vllm_config.parallel_config.worker_cls = "smcsd.vllm_backend.worker.SMCGPUWorker"
```

There is no `model_runner_cls` config field in vLLM. Runner injection is done by
`SMCGPUWorker.init_device()`, which calls `super().init_device()` then replaces
`self.model_runner` with `SMCGPUModelRunner`.

---

## 5. SMC Parameters

SMC hyper-parameters (`n_particles`, `gamma`, `temperature`) are server-level settings,
not per-request. They are passed to `EngineArgs` as a `speculative_config` dict and
land in `vllm_config.speculative_config` (`SpeculativeConfig`):

| Field | Source |
|---|---|
| `smc_n_particles` | `SpeculativeConfig.smc_n_particles` |
| `smc_gamma` | `SpeculativeConfig.num_speculative_tokens` |
| `smc_temperature` | `SpeculativeConfig.smc_temperature` |

`SMCVLLMScheduler.__init__` reads these from `self.vllm_config.speculative_config`.

---

## 6. Request Flow

### Prefill (no SMC-specific changes)

Prefill requests flow through vLLM's base scheduler and base `execute_model`
implementation unchanged.

### Fork (cycle N+1 after prefill completes)

```
Cycle N:   parent request finishes prefill → output_token_ids has 1 token

Cycle N+1:
  SMCVLLMScheduler.schedule()
    → detects: len(req.output_token_ids) == 1 and req.request_id not in smc_groups
    → removes req from self.running
    → calls fork_to_particles():
        kv_cache_manager.fork_blocks(parent_req_id, particle_req_ids, n_decode_blocks)
          prefix blocks: ref_cnt 1 → N (parent relinquishes ownership)
          each particle registered with prefix_blocks + fresh decode_blocks
        builds NewParticleGroupData (block IDs, kv_slots, seed_token_ids, seq_lens)
        stores SMCParticleGroupState in self.smc_groups
    → returns SMCSchedulerOutput(new_particle_groups=[...], ongoing_smc_groups=[])

  SMCGPUModelRunner.execute_model()
    → super().execute_model()              # normal prefill/decode requests
    → for new_group in new_particle_groups:
          add_particle_group(...)          # assigns particle rows in RequestState + BlockTables
          _run_draft_and_verify(new_group) # first draft cycle runs immediately
```

### Ongoing SMC decode

```
Cycle N+K (particles already registered):
  SMCVLLMScheduler.schedule()
    → returns SMCSchedulerOutput(new_particle_groups=[], ongoing_smc_groups=[SMCGroupBatch(...)])

  SMCGPUModelRunner.execute_model()
    → super().execute_model()
    → for batch in ongoing_smc_groups:
          _run_draft_and_verify(batch)     # looks up particle_rows locally, runs draft cycle
```

### Draft cycle (`smc_draft_cycle`)

Runs γ+1 autoregressive decode steps over all N particles in a single loop:

```
block_tables = _gather_block_tables(particle_rows)   # once before loop

for step in 0 .. gamma:
    input_ids  = draft_token_ids[:, step]            # [N]
    positions  = seq_lens_cur                        # [N]
    slot_ids   = kv_slots[:, :, step]               # [num_kv_groups, N]

    attn_metadata        = _build_draft_attn_metadata(block_tables, seq_lens_cur+1, slot_ids)
    slot_mappings_by_layer = build_slot_mappings_by_layer(slot_ids, kv_cache_config)

    with set_forward_context(...):
        hidden_states = model(input_ids, positions)  # [N, hidden_dim]

    logits     = model.compute_logits(hidden_states) # [N, vocab_size]
    log_probs  = log_softmax(logits / temperature)   # [N, vocab_size]
    next_token = multinomial(log_probs.exp())        # [N]

    seq_lens_cur += 1

returns draft_token_ids [N, γ+1], draft_log_probs [N, γ+1, V]
```

---

## 7. Out of Scope (this doc)

- Target scoring (`smc_target_score`) — batched target model pass over draft tokens
- Resampling (`process_decode_result`) — importance weights, particle resampling
- Ongoing SMC group population in `schedule()` — depends on resampling output

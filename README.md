# SGLang — SMC Speculative Decoding Branch (`smc_v0`)

This branch implements **Sequential Monte Carlo (SMC) speculative decoding** in SGLang. SMC runs N particles (parallel generation paths) per request: a lightweight draft model proposes tokens, the target model scores them, and particles are resampled by importance weight so compute focuses on the most promising paths.
<img width="772" height="424" alt="image" src="https://github.com/user-attachments/assets/3cda3320-e257-4079-99b3-93e3a7bec627" />

## Installation

```bash
uv venv --python 3.12
uv pip install -e "python"
```

## Quick Start

```python
import sglang as sgl

engine = sgl.Engine(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    speculative_algorithm="SMC",
    speculative_draft_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    smc_n_particles=4,
    smc_gamma=4,
    page_size=1,
)

prompts = [
    "The capital of France is",
    "In one short paragraph, explain speculative decoding.",
]
sampling_params = {"temperature": 0, "max_new_tokens": 32}

outputs = engine.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"{prompt} {output['text']}")

engine.shutdown()
```

See [`scripts/smc/`](scripts/smc/) for experiment scripts including GSM8K accuracy benchmarks, quality checks, and profiling.

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `--speculative-algorithm SMC` | — | Enable SMC speculative decoding |
| `--speculative-draft-model-path` | — | Path to draft model (required) |
| `--smc-n-particles` | 4 | Particles per request |
| `--smc-gamma` | 4 | Max draft tokens per step |
| `--smc-draft-temperature` | 0.7 | Draft model sampling temperature |
| `--smc-target-temperature` | 1.0 | Target model scoring temperature |
| `--smc-resample-threshold` | 0.5 | ESS ratio that triggers resampling |
| `--smc-resample-method` | systematic | `systematic` or `multinomial` |
| `--page-size` | — | Must be set to `1` for SMC |

## Architecture

See [`smc_design_docs/smc_architecture_overview.md`](smc_design_docs/smc_architecture_overview.md) for the full architecture walkthrough including data flow diagrams, phase descriptions, scheduler integration, KV cache management, and weight mathematics.

### Key Files

All paths relative to `python/sglang/srt/`.

| File | Role |
|---|---|
| `speculative/smc_info.py` | Core data structures, input/output types, resampling algorithms |
| `speculative/smc_worker_v2.py` | Draft & score phase orchestration |
| `speculative/smc_scheduler.py` | Resampling decisions, admission control, stall management |
| `speculative/smc_manager.py` | Group lifecycle: creation, tracking, finalization |
| `speculative/smc_draft_cuda_graph_runner.py` | CUDA graph capture for multi-step draft |
| `managers/scheduler.py` | Main scheduler hooks for SMC integration |
| `managers/scheduler_output_processor_mixin.py` | Post-prefill init, decode result processing |
| `managers/schedule_batch.py` | `SMCGroupSpan`, per-request SMC fields |

### Data Flow Summary

```
User Request
  → Prefill (target model, normal path)
  → Group Creation (spawn N particles, copy parent KV)
  → [Loop]:
      → Draft Phase (draft model generates γ tokens per particle)
      → Score Phase (target model scores proposals, compute importance weights)
      → Weight Update (accumulate log_weight += target_logprob - draft_logprob)
      → Resample Check (if ESS < N × threshold, resample particles + KV cache)
      → Termination Check (all particles finished?)
  → Finalization (select best particle by log_weight, return output)
```

## Design Docs

See **[`smc_design_docs/smc_architecture_overview.md`](smc_design_docs/smc_architecture_overview.md)** for the full architecture reference — covers data flow, two-bucket resample stream, scheduler integration, KV cache management, weight mathematics, and CUDA graph optimization.

## Base Project

This branch is based on [SGLang](https://github.com/sgl-project/sglang), a high-performance serving framework for large language models. See the upstream `main` branch for the full project README.

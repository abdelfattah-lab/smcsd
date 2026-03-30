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

### Ping-Pong Overlap Scheduler

The ping-pong scheduler double-buffers SMC work across two CUDA streams: while one batch runs forward passes on the GPU, the other batch does CPU-side resampling and scheduling on a separate stream. This hides resampling latency and improves throughput for multi-group workloads.

| Flag | Default | Purpose |
|---|---|---|
| `--smc-pingpong-overlap` | `False` | Enable ping-pong double-buffer scheduler |
| `--smc-resampling-overlap` | `False` | Enable single-stream overlap scheduler (implied by ping-pong) |

**Setting `--max-running-requests` for ping-pong:**

`--max-running-requests` is a **global limit shared across both ping-pong slots**. The scheduler checks `slot_0_particles + slot_1_particles <= max_running_requests` before admitting new groups. To have all your requests running concurrently:

```
max-running-requests = num_prompts × smc_n_particles
```

For example, 16 prompts with 8 particles each requires `--max-running-requests 128`. The scheduler load-balances groups across the two slots (~64 particles each), but the limit applies to the combined total.

### CUDA Graph Sizing

CUDA graphs pre-capture GPU kernels at specific batch sizes to avoid launch overhead. For SMC workloads, `--cuda-graph-max-bs` must cover your maximum batch size, and the system auto-generates capture sizes up to that value.

| Flag | Default | Purpose |
|---|---|---|
| `--cuda-graph-max-bs` | auto | Maximum batch size for CUDA graph capture |
| `--cuda-graph-bs` | auto | Manually specify the list of capture batch sizes |
| `--disable-cuda-graph` | `False` | Disable CUDA graph entirely |

**How to set `--cuda-graph-max-bs`:**

The CUDA graph max batch size must be at least as large as the maximum number of tokens the model will process in a single forward pass. For SMC with ping-pong, each slot forwards independently, so the relevant batch size is **per-slot**, not the global total.

A safe setting is:

```
cuda-graph-max-bs >= max-running-requests
```

Note that larger values consume more GPU memory (roughly `cuda_graph_max_bs × 2 GB` reserved). If memory is tight, you can set it to match your expected per-slot batch size instead:

```
cuda-graph-max-bs >= max-running-requests / 2
```

**Example: 16 prompts, 8 particles, ping-pong enabled**

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm SMC \
    --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \
    --smc-n-particles 8 \
    --smc-gamma 4 \
    --smc-pingpong-overlap \
    --max-running-requests 128 \
    --cuda-graph-max-bs 128 \
    --page-size 1
```

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

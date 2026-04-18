# SMC Speculative Decoding

> **Warning:** This repository is under active development. APIs, configuration flags, and internal interfaces may go through breaking changes.

This repository implements **Sequential Monte Carlo Speculative Decoding (SMC-SD)** on top of [SGLang](https://github.com/sgl-project/sglang). SMC-SD is a population-based alternative to rejection-based speculative decoding: N particles maintain parallel generation paths, weighted by target/draft likelihood ratios, and resampled when effective sample size drops. All drafted tokens are accepted (no rejection), and throughput scales with batch size by increasing arithmetic intensity toward the GPU compute bound.

Paper: *Faster LLM Inference via Sequential Monte Carlo*

<img width="772" height="424" alt="image" src="https://github.com/user-attachments/assets/3cda3320-e257-4079-99b3-93e3a7bec627" />

## Installation

This repo vendors a patched SGLang as a git submodule at `3rdparty/sglang` (branch `smc_v2_clean`). Install both in editable mode.

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/abdelfattah-lab/smcsd.git
cd smcsd

# If you already cloned without --recurse-submodules, initialise now:
# git submodule update --init --recursive

# 2. Create a Python 3.12 environment
uv venv --python 3.12
source .venv/bin/activate

# 3. Install the patched SGLang (from the submodule), then this package
uv pip install -e 3rdparty/sglang/python
uv pip install -e .
```

Updating the vendored SGLang later:

```bash
git submodule update --remote 3rdparty/sglang    # pull latest smc_v2_clean
# then commit the bumped submodule pointer
```

## Quick Start

```bash
# SMC-SD throughput on ShareGPT (batch size 1)
python -O -m sglang.bench_offline_throughput \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm SMC \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 8 --smc-gamma 8 \
  --smc-draft-temperature 0.7 --smc-target-temperature 0.7 \
  --attention-backend fa3 \
  --mem-fraction-static 0.60 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --dataset-name sharegpt \
  --num-prompts 200
```

```bash
# SMC-SD accuracy on GSM8K via the dedicated SMCEngine (N=12 particles, gamma=8)
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --max-running-requests 24 \
  --cuda-graph-max-bs 24 \
  --smc-fast-resample \
  --num-questions 400
```

See [scripts/README.md](scripts/README.md) for more benchmark entrypoints.

## SMC-SD Parameters

| Parameter | Flag | Description |
| --- | --- | --- |
| Particles (N) | `--smc-n-particles` | Number of parallel generation paths per request |
| Gamma (K) | `--smc-gamma` | Draft tokens per speculative step |
| Draft temp | `--smc-draft-temperature` | Sampling temperature for draft model |
| Target temp | `--smc-target-temperature` | Scoring temperature for target model |
| Resample threshold | `--smc-resample-threshold` | Resample when ESS < N × threshold (0 = disable) |
| Resample method | `--smc-resample-method` | `systematic` or `multinomial` |

## Architecture

SMC lives in the top-level `smcsd/` package, layered over the patched SGLang via a handful of extension points (`ModelRunner._init_pools`, `ModelRunner._build_dummy_run_spec_info`, `ModelRunner._get_graph_runner_class`, `CudaGraphRunner.get_spec_info`, `Scheduler.init_tp_model_worker`, `TpModelWorker._init_model_runner`).

| Path | Description |
| --- | --- |
| `smcsd/engine.py` | `SMCEngine` — standalone offline engine (bypasses Tokenizer/Detokenizer managers) |
| `smcsd/v2/scheduler.py` | `SMCSchedulerV2` + `SMCCoordinatorV2` — slot-based decode loop and resampler |
| `smcsd/v2/worker.py` | `SMCWorkerV2` — draft AR loop + target scoring + importance weights |
| `smcsd/v2/req_state.py` | `ScheduleBatchSMC` — per-slot decode state |
| `smcsd/v2/stacked_state.py` | `StackedGroupState` — GPU-resident per-group tensors for the fast path |
| `smcsd/v2/info.py` | `SMCDraftInputV2`, `SMCDecodeContext` — spec-info wiring |
| `smcsd/v2/kernels/` | Fused Triton kernels (`fused_collect`, `fused_resample_kv`) |
| `smcsd/managers/smc_tp_worker.py` | `SMCTpModelWorker` — wires `SMCModelRunner` into the target TP worker |
| `smcsd/model_executor/smc_model_runner.py` | `SMCModelRunner` — installs refcounted allocator + SMC warmup spec-info |
| `smcsd/model_executor/smc_cuda_graph_runner.py` | `SMCCudaGraphRunner` — `SMCVerifyInput` during CUDA graph capture |
| `smcsd/mem_cache/allocator.py` | `SMCRefCountedTokenAllocator` + `copy_block_table` |
| `smcsd/common/verify.py` | `SMCVerifyInput` + Triton cache-assignment kernel |
| `smcsd/common/utils.py` | Particle cloning, weight normalization, ESS / resample helpers |

See [docs/smc/architecture.md](docs/smc/architecture.md) for the detailed design overview.

## Citation

```bibtex
@inproceedings{smcsd2026,
  title={Faster LLM Inference via Sequential Monte Carlo},
  author={Yahya Emara, Mauricio Barba da Costa, Chi-Chih Chang, Cameron Freer, Tim Vieira, Ryan Cotterell, Mohamed Abdelfattah},
  year={2026}
}
```

## Roadmap

- [ ] EAGLE support
- [ ] Async/Delayed resampling (CPU/GPU overlap for KV cache rewrites)
- [ ] Disaggregation (draft/target separation)

PRs welcome!

# SMC Speculative Decoding

This repository implements **Sequential Monte Carlo Speculative Decoding (SMC-SD)** on top of [SGLang](https://github.com/sgl-project/sglang). SMC-SD is a population-based alternative to rejection-based speculative decoding: N particles maintain parallel generation paths, weighted by target/draft likelihood ratios, and resampled when effective sample size drops. All drafted tokens are accepted (no rejection), and throughput scales with batch size by increasing arithmetic intensity toward the GPU compute bound.

Paper: *Accelerating LLM Inference with Sequential Monte Carlo*

## Installation

```bash
# Clone and install
git clone https://github.com/abdelfattah-lab/smc_sglang.git
cd smc_sglang
uv venv --python 3.12
uv pip install -e "python"
```

## Quick Start

```bash
# Single request, 8 particles, 8 draft tokens per step
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
  --random-input-len 256 --random-output-len 512 \
  --num-prompts 10

```

```bash
# SMC-SD accuracy (N=4 particles, gamma=4)
python scripts/smc/accuracy_test_gsm8k.py \
  --mode smc \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 400

### ShareGPT Throughput

```bash
# SMC-SD batch inference on ShareGPT
python -O -m sglang.bench_offline_throughput \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm SMC \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 8 --smc-gamma 16 \
  --smc-draft-temperature 0.7 --smc-target-temperature 0.7 \
  --attention-backend fa3 \
  --mem-fraction-static 0.55 \
  --max-running-requests 64 \
  --cuda-graph-max-bs 64 \
  --dataset-name sharegpt \
  --num-prompts 200
```

## SMC-SD Parameters


| Parameter          | Flag                       | Default    | Description                                     |
| ------------------ | -------------------------- | ---------- | ----------------------------------------------- |
| Particles          | `--smc-n-particles`        | 4          | Number of parallel generation paths per request |
| Gamma              | `--smc-gamma`              | 4          | Draft tokens per speculative step               |
| Draft temp         | `--smc-draft-temperature`  | 0.7        | Sampling temperature for draft model            |
| Target temp        | `--smc-target-temperature` | 1.0        | Scoring temperature for target model            |
| Resample threshold | `--smc-resample-threshold` | 0.5        | Resample when ESS < N * threshold (0 = disable) |
| Resample method    | `--smc-resample-method`    | systematic | `systematic` or `multinomial`                   |


## Architecture

The SMC implementation lives in `python/sglang/srt/smc/`:


| File               | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `smc_workers.py`   | Draft AR loop + target scoring + importance weight computation            |
| `smc_resampler.py` | ESS check, systematic resampling, KV cache copy via refcounting           |
| `smc_manager.py`   | Group lifecycle, particle creation, deferred weight updates, finalization |
| `smc_info.py`      | KV allocation, Triton cache assignment kernel, verify input preparation   |
| `smc_utils.py`     | Particle cloning, weight normalization, ESS computation                   |


See [docs/smc/architecture.md](docs/smc/architecture.md) for a detailed design overview.

## Citation

```bibtex
@inproceedings{smcsd2026,
  title={Accelerating LLM Inference with Sequential Monte Carlo},
  author={},
  year={2026}
}
```


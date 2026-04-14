# SMC Speculative Decoding

> **Warning:** This repository is under active development. APIs, configuration flags, and internal interfaces may go through breaking changes.

This repository implements **Sequential Monte Carlo Speculative Decoding (SMC-SD)** on top of [SGLang](https://github.com/sgl-project/sglang). SMC-SD is a population-based alternative to rejection-based speculative decoding: N particles maintain parallel generation paths, weighted by target/draft likelihood ratios, and resampled when effective sample size drops. All drafted tokens are accepted (no rejection), and throughput scales with batch size by increasing arithmetic intensity toward the GPU compute bound.

Paper: *Accelerating LLM Inference with Sequential Monte Carlo*

<img width="772" height="424" alt="image" src="https://github.com/user-attachments/assets/3cda3320-e257-4079-99b3-93e3a7bec627" />

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
# SMC-SD accuracy on GSM8K (N=12 particles, gamma=8)
python scripts/smc/accuracy_test_gsm8k.py \
  --mode smc \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --num-questions 400
```

### ShareGPT Throughput

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
  --max-running-requests 8 \
  --cuda-graph-max-bs 8 \
  --dataset-name sharegpt \
  --num-prompts 200
```

## SMC-SD Parameters


| Parameter          | Flag                       | Description                                     |
| ------------------ | -------------------------- | ----------------------------------------------- |
| Particles          | `--smc-n-particles`        | Number of parallel generation paths per request |
| Gamma              | `--smc-gamma`              | Draft tokens per speculative step               |
| Draft temp         | `--smc-draft-temperature`  | Sampling temperature for draft model            |
| Target temp        | `--smc-target-temperature` | Scoring temperature for target model            |
| Resample threshold | `--smc-resample-threshold` | Resample when ESS < N * threshold (0 = disable) |
| Resample method    | `--smc-resample-method`    | `systematic` or `multinomial`                   |


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


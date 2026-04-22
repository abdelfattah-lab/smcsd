# SMC Experiment Scripts

Ad hoc entrypoints for SMC (Sequential Monte Carlo) speculative decoding
experiments on top of the `smcsd/` implementation.

## Scripts

- **`accuracy_test_gsm8k.py`** — GSM8K accuracy benchmark. Supports
  `smc_engine` (dedicated offline SMCEngine), `native` (Python-level
  reference), and `baseline` (no speculative decoding).
- **`quick_quality_check.py`** — Quick output quality sanity check
  (vanilla vs SMC) on a handful of hardcoded prompts.
- **`smc_profile_engine.py`** — Offline profiler harness for SMC. Use
  `--engine-kind smc_engine` to target the dedicated ``SMCEngine`` path;
  emits Chrome-compatible traces.
- **`tps_benchmark_scripts/`** — Throughput sweeps (shell scripts)
  across (gamma, n) pairs and batch sizes. See
  `tps_benchmark_scripts/BENCHMARK_CONFIGS.md` for details.

## Reproducing GSM8K Accuracy

```bash
source .venv/bin/activate

# Dedicated SMCEngine (recommended) — 8 particles, gamma=8 draft tokens
python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine -N 8 -g 8 --num-questions 200

# Baseline (no speculative decoding) for comparison
python scripts/smc/accuracy_test_gsm8k.py --mode baseline --num-questions 200

# Native/external SMC (Python-level reference, slower but inspectable)
python scripts/smc/accuracy_test_gsm8k.py --mode native -N 8 -g 8 --num-questions 200

# Custom models (Llama 3.1-8B target + Llama 3.2-1B draft)
python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 8 --num-questions 200
```

Key flags for `accuracy_test_gsm8k.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `smc_engine` | `smc_engine` (dedicated SMCEngine), `native` (Python-level), or `baseline` |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Target model |
| `--draft-model` | `meta-llama/Llama-3.2-1B-Instruct` | Draft model for SMC modes |
| `-N` / `--particles` | `4` | Number of SMC particles |
| `-g` / `--gamma` | `4` | Draft tokens per step |
| `--temperature` | `0.7` | Draft temperature |
| `--num-questions` | `80` | Number of GSM8K test questions |
| `--max-new-tokens` | `512` | Max generation length |
| `--batch-size` | `1` | Batch size for engine mode |
| `--mem-fraction-static` | `0.4` | GPU memory fraction (engine modes) |
| `--seed` | `None` | NumPy seed for reproducibility (question order only — GPU sampling is not seeded) |

## Throughput Sweeps

See `tps_benchmark_scripts/` for shell-based sweeps across batch sizes
and (gamma, n) configurations. Sweep scripts emit timestamped CSVs with
columns `method,gamma,n,tps,b`.

## Quick Quality Check

```bash
source .venv/bin/activate

python scripts/smc/quick_quality_check.py --model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-path meta-llama/Llama-3.2-1B-Instruct --mode smc
```

## Profiling

```bash
source .venv/bin/activate

python scripts/smc/smc_profile_engine.py --engine-kind smc_engine \
    --output-dir /tmp/sglang-smc-profile
```

## Notes

- SMC runs through `SMCEngine` (dedicated offline path) + `SMCScheduler`
  (subclass of the base `Scheduler`). There is no "engine-level" SMC via
  the regular `sgl.Engine` factory anymore.
- `FLASHINFER_WORKSPACE_BASE=/tmp/<user>-flashinfer` is often needed on
  shared machines when running GPU-backed experiments.

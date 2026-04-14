# SMC Experiment Scripts

This directory contains ad hoc entrypoints for SMC (Sequential Monte Carlo) speculative decoding experiments.

## Scripts

- **`accuracy_test_gsm8k.py`** — GSM8K accuracy benchmark comparing engine-level SMC, native (Python-level) SMC, and vanilla baseline generation.
- **`quick_quality_check.py`** — Quick output quality sanity check (vanilla vs SMC) on a handful of hardcoded prompts.
- **`smc_profile_engine.py`** — Offline `sgl.Engine(...)` profiler harness for SMC scheduler variants. Emits Chrome-compatible traces.
- **`bench_tputs_smc.sh`** — Single-config SMC throughput benchmark (quick one-off runs).
- **`bench_tputs_standalone.sh`** — Single-config standalone speculative decoding throughput benchmark.
- **`smc_tputs_sweep.sh`** — Sweep SMC throughput across (gamma, n) pairs and batch sizes. Outputs timestamped CSV.
- **`sglang_tputs_sweep.sh`** — Sweep baseline throughput (plain sglang + standalone spec decoding) across batch sizes. Outputs timestamped CSV.

## Reproducing GSM8K Accuracy

```bash
source .venv/bin/activate

# Engine-level SMC (default) — 8 particles, 32 draft tokens per step
python scripts/smc/accuracy_test_gsm8k.py --mode smc -N 8 -g 32 --num-questions 200

# Baseline (no speculative decoding) for comparison
python scripts/smc/accuracy_test_gsm8k.py --mode baseline --num-questions 200

# Native/external SMC (Python-level reference, slower but inspectable)
python scripts/smc/accuracy_test_gsm8k.py --mode native -N 8 -g 32 --num-questions 200

# Full test set (1319 questions)
python scripts/smc/accuracy_test_gsm8k.py --mode smc -N 8 -g 32 --num-questions 1319

# Custom models (e.g., larger target with smaller draft)
## This is our main testing!
python scripts/smc/accuracy_test_gsm8k.py --mode smc \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 32 --num-questions 200
```

Key flags for `accuracy_test_gsm8k.py`:
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `smc` | `smc` (engine-level), `native` (Python-level), or `baseline` |
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Target model |
| `--draft-model` | same as `--model` | Draft model for SMC modes |
| `-N` / `--particles` | `4` | Number of SMC particles |
| `-g` / `--gamma` | `4` | Draft tokens per step |
| `--temperature` | `0.7` | Draft temperature |
| `--num-questions` | `20` | Number of GSM8K test questions |
| `--max-new-tokens` | `512` | Max generation length |
| `--batch-size` | `1` | Batch size for engine mode |
| `--mem-fraction-static` | `0.4` | GPU memory fraction (engine modes) |
| `--seed` | `None` | NumPy seed for reproducibility |

## Throughput Sweeps

The sweep scripts produce CSV files with columns: `method,gamma,n,tps,b` where `tps` is output token throughput (tok/s) and `b` is the number of prompts (batch size).

```bash
# SMC sweep — varies (gamma, n) pairs × batch sizes
bash scripts/smc/smc_tputs_sweep.sh                    # -> results_smc_<timestamp>.csv
bash scripts/smc/smc_tputs_sweep.sh my_results.csv     # custom output path

# Baseline sweep — plain sglang + standalone spec decoding × batch sizes
bash scripts/smc/sglang_tputs_sweep.sh                 # -> results_baseline_<timestamp>.csv
```

Default (gamma, n) pairs for SMC: `(8,8) (8,10) (10,6) (12,8) (16,8) (8,6) (8,4)`. Edit the `GAMMA_N_PAIRS` array in the script to customize.

Each run writes to a timestamped CSV so previous results are never overwritten. Errors are recorded as `ERROR` in the `tps` column.

## Quick Quality Check

```bash
source .venv/bin/activate

python scripts/smc/quick_quality_check.py                              # both vanilla and SMC
python scripts/smc/quick_quality_check.py --mode smc                   # SMC only
python scripts/smc/quick_quality_check.py --temperature 0.8            # custom temperature
```

## Profiling

```bash
source .venv/bin/activate

python scripts/smc/smc_profile_engine.py --output-dir /tmp/sglang-smc-profile
python scripts/smc/smc_profile_engine.py --profile-v2 --decode-only
```

## Notes

- `FLASHINFER_WORKSPACE_BASE=/tmp/cc2869-flashinfer` is often needed on this machine when running GPU-backed experiments.

# TPS Benchmark Configurations

## Common Parameters

| Parameter | Value |
|-----------|-------|
| Draft Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Memory Fraction | 0.60 |
| Input Length | 256 |
| Output Length | 512 |
| Temperature | 0.7 |
| Batch Sizes (sweep) | 1, 4, 8, 16 |
| Dataset | random |

## Benchmark Scripts

### SGLang STANDALONE Speculative Decoding (spec v2)

| Script | Target Model | Attention | TP | Env Vars | Method Label |
|--------|-------------|-----------|-----|----------|--------------|
| `sglang_sd_1b_8b_triton_v2.sh` | Llama-3.1-8B | triton | 1 | `SGLANG_ENABLE_SPEC_V2=True` | `standalone_triton_v2` |
| `sglang_sd_1b_8b_fa3_v2.sh` | Llama-3.1-8B | fa3 | 1 | `SGLANG_ENABLE_SPEC_V2=True` | `standalone_fa3_v2` |
| `sglang_sd_1b_70b_triton_v2.sh` | Llama-3.1-70B | triton | 4 | `SGLANG_ENABLE_SPEC_V2=True` | `standalone_triton_v2` |
| `sglang_sd_1b_70b_fa3_v2.sh` | Llama-3.1-70B | fa3 | 4 | `SGLANG_ENABLE_SPEC_V2=True` | `standalone_fa3_v2` |

### SMC

| Script | Target Model | Attention | TP | Method Label |
|--------|-------------|-----------|-----|--------------|
| `smc_1b_8b_triton.sh` | Llama-3.1-8B | triton | 1 | `smc_triton` |
| `smc_1b_8b_fa3.sh` | Llama-3.1-8B | fa3 | 1 | `smc_fa3` |
| `smc_1b_70b_triton.sh` | Llama-3.1-70B | triton | 4 | `smc_triton` |
| `smc_1b_70b_fa3.sh` | Llama-3.1-70B | fa3 | 4 | `smc_fa3` |

## STANDALONE Speculative Decoding Parameters

| Parameter | Value |
|-----------|-------|
| `--speculative-eagle-topk` | 1 |
| `--speculative-num-steps` | 4 |
| `--speculative-num-draft-tokens` | 5 |

## SMC Parameters

| Parameter | Value |
|-----------|-------|
| Draft Temperature | 0.7 |
| Target Temperature | 0.7 |
| `--max-running-requests` | `b * n` (computed) |
| `--cuda-graph-max-bs` | `b * n` (computed) |

## SMC (gamma, n) Sweep Pairs

| gamma | n |
|-------|---|
| 8 | 8 |
| 10 | 8 |
| 10 | 6 |
| 12 | 8 |
| 16 | 8 |
| 8 | 6 |
| 8 | 4 |
| 12 | 6 |

## Usage

### run_all.sh

Run all or a subset of benchmarks using group filters:

```bash
bash run_all.sh                  # run ALL 8 scripts
bash run_all.sh sglang           # only SGLang STANDALONE (4 scripts)
bash run_all.sh smc              # only SMC (4 scripts)
bash run_all.sh 8b               # only 8B target models (4 scripts)
bash run_all.sh 70b              # only 70B target models (4 scripts)
bash run_all.sh sglang 8b        # combine filters with AND logic (2 scripts)
```

Multiple filters use **AND logic** — a script must match all filters to be included.

Results are saved to a timestamped `results_YYYYMMDD_HHMMSS/` directory with individual CSVs per script and a merged `all_results.csv`.

#### Experiment Groups

| Group | Description | Scripts |
|-------|-------------|---------|
| `sglang` | SGLang STANDALONE (triton + fa3, spec v2) | 4 |
| `smc` | SMC (triton + fa3) | 4 |
| `8b` | 8B target model | 4 |
| `70b` | 70B target model | 4 |

### merge_results.sh

Merge experiment CSVs after the fact (standalone, does not re-run benchmarks):

```bash
bash merge_results.sh                          # auto-find latest results_* dir
bash merge_results.sh results_20260401_120000  # specify a results dir
bash merge_results.sh *.csv                    # specify individual CSV files
```

### collect_csvs.sh

Find and collect scattered CSV files from anywhere on disk into one folder, then merge:

```bash
bash collect_csvs.sh                           # search current dir recursively
bash collect_csvs.sh /path/to/search           # search a specific directory
bash collect_csvs.sh -o my_results /path       # custom output directory
```

Handles filename collisions by prepending the parent directory name. Skips already-merged `all_results.csv` files.

## CSV Output Format

```
method,gamma,n,tps,b
```

- **method**: benchmark label (e.g. `standalone_triton_v2`, `smc_triton`, `smc_fa3`)
- **gamma**: SMC gamma (0 for STANDALONE)
- **n**: SMC n_particles (1 for STANDALONE)
- **tps**: output token throughput (tokens/sec)
- **b**: batch size (num_prompts)

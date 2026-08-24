# SMC Experiment Scripts

Ad hoc entrypoints for SMC (Sequential Monte Carlo) speculative decoding
experiments on top of the standalone `smcsd/` implementation.

## Scripts

- **`accuracy_test_gsm8k.py`** — GSM8K accuracy benchmark (offline). Supports
  `smc_engine` (dedicated offline SMCEngine) and `baseline`
  (no speculative decoding).
- **`accuracy_test_gsm8k_http.py`** — GSM8K accuracy benchmark over an HTTP
  server (online serving). Launches the SMC HTTP server (`smcsd.http_server`)
  or connects to a running one (`--base-url`), then evals via concurrent native
  `/generate`. Same `#### <number>` scoring as the offline test, so results are
  directly comparable.
- **`quick_quality_check.py`** — Quick output quality sanity check
  (vanilla vs SMC) on a handful of hardcoded prompts.
- **`accuracy_test_olympiadbench.py`** — Long-form semantic-TTS eligibility
  pilot on the official text-only English OlympiadBench subset. Saves exact
  output token IDs for fixed-horizon prefix scoring.
- **`continue_olympiadbench_trajectories.py`** — Continues only length-capped
  pilot trajectories from their exact saved token IDs.
- **`offline_pointwise_error_audit.py`** — Scores semantic recoverability at
  fractional or fixed generator-token checkpoints without generator
  likelihoods.
- **`offline_semantic_hybrid_sim.py`** — Cross-validated offline simulation of
  the agreement-conditioned semantic expansion policy.
- **`offline_prefix_pairwise_verifier.py`** — Order-swapped relative verifier
  on correct/incorrect sibling prefixes at identical generator-token horizons.
  This is a diagnostic screen, not a deployable label-independent selector.
- **`offline_dual_semantic_audit.py`** — Fixed-token validity and
  progress-to-completion scoring with exact ordered-label logprobs. Supports
  multiple verifier models without exposing generator likelihoods.
- **`offline_semantic_ensemble.py`** — Problem-level out-of-fold pairwise
  fusion of multiple semantic models and rubrics, with clustered confidence
  intervals and a frozen gate verdict.
- **`online_semantic_particlescale.py`** — True branched semantic-only SMC or
  deterministic top-half/fork-two continuation from common exact-token
  prefixes. Records ancestry, ESS, diversity, model tokens, and accelerator
  cost without generator likelihoods.
- **`offline_semantic_method_comparison.py`** — Aligned problem bootstrap and
  quality/cost dominance report across self-consistency, terminal semantic
  selectors, semantic SMC, and deterministic semantic forking.
- **`online_likelihood_smc_olympiadbench.py`** — Optimized likelihood-only
  `SMCEngine` run on the frozen long-form slice. Preserves every final particle
  and reports posterior, majority, likelihood-weighted majority, maximum
  weight, oracle, returned tokens, and accelerator seconds.
- **`offline_likelihood_semantic_comparison.py`** — Frozen terminal L1+S beta
  sweep over a likelihood-SMC particle pool, with problem-paired bootstrap and
  explicit verifier GPU accounting.
- **`terminal_bench/terminal_semantic_smc.py`** — Live semantic-only SMC over
  coupled Terminal-Bench model/tool/environment particles. Supports score
  differences, beta, ESS resampling, unique-prefix verifier deduplication,
  `N` through 64, and one through eight independent rubric calls.
- **`terminal_bench/analyze_terminal_semantic_smc.py`** — Aggregates live
  sweeps into reward-isolated terminal ranking, selection, population, cost,
  deduplication, and latency tables.
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
python scripts/accuracy_test_gsm8k.py --mode smc_engine -N 12 -g 8 --num-questions 400

# Baseline (no speculative decoding) for comparison
python scripts/accuracy_test_gsm8k.py --mode baseline --num-questions 400

# Custom models (Llama 3.1-8B target + Llama 3.2-1B draft)
python scripts/accuracy_test_gsm8k.py --mode smc_engine \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 8 --num-questions 200
```

Key flags for `accuracy_test_gsm8k.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `smc_engine` | `smc_engine` (dedicated SMCEngine) or `baseline` |
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

## SMC Online Serving (HTTP)

SMC serves over HTTP via `smcsd/http_server.py`, which reuses sglang's standard
serving stack (TokenizerManager + DetokenizerManager + FastAPI) with the SMC
scheduler injected — no SMC source changes. Unlike the offline `SMCEngine`, the
HTTP server multiplexes concurrent requests.

```bash
source .venv/bin/activate
export FLASHINFER_WORKSPACE_BASE=/tmp/$USER-flashinfer   # shared-machine JIT cache

# Standard sglang endpoints become available (/generate, /v1/...)
python -m smcsd.http_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    -N 8 -g 8 --max-running-requests 16 --port 30000 --trust-remote-code
```

`--max-running-requests` is the number of concurrent SMC *groups*; it is expanded
by `(N+1)` internally (each group needs N+1 Req slots). Keep it modest relative to
`--mem-fraction-static` — CUDA-graph capture scales with the expanded value and can
OOM if set too high.

`--mem-fraction-static` defaults to `0.4` for SMC (not sglang's ~0.88): the server
runs two model runners (target + draft), each sizing its own KV-cache pool, so the
fraction is effectively counted twice and a larger value OOMs at draft KV-pool init
on a single GPU. Raise it only if you have headroom.

### GSM8K over HTTP

```bash
# Self-contained: launches the SMC server, evals, tears it down
python scripts/accuracy_test_gsm8k_http.py -N 8 -g 8 --num-questions 200 --parallel 16

# Against an already-running server (launched separately)
python scripts/accuracy_test_gsm8k_http.py --base-url http://127.0.0.1:30000

# Baseline (no spec) reference
python scripts/accuracy_test_gsm8k_http.py --mode baseline --num-questions 200
```

- Uses the zero-shot `#### <number>` format (no stop strings): SMC does not support
  stop strings, so sglang's few-shot `Question:/Answer:` harness can't drive it.
  Scoring matches `accuracy_test_gsm8k.py`, so HTTP and offline numbers are
  directly comparable (verified: offline 11/20 vs HTTP 9/20 on the same 20 GSM8K
  questions at temperature 0.7 — within sampling noise).
- SMC does not populate the EAGLE-style `avg_spec_accept_length` in `/server_info`,
  so accept length shows `n/a`; use output throughput as the speed metric.

## Throughput Sweeps

See `tps_benchmark_scripts/` for shell-based sweeps across batch sizes
and (gamma, n) configurations. Sweep scripts emit timestamped CSVs with
columns `method,gamma,n,tps,b`.

## Long-form semantic verifier pilot

The official symbolic-equivalence judge requires the SymPy-compatible ANTLR
runtime:

```bash
source .venv/bin/activate
uv pip install antlr4-python3-runtime==4.11.1
```

The frozen protocol and all model/split settings are recorded in
`configs/semantic/semantic_verifier_olympiadbench_pilot_v1.json`. The main
stages are generation, exact-token continuation for capped paths, pointwise
scoring at fixed 512/1,024/2,048-token horizons, relabeling those unchanged
prefix scores with the longer outcomes, and cross-validated hybrid simulation.
Use each script's `--help` for hardware-specific SGLang flags and output paths:

```bash
python scripts/accuracy_test_olympiadbench.py --help
python scripts/continue_olympiadbench_trajectories.py --help
python scripts/offline_pointwise_error_audit.py --help
python scripts/relabel_semantic_scores.py --help
python scripts/offline_semantic_hybrid_sim.py --help
python scripts/offline_prefix_pairwise_verifier.py --help
python scripts/offline_dual_semantic_audit.py --help
python scripts/offline_semantic_ensemble.py --help
python scripts/online_semantic_particlescale.py --help
python scripts/offline_semantic_method_comparison.py --help
python scripts/online_likelihood_smc_olympiadbench.py --help
python scripts/offline_likelihood_semantic_comparison.py --help
python scripts/terminal_bench/export_semantic_checkpoints.py --help
python scripts/terminal_bench/score_semantic_checkpoints.py --help
python scripts/terminal_bench/merge_semantic_score_shards.py --help
python scripts/terminal_bench/analyze_semantic_actionability.py --help
python scripts/terminal_bench/terminal_particle_backend.py --help
python scripts/terminal_bench/terminal_particle_controller.py --help
python scripts/terminal_bench/terminal_semantic_smc.py --help
python scripts/terminal_bench/analyze_terminal_semantic_smc.py --help
```

The completed OlympiadBench validity/progress screen uses Qwen3.8-27B and
Qwen3-32B. Its four-score ensemble failed the frozen 1,024-token actionability
gate (52.52% problem-balanced correct/incorrect sibling ranking, 95% CI
41.91--63.93%) and the Qwen3-32B progress allocation policy lost to the
verifier-free agreement control at matched compute. See
`docs/semantic_tts.md` for the full results and serving-cost decision.

The subsequent true online 50-problem experiment starts every method from the
same eight 2,048-token prefixes and samples every branched continuation online.
Semantic SMC reaches 46% accuracy at 46.77 active GPU-seconds/problem, versus
48% at 12.32 GPU-seconds/problem for self-consistency; deterministic semantic
forking reaches 40% at 35.27 GPU-seconds/problem. The frozen online gate fails.

The optimized likelihood follow-up also fails on this draft/target pair.
Qwen3.5-9B/2B likelihood SMC reaches 26% particle-majority accuracy at 13.38
active GPU-seconds/problem and has only 26% pool oracle. Adding one terminal
Qwen3.8-27B semantic factor at every frozen beta changes no answer and raises
cost to 18.53 GPU-seconds/problem. Self-consistency@8 remains best at 48% and
12.32 GPU-seconds/problem; proposal adequacy and ancestry survival are next.

## Quick Quality Check

```bash
source .venv/bin/activate

python scripts/quick_quality_check.py --model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-path meta-llama/Llama-3.2-1B-Instruct --mode smc
```

## Profiling

```bash
source .venv/bin/activate

python scripts/smc_profile_engine.py --engine-kind smc_engine \
    --output-dir /tmp/sglang-smc-profile
```

## Notes

- SMC runs through `SMCEngine` (dedicated offline path) or `smcsd.http_server`
  (online HTTP serving), both backed by `SMCScheduler` (subclass of the base
  `Scheduler`). There is no "engine-level" SMC via the regular `sgl.Engine`
  factory anymore.
- `FLASHINFER_WORKSPACE_BASE=/tmp/<user>-flashinfer` is often needed on
  shared machines when running GPU-backed experiments.

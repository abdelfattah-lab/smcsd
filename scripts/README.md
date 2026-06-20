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
- **`smc_profile_engine.py`** — Offline profiler harness for SMC. Use
  `--engine-kind smc_engine` to target the dedicated ``SMCEngine`` path;
  emits Chrome-compatible traces.
- **`tps_benchmark_scripts/`** — Throughput sweeps (shell scripts)
  across (gamma, n) pairs and batch sizes. See
  `tps_benchmark_scripts/BENCHMARK_CONFIGS.md` for details.
- **`collect_proposal_data.py`** / **`train_proposal.py`** — Proposal
  (draft-model) finetuning loop. See
  [Proposal Finetuning](#proposal-finetuning).

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

## Proposal Finetuning

SMC's efficiency is governed by the per-token KL between the draft proposal
`q_Td` and the tempered target `softmax(alpha * z_target / T_target)` — every
nat removed lowers importance-weight variance, raises ESS, and reduces
resampling (particle degeneracy). The loop is fully offline; the engine
needs no changes to consume a finetuned draft.

```bash
# 1. Collect rollouts on the GSM8K *train* split (eval uses test) with the
#    SMC config you deploy. Dumps every particle trajectory + log-weights +
#    cycle diagnostics as JSONL. SMC_TRACK_ESS=1 is set automatically.
python scripts/collect_proposal_data.py \
    -o /data/proposal_data/gsm8k_train_N8g8.jsonl \
    -N 8 -g 8 --num-prompts 2000 --batch-size 16

# 2. Finetune the draft. Default loss is token-level KL distillation against
#    the EXACT tempered target the engine weights against (temperatures and
#    power alpha are read from the dump's meta line); --loss wsft is a
#    cheaper posterior-weighted SFT baseline. Saves merged bf16 HF
#    checkpoints (epoch_K/ and final/).
#    NOTE: prefer the SMC-direct objective --loss renyi --renyi-beta 2
#    (log-chi^2) over reverse KL — it generalizes across domains and
#    recovers target-level accuracy; see the objective family below.
python scripts/train_proposal.py \
    --data /data/proposal_data/gsm8k_train_N8g8.jsonl \
    --output-dir /data/proposal_ckpts/llama1b-renyi \
    --loss renyi --renyi-beta 2.0 --renyi-beta-start 1.0 \
    --epochs 1 --batch-size 4 --grad-accum 8 --lr 1e-5

# 3. Evaluate: quality on GSM8K test + the proposal-health diagnostics.
python scripts/accuracy_test_gsm8k.py --mode smc_engine \
    --draft-model /data/proposal_ckpts/llama1b-kl/final \
    -N 8 -g 8 --num-questions 200
```

Per-request diagnostics ride the engine's particle side channel:
`smc_n_cycles`, `smc_n_resamples` (always on), and `smc_mean_ess`
(scheduler env `SMC_TRACK_ESS=1`). The payoff experiment is re-sweeping
gamma / N: a better proposal holds quality at higher gamma and lower N
(measured below: a χ²-finetuned 0.6B draft at N=4 beats the base draft at
N=8 on GSM8K accuracy *and* throughput).

> **Caveat — resample rate / ESS is a misleading proxy.** Reverse KL lowers
> rr / raises ESS the most, but by *mode-collapsing* the draft onto the
> target mode, which loses sample diversity and **hurts task accuracy**
> (held-out HumanEval −28pp in the Qwen3 study). The χ² (Rényi-2) objective
> is mass-covering: worse rr/ESS, but it recovers target-level accuracy and
> generalizes across domains. **Gate on task accuracy across domains, not
> rr/ESS.** See [docs/smc/proposal_objective.md](../docs/smc/proposal_objective.md)
> (objective + theory) and [docs/smc/proposal_results.md](../docs/smc/proposal_results.md)
> (Qwen3-8B/0.6B numbers across GSM8K / MATH / HumanEval / MBPP).

Constraints: the draft checkpoint must be a merged HF directory (no LoRA
adapters) sharing the target's vocab; `train_proposal.py` saves exactly
that.

**Measured recipe for a general-purpose draft** (Llama 8B/1B, June 2026):
collect rollouts with the current draft on a deployment-mix prompt set
(prompts-only, source-stratified — e.g. open-perfectblend), train with
**`--kl-direction reverse`**, and optionally iterate (re-collect with the
new draft, continue training at lower lr). The controlled ablation on
identical round-1 data: forward KL generalizes only on low-entropy
domains (math) and slightly degrades held-out chat/IF at any data scale;
swapping ONLY the objective to reverse KL improved every held-out domain
by 0.06–0.11 resample rate (chat 0.736→0.658, IF 0.738→0.654, code
0.574→0.504, math 0.496→0.413 vs base) — the objective is the main
effect. A second on-policy round adds a small uniform bump (~0.01–0.03 rr,
+1.5–2pp GSM8K). Mixed fwd+rev underperformed pure reverse everywhere.
Mode-seeking trade-off: reverse KL is strongest at small N (flat weights
matter most when particles are scarce) and can soften large-N accuracy
slightly — if N is large, also evaluate a 50/50 weight-interpolation with
the base draft (model soup), which gave the best N=8 GSM8K accuracy.

**Check headroom first.** The recipe pays in proportion to the baseline
proposal mismatch. Llama-3.1-8B + 3.2-1B (resample rate 0.50–0.74 across
domains) gained +7–12pp GSM8K and −0.07–0.11 rr everywhere. Qwen3-8B +
Qwen3-1.7B is already well-matched (rr 0.29 on math, 0.53 mixed): the same
pipeline was ~neutral on diagnostics, flat at N=4, and **cost 4.5pp at
N=8** — sharpening with nothing to buy. Run the per-domain holdout
diagnostics with the base draft before training; if rr is already low,
use a smaller draft (e.g. Qwen3-0.6B) where the recipe has headroom AND
the speed payoff is larger, rather than finetuning an already-good one.
Measured on Qwen3-8B + 0.6B (rr 0.51–0.85, GSM8K 64.5%/61.5% at N=8/4):
one reverse-KL round gave +11.5/+13.5pp GSM8K and −0.08–0.12 rr across
all domains — same magnitudes as Llama. It does not close a 3× draft
capacity gap (base 1.7B still wins on quality); the finetuned small
draft is the throughput operating point, not a 1.7B replacement.

## Throughput Sweeps

See `tps_benchmark_scripts/` for shell-based sweeps across batch sizes
and (gamma, n) configurations. Sweep scripts emit timestamped CSVs with
columns `method,gamma,n,tps,b`.

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

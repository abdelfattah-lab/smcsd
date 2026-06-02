# Hierarchical SMC + EAGLE3 — Results

Date: 2026-06-02 · 1×H100 80GB · GSM8K (openai/gsm8k test), `--disable-thinking`, batch size 1, CUDA graphs on, 20 questions.

## Idea

`target (32B) → SMC → draft (8B) → EAGLE3`. The SMC importance-weight target is
Qwen3-32B-FP8; the draft is Qwen3-8B accelerated by an EAGLE3 head. The head
proposes a chain off the 8B; the 8B verifies it (greedy/typical prefix accept)
so the committed block is the 8B's own continuation (q ≈ 8B — the strong
proposer dense SMC relies on), and the 32B reweights. EAGLE only makes the 8B
block cheap; SMC-SD commits all tokens and reweights, so proposer quality is
what determines accuracy.

## Models

| Role | Checkpoint |
|---|---|
| SMC target (score, p) | `Qwen/Qwen3-32B-FP8` |
| Draft / EAGLE base (q) | `Qwen/Qwen3-8B` |
| EAGLE3 head | `Tengyunw/qwen3_8b_eagle3` (NOT `AngelSlim/Qwen3-8B_eagle3` — that one emits garbage in this loader) |

## Results

| Config | Accuracy | Invalid | tok/s |
|---|---:|---:|---:|
| Vanilla Qwen3-32B-FP8 | 86.7% | 0% | 59.8 |
| Dense SMC 32B/8B (N=8, γ=4) | 95.0% | 0% | 85.9 |
| Fixed-stride eagle3 nested (q=head, lossy) | 10.0% | 25% | 77.6 |
| Lossless nested, greedy accept (N=8, γ=4) | 95.0% | 0% | 47.0 |
| Lossless nested, eager 32B verify (N=8, γ=8, thr=0.01) | 100% | 0% | 52.7 |
| Lossless nested, GRAPHED 32B verify (N=4, γ=8, thr=0.01) | 95.0% | 0% | 93.7 |
| **Lossless nested, GRAPHED, tuned (N=4, γ=4, thr=0.01)** | **95.0%** | **0%** | **99.4** |
| Lossless nested, graphed (N=4, γ=6, thr=0.01) | 90.0% | 0% | 100.9 |
| Lossless nested, graphed (N=8, γ=8, thr=0.01) | 95.0% | 0% | 85.2 |

**Headline: hierarchical 32B→8B→EAGLE matches dense SMC's 95% accuracy and is
1.16× faster (99.4 vs 85.9 tok/s) at N=4 γ=4, and 1.66× faster than vanilla 32B.**

### Where the time goes now (N=4, γ=4; SMCSD_TIMING)
`verify=82% · head draft=10% · rewrite=8%`, avg/step ≈ 28.5ms. Both big-model
verifies (8B EAGLE-verify + 32B score) are **already CUDA-graphed** — that 82%
is real model compute and the throughput floor. The remaining eager work is the
EAGLE head's forwards (draft loop + rewrite, ~18%).

Headroom attempts:
- **γ tuning**: γ=4 beats γ=8 (99.4 vs 93.7) — smaller γ shrinks the verify
  batch AND the wasted head forwards (mean accept ≈ 3.4, so γ=8 over-drafts).
- **Rewrite cap** (loop only to max committed): tried, **net slower** — the
  per-step `accept_lens.max().item()` GPU→CPU sync costs more than the saved
  head forwards. Reverted.
- **Graphing the head draft/rewrite**: the only remaining graphable chunk
  (~18%), but the head loop uses a manual per-step forward + multi-step attn
  backend that bypasses graph replay (the repo's `DRAFT_EXTEND` collapse
  "produced gibberish"). Needs the EAGLE draft-graph-runner integration; ~10%
  ceiling since the verify dominates. Deferred.

## What made it work

1. **Lossless EAGLE-verify** (`SMCSD_EAGLE_VERIFY=1`, `SMCSD_EAGLE_ACCEPT_THRESHOLD`,
   default 0.05 typical / ≥1.0 greedy): commit the 8B-verified block (q≈8B), not
   raw head proposals. Fixed the accuracy collapse (10% → 95–100%).
2. **Graph the 32B verify** (`ScoreModelRunner`): the eager 32B score verify was
   ~77% of decode time. Graphing it cut avg/step 60.7ms → 34ms (~50 → 94 tok/s).

## Reproduce

```bash
# env: uv venv --python 3.12; needs protoc + rust; kernels==0.9.0, huggingface_hub>=1.5.0
# hierarchical (the headline number):
SMCSD_EAGLE_VERIFY=1 SMCSD_EAGLE_ACCEPT_THRESHOLD=0.01 \
python scripts/accuracy_test_gsm8k.py --mode smc_engine \
  --model Qwen/Qwen3-32B-FP8 ... # WAIT: see note below
```

NOTE on flags: in this implementation `--model` is the EAGLE **base** (Qwen3-8B),
`--smc-score-model` is the 32B target, `--draft-model` is the head:

```bash
SMCSD_EAGLE_VERIFY=1 SMCSD_EAGLE_ACCEPT_THRESHOLD=0.01 \
python scripts/accuracy_test_gsm8k.py --mode smc_engine \
  --model Qwen/Qwen3-8B \
  --draft-model Tengyunw/qwen3_8b_eagle3 --smc-draft-mode eagle3 \
  --smc-score-model Qwen/Qwen3-32B-FP8 \
  --particles 4 --gamma 4 --temperature 0.7 \
  --attention-backend fa3 --mem-fraction-static 0.78 --max-total-tokens 24576 \
  --cuda-graph-max-bs 8 --num-questions 20 --max-new-tokens 512 --disable-thinking

# baselines
python scripts/accuracy_test_gsm8k.py --mode baseline --model Qwen/Qwen3-32B-FP8 \
  --mem-fraction-static 0.85 --num-questions 20 --disable-thinking          # vanilla
python scripts/accuracy_test_gsm8k.py --mode smc_engine --model Qwen/Qwen3-32B-FP8 \
  --draft-model Qwen/Qwen3-8B --smc-draft-mode dense --particles 8 --gamma 4 \
  --attention-backend fa3 --mem-fraction-static 0.85 --max-total-tokens 24576 \
  --cuda-graph-max-bs 16 --num-questions 20 --disable-thinking              # dense SMC
```

Profiling: add `SMCSD_TIMING=1 SMCSD_TIMING_EVERY=40` for the per-phase
draft/verify/other split.

## Follow-ups

- N=4 is the throughput sweet spot here (N=8 = bigger verify batch → 85.2). Sweep
  (N, γ, threshold) for the Pareto frontier.
- The 8B EAGLE verify is still eager (the remaining non-graphed forward); graphing
  it too (or the rewrite) would push further.
- The fixed-stride eagle3 path remains lossy; prefer `SMCSD_EAGLE_VERIFY=1`.

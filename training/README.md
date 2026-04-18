# smcsd/training — SMC-native draft MVP

Scaffold for training small proposal models by KL distillation from a target LLM,
with the goal of minimizing block chi-squared divergence for SMC-SD.
Full plan: [../smc_native_draft_plan.md](../smc_native_draft_plan.md).

## MVP parameters

- **Target:** `meta-llama/Llama-3.2-1B-Instruct` (hidden=2048, 16 layers, 128K vocab)
- **Proposal:** 4-layer Llama, hidden=2048, FFN=4096, GQA 32/8 — matches target embed dim so embeddings can be warm-started and frozen. ~355M total, ~92M trainable.
- **Data:** GSM8K train (7473 problems × 4 target rollouts × 256 tokens, top-50 logits cached). ~7.5M supervised positions, ~2.4GB on disk.

## Layout

```
training/
  configs/mvp.yaml
  model/proposal.py              builder + warm-start/freeze helpers
  data/gen_target_traces.py      one-shot target rollouts + top-K logit cache
  data/dataset.py                (TODO) torch Dataset over cached shards
  training/distill.py            (TODO) stage-1 KL distill
  training/on_policy.py          (TODO) stage-2 on-policy correction
  training/loss.py               (TODO) top-K soft CE, clipped chi-sq
  eval/block_chi2.py             (TODO) held-out block chi-sq estimator
  eval/ess_harness.py            (TODO) ESS/N on held-out prefixes
  eval/smc_eval.py               (TODO) end-to-end SMC-SD eval via existing engine
  scripts/run_mvp.sh             (TODO) orchestration
```

## Quickstart

Data generation (run in background on `cuda:1`):

```bash
cd /home/yahya/smcsd
python -m training.data.gen_target_traces \
  --target meta-llama/Llama-3.2-1B-Instruct \
  --output-dir training/data_cache/target_traces \
  --device cuda:1 \
  --sort-by-length
```

Smoke test first (20 prompts, ~3 min):

```bash
python -m training.data.gen_target_traces \
  --target meta-llama/Llama-3.2-1B-Instruct \
  --output-dir training/data_cache/smoke \
  --max-prompts 20 --device cuda:1
```

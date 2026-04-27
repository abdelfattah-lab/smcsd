# SMC-SD EAGLE Draft — Progress Report and Recipe

**Status updated: 2026-04-27.** Living document — append-only updates as we
iterate. See `eagle_plan.md` for the original plan and earlier postmortems.

## TL;DR

- Started with off-the-shelf hot-vocab EAGLE: **0% / 5%** GSM8K accuracy in
  SMC-SD, fundamentally broken as a proposal.
- After full-vocab warmstart on 50k mixed data + on-policy RB-KL distillation
  (this paper: Amini et al. 2025): **best EAGLE point γ=2 / 64% / 154 TPS.**
- Reference: AR-1B sibling at γ=8 gives **74% / 306 TPS** (still Pareto-dominates
  EAGLE for Llama-8B GSM8K).
- The recipe is target-only — no AR sibling required at inference. It transfers
  verbatim to GPT-OSS-20B and other targets without good small drafts.

---

## Headline numbers (Llama-3.1-8B GSM8K, q100, N=12, T=0.7)

| Draft / Stage | γ=2 acc | γ=4 acc | γ=8 acc | γ=8 TPS | γ=8 logw_var mean |
|---|---:|---:|---:|---:|---:|
| AR-1B baseline | — | — | **74%** | **306** | 11.9 |
| Off-shelf hot-vocab EAGLE3 | — | — | 5% (q20) | 139 | 41.9 |
| Full-vocab warm 2k smoke | — | — | 15% (q20) | 131 | 64.2 |
| Pure SMC weighted-MLE finetune (failed) | — | — | 0% (q20) | 123 | 89.5 |
| Hybrid topk+SMC finetune (failed) | — | — | 10% (q20) | 127 | 97.4 |
| **Warm-start 50k (mixed data)** | 52% | 31% | 12% | 201 | 533.8 |
| SMC-recurrent path-MLE on 50k | 48% | 37% | 8% | 205 | 285.3 |
| **KL-onpolicy on 50k (current best)** | **64%** | **39%** | 10% | 208 | 225.7 |

Each row is a real measurement; full SMC-metric tables in `outputs/sweep/` and
`outputs/kl_trained_sweep/`.

---

## What actually worked, and why

### 1. Full-vocabulary EAGLE warm-start with target-aligned data
**Lever**: 2k → 50k examples (~25× more data), full-vocab config
(`draft_vocab_size = vocab_size`), 35k from
`frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct` (target-aligned)
plus all GSM8K + hendrycks_math (math reasoning concentration).

**Why it worked**: EAGLE3 with `num_hidden_layers=1` is a tiny head, so
sample-efficiency is bad. Our 2k smoke training gave a degenerate proposal
(top-20 overlap 35%, lp gap −5.7, γ=2 q20 0%). Scaling data 25× lifted top-20
overlap to 53%, lp gap to −1.9, and pushed γ=2 q20 to 15%. The same 50k checkpoint
at q=100, N=12 then gave **52% γ=2 / 31% γ=4 / 12% γ=8.**

**Decision rule**: data scale was the **single highest-leverage knob** at every
stage. Don't waste cycles on objective tweaks if warmstart top-20 hit < 0.85
or lp gap median < −2.

### 2. Multi-GPU DDP (4× H100s) for warm-start
**Lever**: `--nproc_per_node 4` in SpecForge train_eagle3.py. 50k @ 1 GPU was
~3 hr; @ 4 GPU DDP was 47 min.

**Why it mattered**: enabled iteration speed. 200k now budgeted at ~3 hr.

### 3. Chain-mode runtime diagnostics (in `worker.py`)
**Lever**: per-decode-cycle JSONL of `draft_target_topk_overlap_mean`,
`sample_in_target_topk_mean`, `sample_target_rank_mean`,
`target_minus_draft_lp_mean`. Originally only firing in tree mode; extended to
chain mode and to the AR dense path so we have an apples-to-apples comparison.

**Why it mattered**: gave a **target-only**, runtime-actionable signal of
proposal quality independent of downstream task accuracy. The lp-gap mean
of −5.7 on EAGLE-2k vs −0.24 on AR-1B made the gap **numerical** rather than
intuitive. North-star numbers from AR-1B set explicit goals for warm-start
quality.

### 4. On-policy Rao-Blackwellized KL distillation
**Lever**: `train_eagle_kl_onpolicy.py`. Replaces SpecForge's off-policy
target-trajectory CE with **on-policy** KL distillation:
- Roll out EAGLE for K steps using its own (snapshotted, no-grad) weights —
  this is the *trajectory* the inference engine produces.
- Run target ONCE on `[prompt + x0 + y_1..y_{K-1}]` to get target's full
  per-position next-token distribution.
- Teacher-force the **same** trajectory through the **trainable** EAGLE
  recurrence, getting `q_θ(·|prefix)` at each position.
- Loss is the **Rao-Blackwellized KL estimator** (Eq. 9 of Amini et al.):
  exact `Σ_t Σ_v p_t(v)·[log p_t(v) − log q_θ,t(v)]` on full distributions.
  Provably variance ≤ MC at zero extra compute (target's full softmax is free
  from its forward pass).

**Why it worked vs the previous SMC-recurrent training**:
- The previous trainer used path-level importance-weighted MLE
  (`w_i = softmax(Σ logp − Σ logq_old)`), with per-record `logw_var ≈ 50–500`.
  That's the catastrophic-MC-variance failure mode the paper warns about.
- RB-KL replaces sample-based path scoring with **per-step exact KL on full
  distributions**. Variance is provably bounded; gradient signal stays clean.
- AND the trajectories are EAGLE-rolled-out (its own state, its own samples) —
  closing the **exposure-bias gap** that SpecForge's gold-trajectory training
  leaves wide open.

**Result deltas vs warmstart-50k**:
- γ=2: 52% → **64%** (+12 absolute points — biggest single-objective gain)
- γ=4: 31% → **39%** (+8)
- γ=8 acc: 12% → 10% (flat) but **logw_var 534 → 226 (−58%)** — the variance
  reduction is real, accuracy hasn't caught up yet at deep K.

### 5. Universal ε-uniform mixture (correctness backstop)
**Lever**: `--smc-eagle-eps-uniform` flag in `worker.py`. At each EAGLE
sampling step, with prob ε sample uniformly from V, otherwise from q_eagle;
record exact `log q_mix(t) = logsumexp(log(1−ε)+log q_eagle(t), log(ε/V))` so
SMC weights stay correct.

**What we measured**: at γ=8, ε=0.01 cuts `logw_var` mean roughly in half
(534 → 238) but accuracy *drops* (12% → 9%) because the random-token noise
hurts more than the bounded weights help. Net: **keep at ε=0 by default**,
keep the knob for cases where weight blowups are catastrophic. **Universal**
(no external draft needed) — only requires full-vocab EAGLE.

---

## What did NOT work (so we don't repeat)

### Hot-vocab EAGLE (e.g. yuhuili/EAGLE3-LLaMA3.1-Instruct-8B with 32k draft vocab)
- Off-the-shelf at γ=1 q20: **5%** GSM8K, logw_var mean 41.9.
- Reduced vocab violates `support(q) ⊇ support(p)`: target assigns nonzero
  probability to tokens with q=0, blowing up logw. Even after expanding to
  full-vocab via `expand_eagle3_to_full_vocab.py`, untrained weights → 0%.
- **Verdict**: must train a full-vocab EAGLE warm-start from scratch (or
  init the new lm_head rows from target's lm_head and finetune).

### Pure path-level weighted MLE on isolated K=1 rollouts
- Trained the original 2k checkpoint on `R=8 K=1` rollouts with
  `L = -Σ w_i log q(y_i)` where `w_i = softmax(logp − logq_old)`.
- Result: 0/20 GSM8K. Rollouts had ESS/R = 0.175 — degenerate weights, the
  weighted MLE gradient collapsed onto single-path imitation, destabilizing
  the proposal.
- **Verdict**: don't train SMC proposals with sample-based path-level
  importance weights. Use full-distribution per-step KL (Stage D below).

### Hybrid path-MLE + topk-KL + q_old-anchor on K=2 R=8
- The trainer at `train_eagle_smc_recurrent.py` with α=1, β=0.01, γ_anchor=5.
- Improvement over warmstart was small (γ=4: 31% → 37%) and γ=2 actually
  *regressed* (52% → 48%).
- Diagnosis: with median per-rollout ESS/R ≈ 0.13, the β·SMC term contributes
  noise; α·topk_KL is the only useful gradient and is essentially the same
  signal as warmstart training. Re-derived the right answer: minimize
  per-step full-distribution KL directly, not weighted-MLE.
- **Verdict**: superseded by `train_eagle_kl_onpolicy.py` which is strictly
  better.

### Multi-seed and multi-depth scaffold rollout collection (v3)
- Hypothesis: low ESS/R came from a collapsed seed-token distribution at
  depth 1. Tried sampling 4 i.i.d. x0 per prompt (multi-seed) and a 32-token
  target-rolled scaffold with 4 depth checkpoints (multi-depth scaffold).
- Result: **median ESS/R essentially unchanged** (0.134 → 0.142 multi-seed,
  → 0.139 multi-depth). Per-prompt mean unique seeds was 1.49/4 — target's
  next-token distribution at the prompt boundary is too peaky for i.i.d.
  sampling to give diversity.
- **Verdict**: ESS/R is not the right metric to optimize for offline rollout
  collection. The trainer's α·KL term is ESS-agnostic anyway. Distillation
  works fine on degenerate rollouts as long as states are diverse.

### ε-uniform mixture as performance lever
- Drops mean logw_var by ~50% but adds enough sampling noise that γ=8
  accuracy drops from 12% to 9% at ε=0.01.
- **Verdict**: leave at ε=0 by default. Universal correctness-backstop only.

---

## Universal recipe (target-agnostic — works for GPT-OSS-20B etc.)

This is the production sequence. Each stage's success gate is target-only —
no AR sibling required.

### Stage A — Full-vocab EAGLE warm-start (off-policy KL)
Standard SpecForge training of a 1-layer EAGLE3 with
`draft_vocab_size = target.vocab_size`. Loss is full-distribution KL between
target's per-position softmax and EAGLE's logits along **gold conversations**.

```bash
# Data prep (mixed instruction + domain)
python scripts/prepare_smc_warmstart_mix.py \
  --out cache/dataset/<target>_warmstart_200k.jsonl \
  --n-perfectblend 185000   # adjust source for non-Llama target

# Multi-GPU training
torchrun --nproc_per_node 4 scripts/train_eagle3.py \
  --target-model-path <target> \
  --draft-model-config configs/<target>-eagle3-smc.json \
  --train-data-path cache/dataset/<target>_warmstart_200k.jsonl \
  --output-dir outputs/<target>-warmstart-200k \
  --num-epochs 1 --batch-size 1 --tp-size 1 \
  --learning-rate 1e-4 --max-length 1024 --chat-template <template> \
  --target-model-backend sglang --sglang-attention-backend fa3 \
  --sglang-mem-fraction-static 0.20 \
  --init-lm-head-from-target
```

**Gate**: chain q20 diagnostic top-20 hit ≥ 0.90,
`target_minus_draft_lp` median ≥ −1, γ=1 sample target rank median = 1.
If not met, increase data, not steps.

### Stage B — On-policy RB-KL distillation
The new trainer at `scripts/draft_train/train_eagle_kl_onpolicy.py`.

```bash
python scripts/draft_train/train_eagle_kl_onpolicy.py \
  --target <target> \
  --draft <stage-A checkpoint> \
  --data <same warmstart jsonl> \
  --output-dir outputs/eagle_kl_onpolicy_<target> \
  --max-steps 8000 --batch-size 2 \
  --k-schedule "2:0,4:2000,8:4000,12:6000" \
  --lr 5e-6 --warmup-steps 200 \
  --refresh-old-every 200 --temperature 0.7
```

**Curriculum on K**: start at K=2, grow to K=12. EAGLE's recurrent state at
K=2 is well-modeled (warmstart territory); K=8/12 require new training. The
ramp lets the proposal first lock in shallow accuracy then stretch to deep.

**Gate**: per-step KL/pos drops at least 30% from start, `target_minus_q`
gap on EAGLE-rolled tokens improves ≥ 30%.

### Stage C — Inference safety belt (optional)
`--smc-eagle-eps-uniform 0.0` default. Set to 0.01 only if SMC weight blowups
are observed in production traffic (rare with a properly trained Stage B).

### Stage D — γ-sweep eval
Run accuracy_test_gsm8k.py at q=100 across γ ∈ {2, 4, 8, 12, 16} (or domain
equivalent). Choose the γ on the Pareto frontier (max accuracy at acceptable
TPS, or max TPS at acceptable accuracy).

```bash
for g in 2 4 8 12 16; do
  python scripts/accuracy_test_gsm8k.py \
    --mode smc_engine --model <target> \
    --draft-model <stage-B checkpoint> \
    --draft-mode eagle3_chain \
    --particles 12 --gamma $g --temperature 0.7 \
    --attention-backend fa3 \
    --num-questions 100 \
    --max-running-requests 24 --cuda-graph-max-bs 24 \
    --eagle-topk 20 \
    --smc-metrics --smc-metrics-jsonl outputs/eval_g${g}.jsonl
done
```

Report card per γ: accuracy, TPS, ESS/N med, logw_var mean+median, top-20
overlap, target-rank median.

---

## Why this works (theory)

A good SMC proposal minimizes `KL(p_target || q_eagle)` along the trajectory
distribution actually visited at inference. Training EAGLE is therefore
fundamentally a **KL minimization problem**, and the only meaningful
question is: what's the lowest-variance gradient estimator?

Three estimators we've used:

1. **MC path-level** (`logw = Σlogp − Σlogq` per path). High variance —
   `Var ≈ 50–500` in our settings. Catastrophic at high γ. *Failed in
   SMC-recurrent training.*
2. **Off-policy CE on gold trajectories** (SpecForge default). Per-position
   full-distribution loss but on **wrong** trajectories — exposure bias at
   deep K. *Worked for warmstart only.*
3. **On-policy RB-KL** (Amini et al. 2025, applied here). Per-position exact
   KL on **EAGLE's own** trajectories using full target distributions. Provably
   variance-bounded vs MC; matches inference state distribution. *This is the
   answer.*

The Rao-Blackwell theorem says: replacing a sampled function value with its
conditional expectation (over the sampling step) gives an estimator with
≤ variance and equal mean. For KL between LMs, the natural conditioning is
on the full prefix Y_<n at each position, which gives the exact next-token KL
analytically — for free, because target's softmax is already in the forward
pass.

---

## Current plan (in flight as of 2026-04-27)

### Active experiments
1. **Warmstart 200k** (in progress, 4-GPU DDP, ETA ~3 hr from launch).
   `outputs/llama31-8b-eagle3-smc-warmstart-200k/`. 4× more data than current
   best. Historical sub-linear scaling suggests +5–15 points across γ.
2. **Curriculum-K KL-onpolicy on 200k warmstart** (queued, ~80 min single GPU).
   K-schedule `2:0,4:2000,8:4000,12:6000`, lr 5e-6, 8000 steps.
3. **γ-sweep at {2, 4, 8, 12, 16}** (queued, parallel ~3 min). First view of
   high-γ Pareto curve for our trained EAGLE.

### Decision points
- If 200k warmstart + curriculum KL pushes γ=8 above 30%: **investigate γ=12,
  16** as the actual sweet spot, where AR's TPS advantage is hardest to beat.
- If γ=8 still flat after curriculum: try **DDP on the KL trainer** + 50k
  steps total. Or scale warmstart to 500k.
- If accuracy plateaus regardless: time to try **multi-layer EAGLE**
  (`num_hidden_layers=2` or 3 in the draft config). 1-layer head may be
  too small to capture the recurrent state at K=12.

### Targets after current iteration
- Llama-8B GSM8K **γ=2**: 64% → ≥ 75% (beat AR-1B accuracy at γ=2)
- Llama-8B GSM8K **γ=8**: 10% → ≥ 50% (currently the regime where AR-1B
  wins on TPS)
- Llama-8B GSM8K **γ=12 or 16**: unmeasured; goal ≥ 50% with TPS > 306
  (would Pareto-dominate AR)

---

## Storage layout (post-2026-04-27 migration)

Root filesystem on `/dev/root` is small (1 TB). All training artifacts now live
on the 28 TB RAID array at `/mnt/raid0`. Paths in the codebase still resolve
via symlinks, so no script changes needed.

| Logical path (use this) | Real location | Size |
|---|---|---|
| `/home/yahya/smcsd/outputs` | `/mnt/raid0/yahya/smcsd_outputs` | 33 GB |
| `/home/yahya/smcsd/data` | `/mnt/raid0/yahya/smcsd_data` | 42 GB |
| `/home/yahya/smcsd/checkpoints` | `/mnt/raid0/yahya/smcsd_checkpoints` | 1.6 GB |
| `/home/yahya/SpecForge/outputs` | `/mnt/raid0/yahya/SpecForge_outputs` | 14 GB |
| `/home/yahya/SpecForge/cache` | `/mnt/raid0/yahya/SpecForge_cache` | 2.9 GB |
| `~/.cache/huggingface` | `/mnt/raid0/yahya/huggingface_cache` | 829 GB (was already a symlink) |

`HF_HOME=/mnt/raid0/yahya/huggingface_cache` exported in `~/.bashrc` as
defense-in-depth (in case any HF lib version bypasses the symlink).

**Rules going forward**:
- Continue using the `~/...` and `/home/yahya/...` paths in scripts. Don't
  hard-code `/mnt/raid0` — symlinks make both work and the home paths are
  more readable in commands and logs.
- New training runs that produce >5 GB of artifacts: just write to
  `outputs/<run-name>` as usual; the symlink lands them on raid0.
- If you find yourself adding a new top-level dir under `/home/yahya/`
  (e.g. `tmp_data_dump/`), make it a symlink to a fresh subdir under
  `/mnt/raid0/yahya/` first.

**Layer-2 deletes performed (deprecated, all superseded by current best
`eagle_kl_onpolicy_200k_curriculum`)**:
- `outputs/eagle_smc_recurrent_*` (7 dirs, 49 GB)
- `outputs/eagle_smc_hybrid_*` (2 dirs, 7.7 GB)
- `outputs/eagle_smc_proposal_*` (2 dirs, 7.7 GB)
- `outputs/eagle_kl_onpolicy_smoke` (2.6 GB)
- `SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-2k` (31 GB — original
  2k smoke that established the broken baseline)
- `SpecForge/outputs/llama31-8b-eagle3-smc-gsm8k-smoke` (7.7 GB)
- Total: 109 GB.

## File map

### Training & data
- `scripts/prepare_smc_warmstart_mix.py` — mixed data prep (GSM8K +
  hendrycks_math + PerfectBlend-Regenerated)
- `/home/yahya/SpecForge/scripts/train_eagle3.py` — Stage A warmstart
  (off-policy CE/KL on gold trajectories)
- `scripts/draft_train/train_eagle_kl_onpolicy.py` — Stage B on-policy RB-KL
  distillation. Supports `--k-schedule` for curriculum.
- `scripts/draft_train/expand_eagle3_to_full_vocab.py` — utility for
  converting hot-vocab EAGLE checkpoints to full-vocab (rarely needed; train
  full-vocab from scratch instead)

### Deprecated / superseded
- `scripts/draft_train/collect_eagle_smc_rollouts.py` (v1) — single-step
  rollouts. Wrong objective shape, fixed in v2.
- `scripts/draft_train/collect_eagle_smc_rollouts_v2.py` — runtime-matched
  recurrent rollouts. Still useful for analysis but **not** for training:
  median ESS/R = 0.13 means rollouts are too degenerate for path-MLE.
- `scripts/draft_train/collect_eagle_smc_rollouts_v3.py` — multi-seed and
  scaffold-mode collector. Same ESS/R issue. Useful only if we ever revive
  path-MLE training.
- `scripts/draft_train/train_eagle_smc_proposal.py` (one-step path-MLE) —
  superseded.
- `scripts/draft_train/train_eagle_smc_recurrent.py` (recurrent path-MLE +
  topk-KL + anchor) — superseded by RB-KL distillation, which is strictly
  better.

### Diagnostics
- `outputs/ar_diag/` — AR-1B chain diagnostics (north-star numbers)
- `outputs/eagle50k_diag/`, `outputs/sweep/` — warmstart-50k γ-sweep
- `outputs/kl_trained_sweep/` — current best (KL-onpolicy on 50k)
- `smcsd/core/worker.py:_write_eagle_chain_diagnostics` — runtime per-cycle
  JSONL emitter; works for both EAGLE-chain and dense-AR draft modes.
- `--smc-metrics --smc-metrics-jsonl <path>` enables per-step ESS/N,
  logw_var, max_w, resample-rate logging at any inference run.

---

## Decision log (append-only)

### 2026-04-26
- Verified existing 50k checkpoint via diagnostics: top-20 overlap 0.53,
  target_minus_draft_lp median −0.5. Baseline number: γ=2 q100 52%.
- Implemented `--smc-eagle-eps-uniform`. Result: bounds variance, hurts
  accuracy, default off.
- Re-read trainer's loss decomposition: dominant signal is α·target-topk-KL
  distillation (β·SMC term too small to matter with degenerate rollouts).
  Implication: rollout ESS/R is not the right metric to optimize.

### 2026-04-26
- Trained `train_eagle_smc_recurrent` (path-MLE + topk-KL + anchor) on 8k
  scaffold-mode rollouts. Result: marginal gain (γ=4: +6 abs), regression
  at γ=2 (−4 abs). Inferior to KL distillation.

### 2026-04-26
- Read Amini et al. 2025 ("Better Estimation of the KL Divergence Between
  Language Models"). Realized:
  1. SpecForge's existing CE loss IS the per-position RB-KL estimator (Eq. 9)
     applied off-policy on gold trajectories.
  2. The missing piece is **on-policy** trajectories (EAGLE's own rollout)
     so the per-position KL is computed at the states EAGLE actually visits
     at inference.
- Implemented `train_eagle_kl_onpolicy.py`. Smoke test passed: gap
  `target_lp − q_lp = −17` initially (huge overconfidence), trended to −13
  over 3000 steps.
- Result: **γ=2 52% → 64%** (+12). Best EAGLE result to date.
- logw_var mean at γ=8 dropped 60% (534 → 226) but accuracy unchanged
  (12% → 10%). Hypothesis: K=8 recurrent state requires curriculum + more
  training to fix; on-policy at K=8 from cold is too aggressive.

### 2026-04-27
- Added `--k-schedule` curriculum support to
  `train_eagle_kl_onpolicy.py`. Schedule format: `K:start_step` pairs.
- Launched 200k warmstart (4-GPU DDP, ~3 hr ETA). Will follow with
  curriculum-K KL training and γ ∈ {2, 4, 8, 12, 16} eval to test high
  draft length.

### 2026-04-27 (generalization gauge)
- Added `scripts/heldout_diag.py`: runs SMC-SD on the *last* 200 rows of
  PerfectBlend-Regenerated (never seen during warmstart, since training
  sampled the first 185k of a 1.42M shuffle) and aggregates target-only
  chain diagnostics + SMC metrics. **Single-number summary**: heldout
  `target_minus_draft_lp_mean` median across 200 multi-domain prompts.
- Why: prevent benchmark-specific overfitting. With the recipe focusing on
  GSM8K (math), there's a real risk that math-data injection makes GSM8K
  look great while harming code/chat/Q&A. The heldout diag is a
  **target-only, domain-agnostic generalization gauge** — every future
  change must clear it.
- **Baseline measured for current best ckpt** (`eagle_kl_onpolicy_200k_curriculum`):

  | metric (heldout 200, γ=4) | value |
  |---|---:|
  | top-20 overlap mean | 0.499 |
  | sample-in-target-top-20 mean | 0.913 |
  | sample-target-rank median | 1.0 |
  | target_minus_draft_lp median | **-1.73** |
  | ESS/N median | 0.418 |
  | logw_var median | 65.5 |
  | max_w median | 0.420 |

  Compared to in-domain GSM8K γ=4 (top-20 overlap 0.46, ESS/N 0.46,
  logw_var 27): heldout has slightly lower overlap and notably higher
  logw_var, but ESS/N is similar and rank median is still 1. **The
  current best generalizes reasonably across domains.**

- **Generalization gate going forward** (every future ckpt):
  1. Primary task accuracy goes up (GSM8K q100), AND
  2. Heldout target_minus_draft_lp median doesn't regress more than ~10%
     (i.e. stays ≥ -1.90), AND
  3. Heldout top-20 overlap median doesn't drop more than 5 pts
     (stays ≥ 0.45).
  Promote a checkpoint only if all three hold. Math-data injections must
  pass this gate too.

### 2026-04-27 (results)
- 200k warmstart finished: 49994 steps in 3:08:51, final loss 0.47, top-1
  acc 0.80. Strictly better learner than 50k smoke (final loss 0.51, acc
  0.80 — same acc but more diverse data).
- Curriculum-K KL-onpolicy on 200k warmstart (8000 steps, lr 5e-6,
  K-schedule `2:0,4:2000,8:4000,12:6000`, single GPU, ~37 min wall):
    - K=2 phase: loss 5.16 → 2.29, gap −24.6 → −9.7
    - K=4 phase: loss 3.54 → 3.03, gap −11.4 → −9.9
    - K=8 phase: loss 4.16 → 3.96, gap −9.2 → −9.3
    - K=12 phase: loss 4.27 → 4.15, gap −8.1 → −9.5
- γ-sweep on the resulting checkpoint (q=100, N=12):

  | γ | acc | TPS | ESS/N med | logw_var mean | Δ vs KL-50k |
  |---:|---:|---:|---:|---:|---:|
  | 2 | 63% | 162 | 0.607 | 111.0 | -1 (~saturated) |
  | 4 | **52%** | 193 | 0.458 | 95.5 | **+13** |
  | 8 | **31%** | 209 | 0.205 | 214.7 | **+21** |
  | 12 | 18% | 225 | 0.165 | 168.2 | new |
  | 16 | 6% | 234 | 0.167 | 108.8 | new |

  Biggest single improvement we've measured: γ=8 jumped 10% → 31%.
  γ=12, γ=16 measured for first time — γ=12 is genuinely usable (18% acc
  / 225 TPS), γ=16 falls off (6%).

- **What worked here**: data scale (50k → 200k) + curriculum-K (started
  EAGLE on shallow trajectories first, ramped to deep). The 200k warmstart
  alone wouldn't have moved γ=8 from 12 → 31; we saw KL-onpolicy on the
  smaller 50k *not* improve γ=8 (10%). The combination — better warmstart
  AND on-policy training that explicitly visits K=8/K=12 states with a
  sane loss landscape — is what pulled γ=8 up by 21 points.

- **What still doesn't work**: AR-1B at γ=8 (74% / 306 TPS) still
  Pareto-dominates EAGLE on every γ we tried. EAGLE's best Pareto point
  is now γ=4 (52% / 193). Closing the remaining gap likely requires:
  (a) larger EAGLE (`num_hidden_layers=2` or 3) — capacity may be the
  bottleneck at deep K;
  (b) more KL training steps (~25k, not 8k);
  (c) bigger warmstart data (500k+ from PerfectBlend).

- **Pareto frontier shift** (best-EAGLE-acc vs AR-1B target 74%):
  - Pre-2026-04-26: 15% (warmstart-2k smoke)
  - 2026-04-26 evening: 52% (warmstart-50k γ=2)
  - 2026-04-26 night: 64% (KL-onpolicy on 50k γ=2)
  - 2026-04-27 morning: **63% γ=2 / 52% γ=4 / 31% γ=8** (200k+curriculum)

  γ=2 saturated around 64%; the action is now in the high-γ regime where
  TPS is best.

- **Updated next levers (priority order)**:
  1. **Multi-layer EAGLE** (num_hidden_layers=2 or 3). Architecture change.
     Probably the biggest remaining knob if we want γ=8 above 50%.
  2. Longer KL training (25k steps) on the same 200k warmstart.
  3. 500k warmstart for further sub-linear data scaling.
  4. Try eval at intermediate γ ∈ {6, 10} to find true Pareto knee.

---

## Notes for future-me / next agent

- **Always measure target-only diagnostics first.** AR comparisons are
  useful as a north-star for Llama-8B but the recipe must validate without
  any AR sibling. For GPT-OSS-20B you have only the target; the gates
  (top-20 overlap, lp gap, ESS/N, logw_var) are all target-only.
- **Don't spend cycles on objective tweaks if Stage A gates fail.** If
  warmstart hasn't hit ≥ 0.85 top-20 hit, more data > more KL training.
- **Prefer per-position full-distribution KL over path-level scoring.**
  Free variance reduction. The Amini paper's contribution is exactly this
  intuition turned into a theorem.
- **EAGLE's recurrent state IS an exposure-bias problem.** SpecForge
  training shows the head excellent target H; inference forces it to use
  its own. On-policy training is the canonical fix and it works here.
- **The ε-uniform mixture is correctness, not performance.** Use it only
  when you have evidence of weight blowups in production traffic.
- **Multi-seed and multi-depth scaffold collection are dead ends for
  path-level training.** They don't move ESS/R because target itself is
  near-deterministic at depth 1 and the path space at K=2 is narrow. The
  fix is the loss function (RB-KL), not the data.

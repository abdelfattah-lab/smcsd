# EAGLE3 Support in SMC-SD — Investigation Report

**Target model:** Llama-3.1-8B-Instruct
**Draft model:** lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B (pretrained EAGLE3)
**Benchmark:** GSM8K accuracy @ N=12 particles, γ=8 draft tokens, temperature=0.7
**Hardware:** 8× H100 80GB

---

## 1. Problem statement

SMC-SD is a population-based speculative-decoding scheme: `N` particles per request each
propose `γ` draft tokens per cycle, all drafts are *committed* (no rejection), and
target/draft log-probability ratios are used as SMC importance weights with optional
resampling. On a dense Llama-3.2-1B-Instruct draft, this gives **~75% GSM8K / 310 tok/s**
at the reference config.

An earlier `eagle3-experimental` branch attempted to swap in EAGLE3 as the draft. It
reportedly delivered roughly **half the accuracy and half the speed** of the dense
baseline — the opposite of what a good draft should do.

Goal: figure out why, fix what's fixable, and (if needed) train an SMC-native EAGLE3
head that does beat the dense draft on throughput without tanking accuracy.

---

## 2. Phase A — Port + structural fixes (`test-eagle` branch)

### 2.1 What the old branch did

The `eagle3-experimental` branch sat on an older SMC codebase (`smcsd/v2/`). It wired
EAGLE3 into the SMC worker with three main pieces:

1. **Target aux capture.** `CudaGraphRunner` switched to `CaptureHiddenMode.FULL` under
   EAGLE3 so the target emits 3×H-dim aux hidden states at every position during verify.
2. **Decode loop.** γ−1 eager draft forwards, each consuming its own previous hidden
   state, producing x₂…x_γ given the pre-sampled x₁.
3. **Rewrite loop.** After verify, γ+1 eager forwards over the accepted tokens
   [x₁…x_γ, bonus] paired with target aux, to rewrite the draft KV with the correct
   EAGLE3 conditioning. The last rewrite step samples next cycle's x₁.

### 2.2 Port to `smcsd/core/` + correctness fix

The old branch was based on `smcsd/v2/`, which was superseded by `smcsd/core/` (slot-based
refactor). I ported all EAGLE3 paths onto `smcsd/core/`, plus:

- `SMCDraftInput` grew `hidden_state`, `first_draft_token_id`, `first_draft_logprob`.
- `ScheduleBatchSMC` grew per-slot EAGLE3 buffers (lazy-alloc, copied on resample).
- `SMCCudaGraphRunner` added the FULL-aux target capture path, with fall-back layer IDs.
- Scheduler wires the prefill seed scatter + decode write-back.

**Real bug found:** the rewrite loop was reusing pre-built multi-step attention backends
at indices `[γ−1..2γ−1]` whose internal metadata was set up assuming positions
`L+γ−1..L+2γ−1`. Rewrite actually runs at positions `L..L+γ`, so attention was reading
KV at positions that hadn't been written yet (silent corruption, visible as gibberish
output beyond the first couple of tokens). Fix: re-init attention metadata at
`orig_seq_lens` before the rewrite loop, then use backends `[0..γ]`.

### 2.3 Phase A benchmark

| Draft | GSM8K | Throughput |
|---|---|---|
| dense Llama-3.2-1B-Instruct (reference) | **75%** | 310 tok/s |
| pretrained lmsys EAGLE3 (after port + attn fix) | **2.5%** | 371 tok/s |
| eagle3-experimental branch (approx prior state) | ~2% | ~180 tok/s |

**Conclusions from Phase A:**

- Speed was fully recovered — Phase A runs ~1.2× faster than the dense draft; the old
  branch's 50% speed deficit came from the disabled draft graph + the attention bug
  crashing the cache into garbage that the model had to recover from.
- Accuracy was **not** recovered. The pretrained lmsys EAGLE3 delivers ~1/25 of the
  dense draft's GSM8K accuracy under SMC. This is a distributional problem, not a
  plumbing one:
  1. **Pruned 32K hot vocab violates SMC's absolute-continuity assumption** for
     importance weighting — target probabilities for tokens outside the hot set are
     lost forever.
  2. **Trained for trees, used as a chain.** Upstream EAGLE3 is trained with tree draft
     (`topk>1`, shallow, with rejection verify). SMC forces `topk=1` + linear γ-step +
     all-accept commits. This is OOD for the pretrained head past step ≈ 1.

---

## 3. Phase B — Train an SMC-native head (`eagle-train` branch)

If the pretrained head is structurally mismatched, *can we train one that isn't?*

### 3.1 B.1 Hard-label CE, full-vocab tied lm_head (UltraChat)

**Setup**

- Full vocab (128k), lm_head *tied+frozen* to target's, backbone warm-started from lmsys.
- Training data: 5250 rollouts × 128 tokens from UltraChat (≈1.55M positions, 7/8 GPUs,
  rank 2 lost to a co-tenant).
- Loss: hard-label CE on target's greedy next-token.
- 3 epochs, bs=512/GPU × 8, lr=2e-4.

**Result**

- val CE: 3.87 → 3.42 → 3.88 (overfit past step 750).
- GSM8K@40Q: step_750_best = **10% (4/40)**.

This was an encouraging early data point, but **100-question eval later overturned it**
— the "10%" was noise (40-question stderr at p≈0.05 is ±3.5%).

### 3.2 B.2 Top-K distillation (UltraChat)

Hypothesis: hard-label CE overfits fast (one integer label per position = sparse signal).
Distillation (per-token top-K teacher distribution + forward KL) gives ~log(K) more bits
per example.

**Setup**

- Collector upgraded to save teacher top-K=64 log-probs + target-vocab indices per position.
- Trainer switched to `forward_KL(teacher_topK_renorm || student)`.
- Re-collected with 8/8 GPUs → 1.78M positions, 43GB.
- Same architecture (full vocab, tied+frozen lm_head), same hyperparameters.

**Result**

| Checkpoint | val KL | val CE | GSM8K@40Q |
|---|---|---|---|
| distill_500 | 3.14 | 3.93 | 2.5% |
| distill_750 | 3.03 | 3.82 | 2.5% |
| distill_1000 | 2.97 | 3.74 | 0.0% |
| distill_1250 | 2.95 | 3.73 | 0.0% |

Distillation drove val KL monotonically lower (no overfit) but didn't improve GSM8K —
it actually *hurt* argmax accuracy because the loss trains the student to spread mass
across the top-K, which flattens the peak.

Added a mixed `α·KL + (1−α)·CE` loss (`--ce-weight`) for flexibility, and an identically-
configured pure-CE run on the same 1.78M dataset — to isolate whether distillation itself
was the culprit or whether the dataset was the issue.

Pure CE on the new dataset: still 0-5% GSM8K. Same noise floor.

### 3.3 B.3 Pruned-vocab lmsys-faithful + math-heavy data

By this point all three offline variants were scraping the noise floor. Two deeper
architectural concerns became the next target:

1. **Full-vocab tied lm_head.** lmsys shipped a 32k pruned lm_head specialized for the
   hot set (130M trainable params dedicated to output scoring); I'd replaced that with
   a frozen full-vocab projection and given the head zero trainable output-side
   capacity. Reverted to lmsys's exact layout: 32k trainable lm_head + d2t mapping.
2. **Training data distribution.** UltraChat target rollouts teach the draft to match
   the target on conversational tokens; GSM8K tests on mathematical reasoning. Switched
   to a math-heavy mix: ~15% GSM8K-train + ~65% OpenMathInstruct-2 + ~20% UltraChat.
   2.14M positions, 51GB, all 8 GPUs clean.

**Trainer changes:**

- `Eagle3Head` rebuilt with `draft_vocab_size=32000`, `lm_head` trainable, `d2t`+`t2d`
  as buffers, `refresh_t2d()` inverse mapping.
- Full lmsys state dict loaded (midlayer + fc + norm + lm_head + d2t).
- `pruned_distill_loss`: map teacher top-K target indices → draft vocab via `t2d`,
  renormalize over valid, forward-KL. `pruned_ce_loss` with `ignore_index=-1` for
  out-of-hot labels. NaN-safe masked softmax handles all-OOV rows.
- Added `--warmup-steps 200`, `--grad-clip 0.5`, NaN-skip in train loop (a first
  run with `lr=2e-4` no-warmup diverged to NaN at step ~350).

**Runs:**

- 3-epoch run at `lr=5e-5`, warmup=200: val CE 2.53 → 1.37, clean run, best at step 1500.
- 10-epoch follow-up: val CE floored at **1.32** from step 2400 onward. No NaN, no overfit.

OOV fraction (labels outside lmsys's 32k hot set): **4.7%** throughout — confirms hot
vocab adequately covers target's output distribution on this data.

**GSM8K @ 100Q evaluation (more stable than 40Q):**

| Checkpoint | val CE | GSM8K | Invalid | Throughput |
|---|---|---|---|---|
| step_001000_best | 1.41 | 1.0% | 5% | 331 tok/s |
| step_002000_best | 1.35 | 1.0% | 5% | 368 tok/s |
| step_003000_best | 1.33 | 0.0% | 2% | 371 tok/s |
| step_004600_best | 1.32 | 1.0% | 15% | 364 tok/s |
| step_005210_final | 1.33 | 2.0% | 13% | 367 tok/s |

**Side-by-side with baselines at matching eval protocol:**

| Draft | GSM8K@100Q (temp=0.7) | GSM8K@100Q (temp=0) |
|---|---|---|
| pretrained lmsys EAGLE3 | 1% | 1% |
| B.3 (step 5210) | 2% | 0% |
| dense Llama-3.2-1B-Instruct (reference) | 75% | — |

---

## 4. What we learned

### 4.1 The "5% lmsys baseline" was partly noise

My earlier 40-question eval showed lmsys at 2.5% and a first B.1 checkpoint at 10%.
Both numbers turned out to be sample noise (stderr at p≈0.05, N=40 is ≈3.5%). At 100
questions the full range of evaluated drafts — pretrained lmsys, pure-CE-UltraChat,
distill-UltraChat, pruned-vocab-math-distill — all fell in a **0–5% band, statistically
indistinguishable from each other**.

### 4.2 Lower val loss didn't translate to GSM8K

Val CE trajectory across B.1 → B.2 → B.3: **3.42 → 3.73 → 1.32**. The B.3 head assigns
~27% probability to target's argmax token (vs ~3% for B.1 and the pretrained head).
GSM8K accuracy: **1–10%** → **0–2.5%** → **0–3%**. No systematic improvement.

This is a clean falsification of the hypothesis that offline-trained val loss is a
useful proxy for SMC-under-EAGLE3 GSM8K performance.

### 4.3 Root cause: distribution mismatch between training and inference

The EAGLE3 head, however well-trained offline, sees a **different token-context
distribution at SMC inference time** than it was trained on:

- *Offline training distribution:* target's own greedy (or teacher-forced) trajectory.
  Every prefix the head sees during training is a prefix the target would itself have
  produced.
- *SMC inference distribution:* a mix of target-committed tokens (the verify bonus)
  and draft-committed tokens (x₁..x_γ, including the draft's own possibly-wrong
  proposals). These get committed unconditionally because SMC has no rejection. The
  head sees prefixes that contain its own past mistakes.

Offline training can drive val loss as low as you like on target's clean trajectory
and still miss the inference distribution completely.

### 4.4 Single-position training is a secondary bottleneck

During offline training I processed each position independently (one token per
attention call). That makes Q·K trivial — softmax of a single scalar is always 1 —
so **`q_proj` and `k_proj` receive zero gradient.** All offline improvement came from
`fc`, `v_proj`, `o_proj`, MLP, and norms. Q/K stayed pinned at lmsys's tree-decode
values, leaving the attention pattern frozen.

A sequence-based training loop (K-length windows with causal attention inside each)
would fix this, at the cost of significantly more training code.

### 4.5 Offline warm-start is good; offline-only training isn't enough

The pretrained lmsys checkpoint is a competent starting point — the backbone,
attention patterns, and fc fusion are sensibly initialized by a team with good data
and compute. Our math-heavy distillation preserved this and drove val loss lower, but
didn't beat it on GSM8K. Replacing the lmsys head entirely with a from-scratch trained
one would almost certainly be worse.

---

## 5. Recommendations / next steps

### 5.1 On-policy SMC fine-tuning (Phase B.4)

The one thing we haven't tried, and the one most likely to actually move accuracy:

- Run SMC inference with the Phase-A-ported worker (lmsys warmstart as draft).
- Use `Eagle3Collector` (already wired into the SMC worker, commit
  `ac9f9011c`) to capture `(target_aux, committed_token, top-K)` tuples from the
  **actual SMC-committed trajectory** — including the noisy prefixes the head sees
  at inference.
- Periodically stop inference and retrain on the collected on-policy data.
- Iterate: new weights → new SMC rollouts → new data → retrain.

This closes the loop between training and inference distributions without requiring
us to do gradient updates during serving. The collector is already zero-cost when
disabled and correctly handles resample-based particle copying of SMC state.

### 5.2 Sequence-based training

Orthogonal to on-policy but additive: replace the single-position loss with K-length
windows per rollout, so `q_proj`/`k_proj` receive gradient. Requires implementing a
small attention module with KV cache inside the training script. ~2 days of work.

### 5.3 Wider architecture as a fallback

If on-policy + sequence training still can't close the gap, the final lever is
increasing the head's capacity: a second transformer layer, wider hidden, or deeper
fc. This is the expensive / least principled option and should only be pursued after
(5.1) and (5.2) have been tried.

### 5.4 What not to try

- More UltraChat data. We already hit the val-loss floor at 2.14M positions.
- Longer training at current settings. Diminishing returns past 10 epochs on our data.
- Pure distillation with higher K. K=64 already covers >95% of target's mass per position.

---

## 6. Artifacts left in the tree

### On the `eagle-train` branch (local, unpushed):

- `smcsd/core/` — EAGLE3 draft worker (Phase A port + attn fix)
- `smcsd/core/eagle3_collector.py` — on-policy data collector (wired into worker)
- `scripts/collect_target_rollouts.py` — offline data collector (hard-label + top-K,
  multi-dataset with math-heavy mix support)
- `scripts/train_eagle3.py` — pruned-vocab lmsys-faithful trainer (DDP, distill+CE,
  warmup, grad-clip, NaN-safe, standalone-loadable checkpoints)
- Saved checkpoints at `/home/yahya/eagle3_ckpt_math/` (10-epoch final: val CE 1.33,
  GSM8K 2%)
- Training data at `/home/yahya/eagle3_train_math/` (2.14M positions, 51GB)

### On the `smc-eagle3-args` branch of the sglang fork (pushed):

- `smc_draft_mode` dense|eagle3 ServerArg
- `smc_eagle3_collect_path` ServerArg for the on-policy collector

### Key benchmark numbers to remember:

- Dense Llama-3.2-1B-Instruct draft: **75% GSM8K / 310 tok/s**
- pretrained lmsys EAGLE3 (Phase A): **1–5% GSM8K / 371 tok/s**
- B.3 trained (math-heavy distill, lmsys-faithful): **1–2% GSM8K / 367 tok/s**

Offline training consumes about 40 minutes per 10-epoch run on 8×H100 after data
collection (45 min for math-heavy). Iteration is cheap.

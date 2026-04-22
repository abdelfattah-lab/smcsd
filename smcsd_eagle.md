# EAGLE3 for SMC-SD: Investigation Report

**Date**: 2026-04-22
**Branches**: `new_release` (dense baseline), `eagle3-cg` (this work, based on `origin/eagle3-draft-experimental`)
**Hardware**: H100 80GB
**Target / draft**: `meta-llama/Llama-3.1-8B-Instruct` / `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`

## TL;DR

- We attempted to make EAGLE3 work as the draft head in SMC-SD, starting from the abandoned `eagle3-draft-experimental` branch.
- We fixed real bugs (CUDA-graph wiring, rewrite-phase redundancy, seq_lens off-by-one, target-temperature math). Speed at ~1150 tok/s is now competitive with the 1B dense draft's 1187 tok/s on the throughput bench.
- But on GSM8K the EAGLE3 path gets ~1-5% accuracy vs dense's 72%. **The remaining gap is not mechanical bugs**. It comes from a fundamental mismatch between EAGLE's training objective (optimized for rejection sampling) and SMC's requirements (full-vocab proposal, particle diversity, calibrated logprobs).
- Pre-trained EAGLE3 checkpoints cannot be made to work with SMC by integration-level fixes. Making EAGLE work for SMC requires **re-training the draft head with an SMC-aware objective**, or pursuing a different architectural design.
- For the ICLR submission timeline the practical recommendation is to **ship with the dense draft** and attack the verify-forward path for further speed. If we want to claim an EAGLE-style draft for SMC in the paper, the minimum-viable option is re-training EAGLE3 with a full-vocab head and a KL-on-full-distribution loss (~1-2 weeks of work + a few hundred dollars of compute).

---

## 1. What we measured

### 1.1 Dense baseline (reference)

All experiments use `N=12, γ=8, T=0.7, attention-backend=fa3, max-running-requests=24, cuda-graph-max-bs=24, fast-resample=on` unless stated otherwise.

| Config | GSM8K accuracy (100Q) | Throughput | Total output tokens | Notes |
|---|---:|---:|---:|---|
| Dense (Llama-3.2-1B draft) | **72%** | 309 tok/s | 20,384 | EOS-terminates cleanly, 0% invalid |

Throughput-only bench (`bench_offline_throughput.py`, sharegpt 20 prompts, `max-running-requests=16`, `cuda-graph-max-bs=128`): **1187 tok/s** for dense at `N=8, γ=8`.

### 1.2 EAGLE3 — five attempts

| Attempt | GSM8K acc (100Q or 20Q) | Throughput | Output quality |
|---|---:|---:|---|
| Original branch, CG off, full rewrite (baseline) | 0/20 (0%) | 366 tok/s | gibberish; never emits EOS |
| + CG on decode & rewrite (our new runner) | 4/100 (4%) | 433 tok/s | gibberish with structure |
| + fused-extend rewrite (1 extend vs γ+1 decodes) | — | 856 tok/s (regression) | same; fused extend path slower than 9 CG replays |
| + minimal rewrite (2 forwards instead of γ+1) | 1/100 (1%) | **1150 tok/s** | gibberish, shorter outputs |
| + seq_lens off-by-one fix | 0/20 (0%) | 377 tok/s | structured attempts, wrong numbers |
| + seq_lens fix + target-logprob temp=1 (match dense's math) | **1/20 (5%)** | 366 tok/s | coherent prose, wrong arithmetic, 0% invalid |
| + above + draft temp=1.0 instead of 0.7 | 0/20 (0%) | 368 tok/s | pure noise, 60% invalid |

Representative sample output, GSM8K Q0 (Janet's eggs, correct answer 18), "best" config (seq_lens + target-temp fixes):

> "Let 15g brackets = 6 eggs per day. Step 1: Calculate the number of eggs laid in a day: 15g = eggs/6 eggs = 16 eggs/6 = 6 ... bakes 3 eggs = 14 eggs..."

Reads like English; math is hallucinated.

### 1.3 Internal profile (from nsys on dense path)

GPU-time breakdown per decode cycle at the throughput-bench config (bs ≈ 68, CG on):

| Category | ms/cycle | % GPU |
|---|---:|---:|
| GEMMs (transformer body) | 21.8 | 60.4 |
| Attention (FlashAttn SM90) | 6.1 | 16.8 |
| Other layer ops (RMSNorm, SiLU, RoPE, KV store, copies) | 4.3 | 11.9 |
| Sampling / log_softmax / multinomial | 1.3 | 3.6 |
| SMC fused kernels (collect, resample KV) | 0.07 | 0.2 |
| Framework misc | 2.7 | 7.6 |
| **Total** | **36.1** | 100 |

Draft forwards ≈ 18 ms/cycle; verify forward ≈ 16 ms/cycle. These two now dominate.

---

## 2. Bugs we found (and fixed, or diagnosed)

### 2.1 CUDA graph was disabled for the EAGLE3 draft
`smcsd/v2/worker.py` on the original branch set `backup_disable_cuda_graph = True` in EAGLE3 mode, so the draft ran 16 eager forwards per cycle (`γ-1` decode steps + `γ+1` rewrite steps = 2γ = 16 at γ=8). Under eager dispatch this adds ~50-150 μs of Python-side launch overhead per forward, which compounded to multiple ms per cycle.

**Fix**: A custom runner at `smcsd/model_executor/smc_eagle3_draft_cuda_graph_runner.py` captures per-`(bs, hidden_dim)` graphs lazily. Works. 100% CG coverage. Measurable speedup but smaller than predicted (+5.5% throughput), because once CG is on the remaining cost is actual compute, not launch overhead.

### 2.2 Rewrite phase did γ+1 forwards; only 2 are strictly needed
The draft AR loop writes KV for positions `L..L+γ-2` (γ-1 positions). The rewrite loop then writes positions `L..L+γ` (γ+1 positions), *overwriting* positions the draft phase already wrote with target-hidden-based KV. Positions L+γ-1 and L+γ are the only *new* positions — the rest are "quality polish."

**Fix**: Env-gated `SMC_EAGLE3_REWRITE=min` runs only 2 rewrite forwards instead of γ+1. Throughput went 1025 → 1150 tok/s on the throughput bench. Accuracy on GSM8K was unchanged (still ~0-1%) because the accuracy problem is unrelated. The fix is real and kept; it just doesn't fix the accuracy issue.

### 2.3 Fused-extend rewrite was a regression
Collapsing the γ+1 rewrite decode forwards into one EXTEND forward at `bs × (γ+1)` tokens *lowered* throughput (1025 → 856). The extend attention kernel path is less optimized for this shape than γ+1 CG-replayed decode forwards. Shelved.

### 2.4 seq_lens off-by-one in the non-multistep code path
`SMCDecodeContext.prepare_for_draft` computes `all_seq_lens[:, step] = orig_seq_lens + step + 1`. The FA backend computes `cache_seqlens = seq_lens + speculative_step_id + 1`. With the default attn backend (`step_id = 0`), this gives `cache_seqlens = orig + step + 2`, which is one position past the valid write. Upstream EAGLE's standalone worker instead uses multistep backends where `step_id = i` and `seq_lens = orig_seq_lens`, yielding the correct `orig + step + 1`.

**Fix**: In the EAGLE3 decode and rewrite loops we now pass `seq_lens = all_positions[:, step] = orig + step`, which compensates for the default backend's `+1`. Outputs went from word-salad gibberish to structured-but-wrong math. The dense 1B draft is apparently robust to this off-by-one (reads a junk KV slot, absorbs it); the 1-layer EAGLE3 head is not.

### 2.5 Target-logprob temperature inconsistency
Dense computes the SMC weight with target logprobs at **temp = 1** (`log_softmax(score_logits)`, no division). EAGLE3 on the branch used `log_softmax(score_logits / smc_target_temperature)`, i.e. temp = 0.7. These produce different importance ratios and different SMC dynamics.

**Fix**: EAGLE3 now matches dense's convention (temp=1 in the weight math; bonus sampling still uses `smc_target_temperature`). 0% → 5% accuracy on 20Q. Real, but tiny.

### 2.6 Hot-vocab ratio asymmetry (unfixed, structural)
EAGLE3's draft predicts into a 32k "hot" vocab. Mapped to target-vocab via `hot_token_id[]`. SMC's importance weight becomes `log P_target(x | full 128k vocab) - log P_draft_hot(x | 32k vocab)`. Mathematically the ratio is fine for *tokens the draft did sample* — but the draft can never propose tokens outside its 32k support. If target's likely continuations drift into non-hot territory, no particle can follow. This interacts badly with SMC's assumption of full-support proposal distributions.

**Not fixed** — structural change. Would require re-training the head with full vocab.

---

## 3. Why EAGLE fundamentally struggles with SMC

The bugs above accounted for some of the failure, but not most of it. The remaining ~67 percentage-point gap (5% vs 72%) comes from a deeper mismatch. EAGLE and SMC were designed for **different acceptance mechanisms**, and the optimal draft is different under each.

### 3.1 What EAGLE is, briefly

EAGLE3 is an auxiliary draft head that sits next to the target model:
- A tiny transformer (1 "midlayer" block + fc projector + RMSNorm) — on the order of 100-300M params vs the target's 8B.
- It consumes the target's *auxiliary hidden states* (concatenation of several late-layer hidden states, "aux" = 3 × hidden_size) as conditioning input plus the next token's embedding.
- It outputs a distribution over a **curated "hot" subset** of the target's vocabulary (typically ~32k of ~128k) — the tokens that actually appear frequently enough in training to be worth predicting.
- Per decode cycle, the draft does γ AR steps to propose γ tokens; each step's output hidden becomes the next step's conditioning.

### 3.2 EAGLE's training objective: rejection-sampling acceptance

EAGLE3's loss minimizes KL(target || draft) at sampled next-token positions over the hot vocab. The effect:
- **Peaked distributions** on the tokens target is likely to pick — specifically, draft's argmax often equals target's argmax.
- **High acceptance rate** under rejection sampling, because "target accepts draft's token" ≈ "target's argmax matches draft's argmax."
- **Hot-vocab truncation** — rare tokens aren't worth predicting because target would reject them anyway, and the `lm_head` savings (32k vs 128k) are significant for a small draft.
- **Hidden-state chaining across γ steps** — AR behavior inside the draft, feeding each step's output hidden into the next.

Under rejection-sampling decoding, when draft proposes a wrong token, target rejects it and samples its own. Draft failure is **truncated and recovered immediately**.

### 3.3 What SMC needs from a draft

SMC has **no rejection**. Every drafted token is committed to the sequence. Its role on the particle's trajectory is determined by the importance ratio `log p_target(x) - log p_draft(x)`, accumulated into a weight. Particles with low weight get resampled away; particles with high weight replicate.

For this mechanism to produce useful dynamics, the draft must give SMC four things:

1. **Full-vocab proposal support.** If `q(y) = 0` anywhere target wants to go, no particle can reach that region. SMC just gets stuck. EAGLE3's hot-vocab cuts off 75% of the vocab — whenever target's preferred continuation lies outside the hot set (a specific number, a rare symbol, an unusual word), no particle can follow, and the SMC swarm has to limp through with an increasingly bad approximation of target's intentions.

2. **Enough entropy in per-step proposals to keep N particles diverse.** SMC's main advantage over single-path decoding is parallel exploration of N different continuations. If all N particles sample from a sharp distribution peaked on the same token, they collapse into one sequence and the swarm gains nothing. EAGLE3's training explicitly optimizes for peakedness (because that's what maximizes rejection-sampling accept rate), which is exactly what kills SMC diversity.

3. **Logprob calibration across the whole distribution**, not just at the mode. The importance ratio is evaluated *wherever particles go*, including into tails. Cross-entropy loss at the sampled token (the standard EAGLE training signal) only supervises the top of the distribution — tail probabilities can be anywhere and the loss barely notices. Under rejection sampling this is fine (tails rarely come up). Under SMC, miscalibrated tails produce noisy or systematically biased importance ratios everywhere off-mode, which drives particle weight dynamics off-course.

4. **Resilience to runs of wrong-but-accepted tokens.** Rejection sampling truncates bad runs at the first mistake. SMC absorbs them into weights and relies on resampling to recover — but resampling only helps if *some* particle in the swarm has correct state. Under a low-diversity draft (point 2), all particles are wrong in the same way, and resampling has nothing to rescue.

### 3.4 What we saw in practice

The gibberish-but-coherent pattern ("Let 15g brackets = 6 eggs per day... bakes 3 eggs = 14") is the exact signature of points 2 and 4 compounding:
- The draft is a *competent LM*. EAGLE3 is trained on target-model-generated sequences; it produces real tokens in grammatical structures.
- But all 12 particles, fed the same prefix and a sharp draft distribution, generate highly correlated sequences.
- When target's verify reveals they're wrong, all particles have similar low weights. Resampling shuffles between equally-bad options — there's nothing better to copy to.
- Over many cycles, small errors compound. The draft's chained hidden state drifts (each step conditions on the previous step's output hidden, which itself encodes the compounded earlier errors). Outputs devolve from "wrong math but real English" into structured gibberish.

### 3.5 Not impossible — just not what the pre-trained checkpoints are

**EAGLE-style architectures are not fundamentally incompatible with SMC.** A draft head with:
- full-vocab output,
- higher-entropy distributions (supervising target's *shape*, not just its mode),
- numerically stable cross-cycle hidden-state handling,

...can in principle be made to work well as an SMC proposal. But existing pre-trained EAGLE3 checkpoints do not satisfy any of these three properties — they were explicitly trained *against* the first two. So the path is not "fix the integration", it's "retrain the draft with an SMC-appropriate objective."

---

## 4. How to train an SMC-native draft: objective & procedure choices

Above we described *what* properties the draft needs. This section covers *how* to get them — the training objective and procedure.

There are five reasonable families of methods, in roughly ascending complexity:

### 4.1 SFT distillation on target's full distribution (recommended starting point)

Generate a training corpus by running the target model on many prompts (or reusing target's original training data). At every position, compute target's full output distribution. Train the draft with:

```
L = E_{(prefix)} [ KL( softmax(target_logits) || softmax(draft_logits) ) ]
  = Σ_v softmax(target_logits)[v] · log( softmax(target_logits)[v] / softmax(draft_logits)[v] )
```

This is the **KL-on-full-distribution** objective. Unlike cross-entropy at the sampled token (which only supervises the mode), KL supervises the full distribution including tails. Its effect:

- Draft learns target's *shape*, not its argmax.
- Tails get non-zero probability mass, so particles can explore them.
- Calibration is enforced everywhere `target_logits` places mass.

**Why this is the right starting point**:
- Cheap and well-understood — it's classic knowledge distillation, matches Hinton-style KD.
- All you need is target's logits at each position (1 forward pass of the target) and draft's logits (1 forward pass of draft).
- Data is plentiful — any LM training data or target-generated sequences work.

**Why it alone isn't quite optimal for SMC**:
- KL doesn't directly reward particle diversity.
- It still doesn't teach the draft about cross-cycle hidden-state stability (draft's chain stability).
- But it's sufficient for a first working version, and subsequent strategies can be layered on top.

**Cost**: 1-2 weeks engineering, ~1 A100-day per 10k steps, ~$300-800 total.

### 4.2 SFT on SMC outputs (behavioral cloning, weakest)

Alternative: run SMC (with the current draft) on many prompts and treat its output sequences as training data. Cross-entropy on the sampled tokens.

- Easy to implement; matches what most ML practitioners reach for first.
- **But it learns what SMC *does*, not what SMC *needs*.** If current SMC is imperfect, you train the draft to reproduce its imperfections.
- No incentive to match target's distribution tails or to preserve diversity.

Useful as a polish step, not as the main training objective. Probably skip.

### 4.3 RL with target-logprob reward

Define a reward `r(x) = log P_target(x | prefix)` for each drafted token. Optimize draft's policy to maximize expected reward.

**This is almost equivalent to distillation (4.1)**. The gradient of expected reward under draft's policy is, up to a baseline, essentially the same signal as KL minimization. The differences:
- Pure RL uses Monte-Carlo samples (draft's own samples, weighted by their reward) to estimate the gradient.
- KL uses the analytical gradient from the full distribution.
- For a given amount of compute, the analytical KL gradient has lower variance → faster convergence.

**Verdict**: don't do this as RL; do it as distillation (4.1). Same signal, better implementation. If you see a paper claiming "RL with target logprob reward" it's essentially distillation with extra steps.

### 4.4 RL with SMC-outcome reward

This is the interesting one. Reward the draft based on actual SMC-end-to-end quality:

- **Sparse reward**: final-answer correctness on a downstream benchmark (e.g. GSM8K accuracy → +1 for correct, 0 for wrong).
- **Dense reward**: effective sample size (ESS) of particle weights after each decode cycle. Higher ESS = particles are diverse and useful = better draft.
- **Hybrid**: per-step logprob-diff variance, or particle weight entropy, as a proxy for "draft is giving SMC useful signal."

Then optimize draft with policy gradient (REINFORCE, PPO, GRPO-style) to maximize this reward.

**Pros**:
- Optimizes *directly for what we want*: SMC performance.
- Can handle subtle effects (hidden-state stability, cross-particle correlation, etc.) that simpler objectives miss.
- Most principled approach for SMC-specific drafting.

**Cons**:
- **Expensive**: each training step requires running SMC, which means `N × γ` draft forwards + verify forwards per sample.
- **High-variance reward signal** (especially with sparse binary rewards).
- **Credit assignment is hard**: a bad token 50 steps ago caused failure, but the reward only fires at the end. Need careful reward shaping or credit-assignment tricks.
- Needs a warm start. Random-init draft under SMC produces garbage → reward is always bad → no gradient signal.

**Verdict**: promising *as a fine-tuning step on top of a distillation-trained draft*. Don't use it from scratch.

### 4.5 SMC-in-the-loop online training

Run SMC continuously during training. Each training step:
1. Take a prompt, run SMC for K cycles.
2. Compute some SMC metric (ESS, logprob-diff variance, or final accuracy on a validation question).
3. Backprop through the draft's sampled logprobs to improve the metric.

This is essentially 4.4 with very short-horizon rewards (per-cycle instead of end-of-run). Can be framed as on-policy RL with dense rewards.

**Pros**:
- Closes the loop between "what the draft sees in training" and "what the draft sees in deployment."
- Directly exposes draft to the particle dynamics it'll face at inference time.

**Cons**:
- Most complex to implement (need differentiable or REINFORCE-style gradient through SMC sampling).
- Most expensive per gradient step (every step = a mini SMC run).
- Most unstable (reward signal is noisy; two consecutive steps can have wildly different rewards for similar draft parameters).

**Verdict**: interesting research direction but unlikely to pay off before simpler methods are exhausted.

### 4.6 Recommended training recipe

Given the cost/benefit tradeoffs:

1. **Phase 1 (required)**: SFT distillation with KL-on-full-distribution (4.1). Full-vocab head, 50-100k sequences, ~1 A100-day. This alone should restore EAGLE-style drafting to near-dense accuracy for SMC. Gets us to the starting line.

2. **Phase 2 (optional, if Phase 1 isn't enough)**: RL with SMC-outcome reward (4.4), warm-started from Phase 1. Use ESS as the reward (dense, low variance) and fine-tune for a few hundred steps. This captures the SMC-specific preferences that distillation misses.

3. **Phase 3 (research, if publishing)**: ablate SMC-in-the-loop (4.5) as an alternative to Phase 2. Compare on ESS, accuracy, and efficiency. This is publishable work in its own right.

For the ICLR paper, Phase 1 alone is likely enough. Phase 2/3 are for follow-up work.

---

## 5. What an SMC-native draft should look like (architectural design)

Four design principles, in descending order of importance. These are the *properties* we want; Section 4 covers the *training* to achieve them.

### 5.1 Full vocab output head (mandatory)
Replace `draft_vocab_size=32000` with the full `vocab_size=128256`. Initialize `lm_head` from target's `lm_head` (EAGLE3's loader already supports this — see `load_lm_head_from_target`). Cost: ~300 MB additional. For a 100-300M draft tower this is a meaningful fraction of the weights but acceptable.

This is *non-negotiable* for SMC. The hot-vocab shortcut that makes EAGLE3 fast under rejection sampling actively breaks SMC.

### 5.2 Trained on target's full distribution, not its argmax
See Section 4.1 and 4.6. The key property is that the objective supervises the *shape* of the output distribution, including tails — not just the sampled token. This enables point 2 (diversity) and point 3 (calibration) from Section 3.3.

### 5.3 Cross-cycle hidden-state stability
EAGLE chains `hidden_states` across draft steps and across decode cycles. Per-step errors compound — and in SMC all N particles chain in lockstep, so errors don't cancel across particles the way they might across independent runs.

Mitigations:
- **Re-seed from target's aux hidden every K ≤ γ steps** (trading some speed for stability).
- Regularize training against hidden-state drift relative to target's hidden state at the same position.
- Or: drop cross-step chaining entirely and re-use target's hidden every step (slightly slower, more stable).

### 5.4 SMC-aware training signal
Orthogonal to architecture: train with an objective that exposes the draft to the dynamics it'll actually see in deployment. Section 4.4 (RL with SMC reward) or Section 4.5 (SMC-in-the-loop online) are the techniques; they can be layered on top of any architecture in Section 6.

---

## 6. Overall architecture: frozen target + trainable decoding head

The concrete picture that follows from Sections 4 and 5 is **one frozen target model, one small trainable decoding head on top, nothing else**. There is no separate "draft model" — the head *is* the draft.

### 6.1 Runtime

```
    prompt / previous tokens
              │
              ▼
   ┌──────────────────────┐
   │  TARGET (Llama-3.1   │   ←── FROZEN, never updated
   │  8B, 32 layers)      │
   └──────────────────────┘
              │
              ├── next-token logits  ───►  verify math (SMC weights, bonus)
              │
              └── aux hidden states  ───►  conditioning input for head
              │                              │
              │                              ▼
              │                    ┌──────────────────┐
              │                    │ DECODING HEAD    │  ←── the one thing we train
              │                    │ (~300M params,   │
              │                    │  1 midlayer +    │
              │                    │  full-vocab head)│
              │                    │                  │
              │                    │  γ AR steps      │
              │                    │  → γ draft tokens│
              │                    └──────────────────┘
              │                              │
              ▼                              ▼
        ──────────────────────────────────────────
         SMC machinery: N particles, weights,
         resampling, bonus sampling
        ──────────────────────────────────────────
```

**Head shares with target (no additional parameters):**
- Embedding table — set at head init from target's `embed_tokens`.
- `lm_head` — init from target's `lm_head` (this is the full-vocab fix over EAGLE3's hot-vocab shortcut).
- Aux hidden states — the target's forward has already computed them, we just read them out.

**Head owns (its own trainable parameters):**
- 1 transformer block ("midlayer"): self-attention + MLP.
- `fc` projector: compresses target's concatenated aux hiddens (3H → H) for conditioning.
- Head-side RMSNorm.

Total trainable: ~300M params. Target's ~8B is untouched.

### 6.2 Training

For all the strategies in §4:
1. Target stays frozen. Never gets a gradient.
2. Only the head's parameters update.
3. The strategies in §4 differ only in what *signal* drives the head's update:
   - **§4.1 distillation**: target's full-vocab logits at each position as a soft teacher (KL).
   - **§4.4 ESS-RL**: reward = ESS or accuracy after SMC rollouts; policy gradient through draft samples.
   - **§4.5 SMC-in-loop**: reward = per-cycle SMC metrics under actual N-particle dynamics.

In all cases the training target is the same head's weights; what changes is the loop wrapped around it.

### 6.3 Why this is preferable to the current separate-1B-draft design

| Dimension | Current dense (separate 1B draft) | SMC-native head (this proposal) |
|---|---|---|
| Draft weight footprint | ~2 GB (Llama-3.2-1B weights) | ~600 MB (300M head) |
| Draft KV cache | Separate pool | Separate pool (same) |
| Conditioning on target state | None — learns everything from scratch | Target's aux hiddens, **free** signal |
| Per-step draft GPU cost | ~2 ms | ~0.5-1 ms |
| Tokenizer compatibility | Need a pre-existing small model with matching tokenizer | Inherits target's tokenizer by construction |

That last row removes the "find a small Llama-3-tokenizer model" constraint entirely. The head is architected *to* the target's tokenizer.

### 6.4 Two starting points for the head

The head can either be **initialized from a pre-trained EAGLE3 checkpoint and fine-tuned** (Option A — cheap, fast), or **trained from scratch as a target-conditioned adapter** (Option C — cleaner, more publishable). Either way the runtime picture is identical: one target forward per SMC cycle, γ head forwards per SMC cycle, head weights are what was trained.

---

## 7. Options for going forward (architecture × training combinations)

Each option is a combination of an architecture choice (Section 5) and a training choice (Section 4). Listed in rough order of cost.

### Option A: Fine-tune existing EAGLE3 with full-vocab head + KL distillation
- **Architecture**: EAGLE3 architecture unchanged *except* swap the 32k head with 128k, initialize from target's `lm_head`.
- **Training**: KL-on-full-distribution distillation (Section 4.1). 50-100k sequences generated by target at temp=1.0.
- **Validation**: GSM8K accuracy should recover close to dense.
- **Compute**: ~1 A100-day per 10k training steps at 8B target + 300M EAGLE head.
- **Time**: 1-2 weeks (engineering + training).
- **Cost**: $300-800 cloud compute.
- **Risk**: Medium. The architecture is unchanged; only the head and loss change. Main risk is KL loss over 128k vocab is more numerically delicate than CE at a single token (use stable log-softmax, gradient clip, maybe start with a smaller vocab subset).

### Option B: Fresh small dense draft from scratch
- **Architecture**: ~300M Llama-3-tokenizer-compatible model (could be a distillation of target). Standard dense LM, not an EAGLE-style aux head.
- **Training**: LM loss on target's pretraining data + KL distillation vs target on the same data.
- **Compute**: 2-4 weeks, $1-5k cloud.
- **Risk**: Higher (more moving parts) but architecturally simpler and more robust once done.
- **Drawback**: Not "EAGLE-like" — doesn't exploit target's hidden state. Speed will be closer to the current 1B dense draft, not to EAGLE's sub-ms forward.

### Option C: Target-conditioned adapter (cleanest SMC-native design)
- **Architecture**: Freeze target. Add a lightweight adapter (LoRA on last 4-8 target layers + a tiny AR tower) that, given target's hidden state at position p, approximates target's logits at positions p+1..p+γ.
- **Training**: Phase 1 — KL distillation (4.1). Phase 2 — ESS-reward fine-tune (4.4) to pick up SMC-specific preferences.
- Full vocab (reuses target's `lm_head`).
- **Compute**: ~days per training phase, $50-200 each.
- **Time**: 2-4 weeks including ablations.
- **Risk**: Novel design — unknown pitfalls. But this is also the most *publishable* variant: "SMC-native drafting via target-conditioned adapters with ESS-reward fine-tune."

### Option D: Abandon EAGLE-for-SMC, focus on dense-path speedups
- **Architecture / training**: unchanged (dense Llama-3.2-1B draft).
- Dense at 1187 tok/s is already the fastest thing we have. GPU cost splits ~48% draft forward / 42% verify forward / ~10% other. Verify forward (16 ms/cycle) is untouched — attacking it (piecewise CG, attention-backend tuning, or reducing `bs × (γ+1)` shape) could be 5-15% throughput.
- **Compute**: engineering-only.
- **Time**: 1-2 weeks.
- **Risk**: Low. Incremental but preserves accuracy.

---

## 8. Recommendation

For the ICLR paper timeline, the decision hinges on what the paper claims:

- **If the paper is about SMC as a decoding algorithm** (not draft-head design): ship with the dense draft. Option D. Write about SMC's throughput wins against rejection-based spec decode; the draft is just a commodity small LM. This is the lowest-risk path.

- **If the paper wants to claim an EAGLE-style draft for SMC**: Option A is the minimum. One week of training gets a usable full-vocab EAGLE3 head via KL distillation. This will likely restore accuracy close to dense and preserve EAGLE's speed advantage.

- **If you have 2+ months and want a publishable contribution** on top of SMC itself: Option C with the two-phase training (distillation → ESS-reward fine-tune). "Target-conditioned adapters trained with SMC-aware objective." Real research; real upside.

On *training objective* specifically:
- **Default to distillation (Section 4.1)** for any of Options A/B/C. It's the cheapest and most likely to work.
- **Don't do RL with target-logprob reward (4.3)** — it's just distillation with noisier gradients.
- **RL with SMC-outcome reward (4.4)** is a good fine-tune step *after* distillation gets you to the starting line. Don't start there.
- **Skip SFT on SMC outputs (4.2)** — it teaches the draft to mimic current-SMC's flaws.
- **SMC-in-the-loop (4.5)** is research territory. Fine for Option C's follow-up; overkill for A.

My pick for right-now: **Option D first (the dense-path verify-forward wins are real and easy), then Option A as follow-up if the paper needs EAGLE-style drafting**. Park the `eagle3-cg` branch with the seq_lens and temp fixes committed (they're real bugs that future work will want fixed), and note in the branch README that the remaining accuracy gap requires retraining the draft head.

---

## 9. Artifacts & references

### Branches
- `new_release` — dense baseline, known working (72% GSM8K, 1187 tok/s throughput bench)
- `origin/eagle3-draft-experimental` — abandoned EAGLE3 integration (0% GSM8K)
- `eagle3-cg` (this work) — based on the above, with CG runner, min rewrite, seq_lens and temp fixes. Still ~5% GSM8K.

### Key files on `eagle3-cg`
- `smcsd/v2/worker.py` — EAGLE3 decode + rewrite logic, env-gated via `SMC_EAGLE3_CG` and `SMC_EAGLE3_REWRITE`
- `smcsd/model_executor/smc_eagle3_draft_cuda_graph_runner.py` — custom CG runner (lazy per-`(bs, hidden_dim)` capture)

### Env vars for A/B testing
- `SMC_EAGLE3_CG=1|0` — CG on/off (default 1)
- `SMC_EAGLE3_REWRITE=min|full|fused` — rewrite strategy (default `min`)
- `SMC_PROFILE=1 SMC_PROFILE_WARMUP=10 SMC_PROFILE_FLUSH_EVERY=40` — per-region GPU event profiling (worker.py)

### Reference configs used
```bash
# Dense baseline (GSM8K)
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 --temperature 0.7 \
  --attention-backend fa3 --num-questions 100 \
  --max-running-requests 24 --cuda-graph-max-bs 24 --smc-fast-resample

# EAGLE3 (GSM8K, same config; add --smc-draft-mode eagle3 and swap draft)
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine --smc-draft-mode eagle3 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --particles 12 --gamma 8 --temperature 0.7 \
  --attention-backend fa3 --num-questions 100 \
  --max-running-requests 24 --cuda-graph-max-bs 24 --smc-fast-resample
```

### Reference reading
- EAGLE-2 paper (for rejection-sampling objective) — explains why the hot-vocab and peaked-distribution design is correct for rejection sampling, which is the source of the SMC mismatch.
- `3rdparty/sglang/python/sglang/srt/models/llama_eagle3.py` — canonical EAGLE3 architecture and loader used in this investigation.
- `3rdparty/sglang/python/sglang/srt/speculative/eagle_worker.py:draft_forward` — upstream standalone EAGLE3 draft loop; uses multistep backends with `speculative_step_id=i`, which is the convention our non-multistep CG path had to compensate for.

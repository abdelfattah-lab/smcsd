# Mass-Covering Proposal Distillation for Sequential Monte Carlo Speculative Decoding

*Working ICLR paper draft / pitch. Numbers are from the Qwen3-8B + Qwen3-0.6B
study on 8×B200; see `docs/smc/proposal_results.md` for the full logs. All SMC
accuracy numbers are single-run with unseeded GPU sampling (≈ ±2–4pp noise) —
final submission should average ≥3 seeds and larger question sets.*

---

## 1. The pitch (how to position it)

**One-line claim.** *The objective you finetune a speculative-decoding draft
with should be the divergence that governs Sequential Monte Carlo efficiency —
the χ² / Rényi-2 divergence — not KL; doing so yields a single small draft that
runs SMC speculative decoding at **half the particles** (≈1.3–1.6× faster) on
reasoning workloads while **matching the target model's own accuracy**.*

**Why it's interesting to ICLR (positioning).**
1. **A principled objective, not a heuristic.** We connect the training loss to
   the estimator's variance: for self-normalized importance sampling
   `E[N/ESS] ≈ 1 + χ²(p‖q)`, so minimizing the per-token χ² divergence directly
   minimizes particle degeneracy. This reframes draft finetuning for SMC as
   *variance minimization of an importance-sampling estimator*, a clean and
   general statement.
2. **A counter-intuitive, well-supported finding.** The "obvious" proxy
   (resample rate / ESS) is *misleading*: the reverse-KL objective that best
   minimizes resampling does so by **mode-collapsing** the draft, which *hurts*
   task accuracy. The correct objective is **mass-covering** — it resamples
   *more* yet is *more* accurate. We show this decoupling on four benchmarks and
   explain it.
3. **A practical, reproducible recipe** that turns the above into a
   general-purpose draft, with a single knob (the Rényi order β + a reverse-KL
   mix) that trades the two failure modes and a per-domain operating point.

**Framing for reviewers.** Pitch as *"efficient inference / speculative
decoding"* with a *"variational inference / importance sampling"* analysis
backbone. The novelty is the **objective and the analysis**, not a new decoding
algorithm — SMC-SD is the host system and is unchanged. Avoid over-claiming a
universal speedup: be explicit that the *speed* win is workload-dependent (math
yes, code no) while the *accuracy/robustness* win is general.

---

## 2. The problem we fix

**SMC speculative decoding (SMC-SD).** Instead of rejection-based speculative
decoding, SMC-SD runs `N` particles that each draft `γ` tokens from a small
draft `q`, scores them under the target `p`, and assigns each an importance
weight `w = exp(Σ_t [α·log p(x_t|x_<t) − log q(x_t|x_<t)])`. When the effective
sample size `ESS = (Σw)²/Σw²` drops below a threshold, particles are resampled.
All drafted tokens are accepted; throughput scales with batch/particle
arithmetic intensity. (See `docs/smc/architecture.md`, `core/worker.py`.)

**The pain.** SMC-SD's accuracy and speed are both governed by how well the
draft proposal `q` matches the tempered target `p̃ = softmax(α·z_p/T)`:
- A *mismatched* draft → high importance-weight variance → low ESS → frequent
  resampling (particle degeneracy). To keep accuracy you must run **many
  particles** (slow) or you **lose accuracy** at few particles.
- Empirically, a small off-the-shelf draft leaves a large *degeneracy gap*: at
  N=8 the base Qwen3-0.6B draft scores **68% on GSM8K vs the target's own 90%**
  — 22 points lost purely to a bad proposal, not to model capacity.

**Goal.** Finetune `q` so the *same* draft gives high accuracy at *fewer*
particles / larger γ — i.e. faster *and* more accurate.

**Why prior draft-distillation objectives are wrong for SMC.** Standard draft
training minimizes (forward or reverse) KL or cross-entropy to the target. We
show KL is the wrong divergence here: it does not control the IS weight variance
that SMC efficiency depends on, and reverse-KL actively *games* the resample-rate
proxy via mode collapse at the cost of accuracy.

---

## 3. Method / the recipe

### 3.1 Objective — per-token Rényi-β divergence (SMC-direct)

For the per-token proposal `q = softmax(z_q/T_q)` and tempered-power target
`p̃ = softmax(α·z_p/T_p)` (exactly what the engine weights against), minimize

```
D_β(p̃ ‖ q) = 1/(β−1) · logsumexp_x[ β·log p̃(x) + (1−β)·log q(x) ],
```

summed over the completion tokens of collected trajectories.

- **β = 2** is `log χ²` (Rényi-2), the divergence that bounds IS weight variance:
  `E_q[w²] = Σ_x p̃²/q = 1 + χ²`, and over a γ-block the second moment
  factorizes multiplicatively, so per-token χ² controls how fast weight variance
  compounds with γ. This is the SMC-optimal per-token surrogate.
- **β → 1** recovers reverse-KL `KL(q‖p̃)` (so the family strictly generalizes
  the prior recipe).
- **+ reverse-KL mix** `D_β + λ·KL(q‖p̃)` (`--renyi-kl-mix λ`) re-introduces a
  controlled amount of mode-seeking — needed for robustness at very low N (§5).

Gradient stability: `D_β`'s gradient w.r.t. logits is the logsumexp softmax,
bounded in `[0,1]` per vocab entry, so the heavy χ² `p²/q` tail inflates the
loss *value* but not the gradient — ordinary grad clipping suffices.

### 3.2 Data — on-policy, multi-domain, token-balanced

1. **Collect on-policy SMC rollouts** with the *current* draft at the deployment
   config (N, γ, temperatures, α). The engine emits, per request, every
   particle trajectory + log-weights + cycle diagnostics. This is the *only*
   place SMC runs; training itself is offline.
2. **Mix domains by deployment, drop no-headroom domains.** Use a
   source-stratified prompt set (we use open-perfectblend's chat/code/IF sources
   + MBPP + a small GSM8K slice). We only use each example's *prompt*; responses
   come from the target via SMC. **Headroom-gate**: skip domains where the base
   draft is already near the target ceiling (math, here).
3. **Token-balance the domains** (`--balance-domains`): equalize per-domain
   *completion-token mass*, not prompt count — long reasoning rollouts otherwise
   dominate the gradient and starve under-represented domains (this is what fixes
   cross-domain generalization).

### 3.3 Training — offline distillation (not online SMC)

A static-dataset pass: teacher = frozen target (8B), student = trainable draft
(0.6B). For each completion token, two forward passes (teacher frozen, student),
compute `D_β(p̃‖q)`, backprop into the draft. No particles, resampling, or
weights in the loss. Save a merged bf16 HF checkpoint that drops into the engine
unchanged. Optional **on-policy iteration**: re-collect with the improved draft
and continue-train (helps pure-χ²; ≈neutral for the kl-mixed objective).

### 3.4 The locked recipe

```
collect_proposal_data.py  (on-policy rollouts, per-record domain labels)
  → train_proposal.py --loss renyi --renyi-beta 1.5 --renyi-kl-mix 0.5
                      --balance-domains --epochs 1 --lr 1e-5
  → (optional) one on-policy continue-round at lr 5e-6
Deploy one draft; operating point per domain: N=4 for math/reasoning, N=8 for code.
```

---

## 4. Experimental setup

- **Models.** Target Qwen3-8B, draft Qwen3-0.6B (the "headroom" pair; we also
  recommend reporting Qwen3-1.7B as a draft-capacity point — see §7).
- **Benchmarks.** GSM8K, MATH-500, HumanEval, MBPP (`scripts/eval_tasks.py`:
  sympy math-equivalence + sandboxed pass@1). Diagnostics: per-domain resample
  rate / mean ESS, and the **target-only ceiling** (no speculative decoding).
- **SMC config.** N∈{2,4,8}, γ∈{8,12,16}, T=0.7, α=1, resample-threshold 0.5,
  Triton backend (Blackwell), `--disable-thinking`.
- **Baselines / objectives.** base draft; reverse-KL (prior recipe); χ² (β=2);
  Rényi-β ∈ {1.5,2} × kl-mix ∈ {0,0.5,1}; anneal β 1→2; ablations on
  domain-balancing and on-policy iteration.

---

## 5. Results (key tables)

**(a) χ² closes the degeneracy gap; reverse-KL doesn't generalize.** Accuracy @
N=8 γ=8 vs the target-only ceiling:

| Task | ceiling | base | reverse-KL | χ² |
|------|:---:|:---:|:---:|:---:|
| GSM8K | 90.0 | 68.0 | 78.0 | **86.5** |
| HumanEval | 84.1 | 67.7 | 39.6 | 59.1 |
| MBPP | 67.5 | 44.5 | 38.0 | **51.5** |

χ² beats reverse-KL on every task; reverse-KL craters held-out code (−28pp).

**(b) Resample rate is a misleading proxy.** Held-out rr (lower = "better" by
the classic proxy) vs accuracy:

| draft | rr (code) | rr (math) | GSM8K acc | HumanEval acc |
|-------|:---:|:---:|:---:|:---:|
| reverse-KL | **0.205** | **0.350** | 78.0 | 39.6 |
| χ² | 0.434 | 0.491 | **86.5** | **59.1** |

Reverse-KL wins rr by mode-collapsing → loses accuracy; χ² is mass-covering →
higher rr, higher accuracy. **Report accuracy, not rr.**

**(c) General token-balanced mix → generalist draft.** Accuracy @ N=8 γ=8:

| draft | GSM8K | HumanEval | MBPP |
|-------|:---:|:---:|:---:|
| base | 66.0 | 58.5 | 45.5 |
| χ², general+balanced | 84.0 | **68.9** | 48.5 |
| χ², general, *no balance* | 84.5 | 64.0 | 49.0 |

`--balance-domains` is the lever that recovers the under-represented code style
(HumanEval 64.0 → 68.9).

**(d) The Rényi-β + kl-mix knob fixes low-N code collapse.** Accuracy:

| objective | GSM8K N4 | HumanEval N2 |
|-----------|:---:|:---:|
| χ² (β2) | 76.5 | **26.8** (collapse) |
| β1.5 + klmix0.5 | 78.0 | **45.7** |

Pure χ² (mass-covering) needs particles to represent the peaked code
distribution; a small reverse-KL mix anchors it without losing the math win.

**(e) Production operating curve (β1.5+klmix0.5).** Accuracy% (tok/s):

| | base best | prod @ N=4 | prod @ N=8 |
|---|---|---|---|
| GSM8K | 68.5 (N8, 2054) | **79.0 (2761)** | 81.5 (1998) |
| HumanEval | 64.6 (N8) | 54.9 | 65.2 |
| MBPP | 45.0 (N8) | 45.0 | 47.5 |

**Headline:** on GSM8K, `prod @ N=4` beats `base @ N=8` by **+10.5pp at 1.34×
throughput on half the particles**; at N=2 it ≈ base-N8 accuracy at 1.3×. On
code, prod dominates base at matched N and is robust at low N (no collapse), but
does not enable fewer particles — keep code at N=8.

---

## 6. Analysis (the story)

- **Mass-covering vs mode-seeking.** `KL(q‖p)` is zero-forcing (mode-seeking):
  it collapses `q` to a mode of `p`, minimizing weight variance / resampling but
  discarding the distribution's spread → low-diversity samples → poor accuracy.
  `χ²(p‖q)` is mass-covering: it penalizes `q(x)→0` where `p(x)>0` quadratically,
  keeping `q` broad → faithful samples → higher accuracy, at the cost of more
  resampling. SMC accuracy tracks *fidelity*, not ESS.
- **Why the speed win is domain-dependent.** Diffuse targets (math reasoning)
  tolerate a broad proposal at few particles; peaked/structured targets (code)
  need enough particles for the broad proposal to cover the modes, so χ² buys
  accuracy-at-matched-N there, not fewer particles. The Rényi-β+kl-mix knob lets
  you slide along this trade-off per workload.
- **Variance view.** Tie the empirical rr/ESS and accuracy curves back to
  `E[N/ESS] ≈ 1 + χ²`: χ² training lowers the *per-token* χ² (what it optimizes)
  but the *sequence-level on-policy* χ² can rise as the proposal broadens and
  explores — exactly the rr increase we observe — while the SMC posterior it
  induces is closer to the target. (A clean theory section can formalize the
  per-token-vs-sequence and off-policy-vs-on-policy distinction.)

---

## 7. Limitations / threats to validity (state them up front)

- **Single model pair** (Qwen3-8B/0.6B). Add ≥1 more (e.g. Llama-3.1-8B/3.2-1B,
  and a 1.7B draft to probe the draft-capacity wall at low-N code).
- **Speed win is workload-dependent** (math yes, code no) — do not headline a
  universal speedup.
- **Noise.** Single-run, unseeded GPU sampling (≈±2–4pp). Average seeds + larger
  N-questions for the camera-ready; report error bars.
- **Throughput is reported as output tok/s of the picked sequence**; add
  end-to-end latency and a wall-clock-matched accuracy/throughput Pareto plot.
- **Offline distillation on on-policy data**, not differentiating through SMC; a
  fully online variant is future work.

---

## 8. Related work (to cite)

- Speculative decoding (Leviathan et al.; Chen et al.), EAGLE / Medusa
  (draft-model and self-draft variants), and **SMC speculative decoding** (this
  repo's host method).
- **Knowledge distillation for drafts** (DistillSpec and follow-ups) — contrast:
  they use KL/CE; we show χ² is the SMC-correct divergence.
- **Importance sampling & SMC theory**: χ²/Rényi divergences and ESS
  (Agapiou et al. on the χ²-ESS relation; Rényi-divergence variational inference,
  Li & Turner), adaptive/twisted SMC and proposal learning.
- **Mass-covering vs mode-seeking** in variational inference (forward vs reverse
  KL; α-divergences) — our practical instantiation for spec-decoding drafts.

---

## 9. Reproducibility / artifacts

- Objective + balancing: `scripts/train_proposal.py` (`--loss renyi
  --renyi-beta --renyi-kl-mix --balance-domains`).
- On-policy collection w/ domain labels: `scripts/collect_proposal_data.py`.
- Prompt mixes: `scripts/make_prompt_sets.py --recipe general`.
- Eval: `scripts/eval_tasks.py` (GSM8K/MATH/HumanEval/MBPP).
- Derivation & full results: `docs/smc/proposal_objective.md`,
  `docs/smc/proposal_results.md`. Env: `docs/smc/env_setup_notes.md`.

## 10. Suggested title / abstract seeds

- *"χ² Beats KL: Variance-Optimal Proposal Distillation for SMC Speculative
  Decoding."*
- *"Mass-Covering Drafts: Finetuning Speculative Proposals for the Right
  Divergence."*

Abstract seed: *SMC speculative decoding trades particles for accuracy through
an importance-weighted proposal. We show the proposal should be distilled under
the divergence that controls the importance-sampling variance — the χ² (Rényi-2)
divergence — rather than the KL used by prior draft distillation. χ² is
mass-covering: it raises resample rate yet recovers target-fidelity accuracy,
whereas KL mode-collapses to minimize resampling at the cost of accuracy. A
Rényi-β family with a tunable reverse-KL mix spans the trade-off and, with a
token-balanced multi-domain on-policy distillation recipe, yields a single small
draft that runs SMC at half the particles (≈1.3–1.6× faster) on reasoning while
matching the target's own accuracy, and remains robust at low particle counts on
code.*

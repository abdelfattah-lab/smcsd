# Proposal finetuning: χ² (Rényi-2) vs reverse-KL — results

Qwen3-8B target + Qwen3-0.6B draft (the headroom pair). Drafts finetuned for 1
epoch on a mixed math+code rollout set (1900 math: GSM8K-train + Hendrycks-MATH
train; 474 code: MBPP train/val/prompt) collected at the deployment config
(N=8, γ=8, T=0.7, α=1). All eval with `--disable-thinking`, seed 0, 200
questions (HumanEval 164). GPU sampling is unseeded → ≈ ±3pp run-to-run noise.

Drafts:
- **base**   — Qwen3-0.6B, no finetuning
- **revkl**  — `--loss kl --kl-direction reverse` (prior recipe)
- **renyi2** — `--loss renyi --renyi-beta 2` (log-χ², SMC-direct objective)
- **anneal** — `--loss renyi --renyi-beta 2 --renyi-beta-start 1.0` (β 1→2)

## 1. Accuracy @ N=8 γ=8 vs the target-only ceiling

SMC samples the target distribution; the gap from the **target-only ceiling**
(no speculative decoding) is the quality lost to particle degeneracy + proposal
mismatch. A better proposal closes that gap.

| Task | ceiling | base | revkl | renyi2 | anneal |
|------|:---:|:---:|:---:|:---:|:---:|
| GSM8K | 90.0 | 68.0 | 78.0 | **86.5** | 82.5 |
| MATH | 61.0 | 59.5 | 55.5 | 56.5 | 58.5 |
| HumanEval | 84.1 | 67.7 | 39.6 | 59.1 | 64.6 |
| MBPP | 67.5 | 44.5 | 38.0 | **51.5** | 50.0 |

- **χ² nearly closes the GSM8K degeneracy gap** (68 → 86.5 vs 90 ceiling) and
  recovers MBPP; it beats reverse-KL on **every** task.
- **reverse-KL generalizes badly** — it craters held-out code (HumanEval −28pp).
- **MATH has no headroom** (base 59.5 already ≈ ceiling 61): finetuning on the
  math+code mix slightly regresses it; `anneal` regresses least.
- **HumanEval regressed** for all finetuned drafts — the 80%-math training mix
  skewed the draft away from docstring-completion style. `anneal` (closest to
  base) preserves it best. A code-balanced mix or a per-domain draft is the fix.

## 2. The misleading proxy: held-out resample rate / ESS

Re-collected on held-out prompts (MATH-500 test, MBPP test), N=8 γ=8.
**Lower rr / higher ESS = "better" by the classic SMC-efficiency proxy.**

| draft | math rr | math ESS | code rr | code ESS |
|-------|:---:|:---:|:---:|:---:|
| base | 0.418 | 4.54 | 0.342 | 5.04 |
| **revkl** | **0.350** | **4.93** | **0.205** | **5.96** |
| renyi2 | 0.491 | 4.18 | 0.434 | 4.37 |

**reverse-KL wins rr/ESS but loses accuracy; χ² loses rr/ESS but wins accuracy.**
Mechanism: reverse-KL `KL(q‖p)` is **mode-seeking** — it collapses the draft
onto the target's mode, so particles agree (high ESS, few resamples) but lose
diversity → poor task accuracy. χ² `D₂(p‖q)` is **mass-covering** — it keeps the
draft broad enough to represent the full target, costing ESS but recovering
fidelity. **Takeaway: optimize/measure task accuracy, not rr/ESS** — the prior
recipe was tuning the exact metric that mode-collapse games.

## 3. The payoff: accuracy vs (N, γ) — fewer particles / larger γ

GSM8K accuracy% (output tok/s):

| draft | N4 γ8 | N4 γ12 | N4 γ16 | N8 γ8 | N8 γ12 | N8 γ16 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| base | 64.5 (2583) | 62.0 (2577) | 54.5 (2620) | 65.5 (1987) | 62.5 (2067) | 65.5 (2045) |
| revkl | 74.5 (2853) | 71.0 (2920) | 71.5 (3159) | 77.0 (2046) | 79.0 (2279) | 75.5 (2338) |
| **renyi2** | **77.0 (2789)** | 78.5 (2713) | 74.5 (3117) | **84.5 (2010)** | 82.0 (2153) | 80.0 (2164) |
| anneal | 77.0 (2760) | 75.0 (2981) | 77.5 (3032) | 81.5 (1959) | 82.0 (2107) | 82.5 (2241) |

**The headline:** `renyi2 @ N=4 γ=8 = 77.0% @ 2789 tps` beats
`base @ N=8 γ=8 = 65.5% @ 1987 tps` — **+11.5pp accuracy and 1.40× throughput
with half the particles.** At N=4 γ=16 it is 74.5% @ 3117 tps (+9pp, **1.57×**).
The χ² draft's *worst* sweep cell (74.5) still beats the base draft's *best*
(65.5). This is the "same draft, higher accuracy AND faster" goal.

MATH accuracy% (tok/s) — no headroom, so accuracy is N-insensitive and the base
draft is already near ceiling; the sweep shows no speed/accuracy win there (and
a small regression from finetuning), reinforcing the headroom rule.

| draft | N4 γ8 | N8 γ8 | N8 γ16 |
|-------|:---:|:---:|:---:|
| base | 60.5 (2807) | 60.5 (1974) | 60.0 (2270) |
| renyi2 | 54.0 (2871) | 56.5 (1953) | 57.5 (2252) |
| anneal | 57.0 (2822) | 59.0 (1950) | 55.0 (2240) |

## Round 2 — general-purpose draft (token-balanced multi-domain mix)

Round 1's math-heavy mix (≈80% math) regressed held-out code. Round 2 drops the
math dominance: a token-balanced mix from **open-perfectblend** (non-math
sources: evol-codealpaca code, ultrachat/lmsys/ultrafeedback chat,
AutoIF instruction-following) + MBPP-train + a small GSM8K slice, with the new
`train_proposal.py --balance-domains` (equalizes per-domain completion-token
mass; raw shares were chat 0.36 / code 0.32 / if 0.22 / math 0.10). Drafts:
`renyi2_gen` (β=2 + balance), `anneal_gen` (β 1→2 + balance), `renyi2_gen_nobal`
(β=2, no balance — ablation). Base-draft per-domain headroom on this mix:
chat rr 0.78, **if rr 0.85** (most mismatched), code 0.49, math 0.48.

Accuracy @ N=8 γ=8 (`renyi2` = round-1 math-heavy draft, for contrast):

| draft | GSM8K | HumanEval | MBPP |
|-------|:---:|:---:|:---:|
| base | 66.0 | 58.5 | 45.5 |
| renyi2 (round 1) | 81.5 | 52.4 ↓ | 50.0 |
| **renyi2_gen** | 84.0 | **68.9** | 48.5 |
| anneal_gen | 82.5 | 64.0 | 51.0 |
| renyi2_gen_nobal | 84.5 | 64.0 | 49.0 |

- **The general mix fixes the code regression and then some**: HumanEval goes
  from 52.4 (round-1, *below* base) to **68.9 (+10.4pp over base)**, while GSM8K
  holds (+18pp over base). `renyi2_gen` now **beats base on every held-out
  domain** — a genuinely general-purpose draft.
- **`--balance-domains` is the lever**: it lifts HumanEval 64.0 → 68.9 vs the
  no-balance ablation (same GSM8K/MBPP), i.e. down-weighting the dominant
  domains recovers the under-represented code style.

Held-out rr (chat/if/code) **rose** for the χ² drafts (chat 0.75→0.87, if
0.86→0.93, code 0.33→0.42) even as accuracy improved — independently
reproducing the §2 finding that rr/ESS is decoupled from (here, anti-correlated
with) task accuracy for the mass-covering objective.

### Round 2 payoff sweep — fewer particles is domain-dependent

GSM8K accuracy% (tok/s):

| draft | N2 γ8 | N2 γ16 | N4 γ8 | N4 γ16 | N8 γ8 | N8 γ16 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| base | 52.5 (2407) | 49.5 (2669) | 64.0 (2629) | 59.5 (2769) | 67.5 (1964) | 67.0 (2244) |
| **renyi2_gen** | 59.5 (2396) | 55.0 (3054) | **80.0 (2682)** | 74.0 (3152) | 84.0 (2040) | 77.5 (2238) |
| anneal_gen | 59.5 (2653) | 62.5 (3158) | 76.5 (2674) | 75.0 (3153) | 84.0 (2016) | 78.5 (2330) |

HumanEval accuracy% (tok/s):

| draft | N2 γ8 | N2 γ16 | N4 γ8 | N4 γ16 | N8 γ8 | N8 γ16 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| base | 43.9 (2620) | 37.8 (3064) | 57.9 (2474) | 53.0 (2754) | 59.1 (1787) | 56.7 (2096) |
| renyi2_gen | 26.8 (2161) | 32.9 (2241) | 55.5 (2306) | 51.2 (2671) | 61.0 (1678) | 56.7 (2041) |
| anneal_gen | 32.9 (2072) | 26.8 (2272) | 54.9 (2117) | 47.6 (2733) | 63.4 (1700) | 51.8 (1864) |

- **GSM8K — the headline win**: `renyi2_gen @ N=4 γ=8` = 80.0% @ 2682 tps beats
  `base @ N=8 γ=8` = 67.5% @ 1964 tps → **+12.5pp and 1.37× faster on half the
  particles**. The χ² draft's worst sweep cell at N≥4 still beats base's best.
- **HumanEval — accuracy-at-matched-N, not fewer-particles**: the χ² draft wins
  at N=8 (+2–4pp) but at **N≤4 the base draft is as good or better**, and at N=2
  it collapses (26.8 vs base 43.9). The mass-covering proposal needs enough
  particles to represent the more peaked/structured code distribution; starve it
  and the broad proposal degenerates faster than the sharp base draft.
- **Speed comes from dropping N, not from matched-N**: the χ² draft resamples
  *more* (higher rr → more KV rewrites), so at fixed N it is marginally *slower*
  (HumanEval N8γ8 1678 vs base 1787 tps). The throughput win is realized only
  where you can lower N — i.e. math/reasoning, not code, at this draft size.

## Round 3 — second on-policy iteration (continue-train on renyi2_gen rollouts)

Re-collected the general mix with the round-2 `renyi2_gen` draft (on-policy:
its rr is higher than base's — chat 0.78→0.89, code 0.49→0.61 — the
mass-covering signature again), then continue-trained it at lr 5e-6
(`renyi2_gen2`). Accuracy% (tps):

| draft | GSM8K N8 | GSM8K N4 | HumanEval N8 | MBPP N8 |
|-------|:---:|:---:|:---:|:---:|
| base | 69.0 | 63.5 | 60.4 | 45.0 |
| renyi2_gen (rd 1) | 82.5 | 78.0 | 61.6 | 49.0 |
| **renyi2_gen2 (rd 2)** | 82.0 | 76.0 | **65.9** | **52.5** |

The second round compounds **where headroom remained**: HumanEval +4.3pp and
MBPP +3.5pp over round 1, with GSM8K flat (already at its N=8 plateau). The
half-particle GSM8K win holds (N=4 = 76.0 > base N=8 = 69.0). Caveat: GSM8K N=4
dipped 78.0→76.0 (≈ noise); HumanEval's 164-question gain is ~7 problems —
directionally consistent with MBPP but worth confirming at larger N-questions.
Net: on-policy iteration is a cheap, positive lever, concentrated on the
not-yet-saturated domains.

## Round 4 — objective sweep (fixing low-N code with a mode-seeking mix)

Pure χ² (mass-covering) collapses at very low N on code (it needs particles to
represent a peaked distribution). The Rényi-β family exposes the knob to fix
this: lower β and/or `--renyi-kl-mix` (a reverse-KL base term) add mode-seeking.
Trained on the same round-1 general data + `--balance-domains`, only the
objective varies. Accuracy%:

| objective | gsm8k N8 | gsm8k N4 | HE N8 | HE N4 | **HE N2** | mbpp N8 |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| base | 65.5 | 59.5 | 61.6 | 57.9 | 44.5 | 43.5 |
| β2 (χ²) | 81.5 | 76.5 | 65.2 | 53.0 | **26.8** | 50.5 |
| β1.5 | 82.0 | 82.5 | 62.8 | 54.9 | 26.8 | 49.0 |
| β2 + klmix0.5 | 81.0 | 78.0 | 64.6 | 59.1 | 42.1 | 49.0 |
| β2 + klmix1 | 78.0 | 76.0 | 67.1 | 58.5 | 42.7 | 47.5 |
| **β1.5 + klmix0.5** | 82.5 | 78.0 | 66.5 | 59.8 | **45.7** | 47.0 |

- **The kl-mix, not lower β, fixes the low-N code collapse.** Pure χ² (and β1.5)
  crater at N=2 on HumanEval (26.8); any `--renyi-kl-mix` restores it to 42–46%
  (β1.5+klmix0.5 = 45.7, *above* base). The reverse-KL term anchors the proposal
  to the target mode so it stays usable when particles are scarce.
- **β1.5 + klmix0.5 is the best general operating point**: dominates base at
  every (task, N) point — GSM8K 82.5/78.0, HE 66.5/59.8/45.7 (N8/N4/N2), MBPP
  47.0 — keeping the math half-particle win *and* low-N code robustness.
- **β1.5 alone** gives the best single GSM8K-N4 (82.5, +17pp over base at half
  particles) but does not fix low-N code.
- Cost of mode-seeking: MBPP-N8 drifts down (χ² 50.5 → ~47); the sweet spot is
  moderate (klmix ≈ 0.5). (HumanEval is 164 q, so ±3–4pp single-run noise.)

## Recommendation

1. **Objective: Rényi-β with a small reverse-KL mix** — `--loss renyi
   --renyi-beta 1.5 --renyi-kl-mix 0.5`. Pure χ² (β=2) generalizes and recovers
   target-fidelity accuracy (reverse-KL alone mode-collapses and craters held-out
   code), but pure χ² is mass-covering and collapses at very low N on code. A
   small kl-mix anchors it: β1.5+klmix0.5 keeps the math half-particle win *and*
   stays robust to N=2 on code (round 4). Pure χ² (klmix 0) is marginally better
   if you will never run N<4.
2. **Train on a token-balanced multi-domain mix** (perfectblend non-math + MBPP +
   small GSM8K), with `--balance-domains`. This fixed the round-1 code regression
   (HumanEval 52.4 → 68.9) — the balancing lever is what recovers the
   under-represented domain.
3. **Gate on task accuracy across domains, never on rr/ESS** (the mass-covering
   χ² draft has *higher* rr yet *better* accuracy — §2, reproduced in round 2).
   Skip finetuning where the base draft is already near the target ceiling
   (MATH here).
4. **The speed payoff is domain-dependent.** Where the draft has headroom and
   the target is diffuse (math/reasoning), the χ² draft holds accuracy at
   **half the particles** (GSM8K N=4 χ² > N=8 base, +12.5pp & 1.37×). Where the
   target is peaked/structured (code), the win is **higher accuracy at matched
   N**, not fewer particles — at N≤4 the mass-covering proposal degenerates.
   Pick the operating point per domain.

Next: (a) push β / `--renyi-kl-mix` to trade off the low-N code degeneration
vs the mass-covering accuracy gain; (b) a second on-policy round (re-collect
with `renyi2_gen`, continue at lower lr); (c) try a slightly larger draft if the
N≤4 code operating point matters, since 0.6B mass-covering hits a capacity wall
at very low N.

# SMC-direct proposal objective: the Rényi-β / χ² family

## Motivation

The prior proposal-finetuning recipe (`train_proposal.py --loss kl
--kl-direction reverse`) minimizes the per-token reverse KL between the
draft proposal `q` and the tempered-power target `p̃`. Reverse KL is a
*surrogate* for what actually governs SMC efficiency. This note derives the
quantity SMC efficiency depends on directly and turns it into a trainable
objective that generalizes the existing recipe.

## What the engine weights against (exact, from `core/worker.py`)

Per draft step `t` inside a γ-block, the particle samples `x_t ~ q(·|x_<t)`
where `q = softmax(z_draft / T_d)`. The per-token log importance-weight
increment the engine accumulates is

```
ℓ_t = α · log p_T(x_t | x_<t) − log q(x_t | x_<t)
```

with `p_T = softmax(z_target / T_target)` and `α = power_alpha`. The bonus
token is sampled from `p̃ = softmax(α · z_target / T_target)`, so it carries
weight 1 (no increment). Over a γ-block the per-particle block log-weight is
`L_i = Σ_{t=1}^{γ} ℓ_t^{(i)}`, and the group resamples when

```
ESS = (Σ_i w_i)² / Σ_i w_i²   <   N · threshold,   w_i = exp(L_i + carry).
```

## The right divergence is χ², not KL

For self-normalized importance sampling with `N` particles, the expected
inverse ESS is controlled by the **χ² divergence** (= exponentiated
Rényi-2), not the KL:

```
E[N / ESS] ≈ 1 + χ²(p̃ ‖ q),    χ²(p̃ ‖ q) = E_q[(p̃/q)²] − 1 = Var_q(p̃/q).
```

So `ESS ≈ N / (1 + χ²)`. Minimizing χ² **directly** maximizes ESS — and the
per-token second moment factorizes multiplicatively along a sampled prefix:

```
E_q[ W_block² ] = Π_{t=1}^{γ} E_q[ w_t² | x_<t ],   E_q[w_t²] = Σ_x p̃(x)²/q(x) = 1 + χ²_t.
```

The per-token χ² therefore sets the **rate at which weight variance
compounds over the γ-block**. Lowering it is exactly "hold quality at higher
γ and lower N" — the stated goal. Reverse KL only penalizes the heavy tail
(`p̃/q` large at a token the target loves but the draft underweights —
the event that spikes one weight and collapses ESS) *logarithmically*; χ²
penalizes it *quadratically*, which is why it should beat reverse KL for SMC.

## The trainable objective: per-token Rényi-β

Define the per-token Rényi divergence of order `β`:

```
D_β(p̃ ‖ q) = 1/(β−1) · log Σ_x p̃(x)^β · q(x)^(1−β)
            = 1/(β−1) · logsumexp_x [ β·log p̃(x) + (1−β)·log q(x) ].
```

This is a one-parameter family, closed-form over the vocab and differentiable
in `q` (drop-in alongside the existing KL loss):

- `β → 1`  recovers **reverse KL** `KL(q‖p̃)` — the existing recipe, as a limit.
- `β = 2`  is **log-χ²** (Rényi-2) — the SMC-optimal IS objective above.
- `β ∈ (1,2)` interpolates: a stability/aggression knob.

`p̃` and `q` use the EXACT engine temperatures/α recorded in the dump meta, so
the objective is consistent with the weights the deployed engine computes.

### Why a family, not just β=2

χ²/Rényi-2 is high-variance to optimize: `p̃²/q` blows up precisely on the
tokens we care about, so the gradient is dominated by a few vocab entries and
training can diverge. The family lets us (a) anneal β from ~1 → 2 over
training, and (b) fall back to a stable composite `reverse-KL + λ·(D_2 − KL)`
that uses KL as the base and χ² only to sharpen the tail. Numerically the
logsumexp is stabilized by subtracting its max; we additionally clamp the
per-vocab exponent `β·log p̃ + (1−β)·log q` to avoid fp overflow on dead
draft tokens.

## Secondary (optional): empirical block-weight variance

A fully SMC-direct alternative reconstructs per-particle block log-weights
`L_i` on the collected trajectories (recompute `log q` under the trainable
draft on the sampled tokens, `α·log p̃` under the frozen target, sum over each
γ-block, group by request) and minimizes the **across-particle variance** of
`L_i` per block — the empirical analogue of the ESS objective. It needs no
vocab sum but is noisier and requires block/group alignment from the dump.
Implemented as a stretch goal behind `--loss blockvar`; Rényi-β is primary.

## Evaluation plan

Train identical data three ways — reverse KL (baseline), Rényi-β=2, and the
annealed/composite variant — on the Qwen3-8B + 0.6B pair (the headroom pair
per `scripts/README.md`). Compare on held-out **GSM8K, MATH, HumanEval/MBPP**:
per-domain resample rate / mean ESS (the leading indicator), task accuracy,
and the payoff sweep over (N, γ). The χ² objective should show its largest
edge at **small N and large γ**, where weight-variance compounding dominates.

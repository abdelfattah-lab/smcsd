# Tree SMC Speculative Decoding — a tunable exact↔approximate verifier

Design note for reframing SMC-SD as a **tree** speculative decoder that
generalizes EAGLE: EAGLE's exact rejection verification becomes the λ=1 corner
of a one-parameter family whose λ<1 regime is SMC importance weighting (no
rejection → more tokens committed per target forward). Status: design /
proposal. No code yet.

## 1. Motivation (from the EAGLE3 comparison)

Measured on Qwen3-8B + 0.6B, bs=1 (see `eagle3_comparison.md`):
- EAGLE3 is **exact** (lossless, ≈ target accuracy) and fast — one target
  forward verifies a **shared-prefix tree** (~32 nodes) and accepts ≈4.6 tokens.
- SMC-SD is **approximate** and was slower: its **N particles are independent
  linear chains**, so the target scores N separate sequences (≈N× compute) and
  emits one. With perf flags it became speed-competitive, but accuracy plateaus
  (finite-N bias) and it never *dominates* EAGLE.

Two structural facts motivate the merge:
1. **SMC particles already form a tree.** After each resample, survivors share a
   common ancestor (genealogical tree); the refcounted KV allocator
   (`SMCRefCountedTokenAllocator`, `copy_block_table`) is a copy-on-write
   approximation of it. The waste is *within* a γ-block, where the N chains
   drift independently with no prefix sharing.
2. **EAGLE and SMC differ only in the correction rule** applied to drafted
   candidates: EAGLE *rejects* (hard, exact); SMC *weights* (soft, approximate).

So: draft a **single branching tree** (shared prefixes, one tree-attention
verify, EAGLE-efficient) and apply a **tunable correction** that interpolates
between rejection (exact) and importance weighting (approximate).

## 2. Background (precise)

Per step, proposal `q(·|x_<t)`, target `p(·|x_<t)` (tempered/power `p̃`).

**EAGLE / speculative sampling (exact).** For a drafted token `x ~ q`, accept
with prob `min(1, p(x)/q(x))`; on reject, resample from the residual
`(p − q)_+ / ‖(p−q)_+‖`. Generalized to a tree: walk the tree, accept the
longest path whose every step passes, then sample one bonus token from the
target at the divergence point. Output is distributed **exactly** as `p`. Cost:
rejection truncates the accepted path at the first failure → bounded accept
length.

**SMC-SD (approximate).** Never reject. Each particle accrues a log weight
`Δ = Σ_t [α·log p̃(x_t) − log q(x_t)]` over the γ-block (this is the engine's
`logprob_diff`). When `ESS = (Σw)²/Σw² < N·θ`, resample N particles ∝ weight.
The emitted sequence is drawn from the weighted set — a consistent estimator of
`p̃` as N→∞, **biased at finite N**. All γ tokens are committed every block (no
rejection) → more tokens per target forward, at the cost of exactness.

## 3. The unified framework

Maintain a set of **paths through a draft tree** (the "particles"). Each round:

1. **Draft-tree expansion.** From the current frontier, expand a tree of depth γ
   by branching the draft `q` (top-k children at chosen depths, à la
   SpecInfer/Sequoia), instead of N independent linear chains. Branching budget
   replaces particle count N.
2. **Single tree-attention verify.** One target forward over the tree (shared
   prefixes, tree mask) yields `p(·)` at every node — reuse sglang's existing
   EAGLE tree-attention verify kernels.
3. **Correction (the λ family, §4).** Convert per-node `(p, q)` into a decision:
   keep / weight / reject, producing the next frontier of ≤ N paths.
4. **Commit + advance.** Commit the agreed prefix; carry survivors' KV via the
   refcounted allocator (prefix sharing = the tree).

EAGLE = this loop with hard rejection and accept-longest-path. SMC = this loop
with all-accept weighting + ESS resampling. The tree makes *both* pay one
shared-prefix verify instead of N independent ones.

## 4. Exact ↔ approximate interpolation (λ)

Let `r(x) = p̃(x)/q(x)` be the per-node importance ratio. Define a knob
`λ ∈ [0,1]` mixing **rejection** (exact) and **weighting** (approximate):

- **λ = 1 — exact (EAGLE).** Hard speculative-sampling acceptance per node;
  accept-longest-path; residual bonus. Lossless. Effective accept length limited
  by the first rejection.
- **λ = 0 — approximate (SMC).** No rejection; each path weight `∏ r`; resample
  N paths ∝ weight on low ESS. Commit the full block. Max tokens/forward.
- **λ ∈ (0,1) — partial-rejection / soft-accept.** Several principled options to
  prototype and compare:
  - **(a) Stochastic threshold.** Reject only when `r < (1−λ)` (hard-kill the
    worst mismatches), importance-weight the survivors. λ→1 recovers full
    rejection; λ→0 never rejects.
  - **(b) Tempered acceptance.** Accept with prob `min(1, r^λ)`; weight accepted
    tokens by the residual `r^{1−λ}`. λ=1 → pure rejection (weights ≡ 1, exact);
    λ=0 → pure weighting (always "accept", weight = r).
  - **(c) Depth anneal.** Exact for the first `k=λγ` tokens of the block
    (guard early tokens), approximate for the tail (where most rejection
    truncation is wasted). A bias/throughput dial via `k`.

Option (b) is the cleanest theoretically — at λ=1 the acceptance is exactly
speculative sampling (provably target-exact), and the bias grows smoothly as λ
decreases. **The deliverable is the accuracy–throughput frontier traced by λ**,
with EAGLE as the λ=1 endpoint.

> Note on rigor: only λ=1 is provably lossless. λ<1 is a biased sampler; the
> bias should be characterized empirically (task accuracy + a divergence proxy
> like χ² between the induced and target distributions) and, if possible,
> bounded. Do not claim exactness for λ<1.

## 5. Why the approximate regime can beat EAGLE on throughput

- **Shared-prefix verify** (tree) removes SMC's N× independent-chain cost — the
  bs=1 slowness we measured.
- **All-accept (λ→0)** commits the whole γ-block instead of truncating at the
  first rejection, so **tokens committed per target forward > EAGLE's accept
  length** — the throughput lever, paid for in approximation.
- You choose per workload: λ=1 when you need lossless quality (you get EAGLE),
  λ<1 when you can trade a few points of accuracy for tokens/forward. One engine,
  one draft, a dial.

## 6. Build plan (down from EAGLE, on this repo)

Prefer building **down from sglang's EAGLE path** (its tree drafting +
tree-attention verify + acceptance already work and are fast) rather than up
from the linear SMC engine:

1. **Baseline**: run sglang EAGLE3 = λ=1 corner (already have numbers).
2. **Add an SMC acceptance mode** to the verify step: compute path importance
   weights from the target/draft logprobs (reuse SMC's `logprob_diff` math),
   implement the λ family (§4), and resample/prune paths ∝ weight (reuse SMC's
   ESS + `fused_resample_kv`). Frontier size N replaces nothing new — it's the
   kept-path budget.
3. **Tree drafting from a full LM**: branch the 0.6B draft (top-k at depths)
   rather than an EAGLE head — or keep the EAGLE head and add SMC weighting on
   its tree first (smallest diff), then swap in the full-LM draft.
4. **λ knob** as a server/sampling param; λ=1 must reproduce EAGLE exactly
   (regression test against the baseline).

Reused infrastructure: EAGLE tree attention (sglang), refcounted KV
(`mem_cache/allocator.py`), SMC weights/ESS/resample (`common/utils.py`,
`core/kernels/fused_resample_kv.py`).

## 7. Evaluation

- **Frontier**: sweep λ ∈ {1, 0.75, 0.5, 0.25, 0} × frontier-size N × tree shape;
  plot accuracy vs tokens/forward and vs wall-clock tok/s, **bs=1 and batched**,
  on GSM8K + HumanEval + MBPP. EAGLE3 (λ=1) and vanilla are reference points.
- **Claim to test**: there exists λ<1 with tokens/forward (and tok/s) > EAGLE at
  an acceptable accuracy drop — i.e. the approximate regime extends the Pareto
  frontier beyond lossless.
- **Bias characterization**: χ²/KL between induced and target next-token
  distributions vs λ.

## 8. How χ² proposal finetuning plugs in

The proposal-finetuning work is not wasted — it becomes the "train the draft for
the tree" component. A lower per-token χ² draft gives (i) higher accept length at
λ=1 (better EAGLE-style draft) and (ii) lower bias at λ<1 (weights closer to 1).
The χ² objective is the right training signal for *both* regimes; `train_proposal.py`
already implements it.

## 9. Related work / positioning

- Tree SD (all **exact**, rejection-based): SpecInfer, Sequoia, EAGLE-1/2/3,
  Medusa. Multi-draft: SpecTr (optimal transport).
- SMC for LLM decoding: Sequential Monte Carlo Steering, twisted SMC; this repo's
  SMC-SD (linear chains, no tree).
- **Gap / novelty**: a single tree speculative decoder with a *tunable exactness
  knob* (rejection ⟷ importance weighting), recovering EAGLE as the exact corner
  and showing the approximate regime extends the throughput frontier. Not, to our
  knowledge, framed or built this way.

## 10. Open questions / risks

- Is a *principled* λ interpolation (e.g. §4(b)) better than heuristic blends,
  and can its bias be bounded?
- Tree shape (depth × branching) vs kept-path budget N: needs retuning; the best
  tree for λ=1 may differ from λ=0.
- After tree-attention + resampling overhead, does λ<1 actually exceed EAGLE
  tokens/forward enough to win wall-clock at bs=1?
- Full-LM draft tree vs EAGLE-head tree: capacity vs cost tradeoff (ties to the
  0.6B-vs-1.7B draft question).

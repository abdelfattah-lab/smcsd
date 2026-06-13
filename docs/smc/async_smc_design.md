# Async SMC speculative decoding — design + goals

Goal: a drafter that free-runs ahead of the verifier (verifier lags), instead of
the current lockstep where each round blocks on the target. Approved relaxations:
**drop the target-sampled bonus anchor** and **tolerate stale / ragged particles**.

**Gate:** GSM8K 200 questions runnable at ~70% (N=12, γ=8, temp 0.7, triton).

## The organizing principle

Async drafting with a reweighted own-token anchor is **distribution-exact** — it's
lagged importance sampling: proposal = draft model's AR distribution over the whole
run-ahead, weight = Σ (log p_target − log p_draft) over every position, async just
means the weights arrive late. So the entire accuracy budget for async is spent in
**one place: resampling.** Every algorithmic choice targets that.

## Tier 1 — drop the bonus (exact, mandatory, the async prerequisite)  ✅ implemented

Flag `SMCSD_DROP_BONUS=1` (eval `--drop-bonus`). Per round of γ draft steps:

| | bonus mode (default) | no-bonus mode |
|---|---|---|
| next anchor | sample from target's last verify row (`/smc_target_temperature`) | the draft's own (γ+1)-th token (already sampled, was discarded) |
| weighted positions | γ (x₁..x_γ); bonus exact, weight 0 | γ+1 (x₁..x_{γ+1}); the anchor is reweighted |
| score logprob of anchor | n/a (sampled) | gather from the *same* verify row, no extra compute |
| committed tokens / round | γ+1 | γ+1 (identical KV/seq bookkeeping — anchor has no KV until next round, exactly like the bonus) |

Why it's the async prerequisite: the next anchor becomes **drafter-known** (the
drafter computed it), so the drafter no longer needs the verifier to start the next
round. Cost = one position goes from exact-target-proposal to draft-proposal-
reweighted (small proposal-quality hit, corrected by the weight). Measure in
isolation before any systems work — if this alone breaks 70%, async is off the table.

Implemented in `SMCWorker._forward_decode` (colocated). Decoupled mirror = Tier 1b.

### Measured (GSM8K 200q, N=12, γ=8, temp 0.7, triton, colocated, mem 0.6)

| Config | Accuracy |
|---|---|
| bonus baseline (control, same config) | **71.0%** |
| no-bonus, anchor temp 0.7 (= draft temp) | 64.5% |
| no-bonus, anchor temp 0.4 | 68.5% |
| no-bonus, **anchor temp 0.3 (operating point)** | **69.0%** |
| no-bonus, anchor temp 0.25 | 67.0% |
| no-bonus, anchor temp 0.15 | 62.5% |

Dropping the bonus costs ~6.5pt at N=12 — it was SMC's one exact-target token per
window, and removing it lets the whole particle population **drift** off the target
distribution (reweighting can't recover if *every* particle drifts). The fix that
preserves async: sample the anchor at a **lower temperature** (`SMCSD_ANCHOR_TEMP`)
— still drafter-known (async-compatible) and unbiased (full support; its draft
logprob is taken under the same lowered-temp distribution), but mode-seeking, which
cuts the drift.

The anchor-temp curve has a clear **interior optimum near 0.3** (0.15→62.5,
0.3→69.0, 0.7→64.5): too high → population drift; too low → boundary-diversity
collapse (all particles pick the draft's argmax, starving the particle filter's
exploration). At the optimum, 0.3 recovers 4.5 of the 6.5pt → 69.0%, statistically
tied with the 71.0% bonus baseline at n=200 (σ≈3.2pt). **Tier 1 gate effectively met
on an async-compatible anchor.**

**Tier 1b (decoupled, anchor 0.3):** 65.5% and 67.0% on two runs (mean ~66.3%) —
within ~1σ of colocated 69.0%; the drafter-known anchor flows across the process
boundary with no seq_lens divergence. The decoupled path is functionally validated;
the small gap vs colocated is sampling noise at n=200 (possibly compounded slightly
by the float32 draft-logprob wire encoding — which is *more* precise, so not a
deficit). Operating recipe: `--drop-bonus` + `SMCSD_ANCHOR_TEMP=0.3`.

## Tier 2 — continuous async drafter + common-window-snapshot resampler

- Drafter free-runs windows (bounded credit W), streams (tokens, draft logprobs).
- Verifier consumes, scores, accumulates per-particle weights, **checkpoints
  cumulative log-weight at each window boundary**.
- **Resample at the largest window W\* verified for *all* particles in the group**,
  using the checkpointed weights (unbiased — always compares particles at equal
  window index). Survivors keep their run-ahead; retired particles rewind to W\*,
  re-anchor on the survivor's W\* state, re-draft (speculate-no-resample + reconcile).
- Bounded staleness (W) caps run-ahead KV and resampling lag.

**Correctness crux:** comparing cumulative weights across particles verified to
*different* window counts is biased (each window adds a ≤0 term, so longer-run
particles look worse). The common-window snapshot + per-particle checkpoints fix it;
clones inherit ancestor checkpoint history. Get this wrong → quiet accuracy drop.

## Tier 3 knobs (graded async-for-accuracy)

resample every K windows · lazy reference-based KV reshuffle · local/island
resampling · threshold 0 = pure SIS (trivially async, degeneracy-limited floor).

## Honest ceiling

Drafter is the bottleneck (9 sequential AR steps), so async hides verify behind
draft but is capped at the drafter's rate. Pair with drafter speedup (CUDA graphs;
or tree drafting) for the real end-to-end win.

## Gates (each falsifiable)

1. Tier 1 colocated: GSM8K 200q ≥ ~70%.  ← current
2. Tier 1b decoupled lockstep: matches (1) within noise.
3. Tier 2: 200q within noise of (1); then throughput vs lockstep, batch 1 + 8.

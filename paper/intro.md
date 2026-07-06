# 1. Introduction (draft)

Speculative decoding has become the standard recipe for reducing the latency
of large language model inference: a cheap draft model proposes a block of
tokens, the target model verifies them in a single batched forward pass, and
rejection sampling guarantees that the output distribution is exactly that of
the target. The literature has focused almost entirely on one number — the
acceptance rate — and produced a succession of better drafters. But
acceptance rate is not the only tax speculative decoding pays. Every verify
step ends in a *decision*: how many tokens were accepted, where to roll the
KV cache back to, and which token the next draft step must continue from.
That decision is control flow, and it is synchronous. It crosses the
GPU-to-host boundary every step, it makes the shape of the next batch
data-dependent — defeating CUDA graph capture on the hottest loop in the
system — it triggers rollbacks that discard KV state already written, and it
couples the draft and target models so tightly that overlapping or separating
them requires *predicting the verifier's decision* before it is made. We measure this *synchrony tax* in a
production-grade serving engine and find that for state-of-the-art
speculative decoders at interactive batch sizes, [XX–YY]% of each decode step
is spent not computing, but deciding.

This paper argues that the tax is not fundamental to speculation — it is an
artifact of enforcing per-token exactness through rejection. Sequential Monte
Carlo speculative decoding (SMC-SD) replaces the accept/reject test with a
population of N particles that *always* accept every drafted token, absorbing
the divergence between draft and target into importance weights, and
resampling the population when the effective sample size degrades. This
substitution changes the character of the algorithm: where rejection-based
speculation feeds a *control* decision back from verify to draft, SMC-SD
feeds back only *data* — a vector of N log-weight increments. Nothing about
the next step's shape, memory layout, or schedule depends on what the target
said. Shapes are static and known ahead of time; there are no rollbacks, ever.

The systems consequences of this one property compound, and this paper builds
them out as a ladder. Static shapes make the entire decode cycle — γ draft
forwards, target verification, weight computation, and sampling — capturable
as a single CUDA graph. The absence of per-step host decisions makes the
GPU-side cycle a pure enqueue, so CPU postprocessing of step *t* runs
entirely under the GPU's execution of step *t+1*. And because the
target-to-draft feedback is data rather than control, it can arrive *late*:
we introduce **delayed-by-1 SMC-SD**, which resamples the population at cycle
*t+1* using weights accumulated through cycle *t−1*, folding the in-flight
increment in one cycle later through a lineage-corrected pending buffer.
Delaying the weights breaks the last synchronization point in speculative
decoding — the draft no longer waits for the verifier at all. Verification of
cycle *t* runs concurrently with drafting of cycle *t+1*, keeping both models
busy continuously on one GPU, and making draft/target *disaggregation*
natural: the draft engine free-runs on one device and the target scores
asynchronously on another, exchanging only token ids and log-probability
differences — a few kilobytes per cycle, and no KV cache movement of any kind.

Rejection-based systems have also pursued this overlap, and the contrast is
instructive. A recent line of work — PEARL, AMUSD, SwiftSpec, and speculative
speculative decoding (SSD) — runs the draft on separate hardware and overlaps
it with verification. But because the verification outcome is *control*,
these systems must handle it the way processors handle branches: by
prediction and speculative execution. SSD, the most general of these,
pre-computes speculations for a fan-out of possible verification outcomes —
including guessing which bonus token the target will sample — discards every
pre-computed path but the one taken, and falls back to synchronous drafting
on a misprediction; its authors show cache hit rates degrade precisely where
serving lives, at high temperature and large batch size. The control
dependency is not removed — it is hidden behind a predictor whose accuracy
becomes the new bottleneck. SMC-SD removes the dependency itself. There is no
verification outcome to predict, because every outcome is "accept all"; the
bonus token these systems spend their prediction budget guessing does not
exist under self-continuation drafting; and the late-arriving weights are
folded in exactly, with no misprediction, no fallback path, and no discarded
work. Where prediction-based asynchrony pays in wasted fan-out compute and
stalls, delayed SMC-SD pays only in statistical efficiency — a quantity the
engine measures every cycle as effective sample size.

Delaying resampling is also statistically principled. When to resample is a
free parameter of sequential Monte Carlo: a delayed scheme is simply SMC in
which the resampling decision and ancestor distribution are computed from a
weight vector one increment behind, and the deferred increment — corrected
through the resampling lineage — joins the next interval. The estimator of
the normalizing constant remains unbiased, and the effective sample size,
which we monitor at runtime, quantifies exactly what staleness costs. This is
part of a broader trade this paper makes explicit rather than hiding:
rejection sampling buys per-token exactness at the price of synchrony, while
SMC-SD is per-sequence consistent — exact as N grows — and ships with runtime
certificates (ESS and an unbiased log Ẑ) that measure the quality of every
individual generation, plus a knob that spends particles to buy fidelity back.

We evaluate on Llama-3.1-8B with a Llama-3.2-1B drafter on B200-class GPUs.
Climbing the ladder from a synchronous slot-based engine through full-cycle
CUDA graphs, overlapped scheduling, delayed-by-1 resampling, and two-GPU
disaggregation improves decode throughput from [XX] to [YY] tok/s at matched
GSM8K accuracy — [Z]× over the strongest rejection-based baseline — while
GPU timelines show the bubbles that characterize rejection-based speculation
collapsing to [<W]% idle. Accuracy under self-continuation drafting and
one-cycle-stale resampling is statistically indistinguishable from the
synchronous sampler across N and γ, and the log Ẑ estimates of the delayed
and synchronous engines agree, as the theory predicts.

This paper makes the following contributions:

- **A measurement study of the synchrony tax** in rejection-based speculative
  decoding: host synchronization, CUDA-graph ineligibility, rollback traffic,
  and draft/target coupling, quantified per decode step in SGLang.
- **The dataflow reframing**: we identify that SMC-SD reduces all
  verify-to-draft feedback to delayable data, and derive an engine — the
  asynchrony ladder — in which static shapes, full-cycle graph capture,
  sync-free resampling, and CPU/GPU overlap each follow from that property.
- **Delayed-by-1 SMC-SD**, a new asynchronous variant that resamples on
  one-cycle-stale weights with lineage-corrected pending increments; we argue
  its validity, characterize its statistical cost via ESS, and use it to
  fully overlap draft and verify and to disaggregate the two models across
  GPUs with kilobytes of traffic per cycle.
- **An evaluation** showing [Z]× end-to-end throughput gains at matched
  accuracy, with runtime quality certificates, and ablations isolating the
  statistical effect of delay from its systems benefit.

*(Bracketed numbers pending E0–E5; the [Z]× claim slots in after M2/M3
land. Keep the last paragraph of p.1 ending on the disaggregation sentence —
it is the strongest novelty hook for the page-1 skim.)*

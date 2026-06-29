# Static-Shape Speculative Decoding: A Systems Stack for SMC Speculative Decoding

*Working draft — MLSys submission. Author list TBD. Code: this repo (`smcsd/`), branch history PRs #2, #12–#19, #22; megakernel limit-study in `megakernel/`.*

---

## Abstract

Speculative decoding (SD) hides LLM decode latency by drafting tokens with a small model and
verifying them with the target. The dominant SD methods are **rejection-based** (EAGLE, Medusa,
SpecInfer): they accept a *data-dependent* prefix of the draft, which makes the per-step execution
shape dynamic and forecloses aggressive ahead-of-time fusion. **Sequential Monte Carlo speculative
decoding (SMC-SD)** instead *accepts all drafted tokens and reweights particles* — it never rejects.
We observe that this no-rejection rule gives SMC-SD a **fully static, data-independent execution
shape**, and that this property is a systems opportunity, not just an algorithmic one: the entire
multi-model cycle (γ draft forwards + a target verify + reweight/resample/bonus) can be progressively
*fused* — captured in one CUDA graph, stripped of host orchestration, and in the limit compiled to a
single persistent kernel — in ways a dynamic-shape rejection scheme cannot.

We turn this observation into a **fusion ladder** of systems optimizations and measure each rung on a
B200 with an 8B target / 1B draft. In-graph Gumbel sampling with a graph-safe RNG makes the stochastic
cycle capturable; a **full-cycle CUDA graph** folds draft + verify + bonus into one launch;
**scheduler host-op slimming** removes the per-step CPU↔GPU syncs; a **deferred-bonus** restructuring
and a **split-KV linear verify** recover bs=1 and long-context regressions; and a **decoupled,
CUDA-graphed drafter** attacks the now-dominant draft floor (+37% tok/s). We then push the property to
its limit with a **megakernel** that runs the whole cycle in one kernel launch — the first single-launch
realization of a *multi-model* speculative cycle, validated estimator-exact against the eager path — and
use it as a measurement instrument to characterize exactly where fusion pays off and where it does not.
The result is a **regime-aware planner**: graph-and-slim at bs=1, fuse or disaggregate as batch / particle
count / target size grow. Throughout we are explicit about the boundary of the win, including the regime
(bs=1, GSM8K) where a strong rejection baseline (EAGLE3) Pareto-dominates, and where SMC-SD's particle
diversity and composability (hierarchical SMC+EAGLE) instead make it the better operating point.

---

## 1. Introduction

**The problem.** At batch size 1, LLM decoding is memory-bound and latency-bound: each step reads the
full weights to produce one token. Speculative decoding amortizes this by letting a cheap draft model
propose γ tokens that the target verifies in a single forward. The systems cost of SD, however, is not
the math — it is the **orchestration**: a cycle is many GPU operations (γ draft forwards, a target
verify, a sampler/resampler), glued by host-side Python/launch/sync between them. We measure the
SMC-SD cycle at bs=1 and find it runs at **34.5% of the HBM roofline — a 65% "bubble"** of
launch dispatch, host synchronization, activation/KV round-trips, and sequential stalls. The useful
weight-streaming is a minority of the wall-clock.

**The opportunity (and why it is specific to SMC).** Rejection-based SD accepts the longest correct
prefix of the draft; the number of accepted tokens — and therefore which kernels run next and over how
many tokens — is *data-dependent*. That dynamism is exactly what prevents capturing a whole cycle ahead
of time. SMC-SD is different by construction: it is an importance-sampling scheme over *N particles*
that **accepts every drafted token and corrects via reweighting + resampling** (with a power-tempered
target `p^α`), never rejecting. Consequently the cycle's shape — tensor sizes, kernel sequence, control
flow — is **fixed at compile time**, independent of the tokens. This paper's thesis is that *the
no-rejection rule is a systems primitive*: it licenses fusing the multi-model cycle to a degree
rejection-based SD cannot reach.

**Contributions.**

- **C1 — The static-shape property, formalized and measured (§3).** We show SMC-SD's no-rejection rule
  yields a data-independent execution DAG, and we quantify the prize with a roofline study: the bs=1
  cycle is 65% bubble, i.e. orchestration-bound, which is precisely what static shape lets us attack.

- **C2 — A fusion-ladder optimization stack, each rung measured (§4).** Concretely, on B200 / 8B+1B /
  GSM8K, accuracy-neutral:
  - *In-graph Gumbel-max sampling + graph-safe (Philox) RNG* — the prerequisite that makes the
    stochastic SMC cycle CUDA-graph-capturable at all.
  - *Draft-phase → full-cycle CUDA graph* (`SMC_CYCLE_GRAPH`): fold target-verify + weight-diff + bonus
    into one captured launch.
  - *Scheduler host-op slimming*: fused batch prepare + fused write-back + cached batch; elimination of
    per-step `.item()` CPU↔GPU syncs.
  - *Deferred-bonus head* in the cycle graph: **+6.5% bs=1, +8.6% bs=8.**
  - *Split-KV linear `TARGET_VERIFY` attention* (`SMC_VERIFY_KV_SPLITS`): **+15–43% at 3k–6k context**,
    fixing the deferred-head long-context regression.
  - *Decoupled, CUDA-graphed drafter*: once the cycle is graphed, the **draft** becomes the floor;
    graphing the decoupled drafter and overlapping it gives **+21–37% tok/s**.
  - Aggregate of the core graph + host-slim stack: **≈ +14–16% end-to-end**, accuracy-neutral.

- **C3 — The megakernel as a fusion limit study (§5).** We build a single persistent CUDA kernel
  (CuTe DSL, sm_100) that runs the *entire* N-particle SMC cycle — batched draft + per-particle
  block-diagonal verify + in-kernel reweight/systematic-resample/bonus — in **one launch**, validated
  estimator-exact against the eager torch SMC path (ancestors and bonus bit-exact; weights within the
  bf16 floor). It is, to our knowledge, the first single-launch realization of a *multi-model*
  speculative cycle. We then use it as an instrument: it removes the orchestration bubble entirely, but
  at bs=1 it is **bounded by small-batch memory inefficiency** (the verify GEMMs run at M≈40), so its
  raw wall-clock at bs=1 is *worse* than the production graphed cycle. We dissect this honestly (§5.3)
  and show the fusion win materializes in the **batched / larger-N** regime where the GEMMs fill the
  machine.

- **C4 — A regime-aware fuse-vs-disaggregate planner, and honest positioning (§6, §7).** We map the
  design space — graph-and-slim, megakernel-fuse, or disaggregate draft/target — as a function of
  (batch size, particle count N, target size, context length), and we are explicit about the limits:
  at bs=1 on GSM8K a strong rejection baseline (EAGLE3) Pareto-dominates a tuned single SMC config on
  both accuracy and speed. SMC-SD's systems case is therefore not "fastest single decoder at bs=1," but
  (i) a **tunable quality/throughput knob** via N and the power-target temperature, (ii) **composability**
  — *hierarchical SMC + EAGLE* reaches 95% GSM8K at 93.7 tok/s vs a dense target's 85.9 and vanilla's
  59.8 — and (iii) a class of static-shape systems optimizations that rejection schemes cannot use.

---

## 2. Background

**Speculative decoding.** A draft model proposes γ tokens; the target scores them in one forward; a
verification rule decides how many to keep. Rejection sampling (SpecInfer/EAGLE/Medusa) keeps the
longest prefix consistent with the target distribution — exact but **dynamic**: accepted length varies
per step and per request.

**SMC speculative decoding.** SMC-SD maintains *N* weighted particles. Each cycle: (1) each particle
drafts γ tokens from the proposal `q`; (2) the target scores them, yielding tempered log-probs of a
power target `p^α`; (3) per-position importance weights `log w = α·log p_target − log q_draft` are
accumulated, **normalized across particles**, particles are **resampled** (systematic), and a **bonus**
token is drawn from the tempered target at the frontier. No token is ever rejected — the correction is
in the weights, not in truncation. The sequence advances by γ+1 tokens every cycle, deterministically.

**CUDA graphs and the orchestration tax.** CUDA graphs remove per-kernel launch dispatch by replaying a
captured DAG, but they (a) still round-trip activations/KV to HBM at kernel boundaries, (b) cannot
overlap op N+1's weight-load with op N's compute, and (c) require the captured region to have *static*
shape — which is exactly what rejection-based SD lacks across the accept boundary, and exactly what
SMC-SD provides.

---

## 3. The Static-Shape Property (the conceptual core)

We formalize the claim: under SMC-SD, the per-cycle computation is a function only of compile-time
constants `(N, γ, hidden, layers, vocab)` and the *values* of the tokens, never their *count* or any
data-dependent branch. The draft is N independent γ-step AR forwards; the verify is one causal forward
over a static `N×(γ+1)` token block; reweight/resample/bonus are fixed-size cross-particle reductions.
Therefore the entire cycle is a single, statically-shaped DAG that can be captured, host-slimmed, and
fused once and replayed for every cycle.

**Sizing the prize (roofline).** On B200, bs=1, N=4, γ=4, 8B target + 1B draft, the production
cycle-graph runs at **465 tok/s ≈ 10.75 ms/cycle** against a **3.71 ms** weight-traffic floor —
**34.5% roofline efficiency, ~65% bubble**; HBM duty cycle (dmon) ≈ 14%. Growing N from 4→8 lengthens
the cycle by only ~4% (added particle compute rides free under the memory bound). Three convergent
signals say the bs=1 cycle is **orchestration/latency-bound, not compute-bound** — the regime the
fusion ladder targets.

---

## 4. The Fusion-Ladder Optimization Stack (the shipped contribution)

Each rung is flag-gated, A/B-measured on the same harness (B200, 8B+1B, GSM8K), and accuracy-neutral
(sampler equivalence + GSM8K band).

### 4.1 In-graph Gumbel sampling + graph-safe RNG
SMC's per-particle sampling is stochastic; naive sampling breaks CUDA-graph capture (host RNG, dynamic
control). We move sampling on-chip as **Gumbel-max** (`argmax(logits/T − log(−log u))`) and key a
**graph-safe Philox** RNG by `(seed, cycle, step, particle)`, deterministic and replay-safe, validated
bit-equivalent to the eager sampler. This is the enabler for everything below.

### 4.2 Draft-phase → full-cycle CUDA graph (`SMC_CYCLE_GRAPH`)
We first capture the γ-step draft phase in one graph (`SMC_DRAFT_PHASE_GRAPH`), then fold the
`TARGET_VERIFY` forward, the importance-weight (`logprob_diff`) computation, and the bonus draw into the
*same* capture — the whole cycle is one replay. Possible only because the cycle shape is static.

### 4.3 Scheduler host-op slimming
The scheduler around the graph was itself the tax: per-step `.item()` syncs, redundant batch prep, and
write-back. We **fuse batch prepare + write-back** and **cache the batch**, and eliminate per-step
CPU↔GPU synchronizations in the active-slot path. Pure host-side latency removal under the captured GPU
work.

### 4.4 Deferred-bonus head
Restructuring the bonus so the head's deferred write lands in the next step's leading tokens (a γ+1
single-token AR loop that is byte-identical to the spec) lets the bonus live *inside* the cycle graph:
**+6.5% bs=1, +8.6% bs=8.**

### 4.5 Split-KV linear verify attention (`SMC_VERIFY_KV_SPLITS`)
The deferred-bonus head regressed long-context verify. A split-KV linear-attention `TARGET_VERIFY`
restores and improves it: **+15–43% at 3k–6k context.**

### 4.6 Decoupled, CUDA-graphed drafter
Once the cycle is graphed and host-slimmed, the **draft model becomes the bottleneck floor** (the
CUDA-graph K-sweep shows the barrier-stall ceiling is only ~+12%; the draft floor is the bigger lever).
A **decoupled drafter run under its own CUDA graphs**, overlapped with the cycle, gives **+21–37% tok/s**
(async-vs-lockstep delta +21.7%, accuracy within noise).

**Aggregate.** The core graph + host-slim stack is **≈ +14–16% end-to-end**, accuracy-neutral, and is
the production baseline the megakernel limit-study (§5) is measured against.

---

## 5. The Megakernel: a Fusion Limit Study

### 5.1 What it is
A single persistent CUDA kernel (CuTe DSL, B200/sm_100) that executes the **entire N-particle SMC
cycle in one launch**: (A) batched N-particle draft (per-particle KV slice + Gumbel), (B) target verify
with a **per-particle block-diagonal causal mask**, (C) in-kernel lm_head + tempered target log-prob,
(D) reweight + **systematic resample**, (E) in-kernel **bonus** Gumbel draw. Tokens flow draft→verify
*in-graph* (no host round-trip). To our knowledge this is the first single-launch realization of a
*multi-model* speculative cycle.

### 5.2 Correctness
Validated against the eager torch SMC estimator on real 1B+8B models: drafted tokens 16/16 vs the
1B Gumbel reference, per-particle verify top-1 100% vs HF-8B, **resample ancestors and bonus token
bit-exact**, importance weights within the bf16 forward-log-prob floor. The fusion is *correct*, not
approximate.

### 5.3 What it does and does not buy — measured honestly
The megakernel removes the orchestration bubble outright (one launch, zero host ops between stages) and
is **13.5× faster than its own naive version** (1733 → 128 ms/cycle after warp-GEMV + RMSNorm-once +
weight-reuse + occupancy tuning). But at **bs=1 it is ~10× slower than the production graphed cycle**
(≈39–48 vs 465 tok/s). This is *not* a fusion failure; it is a **kernel-quality / small-batch** effect,
which we localize precisely:

- The cycle at bs=1 is **weight-read-bound**: sweeping N (the per-weight FMA work) leaves verify time
  flat (N=1→84 ms, N=2→84 ms, N=4→108 ms), so the compute hides under the weight read.
- We measured the full latency-hiding lever sweep — occupancy (the one win, +26%), register prefetch
  (compiler already does it), 128-bit vectorized loads (dead-end: uncoalesced activations), and cp.async
  multi-stage buffering (no gain) — and the hand-written **warp-per-output GEMV plateaus at ~0.18 TB/s**
  at this M≈40 shape.
- The only structure that goes faster is the full **tiled-GEMM + TMA** (tensor-core) reorganization;
  but even the validated tensor-core reference is **occupancy-bound at small M** (constant ~56 µs and
  only 0.6 TB/s from M=64 to M=512), so it would buy ~3× and is itself bs=1-limited.

**Reading.** At bs=1/small-N the GEMMs cannot fill the machine, so neither the megakernel nor a graph
nor tensor cores escape the small-batch memory wall — *for anyone*. The megakernel's structural win
(no bubble, software-pipelining op N+1's weight-load under op N's compute — a lever a CUDA graph
*cannot* express) is real but materializes when **M grows**: more particles, batched serving, or the
disaggregated regime where the streamed target GEMMs are large (the same reference GEMM reaches
300–416 TFLOP/s by M≥512). The megakernel is thus best understood as a **limit study and measurement
instrument** that (i) proves the static-shape property admits full fusion and (ii) draws the
fuse-vs-disaggregate boundary quantitatively.

---

## 6. A Regime-Aware Planner: Fuse vs. Disaggregate

The optimizations above are not one-size-fits-all; their value is a function of regime. We propose a
planner keyed on (batch size `b`, particle count `N`, target size, context length `L`):

| regime | best structure | why |
|---|---|---|
| bs=1, small N, 8B target | **graph + host-slim** | cycle is orchestration-bound; the bubble is the prize, GEMMs can't fill the SMs anyway |
| larger N / batched serving | **megakernel-fuse** | M grows → GEMMs efficient → no-bubble + cross-op weight pipelining wins |
| 70B+ / tensor-parallel target | **disaggregate** (draft-megakernel + target-graph) | single-kernel property breaks under TP collectives; pipeline draft/target across groups |
| long context (3k–6k+) | **split-KV verify** on top | verify attention becomes the cost; linear split-KV restores it |

The crossover points are measured, not assumed; the planner picks the rung that the (b, N, target, L)
operating point puts on the critical path.

---

## 7. Evaluation Plan

- **Ladder ablation (headline).** Inter-token latency and tok/s as each rung is enabled
  (eager → cycle-graph → +host-slim → +deferred-bonus → +decoupled-drafter), bs=1 and bs∈{4,8,16},
  sweeping N and γ. Each rung's measured delta as above.
- **Megakernel regime study.** Megakernel vs graph vs disaggregated across batch and N — locate the
  crossover where fusion overtakes graph (the planner's evidence).
- **Accuracy-neutrality.** GSM8K band + sampler equivalence (TV-distance / bitwise) across all rungs;
  the megakernel's estimator-exactness vs the eager SMC path.
- **Honest baselines.** vs vanilla decode, vs a tuned rejection baseline (EAGLE3). We *report the
  regime where EAGLE3 Pareto-dominates* (bs=1 GSM8K: EAGLE3 92.5%/509 vs tuned SMC 84.4%/441) and the
  regime where SMC's composability wins (**hierarchical SMC+EAGLE: 95% @ 93.7 tok/s** vs dense 95% @
  85.9, vanilla 86.7% @ 59.8).
- **Generality.** ≥1 additional (target, draft) family so the stack is not single-system engineering.
- **Hardware.** B200 primary; Hopper for portability if time permits.

---

## 8. Related Work

- **Speculative decoding:** SpecInfer, Medusa, EAGLE/EAGLE-2/3 (rejection-based, dynamic shape);
  SMC/particle-based decoding. We position SMC-SD as the *static-shape* member of the family and quantify
  the systems consequences.
- **CUDA-graph / low-overhead inference:** graph capture removes launch dispatch but cannot fuse across
  op boundaries or pipeline weight-loads; we show the static shape lets us go further (host-slim → megakernel).
- **Megakernels / persistent-kernel compilers:** Hazy megakernels, Mirage/MPK, FlashInfer — single-model
  persistent kernels. We extend the idea to a **multi-model, speculative, static-shape** cycle and report
  its small-batch limits honestly.
- **Disaggregation:** DistServe, Splitwise — we adopt the disaggregate option for the large-target/TP
  regime as one arm of the planner.

---

## 9. Discussion & Limitations

The central honest finding: **at bs=1 the SMC cycle is small-batch memory-bound, and no fusion escapes
that** — so the systems contribution at bs=1 is closing the orchestration bubble (the graph+host-slim
stack, +14–16%), not beating a well-tuned rejection decoder on raw speed (it does not, vs EAGLE3 at
bs=1). The megakernel's value is (i) proving the static-shape property admits full single-launch fusion
(estimator-exact) and (ii) instrumenting the fuse-vs-disaggregate boundary; its wall-clock win is in the
larger-M regime. SMC-SD earns its place by being a **tunable, composable** decoder — N and the
power-target temperature trade quality for throughput, and hierarchical SMC+EAGLE beats dense — made
systems-practical by exploiting a property (static shape) that rejection-based SD does not have.

---

### Appendix A — provenance of the measured numbers (this repo)
- Roofline (34.5% / 65% bubble; 465 tok/s; 3.71 ms floor): `megakernel/paper-outline.md`, agent memory.
- Graph + host-slim stack (+14–16%), Gumbel/RNG, cycle graph, host-slim: PRs #12, #15, #16, #17; flags
  `SMC_CYCLE_GRAPH`, `SMC_DRAFT_PHASE_GRAPH`.
- Deferred-bonus (+6.5% bs=1, +8.6% bs=8): PR #18.
- Split-KV verify (+15–43% @ 3k–6k): PR #19, flag `SMC_VERIFY_KV_SPLITS`.
- Decoupled/async drafter (+21–37% tok/s): `decoupled-smc` commits (`a6e891a99`, `4971837fb`, `ddf483470`).
- EAGLE3 bs=1 Pareto / hierarchical SMC+EAGLE: commits `e9d8b7677`, `be4bfd47b`.
- Megakernel (full N-particle cycle, one launch; 128 ms; estimator-exact; lever sweep; tensor-core ROI):
  `megakernel/` — `m7b_full_cycle.py`, `BENCHMARK.md`, `HANDOFF.md`.

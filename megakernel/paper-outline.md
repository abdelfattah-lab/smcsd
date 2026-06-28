# SMC-SD Megakernel — Paper Outline (megakernel-as-centerpiece)

## Thesis (one sentence)
SMC speculative decoding accepts all drafted tokens and reweights particles instead of
rejecting, so its decode cycle has a **fully static, data-independent execution shape** —
and that property is what lets the *entire multi-model speculative cycle compile to a single
persistent GPU kernel (a megakernel)*, a fusion regime that rejection-based SD cannot reach.

## Working title
*"Megakernel Speculative Decoding: Exploiting Static Execution Shape to Fuse the SMC Cycle into One Kernel"*

## Contributions (megakernel-forward)
- **C1 (headline): the complete SMC megakernel.** First single-kernel realization of a
  *multi-model* speculative cycle (γ draft forwards + target verify + reweight + resample +
  bonus), enabled by static shape + on-chip Gumbel sampling. Low-latency single-stream regime.
- **C2: the fusion ladder + what each rung removes.** eager → CUDA graph (one launch) →
  host-op elision → megakernel (one kernel). Already-shipped graph/host-slim stack (+14–16%
  measured) is the stepping stone and the baseline the megakernel must beat.
- **C3: graceful degradation + planner.** TP/large targets break the single-kernel property →
  hybrid (draft-megakernel + target-graph, disaggregated). A regime-aware planner picks
  fuse-vs-disaggregate by (batch, N, target size).

---

## §5 (core): Megakernel design for the SMC cycle

### Execution model
Persistent grid of CTAs, each running an interpreter over a **compile-time instruction
schedule** (static shape ⇒ schedule is fixed). Instructions: GEMM tiles (QKV / O / MLP),
per-step attention, RMSNorm, residual, gumbel-argmax sample, resample reduction. Dependencies
expressed as a DAG; only true deps get a barrier (compiled minimal barrier set, not per-layer
grid-sync). The win over a CUDA graph: **no activation HBM round-trip at kernel boundaries**
and **software-pipelined weight-load of op N+1 overlapping compute of op N**.

### Phase A — Draft (γ+1 steps, model = 1B)
- Same weights reused across all γ steps ⇒ stream draft weights layer-by-layer; each layer
  computes all (N×bs) tokens for the step before advancing (fat GEMM, good arithmetic intensity).
- Per step: RMSNorm → QKV → KV append → attention(incl. this step) → O → resid → MLP → resid;
  final norm + lm_head → logits → **gumbel-argmax** → token feeds step t+1.
- γ grid barriers (AR-sequential anyway → on the existing critical path, not new serialization).
- RNG: Philox keyed by (seed, cycle, step, particle) — deterministic, megakernel-safe,
  validatable against the eager path with the **existing equivalence test harness**.

### Phase B — Verify (model = 8B/70B)
- One causal forward over the static N×(γ+1) drafted block → target logits per position.
- Target weights stream from HBM (memory-bound) → **software-pipelined GEMM is the biggest win
  here** (load layer L+1 while computing L). This is exactly the lever a graph can't pull.
- Fused tempered logprob: (logits/T)[tok] − logsumexp(logits/T), on-chip.

### Phase C — Reweight / resample / bonus
- Per-position α-weighted logprob diff → importance weights (bs, N).
- Normalize across particles (global reduction over N), resample indices, gather states.
- Bonus: gumbel-argmax on tempered target at last position.
- Tiny ops, fold as final instructions (1–2 barriers).

### Design decisions to argue in the paper
- **Barrier granularity**: minimal compiled barrier set vs naive per-layer grid-sync.
- **Two weight-residency regimes** in one kernel: draft resident-ish, target streamed.
- **Persistent grid sizing / per-phase tile shapes** (draft GEMMs vs streamed target GEMMs).
- **On-chip activation budget**: tile over particles when N×bs×hidden won't fit SMEM/regs.

### Why it beats the CUDA graph (the motivation, must be quantified)
Graph removes launch dispatch but still (a) round-trips activations to HBM per kernel boundary,
(b) can't overlap op N+1 weight-load with op N compute, (c) tears down the grid each kernel.
At bs=1 / small N the cycle is **weight-read-bound** (already observed) → weight-load↔compute
overlap is precisely the megakernel-only lever.

### *** Do this BEFORE building (sizes the prize) ***
Profile the **current full-cycle graph's roofline fraction** at bs=1, N=4, γ=4, 8B target.
Rough memory floor: target verify ≈ read 8B params once (~16 GB bf16); draft ≈ (γ+1)×~2 GB.
At ~8 TB/s HBM that's ~3 ms/cycle of unavoidable weight traffic. **If the graph already sits
at ~80% of memory roofline, megakernel upside is the remaining ~20% + cross-op pipelining —
real but bounded.** Measure this first; it decides whether the headline is strong.

---

## Full paper structure
1. Intro — SMC recap; no-rejection ⇒ static shape; thesis; contributions.
2. Background — SMC-SD (particles, importance weights, power target p^α, bonus) vs rejection
   SD (EAGLE/Medusa/SpecInfer) and *why their dynamic shape forecloses full-cycle fusion*.
3. The static-shape property — formalized; the conceptual core.
4. Fusion ladder (shipped): Gumbel in-graph sampling + graph-safe RNG; one-launch full-cycle
   graph incl. verify; host-op elision. ← the +14–16%, and the megakernel's baseline.
5. **The complete SMC megakernel** (above) — the headline.
6. Degradation + disaggregation: TP/large-target hybrid; draft/target pipelining across groups;
   particle parallelism; the regime-aware planner.
7. Evaluation (below).
8. Related work — SD systems; CUDA-graph inference; **megakernels (Hazy Megakernels, Mirage/MPK
   persistent-kernel compilers, FlashInfer)** — position as *multi-model + speculative + static*;
   disaggregation (DistServe, Splitwise).
9. Discussion / limitations — eng cost, portability, fuse-vs-disaggregate crossover.

## Evaluation plan (megakernel-centered)
- Headline: inter-token latency vs the ladder (eager→graph→host-slim→**megakernel**) at bs=1,
  sweeping N and γ.
- Crossover: megakernel vs graph vs disaggregated across batch size → where each wins (planner).
- Accuracy-neutral: GSM8K band + sampler equivalence (TV-distance / bitwise) via existing harness.
- Ablations: barrier granularity, weight-residency policy, software-pipelining on/off.
- Generality: ≥1 more (target, draft) pair / family so it's not one-system engineering.
- Hardware: B200 primary; Hopper for portability if time.

## De-risking roadmap (cheapest signal first)
1. **Roofline measurement** of current graph (no build) — confirm headroom.
2. **Draft-only megakernel** (1B, γ steps, in-kernel gumbel) vs draft-phase graph — isolates
   AR-fusion win; validate with equivalence test. Fastest go/no-go.
3. Add **verify** fused → full single-GPU 8B cycle.
4. Fold **reweight/resample/bonus** → complete one-kernel cycle.
5. **Hybrid** (draft-megakernel + target-graph) for 70B/TP → graceful scaling figure.

## Tooling decision
Hand-written (ThunderKittens/CUTLASS, Hazy-style) vs persistent-kernel compiler (Mirage/MPK).
Recommend: prototype draft-only with a compiler if it supports two-model schedules; else
hand-write the (small, tractable) draft kernel to validate the thesis before investing in target.

## Risks of megakernel-first
- Eng cost — mitigated by milestone 2 giving early go/no-go.
- Novelty vs single-model Llama megakernels — lean on multi-model + speculative + static-shape.
- If megakernel only ties the graph at bs=1 → weak paper. The roofline step (milestone 1) and a
  small-N/large-hidden eval point exist to *find* the regime where overlap clearly wins.

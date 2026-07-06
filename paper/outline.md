# MLSys Paper Outline — Asynchrony-Centered SMC-SD

**Working title:** *Speculation Without Rollback: Asynchronous LLM Inference via Sequential Monte Carlo*

**Alternates:**
- *From Control Flow to Dataflow: Fully Overlapped Speculative Decoding with SMC*
- *Rollback-Free Speculative Decoding: SMC-SD as an Asynchronous Inference Engine*

---

## Thesis (one sentence)

Rejection-based speculative decoding is a **control-flow** algorithm — the verify
result decides rollback position, batch shape, and the next draft token — while
SMC-SD reduces the entire target→draft feedback to a vector of log-weights,
i.e. **dataflow**, and data (unlike control) can arrive late; this one property
yields static shapes, full-cycle CUDA graphs, complete CPU/GPU overlap,
delayed-by-1 resampling with both models always busy, and natural draft/target
disaggregation.

## Contributions

1. **Characterization of the synchrony tax.** Measurement study of
   rejection-based spec decode (EAGLE-style + vanilla two-model) in SGLang:
   per-step host syncs, CUDA-graph breaks from variable accept lengths, KV
   rollback traffic, draft/target idle bubbles — as a fraction of step time.
2. **The asynchrony ladder.** An SMC-SD engine where each optimization is a
   *consequence* of the no-rollback property: slot-for-life state, refcounted
   KV fan-out, fused collect/resample kernels, full-cycle CUDA graphs
   (graph-safe Gumbel-max sampling), overlapped scheduling, deferred bonus.
3. **Delayed-by-1 SMC-SD** (new algorithm + system): resample at cycle *t+1*
   using weights through *t−1*, with lineage-corrected pending weight
   application; validity argument (resampling time is a free parameter in SMC;
   unbiased log Ẑ preserved); enables draft(t+1) ∥ verify(t) overlap and
   draft/target disaggregation with only token-ids forward / logprob-diffs
   backward on the wire (no KV movement).
4. **Evaluation**: throughput at every ladder rung, GSM8K accuracy parity with
   ESS / log Ẑ runtime certificates, delay-depth ablation, disaggregated
   2-GPU prototype. (Approx-to-exact treatment as a defensive subsection, not
   a headline claim.)

---

## Section-by-section

### 1. Introduction (~1.25 pages)
Draft in [`intro.md`](./intro.md). Hook → synchrony tax → root cause is the
control dependency → SMC-SD turns it into dataflow → ladder → delayed-by-1 →
results preview → contribution bullets.

### 2. Background & Motivation: The Synchrony Tax (~1.5 pages)
- 2.1 Rejection-based speculative decoding recap; where the accept/reject
  decision flows: host sync → rollback → dynamic batch shape → next draft seed.
- 2.2 **Measurement study** (the section that wins the paper). Instrument
  SGLang EAGLE3 + vanilla spec decode: stacked per-step time breakdown
  (forward / host sync / rollback+re-alloc / graph-ineligible dispatch), and a
  GPU timeline trace showing bubbles. Same hardware/models as our eval.
- 2.3 Why batching doesn't fix it: the control dependency is per-request;
  bubbles persist at low latency targets (bs=1–8, the interactive regime).

### 3. SMC-SD: Speculation as Dataflow (~1.5 pages)
- 3.1 Algorithm recap (particles, γ-block drafting, weight increments,
  ESS-triggered systematic resampling, posterior finalization, unbiased Ẑ).
- 3.2 **The key property**: all γ+1 tokens always accepted ⇒ shapes static and
  known ahead of time; the *only* target→draft feedback is (a) log-weight
  increments, (b) optionally the bonus token. (a) is delayable data; (b) is
  removable (self-continuation / SIS-seed drafting, §5).
- 3.3 What exactness costs and buys: rejection sampling is per-token exact but
  synchronous; SMC is per-sequence consistent (N→∞) and asynchronous. Set up
  the diagnostics (ESS, log Ẑ) used in §7.

### 4. The Asynchrony Ladder (engine; ~2 pages)
Present the engine as rungs, each earned by the previous. Report tok/s per
rung in §6 with the same labels.
- **L0** Slot-based synchronous engine: slot-for-life state, refcounted KV
  fan-out (zero-copy materialization + resample), fused collect/resample
  Triton kernels, no retraction path.
- **L1** Sync-free GPU cycle: write-back / collect / dispatch as pure
  enqueues; device-resident resample plan (counter-gated worst-case grids);
  pinned-snapshot postprocessing.
- **L2** Full-cycle CUDA graph: draft AR + verify + weight diff + bonus in one
  capture; graph-safe Gumbel-max sampling.
- **L3** Overlapped scheduler: CPU postprocessing of step t under GPU step
  t+1; deferred-bonus draft schedule (γ forwards per cycle).
- **L4** **Delayed-by-1 resampling** (new): self-continuation drafting +
  pending-weight buffer + lineage-corrected fold ⇒ verify(t) ∥ draft(t+1) on
  separate streams; split cycle graphs.
- **L5** **Disaggregation** (new): draft engine free-runs on GPU A; target
  scores on GPU B; wire traffic = bs·(γ+1) token ids forward, bs·(γ+1)
  logprob diffs back, per cycle — no KV ever moves (draft and target hold
  separate caches; only the block table is shared, and in disagg each side
  applies the same device-resident resample plan locally).

### 5. Delayed-by-1 SMC-SD (algorithm; ~1.5 pages)
- 5.1 Self-continuation drafting: seed cycle t+1 from the draft's own sample
  instead of the target bonus (pure-q proposal; weight gains the (γ+1)-th
  column). Removes the last token-level dependency.
- 5.2 Delayed resampling: resample at t+1 on weights through t−1; the cycle-t
  increment is a *pending* increment folded one cycle late, gathered through
  the resample plan's ancestor map (lineage correction).
- 5.3 Validity: resampling timing is a free parameter of SMC; a delayed
  scheme is SMC with two-cycle interval weights; unbiasedness of Ẑ
  preserved; ESS is evaluated on a stale weight vector — bound/discuss the
  effect, measure it in §7.
- 5.4 Pipeline schedule + steady-state timeline; delay depth is
  ⌈verify latency / draft-cycle latency⌉ (=1 in our regime).

### 6. Throughput Evaluation (~1.5 pages)
Ladder tok/s, overlap timelines, disagg scaling. See experiment matrix below.

### 7. Statistical Evaluation (~1.5 pages)
GSM8K accuracy parity across the ladder; ESS/resample-frequency vs delay;
log Ẑ agreement between sync and delayed engines; the approx-to-exact
subsection: consistency in N, diagnostics as runtime certificates, and (if
ready) an exactness-recovering knob.

### 8. Related Work (~0.75 page)
- Spec decode (Leviathan/Chen, Medusa, EAGLE 1–3, tree/block decoding).
- **Prediction-based asynchronous spec decode** — the closest line, must be
  handled head-on: PEARL, AMUSD (prepare only the all-accept outcome),
  SwiftSpec, SpecBranch, Mirror-SD, DSI, and **SSD/Saguaro (Kumar, Dao, May;
  ICLR 2026, arXiv:2603.03251)** which generalizes them: draft on separate
  hardware predicts verification outcomes (accept count + bonus token),
  pre-speculates a fan-out-B cache, falls back to synchronous drafting on a
  miss. Contrast: they *predict* the control decision (speculative-execution
  style — fan-out waste, misprediction stalls, hit rate degrades at high T
  and large batch, per their own Fig. 3); we *remove* the decision (no
  outcome to predict, no bonus token to guess, staleness costs ESS not
  compute). They remain per-token exact; we trade that for consistency +
  certificates. One-liner: *they predict the branch; we delete it.*
- Async & disaggregated serving (Splitwise/DistServe/Mooncake — they
  disaggregate *phases*, we disaggregate *models*).
- SMC for LLMs (twisted SMC, particle Gibbs steering); async particle
  filters / delayed-resampling literature.

### 9. Conclusion (~0.25 page)

---

## Figure list

| # | Figure | Section | Status |
|---|--------|---------|--------|
| 1 | Teaser: control-flow (spec decode: sync→rollback→reshape) vs dataflow (SMC: weights flow, delayable). Two timelines, bubbles vs dense. | 1 | mock now, real trace later |
| 2 | Synchrony-tax stacked bars: per-step time breakdown for EAGLE3 / vanilla SD / SMC-SD | 2.2 | needs instrumentation |
| 3 | Delayed-by-1 pipeline schedule (draft stream / verify stream / fold+resample points, ancestor correction inset) | 5 | drawable now |
| 4 | Engine architecture (slots, refcounted KV, fused kernels, snapshot) | 4 | adapt from docs |
| 5 | Ladder bar chart: tok/s at L0→L5, GSM8K config | 6 | L0–L3 data exists (`ladder_*.csv`); L4–L5 new |
| 6 | GPU utilization timelines: sync vs L4 vs L5 (the "no bubbles" figure) | 6 | needs nsight traces |
| 7 | Accuracy & ESS vs delay depth / threshold; parity table | 7 | new runs |
| 8 | log Ẑ and ESS traces as runtime certificates (sync vs delayed overlay) | 7 | new runs |

## Experiment matrix (GSM8K-first; other tasks deferred until GSM8K is solid)

Models: Llama-3.1-8B-Instruct target + Llama-3.2-1B-Instruct draft, T=0.7,
B200/B300, 400q GSM8K unless noted. Baselines: AR, vanilla two-model SD,
EAGLE3 (throughput; accuracy parity vs AR sampling).

| ID | Question | Config sweep | Metrics | Blocked on |
|----|----------|--------------|---------|-----------|
| E0 | Synchrony tax in baselines | EAGLE3, vanilla SD; bs∈{1,4,8} | step-time breakdown, bubble % | instrumentation only |
| E1 | Cost of self-continuation (drop target bonus) | N∈{4,8,12}, γ∈{4,8}, bonus∈{target, self} | accuracy, tok/s, ESS traces | M0 |
| E2 | Statistical cost of delay | delay∈{0,1}, threshold∈{0.5,0.7}, N∈{8,12} | accuracy, resample freq, ESS, log Ẑ vs sync | M1 |
| E3 | Single-GPU overlap win | L3 vs L4, N=8/12, γ=8, bs∈{1,4} | tok/s, timeline trace | M2 |
| E4 | Disagg win + wire cost | L5 on 2 GPUs; same-node | tok/s, bytes/cycle, both-GPU util % | M3 |
| E5 | Ladder headline | L0→L5, best config each | tok/s (Fig 5) | all |
| E6 | Scale-up sanity (later) | 70B/1B pair, L4 config | tok/s, accuracy (200q) | M2 |

Gate: E1/E2 accuracy within noise of the sync engine (±~1.5% on 400q) before
investing in M2/M3. If self-continuation costs accuracy, check whether +N or
γ↓ recovers it before proceeding.

## Review risks & prepared answers

- **"Rejection SD is exact; yours isn't."** → §3.3 + §7: consistency,
  ESS/log Ẑ certificates, approx-to-exact knob; accuracy parity is evidence,
  the diagnostics are the argument.
- **"SSD/PEARL/AMUSD already overlap and disaggregate draft and target."**
  → §8 + intro ¶4: they predict the control decision (fan-out pre-execution,
  fallback stall on miss, hit rate collapses at high T / large batch — their
  own data); we eliminate it (nothing to predict, no discarded work, cost is
  ESS). Bonus symmetry: SSD's §4.1 exists to guess the bonus token; our
  self-continuation deletes it. Ideally E3/E4 include an SSD-style baseline
  at T=0.7 where its cache hit rate is weakest.
- **"Why not just a better drafter (EAGLE3)?"** → orthogonal: asynchrony
  properties are drafter-independent; SMC composes with EAGLE-style drafters
  (roadmap; small demo if time permits).
- **"GSM8K only?"** → plan: expand to HumanEval + MT-Bench + one long-form
  task once GSM8K pipeline is validated (explicitly staged; see matrix).
- **"Delay must hurt the sampler."** → E2 isolates the *statistical* effect
  of delay at identical wall-clock (M1 runs delayed logic on the sync
  engine) — clean ablation, no systems confound.
- **"Disagg is a toy."** → wire-traffic table (bytes/cycle vs KV-migration
  approaches); same-node 2-GPU is the claim, cross-node is future work.

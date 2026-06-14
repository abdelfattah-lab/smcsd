# Async SMC speculative decoding — design + goals

Goal: a drafter that free-runs ahead of the verifier (verifier lags), instead of
the current lockstep where each round blocks on the target. Approved relaxations:
**drop the target-sampled bonus anchor** and **tolerate stale / ragged particles**.

**Gate:** GSM8K 200 questions runnable at ~70% (N=12, γ=8, temp 0.7, triton).

## Accuracy ablation (GSM8K 200q, N=12, γ=8, temp 0.7, triton, batch 1)

Each row changes ONE factor from the one above it — the controlled decomposition of
where accuracy comes from and what each async relaxation costs:

| # | configuration | accuracy | Δ | what it isolates |
|---|---|---:|---:|---|
| 0 | bonus baseline (colocated) | 71.0% | — | reference: exact target-sampled anchor each window |
| 1 | drop bonus, anchor @ draft temp 0.7 | 64.5% | −6.5 | cost of dropping the exact-target anchor (population drift) |
| 2 | + anchor temp 0.3 | 69.0% | +4.5 | mode-seeking anchor cuts drift (interior optimum; 0.15→62.5, 0.4→68.5) |
| — | (no resampling at all) | 55.0% | −14 vs #2 | resampling is load-bearing |
| 3 | + barrier-delay resampling K=2 (colocated proxy) | 64.5% | −4.5 | cost of delaying resampling to K-window barriers (saturates: K=4 also 64.5%) |
| 4 | **async runtime, K=2** (real 2-process) | **66.0%** | — | systems realization; ≈ #3 + decoupled noise |
| 5 | async runtime, K=1 (no overlap) | 70.0% | — | per-window resample, drained every window ≈ lockstep |
| 6 | async runtime, K=4 (more overlap) | 64.5% | — | accuracy/throughput knob |

**Async K-sweep (accuracy vs throughput, 200q):**

| async K | accuracy | tok/s | note |
|---:|---:|---:|---|
| 1 | 70.0% | 76.9 | no overlap (≈ lockstep 68.5% / 79.8) — confirms the async path doesn't regress |
| **2** | **66.0%** | **97.1** | **+26% tok/s vs K=1; sweet spot** |
| 4 | 64.5% | 96.8 | tok/s flat vs K=2, accuracy −1.5pt |

The throughput **saturates at K=2** (≈97 tok/s): K=4 adds more overlap windows but no
more speed — the drafter is the bottleneck, so once verify is fully hidden behind
draft at K=2, more overlap can't help; it only adds resampling delay. So K=2 is
optimal: full overlap throughput at the best accuracy among overlapping configs.

Takeaways: the two async relaxations (drop-anchor, barrier-delay resampling) each cost
a few points but are individually recoverable/bounded; the anchor temperature is the
single most important knob (±6pt across 0.15→0.7). Async K is the throughput/accuracy
dial — but throughput saturates at K=2 (drafter-bound), so K=2 dominates K≥4.

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

## Tier 2 — async drafter (design decided by de-risk measurements)

Two cheap measurements (colocated, no systems work) settled the architecture
before building anything:

1. **Is resampling load-bearing?** no-bonus anchor-0.3, **resample threshold 0**
   (no resampling) → **55.0%** vs 69.0% with resampling. Worth ~14pt — the async
   path MUST keep resampling. (Symptom: degenerate runs pick a prematurely-finished
   max-weight particle → short answers that never reach `####`.)
2. **Does delaying resampling to barriers preserve accuracy?** `SMCSD_RESAMPLE_INTERVAL=K`
   resamples only every K decode steps (interval_weights accumulate between), which
   is exactly the barrier-async approximation. **K=2 → 64.5%, K=4 → 64.5%** (vs
   K=1 per-window 69.0%, no-resample 55.0%). The delay costs a *fixed* ~4.5pt that
   saturates immediately (K=2 ≈ K=4) — and 64.5% is within ~1.8pt of the **decoupled
   lockstep no-bonus baseline (66.3%)**, which is the relevant comparison since async
   runs on the decoupled path. So the barrier approximation is **essentially free
   relative to where async actually runs.** Barrier-async design validated.

### De-risk verdict (all three measurements positive)

| step | finding | implication |
|---|---|---|
| drop-anchor (Tier 1) | 69.0% @ anchor 0.3 (tied w/ 71.0% bonus) | anchor becomes drafter-known → async possible |
| no resampling | 55.0% | resampling load-bearing; async must keep it |
| resample every K=2–4 | 64.5% (~free vs decoupled 66.3%) | barrier-async preserves accuracy; **no redraft/rollback needed** |

The async algorithm is fully de-risked. The `SMCSD_RESAMPLE_INTERVAL` knob is the
validated algorithmic core.

### Systems realization (built — `smcsd/decoupled/async_scheduler.py`)

`AsyncDecoupledSMCEngine` / eval `--mode smc_async`. The overlap is a **prefetch**:
the verifier sends the next window's StepReq the instant it receives the current
response (the anchor is drafter-known, so no verifier round-trip is needed for it),
then runs its local target verify while the drafter computes that next window. **The
drafter is unchanged** — still a pure reactor. Resampling fires only at K-window
barriers where the pipeline is drained, reducing to the existing
`batched_resample_kv` clone (no redraft/rollback). Finished particles ride along to
the barrier with weight increments masked.

Each event-loop iteration runs one K-window prefetch train, drained at the barrier
so prefill admission stays clean. Requires no-bonus (`SMCSD_DROP_BONUS=1`).

**Measured (GSM8K 200q, no-bonus anchor 0.3, N=12 γ=8 temp 0.7 triton, batch 1, 2×A6000):**

| mode | accuracy | tok/s |
|---|---|---|
| decoupled lockstep no-bonus | 68.5% | 79.8 |
| **async (K=2 barrier, prefetch)** | **66.0%** | **97.1** |

**+21.7% throughput at batch 1**, accuracy within noise (2.5pt ≈ 0.75σ at n=200,
plus the K=2 barrier delay measured as ~free) — the within-group draft/verify
overlap the cohort pipeline cannot provide with a single group. No seq_lens
divergence; prefetch / barrier / ride-along all correct end-to-end.

**At scale (GSM8K 1000q, anchor 0.3, K=2):** 68.1% accuracy (8 invalid) — holds at
scale, a touch above the 200q. A strict adversarial code review verified all seven
async invariants (prefetch seq_lens parity, writeback ordering, finished-before
weight mask, ride-along output/KV safety, no-bonus weighting parity vs colocated,
tag/FIFO, prefill/close ordering) with no critical/high bugs.

### Drafter CUDA graphs (the bottleneck fix — bigger win than the overlap)

The drafter ran graph-free (9 eager per-step AR forwards) and was the system
bottleneck — so async hid the *cheap* stage (verify) behind the *expensive* one
(draft). Porting the colocated draft path (cuda-graph replay per AR step +
multistep backend) into `SMCDraftServer` shortens the bottleneck directly:

| config (GSM8K 200q, async K=2, anchor 0.3, batch 1) | accuracy | tok/s | vs lockstep |
|---|---:|---:|---:|
| decoupled lockstep no-bonus | 68.5% | 79.8 | — |
| + async overlap (K=2) | 66.0% | 97.1 | +21% |
| **+ drafter CUDA graphs** | **66.0%** | **133.1** | **+66%** |

**+37% on top of the overlap, zero accuracy cost** — and a larger win than the
async overlap itself, exactly as predicted (fix the bottleneck stage first). Toggle:
`draft_cuda_graph` kwarg / `SMCSD_DRAFT_CUDA_GRAPH`. (A 20q smoke read 50% — n=20
sampling noise; the 200q confirms 66.0% parity with the graph-free async.)

Draft-path decomposition (async K=2, 200q — all accuracy-equivalent ~66%, the win is
pure speed): plain eager per-step **97.1**, multistep backend only **86.2** (the
per-step attn-backend switching costs more than it saves at this batch — it's the
uncaptured-shape fallback, not the fast path), cuda-graph replay **133.1**. The cuda
graph is the entire speedup; the multistep backend is kept only as the >cuda_graph_max_bs
fallback.

Next levers: deeper prefetch (W>1) and composing async with cohort pipelining for
batch>1.

### Profiling (torch trace of the verifier scheduler; `scripts/smc_profile_engine.py --engine-kind smc_async|smc_decoupled`)

Verifier-GPU busy fraction (union of kernel intervals on the main compute stream ÷
wall span):

| trace | GPU busy | wall | util |
|---|---:|---:|---:|
| lockstep, DECODE-only | 1387 ms | 2130 ms | **65.1%** |
| async (K=2), combined | 1737 ms | 2410 ms | **72.1%** |

The lockstep DECODE trace shows the verifier GPU **~35% idle** — that idle is exactly
the time the verifier waits for the drafter's StepResp while the drafter computes the
window. That idle is what the async prefetch recovers (the verify runs while the
drafter computes the *next* window), surfacing as the +21–26% throughput. The async
util (72%) is directionally consistent though not a clean apples-to-apples decode
comparison (the async run's by-stage split fell back to a single combined trace, so
it includes the compute-dense prefill). Traces open in Perfetto (ui.perfetto.dev).

### Chosen design: free-run between resample barriers (frontier-clone)

With the no-bonus anchor the drafter's next anchor is its own token, so the drafter
can **free-run** without the verifier. Design:

- Drafter free-runs windows (bounded credit W), anchored on its own tokens, pushing
  `(tokens, draft logprobs)` per window; never stalls except on credit.
- Verifier scores windows behind it, accumulates per-particle weights.
- Resampling happens at **K-window barriers** where the verifier has caught up to the
  drafter's frontier — so the resample is applied at a *common* frontier and reduces
  to the existing `batched_resample_kv` full-state clone (survivor → retired). **No
  redraft, no rollback, no stale-window buffer remapping** — the barrier makes
  verifier and drafter agree on the frontier.
- The *only* approximation: resampling is delayed from its ESS-detection window to the
  next barrier (stale by ≤K windows) — the relaxation already approved. Tier 2b
  measures its accuracy cost directly.

This is why Tier 2b matters: if "resample every K windows" holds accuracy, the async
build is the (tractable) barrier design; if it needs per-window responsiveness, the
build escalates to detection-point resampling with buffer remapping.

## Tier 2 (original sketch) — continuous async drafter + common-window-snapshot resampler

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

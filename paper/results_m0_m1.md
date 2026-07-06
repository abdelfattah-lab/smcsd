# M0/M1 Results — Self-Continuation & Delayed Resampling (GSM8K)

Setup: Llama-3.1-8B-Instruct target + Llama-3.2-1B-Instruct draft, 1×B200,
T=0.7 (draft & target), attention-backend triton, resample threshold 0.5 and
γ=8 unless noted. GSM8K accuracy, 400 questions (1σ ≈ 2.3pp per arm).
Flags: `SMC_SELF_BONUS=1` (M0 self-continuation drafting),
`SMC_DELAY_RESAMPLE=D` (M1: resample decision at cycle t uses weights
through t−D; staged increments ring-buffered, lineage-transported through
each resample, folded oldest-first; finalization includes outstanding
increments). All runs 2026-07-06; logs in `paper/e1_results/`,
`paper/e2_results/`, `paper/e2_probes/`.

## Which delay depth corresponds to which system design point

- **delay 0** — the synchronous engine (L0–L3): resample sees weights
  through the cycle just verified.
- **delay 1** — the fully-overlapped single-GPU schedule (M2/L4): verify(k)
  is launched at the end of draft(k) and completes during draft(k+1), so
  resample(k+1) has weights through k−1. Holds whenever verify latency <
  draft-cycle latency (here: ~5ms vs ~11ms).
- **delay 2** — the regime where verify latency exceeds a draft cycle
  (disaggregated slow/fat target, large batch): one more cycle in flight.
- A "stall-until-verify" delay-0 async schedule wins nothing on this pair
  (stall ≈ the serial gap it was meant to hide).

## E1 — Cost of self-continuation (M0), delay 0

Self-continuation removes the token-level verify→draft dependency (the
target-sampled bonus): the draft's over-draft d_γ becomes the (γ+1)-th
emitted token and next cycle's seed; the verify pass scores all γ+1 columns.
Throughput unchanged (±1%) — the over-draft forward was already paid.

| N | target-bonus | self-bonus | Δ (pp) |
|---|---|---|---|
| 4 | 66.2 | 60.2 | −6.0 |
| 8 | 74.5 | 68.8 | −5.7 |
| 12 | 76.2 | 71.0 | −5.2 |
| 16 | 78.5 | — | — |

Consistent ~5–6pp: the bonus is 1 exact p-sample per γ+1 tokens that
re-anchors every particle each cycle; N↑ recovers only partially
(self N=12 ≈ target N≈5–6).

## M1 validation

- 14/14 kernel unit tests pass.
- **Invariant test** (thr=0, resampling disabled — delayed and immediate
  accumulation are then the same algorithm): d0 = 51.0%, d1 = 52.0%
  (100q). Passes; the staging/transport/fold plumbing is exact.
- SIS floor for reference: thr=0 self-bonus ≈ 51–52% — resampling itself
  is worth ~18pp at N=8.

## E2 — Price of delayed resampling (400q)

| config | delay 0 | delay 1 | delay 2 |
|---|---|---|---|
| target-bonus γ=8 N=8 | 74.5 | 66.2 (−8.3) | 62.5 (−12.0) |
| self-bonus γ=8 N=8 | 68.8 | 58.0 (−10.8) | 54.5 (−14.3) |
| self-bonus γ=4 N=8 | 70.5 | — | 60.2 (−10.3) |
| target-bonus γ=8 N=16 | 78.5 | — | 65.5 (−13.0) |
| self-bonus γ=8 N=16 | — | — | 59.5 |

Mitigation probes at delay 2, target-bonus, N=8: thr=0.25 → 61.2,
thr=0.5 → 62.5, thr=1.0 → 61.8 — **threshold recalibration does not help**.
N=16 buys +3–5pp but the gap to its own delay-0 reference stays ~13pp.

**Reading.** Delayed resampling is unbiased (auxiliary-SMC correction,
validated by the invariant test) but the resample decision permanently
misses the outstanding D×(γ+1)-token window of weight information; the
fold restores unbiasedness while re-injecting exactly the variance
resampling was supposed to remove. Empirically the population slides
toward the SIS floor as the window grows: at γ=8, delay-2 lands ~2/3 of
the way from fresh-SMC (68.8) down to SIS (51). Neither threshold nor
(moderate) N recovers it; γ=4 shrinks the window and the penalty, at a
throughput cost (432 vs 533 tok/s at delay 0).

## Combined design points (the honest L4 menu, N=8, γ=8)

| design point | token dep | weight dep | GSM8K | note |
|---|---|---|---|---|
| L3 sync (bonus, delay 0) | yes | yes | 74.5 | current fastest sync engine |
| L4a semi-async (bonus, delay 1) | yes | no | 66.2 | resample off critical path; bonus still blocks full overlap |
| L4b full-async (self, delay 1) | no | no | 58.0 | the single-GPU full-overlap operating point |
| disagg/slow-verify (self, delay 2) | no | no | 54.5 | deep-pipeline regime |

## Implications for the paper

1. The dataflow architecture and its machinery (M0+M1) work exactly as
   designed — staging, lineage transport, fold, finalize are provably and
   empirically exact. What the experiments price is the *statistics* of
   asynchrony, not the systems.
2. "Staleness costs statistical efficiency, not compute" is true but the
   cost is large at small N and γ=8: −10 to −16pp end-task accuracy for
   full asynchrony. The M2 throughput upside (~1.4–1.5× projected from
   draft/verify latencies) does not obviously pay for it at these settings
   — the paper's Pareto framing must include delay as a priced axis, and
   the honest headline is the *price curve*, not a free lunch.
3. Where async can still win cleanly: (a) regimes where particles are
   cheap — large N at bs=1 is weight-read-bound, so N=32+ at delay-1 may
   cross back over sync-N=8 at equal wall-clock (untested); (b) disagg
   with a fat target where sync bubbles are much larger than 30%;
   (c) algorithmic mitigation — an auxiliary *predictor* for the
   outstanding increment (the draft's q-logprobs for the in-flight window
   are already known at decision time) is the natural next experiment and
   a potential headline contribution if it recovers most of the gap.
4. Gate G2 as originally stated (≤1.5pp) is failed by naive delay.
   Recommended reframing before M2 investment: either pursue the
   predictor mitigation, or reposition the paper's centerpiece to the
   measured sync↔async spectrum + the L0–L3 ladder + disagg wire-cost
   analysis, with delayed resampling as a carefully-priced option rather
   than the default.

---

# E3/E4 — Optimistic & Lazy-Apply Resampling (2026-07-06, second session)

## E3a — Resample-event rate (the async-schedule determinant; self-bonus, γ=8, 100q)

| config | resample rate/cycle |
|---|---|
| N=8, thr=0.25 | 0.32 |
| N=8, thr=0.50 | 0.57 |
| N=8, thr=0.75 | 0.76 |
| N=16, thr=0.50 | 0.63 |

At the default threshold, resampling fires on **57% of cycles** ⇒ any
scheme that falls back to serial on resample cycles (rollback-optimistic)
projects to only ~1.1–1.2×.

## E3b — Seed penalty is purchasable with particles (delay 0, 400q)

| N | self-bonus | target-bonus | self tok/s |
|---|---|---|---|
| 8 | 68.8 | 74.5 | 533 |
| 16 | 73.2 | 78.5 | 485 |
| 24 | **75.8** | 78.2 | 418 |
| 32 | 75.2 | — | 384 |

Self-bonus N=24 ≥ target-bonus N=8 (75.8 vs 74.5): full token-level
asynchrony costs ~3× the particles, at −22% sync throughput — which a
~1.4× overlap would repay. Plateau at N≈24–32.

## E4 — Lazy-apply resampling (fresh decision, copy deferred one cycle; 400q)

Decision on fresh weights the moment verify lands; KV/lineage/weight copy
applied at the next boundary (no rollback, no stall, cycle time independent
of resample rate). Implemented as `SMC_LAZY_RESAMPLE=1`.

| config | sync (delay 0) | lazy | Δ | naive delay-1 |
|---|---|---|---|---|
| self-bonus N=8 | 68.8 | 59.8 | −9.0 | 58.0 |
| self-bonus N=16 | 73.2 | 67.2 | −6.0 | — |
| self-bonus N=24 | 75.8 | 68.5 | −7.3 | — |
| target-bonus N=8 | 74.5 | 65.5 | −9.0 | 66.2 |

**Lazy-apply ≈ naive delay.** Making the decision fresh does not recover
the loss; deferring the *application* costs the same as deferring the
*decision*.

Temperature check (sync sb0 N8 vs lazy sb1 N24): gap −3.0 at T=0.5,
−6.0 at T=0.7, −4.2 at T=1.0 — the penalty is not a temperature artifact.

## The invariant cost — an async trilemma

The two async schemes fail identically because they share one property:
**during the overlap window, the population that keeps drafting is the
pre-resample population** — either because the decision hasn't been made
(naive delay) or hasn't been applied (lazy). Every resample event spends
one draft cycle of population diversity on trajectories about to be
culled/duplicated, and at a 0.57 event rate that is ~half of all cycles.

Pick two of three:
- **fresh decision + immediate apply** → synchronous (stall): 74.5% @ serial
- **no stall + immediate apply** → stale decision (naive): 58–66%
- **no stall + fresh decision** → late apply (lazy): 60–69%

## Measured/projected Pareto menu (8B/1B, γ=8, T=0.7)

| design point | GSM8K | tok/s (sync measured) | tok/s (M2 projected) |
|---|---|---|---|
| sync, target-bonus, N=8 | 74.5 | 532 | — |
| stall-on-resample, self-bonus, N=24 | 75.8 (=sync stats) | 418 | ~480–500 (1.15–1.2×) |
| lazy, self-bonus, N=24 | 68.5 | 426 | ~590–620 (1.4–1.45×) |
| lazy, self-bonus, N=16 | 67.2 | 485 | ~680 |

The projections require the M2 two-stream engine to become real numbers.

## Where this leaves the paper

The async section's honest core is the **trilemma + the complete measured
map** — to our knowledge no one has characterized the statistical price of
asynchrony in population-based speculative decoding, and the machinery
(staging ring, lineage transport, lazy plans, fresh/stale decision timing)
is exactly the apparatus needed to measure it. The Pareto menu gives
deployment guidance; M2 realizes the throughput axis. Remaining algorithmic
open door: anticipatory diversification during the shared window (e.g.,
offspring pre-split with extra proposal randomness) — unexplored.

## E4b — True pipelined-lazy (delay-1 decision + lazy apply, 400q)

On a single GPU, verify(k) cannot land before draft(k+1) starts without a
stall, so the real pipelined-lazy engine pays BOTH a one-cycle-stale
decision and a one-cycle-late copy. Emulated via
`SMC_DELAY_RESAMPLE=1 SMC_LAZY_RESAMPLE=1` (ring transported through the
applied held plan; fold after collect).

| N | sync | pure lazy (E4) | true pipelined-lazy |
|---|---|---|---|
| 8 | 68.8 | 59.8 | 57.2 |
| 16 | 73.2 | 67.2 | 64.0 |
| 24 | 75.8 | 68.5 | 62.2 |

The two effects compound at larger N (−13.6pp at N=24). **Verdict: lazy
mode is not the recommended operating point; rollback mode is.**

## Final Pareto menu (accuracy measured, throughput projected until M2)

| design point | GSM8K (400q) | tok/s |
|---|---|---|
| sync, target-bonus, N=8 (L3) | 74.5 | 532 measured |
| **M2-rollback, self-bonus, N=24** | **75.8** (sync stats by construction) | ~480 proj. (1.15×; 1.27× at thr=0.25 with accuracy TBD) |
| M2-lazy, self-bonus, N=16 | ~64.0 | ~700 proj. (1.45×) |

## M2 build plan (pipelined engine; next session)

Design settled by this session's experiments and race analysis:

1. **Worker split**: `SMCWorker.forward_draft_phase` (γ+1 AR forwards +
   token write-back inputs; runs on schedule stream D) and
   `forward_verify_phase` (verify + diff; enqueued on stream V after an
   event on draft completion). Eager path only at first (cycle-graph split
   is later polish). Self-bonus required (the draft must free-run).
2. **Rollback mode (primary)**: the draft AR loop is host-driven per step —
   between steps, host-checks whether verify(k−1)'s decision event fired
   with a resample. On resample: stop issuing steps, rewind the cycle's
   seq_lens/kv_allocated (no committed state exists mid-cycle — tokens
   commit only at cycle-end write-back), apply the plan, redraft the cycle.
   Clean cycles run fully overlapped. Accuracy = sync self-bonus by
   construction.
3. **Lazy mode (ablation)**: the delay+lazy composition as emulated here.
4. **Benign-race analysis** (needed for verify(k) ∥ apply(plan) on separate
   streams): verify outputs for plan-dst rows are discarded via ring
   transport; verify KV writes for dst rows land in pages whose free is
   deferred one full cycle by the existing `kv_freed_buf` double-buffer —
   no reuse window. Block-table reads mid-copy give garbage *values* for
   dst rows only, never corruption.
5. Measure: real tok/s for rollback N∈{8,24} × thr∈{0.25,0.5} and lazy
   N=16; nsight timelines for the paper's utilization figure.

---

# E5 — M2 Pipelined Engine, Rollback Mode (2026-07-06, third session)

Implemented: `SMC_PIPELINE=1` — verify on a dedicated CUDA stream, next
cycle's draft launched optimistically, decision polled between draft AR
steps; on a resample the in-flight cycle aborts (rewind committed-prefix
lengths → dispatch plan → re-advance onto the same pre-allocated pages →
redraft with post-copy seeds). Nothing commits mid-cycle, so the sampler is
distribution-identical to the synchronous engine by construction.
Worker split: `draft_cycle` / `verify_cycle`; write-back split:
tokens-only + persisted per-slot weight cutoff; verify-stream weight fold.

## Benchmark (100q, γ=8, thr=0.5, self-bonus both arms)

| N | sync tok/s | pipelined tok/s | ratio | abort rate | acc sync/pipe |
|---|---|---|---|---|---|
| 4 | 560.2 | 481.3 | 0.86× | 0.42 | 61 / 54 |
| 8 | 528.6 | 434.2 | 0.82× | 0.53 | 65 / 65 |
| 12 | 495.8 | 393.5 | 0.79× | 0.58 | 75 / 65 |

(Accuracy should be identical in distribution; N=4/N=12 deltas are 1.5-2σ
at 100q — a 400q N=12 check is running. Non-monotonicity vs abort rate
suggests noise, not an abort-path bug.)

## Diagnosis: single-GPU pipelining loses because the engine is host-bound

Cycle time ≈ 16.6ms vs ~5ms of theoretical HBM traffic (38GB/cycle at
8TB/s): the eager engine is dominated by host launch latency, not GPU
work. Consequences:
1. The sequential engine has **no GPU bubble during verify** — kernels are
   back-to-back on one stream. There is nothing for stream-overlap to
   reclaim.
2. The pipeline *adds* host work to the critical path (per-step event
   polls, verify-batch prep before the next draft prep, event churn) and
   the abort redrafts add whole extra draft cycles at 0.42–0.58 rate.
3. Abort rate rises with N (0.42 → 0.58), shrinking the optimistic win
   exactly where accuracy wants to operate (N=12–24).

## Where the rollback design actually pays (next steps)

- **Cut host work first**: a draft-phase CUDA graph (one launch per cycle;
  the capture machinery partly exists as SMCFullCycleGraphRunner's base)
  makes the GPU the bottleneck again — only then can overlap reclaim the
  verify window. Note the graph also coarsens abort granularity to
  whole-cycle (redraft cost 2× draft), which at 57% abort rate roughly
  cancels — needs thr≈0.25 or lower resample rates to pay.
- **Second GPU (disagg-lite)**: verify on its own device removes both the
  kernel contention and (with a helper thread) the verify host prep from
  the critical path — this is where the pipelined scheduler's structure
  (decision events, abort/redraft, cross-stream fold) transfers unchanged
  and the 1.15–1.45× projections become plausible again.
- The abort machinery itself is validated and cheap; the loop, events, and
  lineage handling all work first-try — the bottleneck is purely where the
  host time goes.

## E5b — Host-vs-GPU timing (corrects the E5 diagnosis)

`SMC_PIPE_TIMING=1`, N=8: per cycle the host is busy only ~8.4ms
(draft-drive 7.3 + commit 0.2 + verify-prep 1.1), while the draft stream
still holds ~9.5ms of queued kernels after enqueue ends. The draft phase
alone costs ~17ms GPU in the pipelined run vs ~12ms sequential — the
draft kernels are STRETCHED by the co-running verify. Verdict: the engine
is GPU-saturated (bandwidth), not host-bound. Single-GPU overlap cannot
win: the sequential schedule has no idle to reclaim, and co-running only
reshuffles the same ~38GB/cycle of reads.

**Therefore 2-GPU disaggregation is the correct next step** (it adds
bandwidth rather than splitting it). Corrected projections for rollback
mode on 2 GPUs (draft GPU unstretched ~11.9ms, verify hidden):
~1.17× at thr=0.5 (53% aborts), ~1.27× at thr=0.25 — at exact sync
accuracy; the win grows with a larger target (70B verify >> draft ⇒
verify-bound regime, draft fully hidden).

## M3 build plan: draft-on-GPU1 disagg-lite

Put the DRAFT on cuda:1; keep scheduler + slot state + canonical
req_to_token/refcount pool + TARGET on cuda:0 (prefill path and all slot
kernels untouched):
- cuda:1 holds: draft weights, draft KV pool (same page indexing as the
  canonical allocator), a per-cycle mirror of the active block-table rows
  (bs × row-width ints, ~128KB p2p, negligible on NVLink).
- Per-cycle traffic: seeds in (bs int32), drafted tokens + q-logprobs out
  (bs×(γ+1) each) — kilobytes; no KV ever moves.
- Draft prefill: shim the cloned prefill batch (_make_clean_batch) to
  cuda:1; draft decode graphs capture on cuda:1 at init.
- Pipelined loop unchanged in structure: draft driven on cuda:1's stream,
  verify+fold+collect on cuda:0; same decision events, same abort/redraft
  (rewind lens → dispatch on cuda:0 → re-mirror rows → redraft).
- Init risk to de-risk first: constructing the draft ModelRunner with
  gpu_id=1 in-process (torch.distributed world-size-1 assumptions,
  set_device ordering vs the target's init).

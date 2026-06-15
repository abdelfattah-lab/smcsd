# SBP prototype — overnight build report

**Date:** 2026-06-14 · **Branch:** `smc-async-draft` · **Flag:** `SMCSD_SPEC_BARRIER` (default off)

## TL;DR

Speculative Barrier Prefetch (SBP) is **prototyped, working, and fully measured**. It fills
the async decode engine's barrier stall by firing the next train's window-0 draft across the
resample barrier, so the drafter (the bottleneck) never goes idle.

**The honest bottom line: SBP delivers a large throughput win (+32%) but at *fixed N* is NOT
accuracy-neutral — it costs ~4.4 pt of GSM8K accuracy at 1000q (the resample adoption
coupling).** BUT the throughput it frees can be reinvested in more SMC particles, which
recovers the accuracy: **SBP N=24 = 66.9% @ 146 tok/s matches the original N=12 baseline
(66.3% @ 134) and is +9% faster — a Pareto win** (see "Accuracy-recovery experiments"). So
the practical recipe is **SBP + N≈24**: same accuracy, more throughput. The KV-cache
maintenance was verified correct, so the fixed-N gap is purely algorithmic, not corruption.

**Measured (GSM8K, N=12 γ=8 temp 0.7, anchor 0.3, K=2, triton + drafter CUDA graphs,
2×A6000, seed 0):**

| metric | baseline (async K=2) | **SBP** | Δ |
|---|---:|---:|---:|
| Accuracy @200q | 65.0% (130/200) | 65.0% (130/200) | 0.0 pt *(tie — within noise)* |
| **Accuracy @1000q** | **66.3% (663/1000)** | **61.9% (619/1000)** | **−4.4 pt** ✗ |
| Throughput @1000q | 133.7 tok/s | **176.8 tok/s** | **+32%** ✓ |
| Wall time @1000q | 1432.7 s | **1036.1 s** | **−28%** |
| Total tokens @1000q | 191 522 | 183 189 | −4.4% (shorter outputs) |

The 200q tie was within-noise luck (at 200q, σ≈3.4 pt hides a 4-pt gap). At **1000q**
(σ≈1.6 pt) the gap is clear and **reproduces across two seeds: −4.4 pt (s0) and −4.5 pt (s1),
mean −4.45 pt** over 2000 questions; the output is consistently ~4–5% shorter (particles
finish earlier). The **throughput win is real and robust**
(also +33% with resampling disabled — it comes from the barrier overlap, not from changing
resampling), and **+32% far exceeds the design's +11% estimate** because SBP removes the
barrier stall *entirely* (drafter 0% idle) rather than just reducing its frequency.

> All numbers **measured**, not projected. Logs in `tasks/sbp_logs/`.

### Is this a bug or inherent? — inherent (machinery verified correct)

With resampling **disabled** (`--resample-threshold 0`, seed 0, 100q), SBP and baseline are
equivalent within float noise — **baseline 57.0% / 14 922 tok vs SBP 55.0% / 14 876 tok**
(tokens match to 0.3%; the 2-question accuracy diff is float non-determinism in the weight
reduction occasionally flipping the finalize argmax). So the fire/consume/seq_lens/anchor
machinery is correct. The −4.4 pt comes **only when resampling fires**: it is the **adoption
coupling** the design acknowledged but underestimated — a retired particle adopts its
survivor's *exact* speculative window, so the two are identical (same tokens **and** same
weight increment) for one window, reducing effective sample size right after each resample.
At ~50% resample rate this costs ~4 pt, not the "far below detection" the design claimed.

### Frontier comparison (why it's still useful)

| config | throughput | accuracy | note |
|---|---:|---:|---|
| baseline K=2 | 133.7 | 66.3% | the reference |
| larger K=4 (design's K-sweep) | ~148 (+11%) | ~61% (−5 pt) | trade via fewer barriers |
| **SBP (K=2)** | **176.8 (+32%)** | **61.9% (−4.4 pt)** | **trade via barrier overlap** |

SBP **dominates the larger-K knob** (more throughput for the *same* accuracy cost), so as a
"spend accuracy for throughput" lever it is the better one — but it is **not** accuracy-free.

## What it does (one paragraph)

At K=2 the async engine spends half its windows in a barrier stall: at the resample barrier
the verifier stops prefetching (resampling reshuffles particle KV), so the drafter sits idle
through the verifier's ~30 ms verify + ~few-ms resample. SBP fires the **next train's
window-0 `DraftStepReq`** at the last window — *before* verify + resample, *before*
`send_commit` (FIFO) — so the drafter computes it during the barrier. The next train consumes
that reply as its window 0. Resampling keeps most particles alive; the existing
`batched_resample_kv` frontier-clone already copies a survivor's full KV (including its
just-drafted speculative window) into each retired slot for free, and a verifier-side
ancestor gather rewrites each surviving slot's tokens/logprobs/anchor to the survivor it
adopted. The drafter is **unchanged** — still a pure reactor; one `epoch` field was added to
the wire purely as a fail-fast train fence.

## Mechanistic confirmation (SMCSD_TIMING)

The throughput win is explained directly by the per-window **recv (drafter-wait)** time —
how long the verifier blocks on the drafter:

| | recv / window | drafter / window | recv share of loop |
|---|---:|---:|---:|
| baseline | **28.76 ms** | 42.4 ms | 48% |
| **SBP** | **14.14 ms** | 42.4 ms | 31% |

Baseline 28.76 ms = the average of overlapped windows (~14 ms) and barrier-stalled windows
(~42 ms, full draft with zero overlap). SBP collapses it to **14.14 ms** — right at the
overlap floor (draft 42.4 − verify 29.9 ≈ 12.5 ms). The drafter's own per-window time is
unchanged (42.4 ms): SBP changes *when* the verifier waits, not the drafter's work. This is
the barrier stall removed, measured.

## Validation status

| check | result |
|---|---|
| Throughput (200q & 1000q, seed 0) | ✅ **+31.5% / +32%** tok/s |
| SMCSD_TIMING recv drop | ✅ 28.8 ms → 14.1 ms (barrier stall removed, mechanistic) |
| Machinery correct (no-resample A/B) | ✅ SBP ≈ baseline within float noise (tokens 0.3%) — no logic bug |
| KV / req-pool leak (test e) | ✅ clean — runs to completion, zero leak/refcount/OOM markers |
| `_spec_a1_source_rows` unit tests | ✅ 4/4 (survivor identity, retired→survivor, revival, retired-onto-finished) |
| Adversarial code review | ✅ no high-confidence correctness bugs (full edge-case trace) |
| **Accuracy @1000q (2 seeds)** | ❌ **−4.45 pt mean** (s0 −4.4, s1 −4.5) — inherent adoption coupling, reproducible |

## How the implementation diverged from the design doc (and why)

The design doc (`design_speculative_barrier_prefetch.md`) had the right architecture but
three of its mechanisms were wrong in ways that only surface on GPU at scale. The prototype
fixes them; the doc's `_remap_spec_resp` / "drain = discard" / "score over A0" sketches are
**superseded** by what's in the code. Profiling and crashes drove each fix — nothing guessed.

1. **Score over A1 (post-rebuild survivors), not A0 (the full drafted set).**
   The doc scored the spec window over A0 with an in-place ancestor remap. This **crashes
   with an illegal memory access**: the resample can retire an A0 slot onto a particle that
   *finished before this train* (shorter seq_len, ∉ A0); that slot's KV block table at the
   spec-window positions `[S+KG, S+(K+1)G)` is then stale, and the verify forward pass reads
   it regardless of weight masking. The slots that actually end up at the consistent
   `S+(K+1)G` frontier with valid spec KV are exactly **A1** = `slot_state._active_slots_list`
   at consume time (survivors + retired-onto-active + revived-from-finished). So the consume
   builds a fresh verify batch over A1 with `orig_seq_lens = seq_lens[A1] − (γ+1)` and gathers
   each A1 slot's columns from its adopted ancestor's A0 row
   (`_spec_a1_source_rows`: `pos_in_A0[ancestor[s]]`, always in A0). This also *fixes* a
   weight-omission the A0 approach had for revived slots. (`_build_spec_a1`,
   `_spec_a1_source_rows`.)

2. **Don't drain mid-train; don't defer the drain.**
   The doc's "drain guard discards the spec resp" **desyncs surviving particles** (the spec
   already advanced their frontier by G; discarding skips those tokens and anchors the next
   window one position back). The fix: the guard **commits** the spec standalone (scores A1,
   writes back the frontier) so survivors stay consistent. Separately, an early version
   deferred the finalize-drain into the consume and drained mid-train — which **freed slots
   that windows 1..K-1 then prepped against** (a second IMA). With A1-scoring the consume only
   touches always-allocated survivors, so the drain runs normally at barriers and finishes
   *ride along* to the consume train's own barrier, exactly like the cold path.
   (`_event_loop` guard, `_commit_spec_standalone`.)

3. **Fire window 1 before verifying window 0 in the consume.**
   The overlap that makes SBP pay off at K=2 needs the consume to fire window 1's `StepReq`
   *before* `finish_decode` on window 0 — otherwise window 1 (the costly last window before
   the barrier) is itself stalled and the speedup vanishes. (`_consume_spec_window0`.)

The doc's "Step 4: spec FIRE then discard" intermediate is **not realizable** as written
(discarding desyncs the frontier), so the build went straight to the unified consume path.

## Files changed (behind `SMCSD_SPEC_BARRIER`, default off)

- `smcsd/decoupled/io_struct.py` — `epoch: int = 0` on `DraftStepReq`/`DraftStepResp`.
- `smcsd/decoupled/draft_server.py` — echo `epoch=msg.epoch` (the only drafter change).
- `smcsd/decoupled/worker.py` — thread `epoch` through `send_step`/`send_step_req`/
  `start_decode`/`PendingDecodeStep`; assert it in `finish_decode`.
- `smcsd/decoupled/async_scheduler.py` — `SpecState`, the spec fire at the last window,
  `_consume_spec_window0`, `_build_spec_a1`, `_spec_a1_source_rows`, the `_event_loop`
  commit-guard, `_commit_spec_standalone`, ancestor capture in `_barrier_resample`.
- `smcsd/decoupled/tests/test_spec_barrier_remap.py` — unit tests for the A1 gather.

## Reproduce

```bash
cd /home/cc2869/smcsd
COMMON="SMCSD_DROP_BONUS=1 SMCSD_ANCHOR_TEMP=0.3 SMCSD_RESAMPLE_INTERVAL=2 SMCSD_DRAFT_CUDA_GRAPH=1"
ARGS="--mode smc_async --drop-bonus --particles 12 --gamma 8 --temperature 0.7 \
  --attention-backend triton --num-questions 200 --mem-fraction-static 0.6 --seed 0"
# baseline
env $COMMON SMCSD_SPEC_BARRIER=0 .venv/bin/python scripts/accuracy_test_gsm8k.py $ARGS
# SBP
env $COMMON SMCSD_SPEC_BARRIER=1 .venv/bin/python scripts/accuracy_test_gsm8k.py $ARGS
# recv-drop diagnostic: add SMCSD_TIMING=1, grep "ASYNC_TIMING"
# unit tests
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m unittest smcsd.decoupled.tests.test_spec_barrier_remap
```

## Accuracy-recovery experiments (can we push it back up?)

The gap is the **adoption coupling on resample barriers**, so I swept the knobs that change
the coupling rate (500q, seed 0). **The KV-cache maintenance was independently verified
correct first** — a code audit + runtime invariant assertions (`SMCSD_SPEC_BARRIER_DEBUG=1`)
+ a forced-95%-resample run under `CUDA_LAUNCH_BLOCKING=1` all came back clean (zero IMA,
zero invariant violations, zero leaks), so the gap is purely algorithmic, not memory
corruption.

| config | accuracy | tok/s | reads |
|---|---:|---:|---|
| baseline thr0.5 (peak) | **67.8%** | 133.6 | best accuracy, needs high resampling |
| baseline thr0.2 | 64.2% | 131.5 | less resampling → −3.6pt for baseline |
| SBP thr0.5 (default) | 62.4% | 176.4 | −5.4pt vs peak (the coupling) |
| **SBP thr0.2** | **63.8%** | 174.5 | **+1.4pt; gap to baseline-thr0.2 now only −0.4pt** |
| SBP anchor-temp 0.5 | 61.8% | 176.4 | raising anchor temp does **not** help |

**The decisive finding: the SBP gap scales with the resample rate** — definitive proof the
coupling is the cause. Lowering the ESS threshold cuts coupling (SBP 62.4→63.8%) but *also*
degrades baseline (67.8→64.2%, less resampling = more weight degeneracy), so the two
**converge** at thr0.2 (within noise) with SBP +30% faster. The catch: the *absolute* peak
(67.8%) lives at high resampling, exactly where SBP couples — so the knob trades the gap for
absolute accuracy; it does not buy back the peak. Anchor temperature is a dead end.

### The trick that works: spend SBP's throughput on more particles (N)

SBP frees ~30% throughput; reinvest it in more SMC particles. **Confirmed at 1000q, seed 0:**

| config | accuracy | tok/s | vs original baseline (N=12) |
|---|---:|---:|---|
| baseline N=12 (original) | 66.3% | 133.7 | — |
| SBP N=12 | 61.9% | 176.8 | −4.4 pt, +32% |
| baseline N=24 | 71.3% | 106.7 | +5.0 pt, −20% (slow) |
| **SBP N=24** | **66.9%** | **146.1** | **+0.6 pt acc (≈equal), +9% faster** |

**SBP N=24 (66.9%, 146.1 tok/s) matches the original N=12 baseline accuracy and is +9%
faster — a Pareto improvement over it.** The accuracy lost to coupling at fixed N is bought
back by particles: SBP N=12 → N=24 is **+5.0 pt** (61.9 → 66.9%), landing on baseline-N12.

**Honest mechanism (corrected from the 500q hint):** the coupling gap is **constant ~−4.4 pt
at both N=12 and N=24** (a 500q run suggested it shrank to −2.8 pt, but that was noise). More
particles does **not** dilute the coupling — it lifts *both* curves ~+5 pt, so the particle
gain *offsets* the coupling loss. SBP's speed is what makes the larger N affordable: baseline
N=24 is 20% *slower* than baseline N=12, whereas SBP N=24 is 9% *faster*. So **SBP + N≈24**
delivers the original accuracy at better-than-original throughput.

(The two weaker levers, for the record: resample threshold is a noisy ~±1 pt knob that trades
the gap for absolute accuracy; anchor temperature is weak/unreliable for SBP — a re-sweep with
SBP on @1000q gave 0.20→61.4%, 0.25→64.2%, 0.30→61.9%, 0.40→62.6%, i.e. mostly flat ~62% with
a lone non-monotonic 0.25 bump that beats *both* neighbors and is likely seed noise. N is the
strong lever; barrier-bonus below is the clean +2 pt one.)

A residual gap to **baseline at the same N** remains (−4.4 pt) — that is the coupling itself,
removed only by the redraft (below). But for practical use, N≈24 already recovers the
original operating point at higher throughput.

## Reviving the bonus token (a separate, larger accuracy axis)

The async path uses **no-bonus** (drafter-known anchor) because the prefetch fires the next
`DraftStepReq` *before* the verify, while the real **bonus** (the target's exact sample at the
verify's last row) is only known *after*. The two are a causality trade-off, not an
engineering gap (the scheduler hard-enforces `SMCSD_DROP_BONUS`). Measured (seed 0):

| config | accuracy | tok/s | bonus coverage | overlap |
|---|---:|---:|---|---|
| **lockstep, full bonus** (`smc_decoupled`, no `--drop-bonus`) | **73.5%** (200q) | 106.6 | every window exact | none |
| base async, no-bonus (anchor 0.3) | 66.3% (1k) | 133.7 | none | full prefetch |
| **base async + barrier-bonus** (`SMCSD_BARRIER_BONUS=1`) | **68.3%** (1k) | 133.2 | 1/K windows exact | full prefetch |
| SBP, no-bonus | 61.9% (1k) | 176.8 | none | + barrier overlap |

**The full bonus is a big lever — +7.2 pt (66.3 → 73.5%) — but it is fundamentally
lockstep-only** (every window's anchor must wait for that window's verify, killing the
prefetch). It is the **highest-accuracy config measured**, beating even baseline N=24 (71.3%).

**Partial revival that keeps the overlap — `SMCSD_BARRIER_BONUS` (new, implemented).** The one
window per train where the bonus *is* available pre-StepReq is the **drained barrier** (verify
done before the next train's window-0 fires). So the last window of each train seeds its anchor
from the exact target sample; interior windows stay no-bonus for the prefetch. Cost: it is
**mutually exclusive with SBP** (both want the barrier window — SBP fires it pre-verify), so it
runs on the *base* async (133 tok/s, no SBP +32%). Gain: **+2.0 pt confirmed at 1000q (68.3%
vs 66.3%, identical 133 tok/s)** — a real free recovery, though only ~2 of the full bonus's
+7 pt, because the bonus's benefit is *cumulative* (it fights per-window population drift), so
one exact anchor per K windows recovers a sublinear fraction; you need it *every* window
(lockstep) for the full +7 pt.

**So the bonus is a throughput↔accuracy axis, orthogonal to SBP:** want max accuracy → lockstep
full bonus (73.5% @ 107 tok/s); want max throughput → SBP no-bonus (62% @ 177); the middle is
no-bonus async / SBP+N24 (~66–67% @ 133–146). `mechanism note:` no-bonus is already
*unbiased* (IS-reweighted); the bonus only cuts *variance*, and anchor-temp 0.3 already
recovered 4.5 of the 6.5 pt colocated — the remaining lockstep-only gain is real but costs the
overlap.

## Recommendation

Two viable ways to use SBP, plus the off switch:

1. **SBP + more particles (the recommended use).** At fixed N it regresses (−4.4 pt), but its
   ~30% throughput headroom buys a larger N that recovers it. **Confirmed at 1000q: SBP N=24 =
   66.9% @ 146.1 tok/s vs the original baseline N=12 = 66.3% @ 133.7 — equal accuracy, +9%
   throughput (a Pareto win).** Use the speed to afford more SMC particles.
2. **SBP at fixed N** — only if you explicitly want to spend accuracy for throughput (it does
   so more efficiently than larger K, but it is a real −4.4 pt regression vs same-N baseline).
3. **Off (default).** Still the default flag state; flip to mode (1) where the throughput
   matters. The same-N coupling is removed only by the redraft below.

### How to recover accuracy-neutrality (the real follow-up)

The coupling is fundamental to "fire before the resample plan is known, then adopt": every
retired particle inherits its survivor's *single* speculative draw. To break it **without
losing the overlap for survivors**, redraft only the retired minority:

- Survivors (the majority, `ancestor[s]=s`) keep their adopted spec window — fully overlapped,
  zero coupling (they are already independent).
- Retired/revived slots (`ancestor[s]≠s`) get a **second, non-overlapped drafter round-trip**
  at consume time: roll their frontier back to `S+KG`, redraft a fresh independent window from
  the survivor's prefix, and overwrite the cloned spec window. This restores baseline's
  "resample-then-independent-redraft" law exactly.

Cost: it reintroduces a **drafter KV rollback** (the SpecActor machinery this port
deliberately dropped) and a small extra draft on the retired fraction each barrier — so the
throughput win shrinks from +32% toward the overlap-only floor (still clearly positive,
since only the retired fraction pays). This is a real design change (~a day), not a tweak,
and is why it was **not** attempted in this overnight build — leaving a half-built rollback
path would be worse than a clean, honest characterization.

A cheaper partial knob to explore first: raise the ESS resample threshold (resample less
often → less coupling) and re-measure the accuracy/throughput trade.

## Other notes

- **Default-on**: NO — see Recommendation.
- **K=1**: gets only alternating-train overlap (review note, not a bug). Gate is K=2.
- **Determinism**: the no-resample A/B above is the practical determinism check (machinery
  equivalent to baseline within float noise).

---

### Appendix — full run table

| run | acc | tok/s | tokens | log |
|---|---:|---:|---:|---|
| baseline 200q s0 | 65.0% | 133.7 | 37 964 | `base_seed0_200.log` |
| **SBP 200q s0** | 65.0% | 175.8 | 36 533 | `sbp_seed0_200_v3.log` |
| baseline 1000q s0 | 66.3% | 133.7 | 191 522 | `base_seed0_1000.log` |
| **SBP 1000q s0** | 61.9% | 176.8 | 183 189 | `sbp_seed0_1000.log` |
| baseline no-resample 100q s0 | 57.0% | 86.3 | 14 922 | `noresample_spec0_100.log` |
| SBP no-resample 100q s0 | 55.0% | 115.0 | 14 876 | `noresample_spec1_100.log` |
| baseline TIMING (recv) | — | — | 28.76 ms/win | `base_timing40.log` |
| SBP TIMING (recv) | — | — | 14.14 ms/win | `sbp_timing40.log` |
| baseline 1000q s1 | 66.6% | 132.2 | 192 792 | `seed1_spec0_1000.log` |
| **SBP 1000q s1** | 62.1% | 177.8 | 183 132 | `seed1_spec1_1000.log` |

**Two-seed confirmation (1000q each, 2000 questions total):** the trade-off is stable, not
a fluke — both seeds land at ~−4.5 pt accuracy and ~+33% throughput:

| | baseline acc | SBP acc | Δacc | SBP tok/s | Δtput |
|---|---:|---:|---:|---:|---:|
| seed 0 | 66.3% | 61.9% | −4.4 pt | 176.8 | +32% |
| seed 1 | 66.6% | 62.1% | −4.5 pt | 177.8 | +35% |
| **mean** | **66.45%** | **62.0%** | **−4.45 pt** | **177.3** | **+33%** |

**Accuracy-recovery experiments (seed 0):**

| run | acc | tok/s | log |
|---|---:|---:|---|
| knob sweep 500q: base thr0.5 / thr0.2 | 67.8% / 64.2% | 133.6 / 131.5 | `trick_base_thr*.log` |
| knob sweep 500q: SBP thr0.5 / thr0.2 / anchor0.5 | 62.4 / 63.8 / 61.8% | ~176 | `trick_sbp_thr*.log` |
| **N=24 @1000q: baseline** | 71.3% | 106.7 | `N24_base_1000.log` |
| **N=24 @1000q: SBP** | **66.9%** | **146.1** | `N24_sbp_1000.log` |

**KV-cache verification:** code audit (clean) + runtime invariant asserts + forced-95%-
resample @ `CUDA_LAUNCH_BLOCKING=1` (60q, 0 IMA / 0 invariant-violations / 0 leaks):
`sbp_kvstress.log`. Confirms the gap is algorithmic, not memory corruption.

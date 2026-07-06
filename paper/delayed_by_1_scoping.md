# Delayed-by-1 SMC-SD — Implementation Scoping

Target: the L4 rung of the paper's asynchrony ladder (see `outline.md` §5).
Grounded against the code as of `main` (2026-07-06).

## 1. What has to change, conceptually

The current cycle has two verify→next-draft dependencies:

1. **Token dependency**: the next cycle's seed token (`verified_ids`) is the
   *target-sampled bonus* (`_resample` extracts it from
   `result.next_draft_input.verified_id`, `smcsd/core/scheduler.py:855-864`;
   written into slots by `write_back_gpu` step (b), `smcsd/core/req_state.py:832`).
2. **Weight dependency**: resampling at the end of cycle *t* reads
   `interval_weights` that already include cycle *t*'s increment
   (`write_back_gpu` step (d) at `req_state.py:931-933` runs before
   `coordinator.collect_resample_jobs_batch` at `scheduler.py:881`).

Delayed-by-1 removes both:

1. **Self-continuation drafting**: the draft samples its own (γ+1)-th token as
   next cycle's seed; the target bonus is dropped. All γ+1 accepted tokens are
   draft samples, so `logprob_diff` gains a column: shape `(bs, γ+1)` instead
   of `(bs, γ)`.
2. **Pending-weight buffer**: verify(t)'s summed increment `d_t` lands in a
   slot-indexed `pending_diff` buffer, *not* in `log/interval_weights`. It is
   folded into the weights one cycle later, gathered through the resample
   plan's ancestor map. Resampling at cycle t+1 therefore sees weights through
   t−1.

### Steady-state schedule (two streams, one or two GPUs)

```
stream D (draft+control):  [resample(k) | draft AR(k) | write_back_tokens(k) | wait ev_V(k−1) | fold pending(k−1)] → cycle k+1
stream V (verify):                                     [verify(k) → diff → pending_diff write → record ev_V(k)]
```

- `resample(k)` uses weights folded through k−2 at that point in the stream;
  by the time cycle k+1's resample runs, fold(k−1) has landed → the resample
  at k+1 uses weights through k−1. That is the delayed-by-1 semantics.
- Effective delay depth is ⌈verify latency / draft-cycle latency⌉. One
  TARGET_VERIFY forward vs γ sequential draft forwards ⇒ depth 1 in our
  regime; assert/measure, don't assume.

### Lineage correction (the one subtle invariant)

`pending_diff` is written by stream V at cycle-k slot identities. Exactly one
resample (cycle k+1's) can dispatch between that write being *scheduled* and
the fold consuming it — and the dispatch may run **before** verify(k) has
written (race). So the resample kernel must NOT copy `pending_diff` in its
phase-3 lineage list. Instead:

- `collect` already emits the plan as flat `(dst, src)` pairs
  (`fused_collect.py:230`, `BatchedResampleResult`). Within one plan, dsts
  (count 0) and srcs (count ≥ 1) are disjoint — no chains — so a single-level
  ancestor map suffices.
- Keep a device tensor `ancestor: (max_slots,) int32`, reset to identity each
  cycle, scattered by the plan (`ancestor[dst] = src`; counter-gated
  worst-case grid, same pattern as `batched_resample_kv`).
- Fold: `w = pending_diff[ancestor]; log_weights += w; interval_weights += w;
  pending_diff.zero_()` — pure enqueues, after `wait ev_V(k−1)`.

This is correct because a dst slot's history after resample(k+1) includes its
src's cycle-k tokens, so it must inherit src's cycle-k increment.

### EOS / weight-cutoff split

Finish detection needs only tokens (all draft-side now) — stays in the
token write-back. The weight cutoff (`req_state.py:902-919`: mask post-EOS
columns) is computed at token-write-back time and stored per-slot
(`pending_cutoff`), read by stream V when it reduces `logprob_diff` → `d_t`.
Stream V also needs cycle k's dense→slot mapping: snapshot `active_slots`
per in-flight cycle (it only changes at materialize/finalize, but snapshot
anyway — cheap).

### log Ẑ bookkeeping

`resample_logZ_increment` (`req_state.py:965`) reads `interval_weights`
before collect zeroes them — unchanged mechanically; the interval now spans
the increments folded since the last resample (two-cycle intervals in steady
state). The unbiased-product structure is preserved because the pending
increment always joins exactly one interval (the next one), never dropped or
double-counted. Correctness check: log Ẑ distribution agreement with the
sync engine over many seeds (statistical, not bitwise — RNG consumption
differs).

## 2. Why this codebase is ready for it

- Resampling is already a self-contained, device-resident, sync-free stage
  (`_resample`, `scheduler.py:829-887`): counter-gated plans, no `.item()`,
  freed pages deferred to postprocessing. Reordering it within the cycle is a
  scheduling change, not a rewrite.
- Weights are flat slot-indexed tensors with one vectorized accumulation
  site (`req_state.py:931-933` / `fused_write_back`) — adding `pending_diff`
  is one more tensor of the same shape.
- Draft and target share one block table (`worker.py:373`:
  `draft_pool.req_to_token = target_pool.req_to_token`), so the existing
  single `batched_resample_kv` launch already re-links *both* models' KV.
  (In disagg/M3 this aliasing breaks — each device replays the same plan on
  its own block table.)
- The overlap loop (`_event_loop_overlap`, `scheduler.py:414`) already
  established the one-step-late discipline (pinned snapshots, headroom
  check, drain-one-late semantics) that delayed weights extend.

## 3. Milestones

### M0 — Self-continuation drafting, synchronous (`SMC_SELF_BONUS=1`)
Drop the target bonus; draft supplies the seed; weight over γ+1 columns.
No pipeline/stream changes. **This is a paper result on its own (E1):** the
accuracy cost of the pure-q proposal, and the gate for everything after.

- `smcsd/core/worker.py`: in `_forward_decode` / the verify+bonus path
  (~L687-745), skip bonus sampling; `next_draft_input.verified_id` becomes
  the draft's own (γ+1)-th sample; `logprob_diff` gets its (γ+1)-th column
  (the verify forward already scores all γ+1 positions — the logits are
  there, we currently just don't diff the last one).
- `smcsd/core/req_state.py` + `kernels/fused_write_back.py`: cutoff logic
  parameterized on `n_weight_cols = γ+1`.
- Interaction: `SMC_DEFER_BONUS`'s 2-token head (`worker.py:577+`) exists
  *because* the seed came from the target; under self-bonus the draft's KV
  for the seed token is already written when it was sampled, so the head
  trick disappears (γ single-token forwards, no head). Simplification, not
  new complexity — but it forks the cycle-graph capture path, which is most
  of this milestone's work.
- Estimate: ~300 LOC touched, mostly `worker.py`. **~2-4 days incl. GSM8K runs.**

### M1 — Delayed weights on the synchronous engine (statistical dry-run)
Same stream, same wall-clock; only the *semantics* change: `pending_diff`,
`ancestor`, fold-one-late, resample on stale weights. This isolates the
statistical effect of delay for E2 with zero systems confound — and it is
the entire algorithmic risk of the paper, retired early.

- `req_state.py`: add `pending_diff (max_slots,) f64`, `pending_cutoff`,
  `ancestor (max_slots,) i32`; split `write_back_gpu` into
  `write_back_tokens_gpu` (steps a-c) and `stage_pending_weights_gpu`
  (step d → pending) + `fold_pending_weights_gpu` (gather-via-ancestor,
  accumulate, zero).
- `scheduler.py::_resample`: reorder to fold(k−1) → collect/dispatch →
  (tokens were already written) ; keep `resample_logZ_increment` reading
  pre-collect.
- New tiny kernel (or extend `fused_collect` to also emit it): identity-reset
  + plan-scatter of `ancestor`. ~40 LOC Triton, same counter-gating idiom as
  `batched_resample_kv`.
- Estimate: ~400 LOC. **~1 week incl. E2 sweeps.**

### M2 — Two-stream overlap, split cycle graphs (the throughput rung)
- Split `SMC_CYCLE_GRAPH`'s single capture (draft AR + verify + diff +
  bonus, `worker.py:771+`) into a **draft-cycle graph** (stream D: resample
  + γ draft forwards + token write-back) and a **verify graph** (stream V:
  TARGET_VERIFY + diff + pending write). Cross-stream ordering via events
  recorded *between* graph replays (no in-graph events needed).
- Scheduler event loop: stream V handle, `ev_V[phase]` ping-pong (mirror the
  existing 2-phase pinned-snapshot pattern), fold gated on `wait ev_V`.
- Hazards to design for: (i) verify(k) reads KV/block-table state that
  dispatch(k+1) mutates — verify must snapshot its `out_cache_loc`/block
  rows or be ordered before dispatch via event (cheap: dispatch(k+1) waits
  ev_V(k); measure whether that wait ever stalls); (ii) allocator headroom
  logic (`scheduler.py:449-455`) now has two in-flight cycles of unfreed
  pages — widen the check.
- Estimate: **~2 weeks.** Largest single chunk; only worth starting after
  E1/E2 gates pass.

### M3 — Disaggregation prototype (2 GPUs, same node)
- Target worker on `cuda:1`; per-cycle p2p copies: token block
  `(bs·(γ+1)) i32` + positions forward, `logprob_diff (bs·(γ+1)) f32` back.
- Block-table aliasing (`worker.py:373`) breaks: keep two block tables, replay
  the same device-resident resample plan on both devices (plan tensors are
  tiny; copy or mirror the collect launch).
- Target-side prefill also moves to `cuda:1` (parent prefill already goes
  through the score model — route it).
- Estimate: **~1-2 weeks** for the prototype that produces E4's numbers;
  productionizing (NCCL, multi-node) is future work and framed as such.

## 4. Decision gates & risk register

**G1 verdict (2026-07-06, 400q GSM8K, γ=8, T=0.7, thr=0.5):** self-continuation
costs a consistent −5.2 to −6.0pp (N=4: 66.2→60.2, N=8: 74.5→68.8, N=12:
76.2→71.0). +N recovers only partially (self N=12 ≈ target N≈5-6). The
target bonus (1 exact p-sample per γ+1 tokens) does real statistical work.
Paper implication: present the async spectrum honestly — L4a (delay weights,
keep bonus) vs L4b (full self-continuation) — with this table as the cost of
full token-level asynchrony. Mitigations to explore: smaller γ (E2 probes
γ=4), thr sweep, larger N.

**Delay-depth analysis (2026-07-06, corrected):** the fully-overlapped M2
schedule is **delay-1** whenever verify latency < draft-cycle latency
(here ~5ms vs ~11ms): verify(k) launches at the end of draft(k), completes
during draft(k+1), so resample(k+1) sees weights through k−1. Delay-2 is
the disagg/slow-verify regime (verify > one draft cycle). M1 implements
SMC_DELAY_RESAMPLE=D (ring of D staged increments, lineage-transported
through each resample via `transport_pending`; finalize adds outstanding
ring rows), and E2 measured D∈{0,1,2}. Invariant test passed: thr=0
(never resample) gives d0=51%, d1=52% at 100q — plumbing exact.

**G2 verdict (2026-07-06):** naive delayed resampling fails the ≤1.5pp
gate decisively — see `results_m0_m1.md` for the full price table
(delay-1: −8 to −11pp; delay-2: −12 to −14pp; threshold recalibration
does not help; N=16 recovers only +3-5pp; γ=4 shrinks the window and the
penalty). Recommended: pursue the auxiliary-predictor mitigation (draft
q-logprobs of the in-flight window are known at decision time) or
reposition the paper on the measured sync↔async price curve.

| Gate | Question | Kill/pivot criterion |
|------|----------|---------------------|
| G1 (after M0/E1) | Does self-continuation hold GSM8K accuracy? | **RESULT: −5-6pp, partial recovery via N — pivot to presenting L4a/L4b spectrum; mitigation probes in E2** |
| G2 (after M1/E2) | Does 1-cycle-stale ESS hold accuracy/resample-freq? | Same criterion; fallback: adaptive delay (resample eagerly when ESS crashes, delayed otherwise) — actually a nice contribution if needed |
| G3 (after M2/E3) | Is the single-GPU overlap win material (>15%)? | If co-running draft+verify contend too much on one GPU, lead with disagg (M3) as the headline systems result and report M2 honestly |

Known non-issues (already handled by existing machinery): RNG under graphs
(Gumbel-max + Philox `step_counter`), freed-page timing (2-phase capture
buffers), hybrid models (already gated out of overlap, keep gated).

## 5. Suggested order of work

1. M0 (gates the algorithm) → E1 on GSM8K.
2. M1 → E2. At this point the paper's §5 and §7 are fully backed.
3. Instrument baselines for E0 (independent of M-track; can parallelize).
4. M2 → E3, then M3 → E4/E5.

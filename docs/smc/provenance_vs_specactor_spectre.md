# Decoupled SMC — component provenance vs SpecActor (#22520) and SPECTRE (#22272)

What our decoupled/async SMC build (branch `smc-async-draft`) took from each upstream
SGLang decoupled-spec design, what it deliberately did not, and what is novel to the
SMC port. All three columns are source-grounded (SpecActor from the pinned prototype
under `.scratch_decoupled/proto/`; SPECTRE from PR #22272 head `1a8520879c…` via `gh`;
ours from the branch source). Companion to `async_smc_design.md`.

## The one-line summary

We took **SpecActor's *shape*** (dedicated drafter engine, named-dataclass wire
protocol over ZMQ PUSH/PULL, sync/commit/close lifecycle) and **SPECTRE's *control
flow*** (target-initiated, one-window-then-stop), and threw away the heavy parts of
both (SpecActor's `DraftTailBuffer`/run-ahead/rollback; SPECTRE's C++ transport,
snapshot-diff, circuit breakers) because **SMC's reweighting verify + our lockstep
barrier make a reconciliation layer unnecessary**. The genuinely new machinery
(no-bonus anchor, particle materialization, frontier-clone resample, barrier-delayed
resampling, per-position draft logprobs) is SMC-specific and has no analog in either.

## Control model

| | SpecActor #22520 | SPECTRE #22272 | **Ours (SMC)** |
|---|---|---|---|
| Who drives | drafter **free-runs ahead**; verifier consumes a token stream | **target push-initiates** each round; drafter does one window then **pauses** | **verifier drives**; drafter is a **pure reactor** (one authorized window per `DraftStepReq`) |
| Verify | EAGLE rejection + bonus | EAGLE rejection (reused) | **SMC importance reweighting** — no rejection, `logprob_diff` only |
| Drafter tenancy | dedicated engine, M:N mesh | **shared** multi-tenant server (runs own traffic) | dedicated `SMCDraftServer`, 1:1 |
| Overlap | run-ahead window (≤ 2× draft tokens) + sleep/wake | one-round pipeline draft(t+1)∥verify(t), 200ms bounded wait | **cohort pipeline** (cross-group) + **prefetch** (within-group); lockstep-bounded |
| Async safety | `DraftTailBuffer` reconciliation + KV rollback | snapshot fork-point diff + local rollback / re-prefill | **none needed** — "mirror + assert" + frontier-clone at a drained barrier |

## Component-by-component mapping

Verdict legend: **◀SA** taken/adapted from SpecActor · **◀SP** taken/adapted from
SPECTRE · **✦SMC** novel to this port · **✗** deliberately not taken.

| Axis | SpecActor | SPECTRE | Ours | Verdict |
|---|---|---|---|---|
| **Dedicated drafter engine** | dedicated engine + `DECOUPLED_DRAFT` role/plugin registry | shared server | `SMCDraftServer` separate process, spawned by the engine; **no** plugin/role registry (monkey-patched scheduler instead) | **◀SA** (the shape, not the registration) |
| **Wire protocol style** | stateful deltas: `DraftSync`/`VerifyCommit`/`DraftClose` + `DraftTailStreamOutput`, `DraftReqKey`, control batches | snapshots: full `output_ids` per `DRAFT`, `(rid, spec_cnt)` round counter | stateful deltas: `DraftPrefillReq/Resp`, `DraftMaterializeGroup`, `DraftStepReq/Resp`, `DraftCommitResample`, `DraftCloseGroup` | **◀SA** style (named dataclasses, sync/commit/close lifecycle); **✗** SPECTRE's snapshot model |
| **Transport** | Python ZMQ PUSH/PULL, pickle, M:N mesh | compiled C++ ZMQ (`cpp_zmq/`), msgpack, ROUTER/DEALER, 5 IO threads, heartbeats | Python ZMQ PUSH/PULL, `send_pyobj` (pickle), single FIFO channel, 1:1 | **◀SA**; **✗** SPECTRE's C++/msgpack/router stack |
| **Request identity** | `DraftReqKey(src_verifier_rank, rid)` for M:N | `(rid, spec_cnt)` | `group_id` + slot ids; `tag` (monotonic) for FIFO matching | **✦SMC** (our `tag` ≈ SPECTRE's `spec_cnt` in spirit — a round/echo counter, but for FIFO assert, not snapshot dedup) |
| **Verifier reconciliation** | `DraftTailBuffer` (stale-base floor, pending-expected queue, echo-back, snapshot immutability) | fork-point diff (`_find_fork_point`), `SpectreDraftStateManager`, epoch-stale drops | **none** — `seq_lens` assertion + FIFO ordering | **✗** both (lockstep+barrier removes the tail-to-reconcile) |
| **KV rollback / truncation** | `apply_verifier_commit_segment` token-granular truncate + re-install + single-token-mismatch contract | `SpectreKVRollbacker.local_rollback` / re-prefill fallback | **none** — resampling is a forward `dst←src` clone, never a rewind | **✗** both |
| **Run-ahead / sleep-wake** | `_draft_ahead_window`, `sleep_overrun`/`wake_draft_sleeping` | "one window then PAUSE" + bounded recv | **none** — drafter blocks on `recv`; never runs past authorized state | **✗** both (SMC forbids run-ahead: anchor is target-derived + resampling reshuffles) |
| **Echo-back** | drafter re-streams committed tokens to drain `pending_expected` | n/a | **none** | **✗** (no pending queue exists) |
| **Verify compute** | `VerifyWorker`, linear-topk1 chain, EOS pad, accept+bonus | `SpectreWorker`, reused EAGLE, dynamic ntpb | `DecoupledSMCWorker.finish_decode`: target forward → `logprob_diff` (no accept/reject) | **✦SMC** |
| **Anchor for next round** | target bonus / corrected token (rejection) | target accepted/corrected token | **no-bonus**: the draft's own (γ+1)-th token, reweighted (`SMCSD_DROP_BONUS`) + `SMCSD_ANCHOR_TEMP` | **✦SMC** (uniquely possible because SMC reweights instead of rejecting) |
| **Particle materialization** | n/a (1 chain/req) | n/a | `DraftMaterializeGroup` + `copy_block_table` N-way refcounted KV fanout | **✦SMC** (N-particle population) |
| **Resampling** | n/a | n/a | `DraftCommitResample` → `batched_resample_kv` **frontier-clone**; **barrier-delayed** (`SMCSD_RESAMPLE_INTERVAL`); row-masked collect for cohorts | **✦SMC** |
| **Draft logprobs on the wire** | n/a (rejection re-verifies) | carries `draft_logprobs` (telemetry) | per-position draft logprobs in `DraftStepResp` — **load-bearing** for SMC weights | **✦SMC** |
| **Cross-group overlap** | implicit (two free-running schedulers) | one-round pipeline within a req | `PipelinedDecoupledSMCScheduler` cohorts (explicit) | **◀SP** in spirit (pipeline), **✦SMC** in form (cohorts, row-masked resample) |
| **Within-group overlap** | run-ahead | draft(t+1)∥verify(t) | **prefetch** (`send_step_req` split: fire next StepReq before local verify) | **◀SP** in spirit; **✦SMC** mechanism (verifier-side prefetch, drafter unchanged) |
| **Modes (lockstep→parallel)** | `allow_partial` env | n/a | `smc_decoupled` / `smc_pipelined` / `smc_async`; barrier interval K | **◀SA** (lockstep-first bring-up); our "parallel" = prefetch+barrier, not free-running |
| **Failure handling** | fail-fast `RuntimeError`s | heartbeats, `DraftCircuitBreaker`, fallback-to-plain-decode, GC | fail-fast (`seq_lens` divergence, tag-mismatch, prefix assert) | **◀SA**; **✗** SPECTRE's resilience (a noted future borrow) |
| **Drafter CUDA graphs** | (verify worker graphs) | two ntpb buckets | ported from the colocated `SMCWorker` (graph replay per AR step + multistep fallback) | **✦SMC** (from our own colocated path, not upstream decoupled) |
| **Constraints** | `page_size=1`, `disable_radix_cache` | `page_size=1` for local rollback | `page_size=1`, `disable_radix_cache`, `disable_piecewise_cuda_graph` (drafter) | **◀SA** (SMC already satisfied these) |

## Taken from SpecActor (the structural skeleton)
- Dedicated drafter-engine topology (separate process, own GPU/pools).
- Named-dataclass wire protocol over ZMQ PUSH/PULL with `send_pyobj`; the
  Sync→Commit→Close request lifecycle (our `DraftPrefillReq`/`DraftCommitResample`/
  `DraftCloseGroup` are the SMC analogs of `DraftSync`/`VerifyCommit`/`DraftClose`).
- Lockstep-first bring-up, then add overlap (lesson 06's progression).
- Fail-fast correctness (assertions over recovery machinery).
- The constraints (page_size=1, disable_radix_cache) — SMC already met them.

## Taken from SPECTRE (the control flow)
- Target/verifier-initiated, **one-window-then-stop** drafter (vs SpecActor's
  free-running drafter). Our drafter is a pure reactor — the strongest form of
  "do one window, wait."
- The **one-round pipeline** idea (overlap draft of the next unit with verify of the
  current) — realized as cohort pipelining (across groups) and prefetch (within a
  group), not as a within-request continuation.

## Deliberately NOT taken (and why)
- **SpecActor's `DraftTailBuffer` + run-ahead + KV rollback + echo-back, and SPECTRE's
  snapshot fork-point diff + rollbacker.** Both exist to reconcile a drafter that has
  speculated *past* the verifier. Our drafter never does: the no-bonus anchor makes
  the next token drafter-known (no guessing), and the K-window **barrier drains the
  pipeline before any frontier disagreement can exist**, so resampling is a clean
  `dst←src` clone with nothing to truncate or reconcile. No tail ⇒ no buffer.
- **Free-running drafter / sleep-wake.** SMC *forbids* run-ahead: round t+1's anchor
  is derived from round t's verify, and resampling reshuffles particle KV at the
  boundary — a free-running drafter would speculate on both. (See `async_smc_design.md`
  §"Why lockstep".)
- **SPECTRE's C++/msgpack transport and resilience stack** (circuit breaker,
  heartbeats, fallback-to-plain-decode). Over-engineering for a single-tenant
  research prototype; Python PUSH/PULL + fail-fast suffices. The circuit-breaker /
  fallback ideas are flagged in lesson 07 as worth borrowing later.
- **The plugin role registry** (`DECOUPLED_VERIFY`/`DECOUPLED_DRAFT`,
  `validate_server_args`). We monkey-patch a scheduler/engine instead, per the repo's
  standalone-overlay style.

## Novel to the SMC port (neither upstream has an analog)
- **No-bonus / drop-anchor + anchor temperature** — commit the draft's own token and
  reweight. EAGLE-based designs *cannot* do this (rejection needs the target token);
  SMC's importance weighting *can*. This is the keystone that makes async feasible.
- **Particle materialization** — N-way refcounted KV fanout of the prompt
  (`copy_block_table`).
- **Frontier-clone resampling** + **barrier-delayed resampling** + **row-masked
  resample collect** — SMC's resampling step, mirrored to the drafter as a forward
  clone with no rollback.
- **Per-position draft logprobs on the wire** — load-bearing for the SMC weight, not
  telemetry.
- **Ride-along finishes + finished-before weight mask** — keeping finished particles
  in a fixed train batch with masked weights so prefetch stays consistent.
- **Cohort pipelining + verifier-side prefetch** — overlap forms specific to the
  reactor drafter (drafter needs zero changes).
- **Drafter CUDA graphs** — ported from our own colocated `SMCWorker`, the headline
  perf win (+37%).

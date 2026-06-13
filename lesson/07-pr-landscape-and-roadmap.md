# 07 — PR Landscape & Roadmap (read this to avoid the trap)

The **single biggest source of confusion** in this feature: there are **two
unrelated, competing PRs** plus a **slice stack** that re-lands the design
cleanly. Get this wrong and you read the wrong code.

> **Correction note (verified via `gh`):** an earlier draft of this lesson claimed
> #22520's diff was "only 7 example files." That was wrong — it came from an
> unreliable web summary. The authoritative GitHub files API shows **#22520 =
> 59 files, +11,182**, including the full engine. Always verify file inventories
> with `gh api .../pulls/<n>/files`, never a prose summary.

## The map

| PR / Issue | Name | Author | What it is | Has the engine? |
|---|---|---|---|---|
| **#22520** | **SpecActor** | sisyphus111 / ByteDance | **Our architecture's prototype — and it is complete.** 59 files: the IPC protocol, `DraftTailBuffer`, `VerifyWorker`, the 2189-line scheduler mixin, both ZMQ transport threads, Mamba/GDN hybrid-attention rollback support, and Ray launch examples. **Port from this.** | ✅ full |
| **#22272** | **SPECTRE** | xq25478 | **Different competing design.** 29 files, **+5,896** (≈3.1k Python + **2.1k C++ that must compile** + 0.7k glue; no examples/tracing). A compiled `cpp_zmq/` async ZMQ extension (`setup.py` + `build_cpp_zmq.sh`, msgpack), separate drafter/verifier scheduler mixins, own `spectre_kv_rollbacker.py` + `spectre_state_manager.py`. Reuses EAGLE verify. **No** DraftTailBuffer/VerifyWorker. | ❌ (other design) |
| **#27634** | `[Spec][1/N]` | zhendonghua | Clean re-land slice 1: `decoupled_spec_io.py` + server flags. **Byte-for-byte logic copy** of #22520's IPC layer, plus explanatory docstrings (e.g. a threading-hazard note on `VerifierCommitSegment.append_message`). | ✅ (IPC only) |
| **#27982** | `[Spec][2/N]` | zhendonghua | Clean re-land slice 2: `draft_tail_buffer.py` + **27 CPU tests**. Logic-identical to #22520's buffer; only adds a lazy-diagnostic O(N²)→O(N) fix + docstrings. | ✅ (buffer only) |
| **#27462** | roadmap (epic) | — | Tracks the slice plan; lists #22520 and #22272 as the two "initial PRs." | n/a |

### The crucial correction
The task brief names **#22272**, but the named symbols
(`DraftSync`/`VerifyCommit`/`DraftClose`/`DraftTailStreamOutput`/`DraftReqKey`/
`DraftTailBuffer`/`VerifyWorker`) are **SpecActor (#22520)**, not SPECTRE
(#22272). SPECTRE is a different team's C++-ZMQ proposal for the same goal. **For
our port: read #22520's actual source (now downloaded), and treat the #27634/#27982
slices as the same code with nicer docstrings + tests.**

## Where each symbol actually lives (all in #22520, file paths under `python/sglang/srt/`)

| Symbol | File | Also re-landed in |
|---|---|---|
| `DraftReqKey`, `DraftSync`, `VerifyCommit`, `DraftClose`, `DraftTailStreamOutput`, `DraftControlBatch`, `DraftMeshMessage`, `VerifierCommitSegment`, `DraftControlInbox`, `ReadyDraftControls`, `DecoupledSpecIpcConfig` | `speculative/decoupled_spec_io.py` (+376) | #27634 (+384, identical logic) |
| `DraftTailBuffer`, `RequestDraftTailState`, `DraftTailSnapshot` | `speculative/draft_tail_buffer.py` (+666) | #27982 (+695, identical logic) |
| `VerifyWorker` | `speculative/decoupled_verify_worker.py` (+451) | not yet sliced |
| `SchedulerDecoupledSpecMixin`, `DraftReqState`, `DraftKVTruncation`, `DraftBatchMetadataUpdate` (run-ahead, sleep/wake, commit, KV rollback, Mamba ckpt) | `managers/scheduler_decoupled_spec_mixin.py` (+2189) | not yet sliced |
| `DraftProxyThread` | `speculative/draft_proxy.py` (+171) | not yet sliced |
| `TokenSyncThread` | `speculative/token_sync_thread.py` (+259) | not yet sliced |
| `DecoupledVerifySpecAlgo`, `DecoupledDraftSpecAlgo`, `DECOUPLED_VERIFY`/`DECOUPLED_DRAFT` registration, `validate_server_args` | `speculative/spec_info.py` (+202), `spec_registry.py` | not yet sliced |
| consistency design doc | `speculative/decoupled_speculation_consistency.md` (+227) | — |

So: the IPC + buffer slices are merged-track and stable; the **verify worker,
scheduler mixin, transport, and registration are only in #22520's branch** so far
(the later roadmap slices haven't landed). The prototype is the source of truth for
those.

## Architecture comparison: SpecActor vs SPECTRE (source-verified, both diffs read)

Same goal, two different distributed-system designs:

| Axis | SpecActor #22520 | SPECTRE #22272 |
|---|---|---|
| Control model | drafter **free-runs ahead**; verifier consumes a stream | **target push-initiates each round**; drafter does one window then **pauses** |
| Overlap | run-ahead ≤ 2× window + sleep/wake | one-round pipeline: draft(t+1) overlaps verify(t); 200 ms bounded wait (`SPECTRE_RECV_TIMEOUT_MS`) |
| Wire semantics | **stateful deltas** (token stream + commit segments) | **snapshots**: every `DRAFT` carries full `output_ids`; correlated by `(request_id, spec_cnt)` round counter |
| Reconciliation | `DraftTailBuffer` state machine (stale-base floor, pending queue, echo-back) | fork-point **diff of snapshots** + epoch-stale drops; optional sync retry |
| Rollback signal | explicit `VerifyCommit` → surgical token-granular KV truncation | **no commit/rollback message** — drafter infers divergence by diffing; local rollback or **full re-prefill** fallback |
| Drafter tenancy | dedicated engine; M:N mesh + least-loaded assignment | **shared** sglang server also serving own traffic; `REJECT` under load; Dealer registry but effectively 1:1 today (`_zmq_send` → `all_drafts_identity[0]`) |
| Failure handling | fail-fast `RuntimeError`s (degradation = roadmap 5c) | heartbeats, `DraftCircuitBreaker` (CLOSED/OPEN/HALF_OPEN), per-batch fallback to plain decode, stale-state GC |
| Verify compute | EAGLE, fixed-width topk=1 chain, EOS-pad+cut | EAGLE, **dynamic `num_draft_tokens`**, two CUDA-graph ntpb buckets (1 and full) |
| Transport | Python PUSH/PULL full mesh, pickle | C++ Router(target-bind)/Dealer(drafter-dial), msgpack, 3 channels/pair, 5 IO threads |

Takeaways: SpecActor = replicated-state-machine optimized for deep asynchrony;
SPECTRE = round-based snapshot pipeline optimized for resilience/degradation.
The `smcsd/decoupled/` port is a deliberate hybrid: SpecActor's *shape*
(roles, dataclass wire protocol, lifecycle) with SPECTRE-like *control flow*
(per-round target-initiated, window-then-wait), since SMC forbids run-ahead.
SPECTRE ideas worth borrowing later: circuit-breaker + fallback-to-plain-decode;
dynamic-ntpb graph buckets if gamma varies.

## Roadmap slice plan (issue #27462)

Dependency-ordered breakdown (the prototype is being landed piece by piece):

1. **[1/N] IPC protocol & transport foundation** — message dataclasses + server
   flags (#27634); `DraftTailBuffer` + tests (#27982); ZMQ + a "fake" transport
   behind one interface, health/abort hooks `[1c]`.
2. **[2/N] Algorithm & verify compute** — register `DECOUPLED_VERIFY` /
   `DECOUPLED_DRAFT`; `BaseVerifyWorker` + eager `VerifyWorker`; verify CUDA-graph
   capture.
3. **[3/N] Scheduler split & verifier integration** — extract a shared base mixin;
   add the verify-half mixin + metrics; verifier fault tolerance.
4. **[4/N] Drafter integration** — drafter-half mixin; lockstep mode; parallel/
   draft-ahead mode.
5. **[5/N] Hybrid drafters** — linear-attention state checkpoint/rollback (already
   present in the prototype via the Mamba ring; see [05](05-drafter-runahead-and-rollback.md)).

Note: the roadmap also mentions an `ignore_decode_budget` admission switch as an
independent item — **it is NOT in the #22520 prototype**; there, run-ahead memory is
bounded purely by the sleep/wake window. Don't assume it exists yet.

## Status snapshot (as of 2026-06-12)

- #22520, #22272, #27634, #27982 are **all OPEN / unmerged**.
- #27462 is open, assigned to Qiaolin-Yu and zhendonghua.
- Practical takeaway: there is **no merged upstream reference** yet, but the full
  prototype is readable and self-consistent. We port from the prototype SHA
  (pinned in the README) + the two landed-track slices. Pin to that SHA when we
  implement, since the slice stack is a moving target.

---
*Confidence:* High — file inventories and per-file logic verified directly from the
GitHub contents/files API at the pinned SHAs.

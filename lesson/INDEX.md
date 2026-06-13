# INDEX — Decoupled Speculative Decoding Knowledge Base

Retrieval index for implementation agents. **Start here**, map your task to the
right lesson(s) + the real source file, then read. For *how* to use this KB
(source-of-truth order, fetch protocol, warnings) see [CLAUDE.md](CLAUDE.md).

```
lesson/
├── CLAUDE.md   ← retrieval contract: HOW to use this KB (read first)
├── INDEX.md    ← this file: WHERE to look (routing + reverse maps)
├── README.md   ← narrative overview + glossary + reading order
├── 01 architecture-and-roles      ── engines, transport threads, mesh, flags
├── 02 ipc-protocol                ── ZMQ message dataclasses
├── 03 draft-tail-buffer           ── verifier-side reconciliation (correctness core)
├── 04 verify-worker               ── verify compute (linear chain, accept)
├── 05 drafter-runahead-and-rollback ── run-ahead, sleep, commit, KV rollback
├── 06 modes-and-constraints       ── lockstep vs parallel, validate_server_args
├── 07 pr-landscape-and-roadmap    ── which PR is which (anti-confusion) + architecture comparison
├── 08 smcsd-implications          ── mapping onto SMCSD (Q1–Q5 RESOLVED; port: smcsd/decoupled/)
└── 09 workflows                   ── flow charts: SpecActor vs SPECTRE vs SMCSD-port request flows
```

## 1. Topic hierarchy (concept → lesson → real source)

Source paths are relative to `python/sglang/srt/` in PR #22520; the downloaded
copies are under `.scratch_decoupled/proto/` (see [CLAUDE.md](CLAUDE.md) for the SHA).

```
Decoupled Speculative Decoding
│
├─ A. System shape ........................... [01] [README]
│   ├─ Two engines: VERIFIER / DRAFTER ....... spec_info.py, spec_registry.py
│   ├─ Transport threads (ZMQ PUSH/PULL) ..... draft_proxy.py, token_sync_thread.py
│   ├─ M:N full mesh + entry-rank gating ..... scheduler_decoupled_spec_mixin.py
│   └─ Server flags / PortArgs / IPC config .. server_args.py, decoupled_spec_io.py
│
├─ B. Wire protocol .......................... [02]
│   ├─ DraftReqKey + scheduler-rid ........... decoupled_spec_io.py
│   ├─ Control (Sync/Commit/Close) ........... decoupled_spec_io.py
│   ├─ Data (TailStreamOutput) ............... decoupled_spec_io.py
│   └─ Staging (CommitSegment/Inbox) ......... decoupled_spec_io.py
│
├─ C. Verifier side .......................... [03] [04]
│   ├─ DraftTailBuffer reconciliation ........ draft_tail_buffer.py (+ tests)
│   ├─ VerifyWorker compute (chain+accept) ... decoupled_verify_worker.py
│   └─ Snapshot/sync/submit + metrics ........ scheduler_decoupled_spec_mixin.py
│
├─ D. Drafter side ........................... [05]
│   ├─ Run-ahead window + sleep/wake ......... scheduler_decoupled_spec_mixin.py
│   ├─ Commit apply + KV rollback ............ scheduler_decoupled_spec_mixin.py
│   ├─ Echo-back (drain pending) ............. scheduler_decoupled_spec_mixin.py
│   └─ Mamba/hybrid state-ckpt ring .......... scheduler_decoupled_spec_mixin.py
│
├─ E. Modes & constraints .................... [06]
│   ├─ lockstep vs parallel (allow_partial) .. draft_tail_buffer.py, scheduler_…mixin.py
│   └─ validate_server_args (hard rules) ..... spec_info.py
│
└─ F. Meta .................................... [07] [08]
    ├─ PR landscape (SpecActor vs SPECTRE) ... (see 07)
    └─ SMCSD port mapping + open questions ... (see 08)
```

## 2. Task → lessons routing

Pick the row matching what you're about to implement.

| If you're implementing… | Read lessons | Then open source |
|---|---|---|
| Engine roles / algorithm registration / role flags | 01, 06 | `spec_info.py`, `spec_registry.py` |
| Server CLI flags + `DecoupledSpecIpcConfig` + PortArgs | 01, 06 | `server_args.py`, `decoupled_spec_io.py` |
| ZMQ message dataclasses (the wire schema) | 02 | `decoupled_spec_io.py` (= #27634, stable) |
| ZMQ transport threads (proxy / token-sync) | 01 | `draft_proxy.py`, `token_sync_thread.py` |
| Verifier reconciliation state machine | 03 | `draft_tail_buffer.py` (+ `test_draft_tail_buffer.py`) |
| Verify compute (linear chain, accept, EOS pad) | 04 | `decoupled_verify_worker.py` |
| Verifier scheduler half (snapshot/sync/submit, metrics) | 03, 04 | `scheduler_decoupled_spec_mixin.py` (verifier half) |
| Drafter run-ahead + sleep/wake backpressure | 05 | `scheduler_decoupled_spec_mixin.py` |
| Commit apply + KV truncation/rollback | 05, 03 | `scheduler_decoupled_spec_mixin.py` |
| Echo-back / pending-expected draining | 05, 03 | `scheduler_decoupled_spec_mixin.py`, `draft_tail_buffer.py` |
| Hybrid (Mamba/GDN) state rollback | 05 | `scheduler_decoupled_spec_mixin.py` + hybrid-attn files |
| Config validation / hard constraints | 06 | `spec_info.py` (`validate_server_args`) |
| Lockstep vs parallel mode toggle | 06, 03 | env `SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL`; `draft_tail_buffer.py` |
| Deciding what maps onto SMCSD | 08, 07 | `smcsd/decoupled/` (lockstep port) + `tasks/decoupled_smc_design.md` |
| "Wait, which PR/design is this?" | 07 | — |
| End-to-end request flow / who-talks-to-whom-when | 09, 01 | flow charts; then the mixin / `smcsd/decoupled/` |

## 3. Symbol → lesson → source (reverse index)

| Symbol | Lesson | Source file |
|---|---|---|
| `DraftReqKey`, `build/parse_draft_scheduler_rid` | 02 | `decoupled_spec_io.py` |
| `DraftSync` / `VerifyCommit` / `DraftClose` | 02 | `decoupled_spec_io.py` |
| `DraftTailStreamOutput(Batch)` | 02 | `decoupled_spec_io.py` |
| `DraftControlBatch`, `DraftMeshMessage`, `DraftMeshMessageType` | 02 | `decoupled_spec_io.py` |
| `VerifierCommitSegment`, `DraftControlInbox`, `ReadyDraftControls` | 02 | `decoupled_spec_io.py` |
| `DecoupledSpecIpcConfig` | 01, 02 | `decoupled_spec_io.py` (+ `server_args.py` PortArgs) |
| `DraftTailBuffer`, `RequestDraftTailState`, `DraftTailSnapshot` | 03 | `draft_tail_buffer.py` |
| `committed_len`, `tail_tokens`, `pending_expected_tokens`, `can_accept_prefix_len` | 03 | `draft_tail_buffer.py` |
| outcome strings (`appended`/`stale_base`/`pending_expected_*`/…) | 03 | `draft_tail_buffer.py` |
| `VerifyWorker`, `_build_linear_topk1_tree_metadata`, `_get_pad_token_id` | 04 | `decoupled_verify_worker.py` |
| `_assert_verify_output_within_snapshot_tail` | 04 | `decoupled_verify_worker.py` |
| `SchedulerDecoupledSpecMixin` | 05 | `scheduler_decoupled_spec_mixin.py` |
| `DraftReqState`, `DraftKVTruncation`, `DraftBatchMetadataUpdate` | 05 | `scheduler_decoupled_spec_mixin.py` |
| `verifier_committed_prefix_len` (invariant anchor) | 05, consistency | `scheduler_decoupled_spec_mixin.py` |
| `_draft_ahead_window`, `sleep_overrun_draft_requests`, `wake_draft_sleeping_requests` | 05 | `scheduler_decoupled_spec_mixin.py` |
| `apply_verifier_commit_segment` (commit + rollback) | 05 | `scheduler_decoupled_spec_mixin.py` |
| `_apply_ready_verifier_commit_segments` (echo-back) | 05, 03 | `scheduler_decoupled_spec_mixin.py` |
| `_draft_mamba_ckpt_slot`, `commit/_prune_draft_mamba_ckpts` | 05 | `scheduler_decoupled_spec_mixin.py` |
| `_snapshot_verify_inputs`, `_sync_verify_requests`, `submit_verify_updates`, `validate_verify_outputs` | 03, 04 | `scheduler_decoupled_spec_mixin.py` |
| `DraftProxyThread` | 01 | `draft_proxy.py` |
| `TokenSyncThread` | 01 | `token_sync_thread.py` |
| `DecoupledVerifySpecAlgo` / `DecoupledDraftSpecAlgo` / `DECOUPLED_VERIFY` / `DECOUPLED_DRAFT` | 01, 06 | `spec_info.py`, `spec_registry.py` |
| `validate_server_args` | 06 | `spec_info.py` |
| `SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL` (env) | 06 | scheduler mixin (`envs.…`) |

## 4. Source-file → lesson (forward index)

All under `python/sglang/srt/` in #22520; local copies in `.scratch_decoupled/proto/`.

| Source file | Lines | Role | Lesson | Re-landed slice |
|---|---:|---|---|---|
| `speculative/decoupled_spec_io.py` | 376 | wire dataclasses + staging + IPC config | 02 | #27634 (identical) |
| `speculative/draft_tail_buffer.py` | 666 | verifier reconciliation state machine | 03 | #27982 (identical) |
| `speculative/decoupled_verify_worker.py` | 451 | `VerifyWorker` verify compute | 04 | not yet |
| `managers/scheduler_decoupled_spec_mixin.py` | 2189 | runtime core: both scheduler halves | 03,04,05 | not yet |
| `speculative/draft_proxy.py` | 171 | verifier-side ZMQ thread | 01 | not yet |
| `speculative/token_sync_thread.py` | 259 | drafter-side ZMQ thread | 01 | not yet |
| `speculative/spec_info.py` | 445 | algo classes + `validate_server_args` | 01,06 | not yet |
| `speculative/spec_registry.py` | 129 | `CustomSpecAlgo` registry | 01 | not yet |
| `speculative/decoupled_speculation_consistency.md` | 227 | authoritative prefix-consistency spec | 03,05 | — |
| `server_args.py` (decoupled parts) | ~80 | CLI flags + PortArgs IPC config + GPU-mem exclusion | 01,06 | #27634 (subset) |
| `speculative/tracer.py`, `decoupled_spec_nvtx.py` | 993/41 | tracing (optional) | — | not yet |
| `examples/runtime/engine/decoupled_speculation/*` | ~3.5k | Ray launch + benchmark harness | 01 | (this is #22520's own example diff) |

## 5. Pinned commits (single source of truth: [README](README.md))

| Ref | SHA | Contents |
|---|---|---|
| Prototype #22520 | `16c8cf4a30e590c766f5b3b979f2446e3c92dbb2` | full engine (59 files) |
| Slice #27634 | `d1b4e1807450c1a87408a5e3392389962b131960` | IPC protocol |
| Slice #27982 | `0f8ad97ae2c025f47da08827e635765fbd21d145` | DraftTailBuffer + tests |

Fetch any file: `gh api repos/sgl-project/sglang/contents/<path>?ref=<SHA> -H "Accept: application/vnd.github.raw"`

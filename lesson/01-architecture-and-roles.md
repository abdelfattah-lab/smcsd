# 01 — Architecture & Roles

How the two engines are wired, who does what, and how requests are addressed
across the mesh.

## Two engine roles (selected by speculative-algorithm)

The same SGLang engine binary runs in one of two roles, chosen by the
spec-algorithm name:

| Role | Algorithm | Worker | Owns |
|---|---|---|---|
| **Verifier** | `DECOUPLED_VERIFY` | `VerifyWorker` | Target model, the **authoritative output**, `DraftTailBuffer`, client streaming, request lifecycle. |
| **Drafter** | `DECOUPLED_DRAFT` | normal `TpModelWorker` (no spec worker) | Draft model, private *mirror* requests keyed by `DraftReqKey`, run-ahead + rollback. |

The drafter deliberately uses the **ordinary** decode worker — it is "just" a
continuous-batching decode engine whose requests happen to be opened/closed by
the verifier instead of by end users. That is what lets it inherit chunked
prefill, radix cache, CUDA graphs, and TP/PP for free. This is enforced two ways:
`DecoupledDraftSpecAlgo.create_worker()` **raises**, and the `DECOUPLED_DRAFT`
worker factory **raises** — both say *"decoupled_draft uses the normal TP worker
instead of create_worker()."*

## Responsibilities

**Verifier (authoritative):**
- Prefill + decode the target model.
- Each step: verify the mirrored draft tail in **one** target forward, accept the
  **longest matching draft prefix + 1 bonus token**, commit.
- Drive the drafter with control messages: `DraftSync` (open), `VerifyCommit`
  (advance committed prefix), `DraftClose` (finish/abort).
- Maintain a `DraftTailBuffer` per request to reconcile the asynchronous draft
  stream against its own commits.
- The **only** side that streams tokens to the client.

**Drafter (speculative, never autonomous):**
- Plain autoregressive decode of the draft model, ahead of the verifier.
- Stream **every** produced token back as `DraftTailStreamOutput`.
- On a divergent commit: truncate output + roll back KV to the committed point,
  re-install the verifier's chosen token, resume.
- **Sleep** (throttle) a request when it gets too far ahead (memory bound).
- Never decides when a request ends — waits for `DraftClose`.

## Transport threads (the ZMQ glue)

Each engine runs a dedicated thread bridging ZMQ and the scheduler:

- **`DraftProxyThread`** (verifier side, `draft_proxy.py`):
  PUSH control sockets → each drafter rank; PULL result socket ← draft tail
  batches. It applies control batches to the local `DraftTailBuffer` *before*
  forwarding, and appends incoming tail batches to it.
- **`TokenSyncThread`** (drafter side, `token_sync_thread.py`):
  PULL control socket ← verifier; PUSH result sockets → each verifier rank.
  Buffers control into a `DraftControlInbox`; drains outgoing
  `DraftTailStreamOutputBatch` results.

Sockets are PyZMQ **PUSH/PULL**, payloads are pickled dataclasses
(`send_pyobj`). Contrast: SPECTRE (#22272) used C++ Dealer/Router + msgpack — see
[07](07-pr-landscape-and-roadmap.md).

**Exact socket wiring (verified):** each engine **binds exactly one PULL** on its
own `bind_endpoint` (its inbox) and **connects one PUSH per peer** to each entry in
`connect_endpoints` (peer rank = list index). The verifier's inbox receives draft
tails; the drafter's inbox receives control. Both ends validate the `dst_*_rank`
field on every message (the proxy *raises* on a wrong-verifier batch; the sync
thread *silently drops* control not addressed to it). Only the **entry rank** of
each engine (`pp_rank == attn_tp_rank == attn_cp_rank == 0`) owns a transport
thread; peer TP ranks receive everything via `broadcast_pyobj` (snapshots and
`ReadyDraftControls`), so every rank acts on identical data.

Receive loops poll on tiny timeouts (proxy: 1 ms; token-sync: 0.5 ms idle wait)
and break on `zmq.error.ContextTerminated`. `close()` sets a flag, closes the
buffer, joins the thread, and closes every socket with `linger=0`.

## M:N topology & addressing

- Arbitrary **M drafters : N verifiers** (e.g. one TP1 drafter feeding several
  TP8 verifiers). Multi-verifier note: `--batch-size` must be divisible by the
  verifier replica count (each verifier gets one contiguous slice).
- Wiring is a **static full mesh**: *every verifier connects to every drafter
  control endpoint; every drafter connects to every verifier result endpoint.*
- Each engine has one `bind_endpoint` (where it listens) and an ordered list of
  `connect_endpoints` (peers, ordered by peer rank), plus its own `rank`.
- **`DraftReqKey(src_verifier_rank, request_id)`** is the cross-process id. A
  `request_id` is unique only *within* its owning verifier; the verifier rank
  disambiguates when several verifiers share one drafter.

## Entry points / server args (PR #27634)

CLI flags added to `server_args.py`:

- `--decoupled-spec-bind-endpoint` (str) — this engine's listen endpoint.
- `--decoupled-spec-connect-endpoints` (JSON list) — peer endpoints, ordered by rank.
- `--decoupled-spec-rank` (int) — this engine's rank.
- `--spec-trace-dir` (str) — optional tracing output.

These are frozen into `DecoupledSpecIpcConfig(bind_endpoint, connect_endpoints,
rank)` and hung off `PortArgs.decoupled_spec_ipc_config` (built in
`PortArgs.init_new` when the algorithm is decoupled; raises if endpoints are
missing).

**Memory:** `_handle_gpu_memory_settings` **excludes** the decoupled roles from
the extra draft-model memory reservation — the two models live in separate
engines, so the verifier must not reserve draft memory (and vice versa).

## Launch harness (the actual content of PR #22520)

Ray-based examples under `examples/runtime/engine/decoupled_speculation/`:
`DraftActor` wraps a `DECOUPLED_DRAFT` engine, `TargetActor` wraps a
`DECOUPLED_VERIFY` engine; `create_remote_decoupled_spec_topology()`,
`launch_draft_actors()` (node-affinity scheduling), `plan_draft_placement()`,
`reserve_tcp_port()`, `format_tcp_address()` build the mesh. This is the only
code actually shipped in #22520's diff today.

---
*Confidence:* High — role split, transport threads, socket wiring, entry-rank
gating, and server flags read directly from `draft_proxy.py`, `token_sync_thread.py`,
`spec_info.py`, and `server_args.py` at the pinned prototype SHA. The transport
classes aren't in a merged slice yet, so names *could* shift when sliced — but the
already-landed IPC/buffer slices kept prototype names verbatim.

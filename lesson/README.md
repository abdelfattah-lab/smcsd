# Decoupled (Parallel) Speculative Decoding — Lesson Index

> Knowledge base built for **Job Step 1 (Learning)** of porting decoupled
> draft/verify into **SMCSD**. Top-down: start here, then drill into the
> numbered files.
>
> **Verification status:** every claim below is grounded in the **actual source**
> (read via `gh`, not summarized). Pinned commits:
> - Prototype **PR #22520** — `sisyphus111` fork, branch `v0.5.9-decoupled-spec-dev`,
>   SHA **`16c8cf4a30e590c766f5b3b979f2446e3c92dbb2`** (59 files, +11,182).
> - Slice **PR #27634** (IPC protocol) — SHA `d1b4e1807450c1a87408a5e3392389962b131960`.
> - Slice **PR #27982** (DraftTailBuffer + tests) — SHA `0f8ad97ae2c025f47da08827e635765fbd21d145`.
>
> Source was downloaded to `.scratch_decoupled/` (git-excluded) for the deep read.
> Re-fetch any file with:
> `gh api repos/sgl-project/sglang/contents/<path>?ref=<SHA> -H "Accept: application/vnd.github.raw"`

## Navigation (for implementation agents)

This dir is a retrievable knowledge base. The hierarchy:
- **[CLAUDE.md](CLAUDE.md)** — *how* to use this KB (retrieval protocol, source-of-truth order, warnings). Read first.
- **[INDEX.md](INDEX.md)** — *where* to look (task→lesson routing, symbol→source reverse map, topic tree).
- **README.md** (this file) — *what it is* (overview, glossary, reading order).
- **01–08** — the content. **`.scratch_decoupled/proto/`** — the real upstream source (ground truth).

## Source material

| Source | What it is | Confidence |
|---|---|---|
| **Issue [#27462]** | Parallel Spec-Decoding **roadmap** (epic). Good for concepts + slice plan; some details (e.g. `ignore_decode_budget`) are roadmap aspirations not in the prototype. | High (concepts) |
| **PR [#22520]** "SpecActor" | The **prototype** — and it contains the **full engine** (59 files: `decoupled_spec_io.py`, `draft_tail_buffer.py`, `decoupled_verify_worker.py`, `scheduler_decoupled_spec_mixin.py` +2189, `draft_proxy.py`, `token_sync_thread.py`, Mamba/GDN hybrid-attention changes, examples). **This is the authoritative reference to port from.** | High (read in full) |
| **PR [#22272]** "SPECTRE" | A **different, competing** parallel-spec design (C++/msgpack ZMQ, `SpectreRequest`). **Not** our architecture. | High |
| **PR #27634** `[Spec][1/N]` | Clean re-land slice 1: `decoupled_spec_io.py` + server flags. **Byte-for-byte logic copy** of the prototype's IPC layer (+ explanatory docstrings). | High |
| **PR #27982** `[Spec][2/N]` | Clean re-land slice 2: `draft_tail_buffer.py` + **27 CPU tests**. Logic-identical to the prototype's buffer (+ an O(N²)→O(N) diagnostic fix). | High |

[#27462]: https://github.com/sgl-project/sglang/issues/27462
[#22520]: https://github.com/sgl-project/sglang/pull/22520
[#22272]: https://github.com/sgl-project/sglang/pull/22272

> ⚠️ **The one trap to avoid:** the architecture named in the task brief
> (`DraftSync`/`VerifyCommit`/`DraftTailBuffer`/`VerifyWorker`) is **SpecActor
> (#22520)**, *not* SPECTRE (#22272). SPECTRE is a different team's C++ design for
> the same goal. See [07](07-pr-landscape-and-roadmap.md).

## TL;DR — the big picture

Classic speculative decoding runs **draft → verify → draft → verify** on one
critical path: the target GPU idles while the draft model proposes, and vice
versa. Decoupled spec-decoding splits these into **two independent SGLang
engines** (separate processes, usually separate GPUs) talking over **ZMQ**:

- The **VERIFIER** runs the *target* model. It is **authoritative** — it owns the
  request, decides which tokens are final, and is the only side that streams to
  the client.
- The **DRAFTER** runs the *draft* model. It **runs continuously ahead**,
  streaming speculative tokens, and **rolls back** its KV cache whenever the
  verifier's committed prefix diverges from what it guessed.

Because the two run in parallel, the target model stays busy during draft
latency. The price is a cross-process **protocol + reconciliation + rollback**
layer — that layer *is* the feature, and it is what these lessons document.

```
                        client request (prompt)
                                   |
                                   v
   +================================================================+
   |            VERIFIER  --  target model  (authoritative)         |
   |   +-------------------------+   +--------------------------+    |
   |   | VerifyWorker            |   | DraftTailBuffer          |    |
   |   | verify draft tokens;    |   | mirror of the drafter's  |    |
   |   | accept longest matching |   | streamed tail; reconcile |    |
   |   | draft prefix + 1 bonus  |   | it against the commits   |    |
   |   +-------------------------+   +--------------------------+    |
   +================================================================+
      |  committed prefix to align to            ^  speculative
      |  (DraftSync / VerifyCommit / DraftClose) |  draft tokens
      v             [ ZMQ ]                      |  (DraftTailStreamOutput)
   +================================================================+
   |        DRAFTER  --  draft model  (runs continuously ahead)     |
   |   +-------------------------+   +--------------------------+    |
   |   | Draft decode loop       |   | Commit apply / rollback  |    |
   |   | generate tokens ahead   |   | align to committed prefix|    |
   |   | of the verifier, from   |   | truncate + roll back KV  |    |
   |   | the committed prefix     |  | on mismatch; sleep if    |    |
   |   +-------------------------+   | too far ahead            |    |
   +================================================================+
                                   |
              committed tokens -> stream_output -> client
```

## Why SMCSD is a good fit

The user's thesis: **"SMCSD is very suitable for this as we have no-rejections."**
In standard spec-decoding the verifier frequently *rejects* draft tokens, so the
drafter's KV rollback path is hot. If SMCSD's verification means the drafter's
tokens are (nearly) always accepted, the **expensive rollback path becomes cold**
and we capture nearly pure parallel overlap with little reconciliation churn.
Validating that against SMC's actual accept rule is the key open item —
see [08-smcsd-implications.md](08-smcsd-implications.md).

## Reading order

1. [01-architecture-and-roles.md](01-architecture-and-roles.md) — the two engines, transport threads, M:N mesh, server flags.
2. [02-ipc-protocol.md](02-ipc-protocol.md) — every ZMQ message dataclass, fields, direction.
3. [03-draft-tail-buffer.md](03-draft-tail-buffer.md) — the verifier-side reconciliation state machine (the correctness core).
4. [04-verify-worker.md](04-verify-worker.md) — the verify compute: linear top-k=1 chain, longest-prefix + bonus accept.
5. [05-drafter-runahead-and-rollback.md](05-drafter-runahead-and-rollback.md) — run-ahead, sleep/wake, commit apply, KV truncation, the echo-back trick.
6. [06-modes-and-constraints.md](06-modes-and-constraints.md) — lockstep vs parallel modes; the exact `validate_server_args` constraints.
7. [07-pr-landscape-and-roadmap.md](07-pr-landscape-and-roadmap.md) — which PR is which; the slice roadmap. **Anti-confusion lesson.**
8. [08-smcsd-implications.md](08-smcsd-implications.md) — mapping onto SMCSD + open questions (forward-looking).

## Glossary

| Term | Meaning |
|---|---|
| **Verifier** | Engine running the target model; authoritative; streams to client. |
| **Drafter** | Engine running the draft model; runs ahead; owns no requests of its own. |
| **Committed prefix** | Output tokens the verifier has finalized; both sides must agree on it. Tracked as `verifier_committed_prefix_len` (drafter) / `committed_len` (verifier buffer). |
| **Draft tail** | Speculative tokens the drafter produced *beyond* the committed prefix. |
| **DraftTailBuffer** | Verifier-side mirror of the drafter's tail; reconciles it vs commits. |
| **Bonus token** | The 1 always-correct token the verifier appends after the accepted draft prefix. |
| **Run-ahead window** | `speculative_num_draft_tokens * 2` — how far ahead the drafter may get before it sleeps. |
| **`DraftReqKey`** | `(src_verifier_rank, request_id)` — globally unique request id across the mesh. |
| **Stale base** | A draft token computed from a prefix older than a detected mismatch; rejected via `can_accept_prefix_len`. |
| **Pending-expected** | Verifier-committed tokens the drafter hasn't streamed back yet ("verifier ahead of drafter"). |
| **Echo-back** | After applying a commit, the drafter re-streams the committed token so the verifier's pending-expected queue can resolve. |

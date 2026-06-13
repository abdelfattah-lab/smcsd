# 02 — The IPC Protocol

All message dataclasses live in **`python/sglang/srt/speculative/decoupled_spec_io.py`**
(PR #27634). They are stdlib-only dataclasses with no behavior — pure wire schema.
Direction matters: **control** flows verifier→drafter, **data** flows drafter→verifier.

## Request identity

```
DraftReqKey(frozen):
    src_verifier_rank: int     # which verifier owns this request
    request_id: str            # unique only within that verifier
```
Helpers map it to a scheduler rid string: `build_draft_scheduler_rid` /
`parse_draft_scheduler_rid` → `"draft:{rank}:{request_id}"`.

## Control plane — Verifier → Drafter

Batched into one `DraftControlBatch` per drafter rank:

```
DraftControlBatch:
    dst_drafter_rank: int
    sync_messages:          list[DraftSync]
    verify_commit_messages: list[VerifyCommit]
    close_messages:         list[DraftClose]
```

### `DraftSync` — open / (re)sync a mirror request
```
request_id, src_verifier_rank, dst_drafter_rank
prompt_token_ids:      list[int]   # the prompt to mirror
committed_output_ids:  list[int]   # output already committed at sync time
```
The drafter creates a private decode request seeded from
`prompt_token_ids + committed_output_ids`.

### `VerifyCommit` — advance the committed prefix
```
request_id, src_verifier_rank, dst_drafter_rank
pre_verify_committed_len:  int        # committed length BEFORE this verify step
committed_token_ids:       list[int]  # the freshly accepted tokens
```
Semantics: the committed segment is
`output_ids[pre_verify_committed_len : pre_verify_committed_len + len(committed_token_ids)]`.
`pre_verify_committed_len` is the **anchor** that lets the drafter detect whether
this commit agrees with what it already guessed. `validate_committed_token_ids()`
raises if `committed_token_ids` is empty **or** `pre_verify_committed_len < 0`
(commits are always non-empty contiguous segments).

### `DraftClose` — release a request
```
request_id, src_verifier_rank, dst_drafter_rank
reason: str        # "finished" | "abort"
```

## Data plane — Drafter → Verifier

Batched into `DraftTailStreamOutputBatch(outputs: list[DraftTailStreamOutput])`:

```
DraftTailStreamOutput:    # ONE streamed speculative token
    src_drafter_rank, dst_verifier_rank, request_id
    base_committed_len: int   # committed-prefix length the drafter used as base
    new_token_pos:      int   # 0-based output position of this token
    new_token_id:       int
```
`base_committed_len` is the key anti-staleness field: it lets the verifier reject
tokens that were computed from a prefix older than where divergence was detected
(see [03](03-draft-tail-buffer.md), "stale base").

## Envelope

```
DraftMeshMessageType(str Enum): CONTROL_BATCH | TAIL_STREAM_OUTPUT_BATCH
DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch:            Optional[DraftControlBatch]
    tail_stream_output_batch: Optional[DraftTailStreamOutputBatch]
```
Constructed via `from_control_batch` / `from_tail_stream_output_batch`; sent with
`send_pyobj` (pickle).

## Drafter-side staging structures (also in `decoupled_spec_io.py`)

The drafter cannot apply commits the instant they arrive — they must be coalesced
and applied at a safe boundary in the scheduler loop:

- **`VerifierCommitSegment`** — coalesces *contiguous* `VerifyCommit`s for one
  request. `append_message` enforces same-`draft_key`, same-`dst_drafter_rank`, the
  validator above, and the **contiguity invariant**
  `message.pre_verify_committed_len == segment.end_committed_len` (no gaps).
  `extract_prefix(n)` peels off the first `n` tokens as a new segment and advances
  `self.pre_verify_committed_len += n`.
- **`DraftControlInbox`** — `sync_messages`,
  `verifier_commit_segments: dict[DraftReqKey, VerifierCommitSegment]`,
  `close_keys: set[DraftReqKey]`. `add_control_batch_locked` processes **close
  first** (a closed key suppresses syncs/commits in the same batch), then syncs,
  then commits. `extract_ready_controls_locked(consumable_commit_len)` drains all
  syncs/closes and only the *consumable* prefix of each commit segment (the
  callback decides how much the drafter has materialized) → `ReadyDraftControls`.
- **`ReadyDraftControls`** — `sync_messages`, `close_keys`,
  `ready_commit_segments: list[VerifierCommitSegment]`.
- **`DecoupledSpecIpcConfig`** (frozen) — `bind_endpoint`,
  `connect_endpoints: tuple[str,...]`, `rank`.

## Mental model

- Verifier pushes **three control verbs**: open (`DraftSync`), advance
  (`VerifyCommit`), close (`DraftClose`).
- Drafter pushes **one data verb**: here-is-a-token (`DraftTailStreamOutput`),
  each tagged with the base it was computed from.
- Everything is batched per peer rank to amortize ZMQ round-trips.

---
*Confidence:* High — field names cross-confirmed across both research passes and
consistent with the #27634 schema-only PR and the prototype consistency doc.

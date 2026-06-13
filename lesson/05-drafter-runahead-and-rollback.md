# 05 — Drafter: run-ahead, sleep, commit & KV rollback

The drafter is a normal continuous-batching decode engine whose requests are
opened/closed by the verifier. All logic lives in `SchedulerDecoupledSpecMixin`
(`managers/scheduler_decoupled_spec_mixin.py`, 2189 lines). Source-verified against
the #22520 prototype; line numbers below are from that file.

## Internal dataclasses

```python
@dataclass
class DraftReqState:                       # L47
    key: DraftReqKey
    req: Optional[Req] = None
    verifier_committed_prefix_len: int = 0   # the central anchor
    is_sleeping: bool = False
    mamba_checkpoint_positions: set[int] = field(default_factory=set)
    mamba_checkpoint_slots: Optional[torch.Tensor] = None

@dataclass(frozen=True)
class DraftKVTruncation:                    # L57
    req_pool_idx: int
    kv_start: int
    kv_end: int

@dataclass(frozen=True)
class DraftBatchMetadataUpdate:             # L64
    req_batch_idx: int
    new_seq_len: int
    new_tail_token_id: int
```

State tables: `draft_req_table: Dict[DraftReqKey, DraftReqState]` and
`draft_sleeping_reqs: Dict[DraftReqKey, Req]`.

Draft requests are created (`_create_draft_request`, L1106) with
`SamplingParams(max_new_tokens=1<<30, temperature=0.0, top_k=1, ignore_eos=True)`
— they decode until `DraftClose`. There is **no decode budget**; memory is bounded
purely by the sleep/wake window below.

## 1. Run-ahead + sleep (backpressure)

```python
def _draft_ahead_window(self):                      # L1382
    return max(0, int(self.server_args.speculative_num_draft_tokens or 0) * 2)
def _draft_req_ahead(self, state):                  # L1386
    return len(state.req.output_ids) - int(state.verifier_committed_prefix_len)
```

- **Window** = `speculative_num_draft_tokens * 2`. Per-request "ahead" = uncommitted
  suffix length.
- **Sleep** (`sleep_overrun_draft_requests`, L1435): any req with `ahead >= window`
  is set `is_sleeping = True`, parked in `draft_sleeping_reqs`, and removed from the
  running batch via `batch.filter_batch(keep_indices=...)`. Bounds *ahead-distance*,
  not total length.
- **Wake** (`wake_draft_sleeping_requests`, L1474): re-admits parked reqs once
  `ahead < window` and there's headroom vs `max_running_requests` /
  `pp_max_micro_batch_size`; rebuilds a decode batch (`_build_draft_decode_batch`)
  and merges into `running_batch`.

Every newly decoded tail token is streamed out (`flush_draft_updates`, L223) as a
`DraftTailStreamOutput` tagged with `base_committed_len` and `new_token_pos`;
tokens at positions `< verifier_committed_prefix_len` are skipped.

## 2. Applying a commit (`apply_verifier_commit_segment`, L758)

Commits arrive via `TokenSyncThread` → `DraftControlInbox`, coalesced into a
`VerifierCommitSegment`, and applied at a safe boundary. The drafter matches the
committed segment against its **own** speculative output:

```python
while (matched_segment_len < max_possible_match_len               # L850
       and int(req.output_ids[pre_verify_committed_len + matched_segment_len])
           == committed_token_ids[matched_segment_len]):
    matched_segment_len += 1
```

- **Full match** (L857): every committed token already matches → just
  `state.verifier_committed_prefix_len = new_committed_len` and prune Mamba
  checkpoints. **This is the cheap, common path — and the path SMCSD's
  "no-rejections" should make dominant** (see [08](08-smcsd-implications.md)).
- **Mismatch** (L863): the remaining mismatched tail must be **exactly one token**
  — callers split multi-token mismatches first, else it raises. Then roll back ↓.

## 3. KV rollback on mismatch (the load-bearing path on rejection)

```python
truncate_from   = committed_token_pos              # L908  (output-id index)
removed         = output_len - truncate_from       # L912
kv_truncate_from = prompt_len + truncate_from      # L918  (full-seq KV coord)
```

Then:
1. `req.grammar.rollback(removed)`.
2. Queue `DraftKVTruncation(req_pool_idx, kv_start=kv_truncate_from,
   kv_end=trimmed_end)` where `trimmed_end = min(kv_allocated_len, prompt_len +
   len(output_ids))`.
3. Lower the pointers: `kv_committed_len`, `kv_allocated_len`,
   `cache_protected_len` all `= min(current, kv_truncate_from)`.
4. `del req.output_ids[truncate_from:]` (+ slice the 6 logprob arrays and
   `hidden_states` if `return_logprob`).
5. **Re-install** the verifier's chosen token: `req.output_ids.append(
   committed_token_id)`, `req.grammar.accept_token(...)`, and reset finish state.
6. Set `state.verifier_committed_prefix_len = new_committed_len`; prune Mamba ckpts.
7. If the req is in `running_batch`, queue a `DraftBatchMetadataUpdate(req_batch_idx,
   new_seq_len = len(origin_input_ids) + max(len(output_ids)-1, 0),
   new_tail_token_id)` so the in-flight decode batch continues from the corrected
   prefix.

KV truncations and metadata updates are **batched and flushed**
(`_flush_draft_kv_truncations` frees the KV slices; `_flush_draft_batch_metadata_updates`
runs a Triton kernel to fix `seq_lens`/`output_ids` on device).

> **Why `page_size == 1` is mandatory:** rollback truncates KV at an *arbitrary
> token boundary*. With multi-token pages you couldn't free a partial page.

## 4. The echo-back trick (subtle but essential)

After applying ready commit segments (`_apply_ready_verifier_commit_segments`,
L1220), the drafter **re-streams the last applied committed token back to the
verifier** as a `DraftTailStreamOutput`. Why: when the verifier committed a token
the drafter hadn't produced yet (the "verifier ahead" / *pending-expected* case in
[03](03-draft-tail-buffer.md)), the verifier's `DraftTailBuffer` is waiting for the
drafter to *confirm* that token before advancing `committed_len`. The echo is that
confirmation. Without it, `pending_expected_tokens` would never drain.

## 5. Mamba / hybrid linear-attention rollback (the [5/N] roadmap item, already here)

Dense KV truncation isn't enough when the draft model has recurrent state. The
prototype keeps a **ring of state-checkpoint slots** sized to the ahead window:
- `_ensure_draft_mamba_ckpt_slots` allocs `window` slots from the Mamba pool.
- `_draft_mamba_ckpt_slot(state, token_pos, for_write=...)` maps a position to a
  ring slot via `token_pos % slot_count`. On **write**, if a *different* live
  position maps to the same slot it raises *"mamba checkpoint ring would overwrite
  a live checkpoint"* — i.e. the drafter exceeded its rollback window.
- `commit_draft_mamba_ckpts` records the new position after each forward;
  `_prune_draft_mamba_ckpts` drops checkpoints outside the uncommitted suffix
  (always keeping the current tail). On rollback, the slot for `committed_token_pos`
  is asserted to exist before truncation.

This needs `mamba_scheduler_strategy="no_buffer"` (auto-set; see
[06](06-modes-and-constraints.md)). SMCSD likely won't need this unless its draft
model is hybrid.

## 6. Event-loop order each scheduler step (`sync_draft_requests`, L1323)

1. Entry rank collects ready controls from the inbox via a `consumable_commit_len`
   callback (`_collect_ready_draft_controls`, L1200) — only the portion of a commit
   segment the drafter has actually **materialized** is released
   (`_draft_commit_segment_consumable_len`, L1149: requires
   `pre_verify_committed_len == current_committed_len`).
2. Broadcast `ReadyDraftControls` to all TP ranks.
3. **Close keys first** (`release_draft_request`), then
4. **Sync messages** (create + queue new draft requests), then
5. **Apply ready commit segments** (the rollback engine above), then echo-back,
   flush KV truncations + metadata updates.

(Around the forward: `prepare_draft_mamba_routing` before, `commit_draft_mamba_ckpts`
after; `sleep_overrun_draft_requests` / `wake_draft_sleeping_requests` around batch
scheduling.)

## The single invariant everything serves

```
drafter.output_ids[:verifier_committed_prefix_len]
        == verifier.output_ids[:verifier_committed_prefix_len]
```

Run-ahead may speculate arbitrarily far past that point; rollback exists precisely
to restore this equality whenever a commit says the speculation was wrong.

---
*Confidence:* High — every method name, line number, and expression above is read
directly from `scheduler_decoupled_spec_mixin.py` at the pinned prototype SHA.

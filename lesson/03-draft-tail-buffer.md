# 03 — DraftTailBuffer (the correctness core)

Verifier-side state machine that **mirrors the drafter's streamed tail and
reconciles it against the verifier's own commits**. Lives in
`python/sglang/srt/speculative/draft_tail_buffer.py` (PR #27982), with **27 CPU
unit tests** at `test/registered/unit/spec/test_draft_tail_buffer.py`. It is
thread-safe (`_lock` + `_condition`): the transport thread writes, the scheduler
thread reads.

> This file does **NOT** touch the KV cache. KV truncation belongs to the
> *drafter* scheduler layer that receives `VerifyCommit` (see
> [05](05-drafter-runahead-and-rollback.md)).

## The invariant it protects

```
drafter.output_ids[:verifier_committed_prefix_len]
        ==
verifier.output_ids[:verifier_committed_prefix_len]
```
"The drafter's committed prefix is aligned with the verifier before accepting any
future draft-tail tokens." Everything below exists to keep this true under
asynchrony, where the drafter may be *ahead of* or *lagging* the verifier.

## Per-request state — `RequestDraftTailState`

| Field | Meaning |
|---|---|
| `drafter_rank` | which drafter is paired to this request |
| `committed_len` | prefix length confirmed between verifier and drafter |
| `tail_tokens: list[int]` | contiguous draft tokens received *after* `committed_len` |
| `pending_expected_tokens: deque[int]` | verifier-committed tokens still awaiting drafter confirmation (drafter is *lagging*) |
| `can_accept_prefix_len` | **stale-base floor**: lowest base from which a draft token is still acceptable |

Buffer-level: `verifier_rank`, `required_tail_len` (min buffered tokens for a
snapshot), `_states: dict[request_id, RequestDraftTailState]`, plus one
`threading.Condition` over a single `Lock` (scheduler thread + transport thread
share it).

**Key invariant:** `tail_tokens` and `pending_expected_tokens` are **never both
non-empty**. They model opposite directions of skew — `tail_tokens` = "drafter
ahead of verifier", `pending_expected_tokens` = "verifier ahead of drafter". The
`consumable_tail_*` accessors return *nothing* while anything is pending, so
"committed" ≠ "consumable": a request can have advanced `committed_len` yet expose
an empty snapshot tail until the drafter confirms the pending tokens.

## Rule set 1 — applying a `VerifyCommit` (`apply_verify_commits`)

**Contiguity guard first:** `pre_verify_committed_len` must equal
`committed_len + len(pending_expected_tokens)`, else it raises. (A commit for an
unknown/closed request is a silent no-op.) Then match the committed segment against
buffered `tail_tokens`, longest-prefix style:

1. **Full match** — every committed token matches the buffered tail → drop the
   matched tail, advance `committed_len`. *No boundary change.*
2. **Tail too short** — buffer ran out before the commit did → append the missing
   committed tokens to `pending_expected_tokens`; **leave `can_accept_prefix_len`
   unchanged** (no divergence has been *observed* yet, the drafter is just behind).
3. **Actual mismatch** — a committed token disagrees with the buffered tail →
   drop the unmatched tail, queue the remaining committed tokens into
   `pending_expected_tokens`, and **advance `can_accept_prefix_len =
   committed_len`**. This is the divergence case; raising the floor lets us reject
   any in-flight stream the drafter computed from the now-dead prefix.

Key subtlety: `can_accept_prefix_len` advances **only on an observed mismatch at
the same `token_pos`**, never merely because the drafter is behind. Confusing
"behind" with "diverged" is the classic bug this design avoids.

## Rule set 2 — accepting a `DraftTailStreamOutput` (`append_draft_stream_batch`)

Each token is classified by `_push_one_locked` into an outcome string. The three
that matter most:

- **Stale-base rejection** (checked first): require
  `base_committed_len >= can_accept_prefix_len`. If not → `"stale_base"`, drop.
  This is what makes run-ahead safe: tokens computed before a divergence can't
  sneak in.
- **Pending-expected prefix** (drafter catching up after it lagged): the next
  output must satisfy *all of*
  `base_committed_len == committed_len`,
  `new_token_pos == committed_len`,
  `new_token_id == pending_expected_tokens[0]` →
  `"pending_expected_match"` (pop it, advance `committed_len`), else
  `"pending_expected_reject"` / `"pending_expected_gap"`.
  The confirming token arrives via the drafter's **echo-back**: after the drafter
  applies the same commit, it re-streams the committed token so this branch can
  drain the pending queue (see [05](05-drafter-runahead-and-rollback.md) §4).
  Without the echo, `pending_expected_tokens` would never empty.
- **Normal contiguous append**:
  `new_token_pos == committed_len + len(tail_tokens)` → `"appended"`.

Other outcomes: `"duplicate"`, `"already_committed"`, `"stale_gap"`,
`"unknown_request"`. (Diagnostics for the raising paths are built **lazily**, only
when actually raising — a deliberate perf fix to avoid O(N²) work + long lock
holds on the hot receive path.)

## Rule set 3 — snapshots for the verify step (`get_draft_snapshots`)

Before each verify forward, the verifier reads an **immutable**
`DraftTailSnapshot` of the current tail (so concurrent appends can't mutate the
in-flight batch):

- `allow_partial=True` (default) → return whatever tail is ready now (may be
  empty → that request just skips speculation this round).
- `allow_partial=False` → **block** in `_wait_for_draft_tokens_locked()` on
  `_condition` until `required_tail_len` tokens exist (or pending resolves). This
  is the lockstep mode (see [06](06-modes-and-constraints.md)).

`open_requests(list[DraftSync])` / `close_requests(list[DraftClose])` manage state
lifecycle.

## Why this is "the core"

Every other piece (verify compute, KV rollback) is mechanical. This buffer is the
single place where the asynchronous, possibly-divergent draft stream is turned
back into a provably-consistent committed prefix. The 27 unit tests cover match /
partial / mismatch / stale / pending match-reject-gap / duplicate / unknown /
protocol-violation-raises / wait / allow_partial — a good template for our own
test suite.

Two operational details worth porting: (1) the buffer-level `close()` (distinct
from per-request `close_requests`) sets `_closed`, clears all state, and wakes any
blocked snapshot waiter with a `RuntimeError` — clean shutdown. (2) The whole-batch
error diagnostic string is built **lazily, only on the raise path**
(`_format_batch_outputs_diag`) — the cleaned slice's one real change over the
prototype, removing O(N²) work + lock-hold on the hot receive path.

---
*Confidence:* High — read in full from `draft_tail_buffer.py` (+666 prototype /
+695 slice) and the 27 unit tests; every branch, outcome string, and invariant
above is taken from the source, cross-checked against the consistency design doc.
The cleaned slice (#27982) is logic-identical to the prototype.

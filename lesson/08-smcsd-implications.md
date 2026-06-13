# 08 — Implications for SMCSD (forward-looking notes)

> **Not a design.** Step 1 is learning. This file maps the upstream concepts onto
> SMCSD modules and records open questions to resolve before we design the port.
> Treat every claim about SMC internals here as a hypothesis to verify against the
> code.

> **UPDATE (2026-06-12): the port exists.** A lockstep decoupled prototype lives in
> `smcsd/decoupled/` (design doc: `tasks/decoupled_smc_design.md`). Answers to the
> open questions below, verified against `smcsd/` source while building it:
>
> - **Q1 — resolved.** "No rejections" is exact: `SMCWorker._forward_decode` always
>   returns `accept_lens == gamma+1` (gamma draft tokens + 1 target-sampled bonus).
>   Commits never truncate, so the upstream rollback/echo-back machinery has **no
>   trigger in lockstep** and was omitted entirely.
> - **Q2 — resolved.** The drafter holds all N particle KV states (each particle
>   drafts from its own context) but no weights/resample logic; the verifier sends
>   the resample plan (`dst_slots`/`src_slots`) and the drafter replays the same
>   fused block-table copy (`batched_resample_kv`) on its own pools.
> - **Q3 — resolved.** There IS a separate draft model (e.g. Llama-3.2-1B), so
>   "two engines, two GPUs" maps cleanly.
> - **Q4 — resolved.** Termination (EOS / length) is decided verifier-side from
>   slot state; the drafter just stops receiving the slot in `DraftStepReq.slots`
>   and frees on `DraftCloseGroup`. No `max_new_tokens=1<<30` run-until-close.
> - **Q5 — superseded.** SMC cannot run ahead *at all*: round t+1's anchor token is
>   the **target-sampled bonus** of round t, and resampling reshuffles particle KV
>   at round boundaries. Upstream's DraftTailBuffer/run-ahead solves an asynchrony
>   SMC's algorithm forbids, so the port keeps only the upstream *shape* (roles,
>   wire protocol, sync/commit/close lifecycle) and runs lockstep. Future overlap =
>   pipeline different groups (draft A while verifying B), not run-ahead.

## The core thesis: "no rejections" makes SMCSD an unusually good fit

In standard spec-decoding the verifier rejects draft tokens often, so the
drafter's **KV rollback path** (file [05](05-drafter-runahead-and-rollback.md)) is
hot and load-bearing. The user's claim is that SMCSD's verification has
**no rejections** — if true, then:

- The **rollback / KV-truncation machinery becomes a cold path** (rarely or never
  taken). That removes the single most complex and bug-prone part of the upstream
  design from the steady state.
- `apply_verifier_commit_segment`'s **full-match branch dominates** → commits just
  advance `verifier_committed_prefix_len` with no truncation.
- `DraftTailBuffer`'s **mismatch / stale-base branches** (file
  [03](03-draft-tail-buffer.md)) become rare → reconciliation is mostly the cheap
  "full match" path.
- We may still want the rollback path implemented for correctness/safety, but it
  need not be fast.

**Open question Q1:** What exactly does "no rejections" mean for SMC's accept
rule? Is the committed prefix *always* equal to the draft prefix (zero rollback
ever), or only in expectation? This determines whether rollback can be a
correctness-only fallback or must be omitted entirely. → verify in
`smcsd/common/verify.py` and `smcsd/core/req_state.py`.

## Where each upstream concept would land in SMCSD

| Upstream piece | Likely SMCSD home | Notes |
|---|---|---|
| `DECOUPLED_VERIFY` / `DECOUPLED_DRAFT` roles | new role flag on `smcsd/engine.py` (`SMCEngine`) | SMC already injects a custom scheduler process; a role split is a natural extension. |
| Verify compute (`VerifyWorker`) | `smcsd/core/worker.py` + `smcsd/common/verify.py` | SMC already has its own verify/resample logic — the decoupled verifier wraps it, not EAGLE. |
| `DraftTailBuffer` reconciliation | new module under `smcsd/core/` | Portable almost verbatim from #27982; it's pure CPU logic. |
| IPC protocol (`decoupled_spec_io`) | new module under `smcsd/managers/` or `smcsd/common/` | Pure dataclasses; portable verbatim. |
| Transport threads (DraftProxy / TokenSync) | alongside `smcsd/managers/smc_tp_worker.py` | SMC already runs a ZMQ scheduler loop — reuse that wiring style. |
| Drafter run-ahead + sleep | `smcsd/core/scheduler.py` (`SMCScheduler`) | Continuous-batching decode loop already exists here. |
| KV truncation / rollback | `smcsd/mem_cache/allocator.py` + `smcsd/core/worker.py` | SMC uses `page_size=1`, `disable_radix_cache=True` already — *aligns with the upstream constraints in [06](06-modes-and-constraints.md).* |

## Constraints SMCSD already satisfies (good news)

From the existing lessons (`tasks/lesson.md`) and SMC setup, SMC already runs with
`page_size=1` and `disable_radix_cache=True` — two of the hard constraints in
file [06](06-modes-and-constraints.md). That removes friction.

## Subtle mechanisms to preserve when porting

Even if "no-rejections" makes rollback cold, these are load-bearing for
*correctness* and easy to drop by accident:
- **Echo-back** ([05](05-drafter-runahead-and-rollback.md) §4): the drafter must
  re-stream committed tokens it didn't itself produce, or the verifier's
  `pending_expected_tokens` never drains and snapshots stall.
- **Single-token-mismatch contract**: `apply_verifier_commit_segment` requires the
  mismatched tail to be exactly one token; callers split multi-token mismatches.
- **Stale-base floor** (`can_accept_prefix_len`): only advances on an *observed*
  mismatch, never merely because the drafter lagged — confusing the two is the
  classic bug.
- **Snapshot immutability**: bind a frozen per-forward `draft_buffer` and broadcast
  it to all TP ranks; never let the verify forward read a live, mutating buffer.

## Porting reference is available locally

The **full prototype source** (59 files, +11,182) is downloaded under
`.scratch_decoupled/proto/` (git-excluded) at the pinned SHA in the
[README](README.md). When we implement, read the real `scheduler_decoupled_spec_mixin.py`
and `decoupled_verify_worker.py` rather than these summaries — and pin to that SHA,
since the upstream slice stack is still moving.

## Open questions to resolve before designing the port

- **Q1** (above) — exact meaning of "no rejections"; can rollback be omitted?
- **Q2** — SMC expands `max_running_requests *= (n_particles + 1)` and runs N
  particles per request. How does particle expansion interact with a decoupled
  drafter? Does the drafter run particles too, or only the verifier?
- **Q3** — SMC's "draft" and "target" may already be coupled differently than
  EAGLE. Is there even a separate draft *model*, or is SMC's speculation
  intra-model? This decides whether "two engines, two GPUs" maps cleanly.
- **Q4** — SMC rejects stop strings (existing lesson). The decoupled drafter
  decodes with `max_new_tokens = 1<<30` until `DraftClose`; confirm SMC's
  termination (EOS / `stop_token_ids`) drives `DraftClose` cleanly.
- **Q5** — Start in **lockstep mode** ([06](06-modes-and-constraints.md)) to
  validate the protocol against a single-process SMC baseline, then enable
  parallel. What's the cheapest correctness diff we can build?

## Suggested next step (Job Step 2 preview)

Before any code: a **code-exploration pass over `smcsd/`** to answer Q1–Q4 (read
`engine.py`, `core/scheduler.py`, `core/worker.py`, `common/verify.py`,
`core/req_state.py`), then a short design doc that decides drafter/verifier
boundaries for SMC's particle model. Do **not** start porting the IPC layer until
Q3 (is there a separate draft model?) is answered — it changes the whole topology.

---
*Confidence:* Low/forward-looking by design. The SMCSD module mapping is inferred
from directory structure + prior lessons, not yet from reading the SMC source.

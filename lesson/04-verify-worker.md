# 04 — VerifyWorker (the verify compute)

The target-side worker that turns a mirrored draft tail into accepted tokens in
**one** target forward. File: `speculative/decoupled_verify_worker.py`
(class `VerifyWorker`, 451 lines). Source-verified against the #22520 prototype.

## What it is

- **A plain class** (`class VerifyWorker:`), **not** a subclass of `EAGLEWorker`.
  It *borrows* EAGLE machinery two ways:
  - Class-level method steal: `_mamba_verify_update = EAGLEWorker._mamba_verify_update`.
  - Imports the EAGLE verify primitives: `EagleVerifyInput`, `EagleVerifyOutput`,
    `TreeMaskMode`, `build_tree_kernel_efficient`.
- It holds the **target model**, not a draft model: `self.target_worker`
  (a `TpModelWorker`), `self.model_runner`, the target's
  `req_to_token_pool` / `token_to_kv_pool_allocator`.
- `self.topk = 1` is hard-pinned. There is **no in-process drafter** — drafts
  arrive externally via `req.draft_buffer` (the snapshot bound by the scheduler).

## Per-step flow (`forward_batch_generation`)

The orchestrator-facing entry is `forward_batch_generation(batch) ->
GenerationBatchResult`, which drives a **draft → verify** pair:

1. **Bypass:** if the batch is extend/idle (prefill), just call
   `target_worker.forward_batch_generation(...)` — no spec logic.
2. `spec_info = self.draft(batch)` — *no model forward*; pure metadata: build the
   linear chain `EagleVerifyInput` from each request's external `draft_buffer`.
3. `(logits_output, verify_output, can_run_cuda_graph) = self.verify(batch, spec_info)`
   — runs the **target** forward, then EAGLE's `EagleVerifyInput.verify` to accept
   the longest matching prefix + 1 bonus.
4. `_assert_verify_output_within_snapshot_tail(...)` guard (below).
5. Assemble `GenerationBatchResult(next_token_ids=accept_tokens,
   num_correct_drafts, num_correct_drafts_per_req_cpu, ...)`.

## The linear top-k=1 chain

`_build_linear_topk1_tree_metadata(batch_size, spec_steps, device)` returns:

```
selected_index = [0, 1, 2, …, spec_steps-1]   (per row)
parent_list    = [-1, 0, 1, …, spec_steps-2]  (per row)   # node i's parent = i-1
```

Fed into EAGLE's `build_tree_kernel_efficient` with `topk=1` and
`TreeMaskMode.FULL_MASK`, this degenerates the EAGLE token *tree* into a single
*chain* — every node has exactly one child, no siblings. This reuses all EAGLE
verify kernels without a new verify path.

## EOS padding + the un-acceptability cut (the 3-part guarantee)

Real draft tails vary per request, but the batched forward needs a fixed width
(for CUDA-graph shape stability). Three mechanisms together guarantee padded
tokens are **never** accepted:

1. **Pad** — `_build_req_verify_tokens` builds `[tail_token, *draft_buffer[:spec_depth]]`
   and right-pads short tails with an **EOS** id from `_get_pad_token_id()`
   (resolved from generation/hf config; raises if no EOS exists).
2. **Cut** — at the end of `draft()`:
   `retrieve_next_token[row, real_tail_len] = -1`, where `real_tail_len =
   min(len(draft_buffer), speculative_num_draft_tokens - 1)`. This truncates the
   chain at the last *real* draft token, so the verify kernel can't traverse into
   the padded positions.
3. **Assert** — `_assert_verify_output_within_snapshot_tail` checks
   `num_correct_drafts_per_req <= real_tail_len` for every request (and that
   `accept_tokens` is non-null). Runtime proof that no pad was accepted.

## Accept logic + outputs

EAGLE's `spec_info.verify(...)` walks the chain, accepts the longest matching
prefix, and emits the bonus token. `VerifyWorker` surfaces:
- `accept_tokens` → `next_token_ids` (includes the bonus).
- `num_correct_drafts_per_req_cpu` → per-request accepted-draft counts.
- `num_correct_drafts` (sum) and running totals `total_accept_length` /
  `total_num_verified_reqs`.

The **bonus token is always safe** because it is the target model's own
next-token prediction after the accepted prefix — authoritative regardless of what
the drafter guessed.

## Snapshot contract (why it's correct under concurrency)

`req.draft_buffer` is a **per-forward snapshot bound before verify** (by the
scheduler's `_bind_verify_snapshots`, broadcast across TP ranks). The worker only
*reads* it (slices a copy — never pops). Any tokens the transport thread appends
concurrently belong to *later* verify rounds. Because every TP rank verifies the
exact same frozen snapshot, ranks can't disagree on accept length and corrupt KV.
After each round `batch.spec_info = None` — there is no carried draft state.

## A second guard lives in the scheduler, not the worker

The scheduler's `validate_verify_outputs` independently re-checks that the
accepted tokens are a **prefix of `req.draft_buffer`** and that the verified
segment equals `req.output_ids[pre:pre+accept_len+1]` — raising on any mismatch.
So "no padded/out-of-snapshot token is ever committed" is enforced in **two**
places (worker assert + scheduler validation). See
[05](05-drafter-runahead-and-rollback.md) §verifier-half.

## CUDA-graph notes

- A request whose real tail is shorter than the full token count disables the
  full-graph path: the reported `can_run_cuda_graph` is
  `can_run_cuda_graph and (spec_info.draft_token_num == speculative_num_draft_tokens)`.
- Fixed-width EOS padding is precisely what keeps tensor shapes graph-stable.
- `CaptureHiddenMode.NULL`; hidden-state return disabled (no downstream drafter to
  feed).

---
*Confidence:* High — read in full from `decoupled_verify_worker.py` at the pinned
prototype SHA. Symbol names are stable: the IPC/buffer slices kept prototype names
verbatim, so the verify worker's names should carry through when it is sliced.

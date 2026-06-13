# 06 — Operating Modes & Hard Constraints

## Two operating modes (same protocol code)

The design runs in two modes that exercise the **same** IPC + reconciliation code
— the first is for validation, the second is the real win.

### A. Lockstep validation mode
- Drafter produces one speculation window, streams the tail, then **blocks until
  that window's commit arrives** before starting the next window.
- Verifier reads the tail with `allow_partial=False` → `DraftTailBuffer.
  get_draft_snapshots` **blocks** in `_wait_for_draft_tokens_locked` on its
  condition variable until `max(1, required_tail_len)` raw tail tokens exist and
  nothing is pending.
- Deterministic; validates the whole protocol + reconciliation end-to-end.
- **Not scaffolding** — it runs the production protocol code, just without overlap.

### B. Parallel / draft-ahead mode (real overlap)
- **Drafter lever:** the drafter runs continuously ahead, bounded only by the
  run-ahead window / sleep (see [05](05-drafter-runahead-and-rollback.md)).
- **Verifier lever:** `allow_partial=True` → verify whatever tail is ready now; if
  the drafter hasn't caught up to the committed output (`snapshot.committed_len <
  len(req.output_ids)`), `req.draft_buffer` is set `None` and that request just
  decodes one token with no speculation this step.
- Now the drafter can be a full window ahead on a **diverged** prefix, so
  **rollback** and **stale-base rejection** become load-bearing (they were inert
  in lockstep).

**Toggle:** environment variable **`SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL`**
(read in the scheduler as `envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.get()`,
default **true** = parallel). ⚠️ It is an **env var, not a server CLI flag** — it
does not appear in `server_args.py`.

### Bring-up order for SMCSD
Come up in **lockstep first** (deterministic, easy to diff against a single-process
SMC baseline), then flip the env to parallel for performance.

## Run-ahead memory bound (there is NO `ignore_decode_budget`)

> **Correction:** the roadmap (#27462) mentions an `ignore_decode_budget` admission
> switch, but it **does not exist in the #22520 prototype** (grep-confirmed: zero
> hits in `server_args.py` and the scheduler mixin). Instead:
- Draft requests are created with `SamplingParams(max_new_tokens=1<<30,
  temperature=0.0, top_k=1, ignore_eos=True)` — they decode until the verifier
  sends `DraftClose`.
- Memory is bounded **entirely** by the sleep/wake run-ahead window
  (`speculative_num_draft_tokens * 2`), not by a decode budget.

## Hard config constraints (`validate_server_args`)

Enforced per role. Some **raise**, some **silently auto-mutate** `server_args` in
place. This is also a checklist of assumptions to port into SMCSD.

### Common to both verifier and drafter
| Check | Behavior |
|---|---|
| `enable_dp_attention` set | **raise** — DP attention unsupported |
| `dp_size != 1` | **raise** |
| missing `decoupled_spec_bind_endpoint` / empty `connect_endpoints` / `rank is None` / `rank < 0` | **raise** |
| `speculative_num_steps is None` | **raise** |
| `speculative_eagle_topk != 1` | **raise**, then force `= 1` (linear chain only) |
| `speculative_num_draft_tokens != num_steps + 1` | **auto-set** to `num_steps + 1` (+warn) |
| `max_running_requests is None` | **auto-set to 64** (+warn) |
| `disable_overlap_schedule` false | **auto-disable** overlap scheduler (+warn) |
| `enable_mixed_chunk` true | **auto-disable** (+warn) |

### Verifier-only
| Check | Behavior |
|---|---|
| `page_size > 1` | **raise** — *"token rollback is token-granular"* (so `page_size == 1` required) |

### Drafter-only
| Check | Behavior |
|---|---|
| `disable_radix_cache` false | **auto-set True** (+warn) — mirror requests don't share prefixes |
| `mamba_scheduler_strategy != "no_buffer"` | **auto-set `"no_buffer"`** (+warn) — needed for the state-slot rollback ring ([05](05-drafter-runahead-and-rollback.md)) |

> Note: the **verifier** enforces `page_size == 1`; the drafter's own
> `validate_server_args` does not re-check it, but it is the side that actually
> does token-granular KV truncation — keep both at `page_size == 1`.

Multi-verifier deployments also require `--batch-size % num_verifiers == 0` (each
verifier gets a contiguous slice).

## Stated limitations / TODOs

- No `overlap_schedule` support with Spec-Decoding-v2.
- Adaptive speculative decoding not implemented.
- `VerifierCommitSegment.append_message` raising on the `TokenSyncThread` will
  **kill the drafter control thread** (the loop only catches
  `zmq.error.ContextTerminated`); the slice docstring flags this and proposes
  per-request quarantine + verifier autoregressive fallback (roadmap phase 5c) as
  future hardening. Relevant if we ever feed it adversarial/buggy peers.

---
*Confidence:* High — modes, the `allow_partial` env, the absence of
`ignore_decode_budget`, and every constraint above are quoted from the prototype's
`spec_info.py` / `scheduler_decoupled_spec_mixin.py` / `server_args.py`.

# Async-Bonus SMC ("ragged-now") — design + implementation spec

**Date:** 2026-06-16 · **Branch:** `smc-async-draft` · **Flag:** `SMCSD_ASYNC_BONUS` (default off)

## TL;DR

A flag-gated, async, per-particle, **FULL-bonus** SMC decode path in `smcsd/decoupled`. Goal:
reach Mode-A's full-bonus accuracy (**~71% GSM8K**) at **~no-bonus throughput (~130–135 tok/s)**
— vs synchronous Mode A's **71.1% @ 109** and no-bonus async's **66.3% @ 135**. Coexists with
the current modes (no-bonus async, `SMCSD_BARRIER_BONUS`, `SMCSD_BET_KEEP/DISCARD`,
`SMCSD_SPEC_BARRIER`) behind the flag, via the mutual-exclusion guard at
`async_scheduler.py:125-136`.

**Mechanism ("ragged-now"):** each particle run-ahead-drafts its NEXT window seeded by its own
drafter anchor `x_g1` (the *bet*), overlapping the verify of its current window. The verifier
**always** commits the target bonus `b`. If `x_g1==b` the banked run-ahead is kept; else it is
discarded and re-drafted from `b`. The committed anchor is **always** `b` (no skipped bonus, no
Mode-B seam). A **ragged verify** scores particles at different per-particle seq_lens in one
batched forward; ESS resample fires on a cadence, copy-ahead cloning carries banked windows for
free, and weights reset to zero after each resample.

> **The three adversarial reviews returned: CUDA-graph GO; correctness CONDITIONAL with one
> INVALID design point (rewritten below); protocol/KV two BLOCKERS (both repaired below).** The
> graph machinery is sound and needs no kernel/forward change. The real gates are (1) the
> weight/finalize semantics under stagger, and (2) the CUDA-graph-bs + per-slot-rollback handling
> that the original slice plan deferred to an "optional" S6 — pulled into S4 here.

---

## 0. Verdicts at a glance (fold of the three reviews)

| Review | Verdict | Net effect on this design |
|---|---|---|
| **CUDA-graph / ragged-verify** | **GO** | Both captured graphs (draft AR, linear target-verify) replay correctly with ragged per-particle `orig_seq_lens`. No kernel/forward/capture change. The only preserved invariants are uniform **batch size** (hit a captured bs; padding handles the rest) and uniform **per-step token count** (γ+1 everywhere). |
| **SMC-correctness** | **CONDITIONAL**; one point **INVALID** | (d)'s weight formula was wrong and is **rewritten** (two-branch: matched=identical-prefix exact increment, mismatched=discard→zero). (a) reset-to-zero is unbiased for the *target* (variance/diversity op only). (b) "ragged window counts at collect" prose is **wrong** — the drain-first path equalizes frontiers at the decision point; kept drain-first, prose cut. (c) copy-ahead bounded at D=1, CONDITIONAL on a dst-capacity KV guard. |
| **Protocol / KV / liveness** | **CONDITIONAL**, 2 BLOCKERS | **B1:** `log_weights` IS zeroed by the kernel every resample → the "lifetime weight" finalize plan is void; repaired with a net-new `finalize_score` tensor. **B2:** subset re-draft bs + per-slot rollback are **required in S4**, not optional S6. Plus finished-rider-as-survivor (3d-i), drafter `SMCSD_DROP_BONUS` footgun, FIFO/epoch ordering — all repaired below. |

**GO/NO-GO header:** **GO.** The make-or-break ragged-verify enabler is empirically reachable
and statically sound. The residual risk is **not** graph correctness — it is the SMC accuracy
hypothesis (mᴺ break via reset-to-zero at staggered frontiers) and the finalize/finished-rider
semantics, both of which the slice plan isolates before they can silently bias accuracy.

---

## 1. Design + rationale (concise)

**What it replaces.** Today the async path (`AsyncDecoupledSMCScheduler`) runs **synchronous trains
of K windows** with all particles locked at one frontier, then a **barrier resample**. Full bonus
under that regime pays the mᴺ "all N particles must match" amplification (`1 − 0.84^12 ≈ 88%`
per-window re-draft probability) — which is exactly why synchronous Mode A drops to 109 tok/s.
Per-particle run-ahead + ragged verify breaks the amplification: only the *mismatched minority*
re-drafts, and it does so without a global sync.

**Core invariants.**
1. **Per-particle RUN-AHEAD depth 1.** Each particle's drafter speculatively drafts its next
   window seeded by its own `x_g1`, overlapping the verify of its current window. Bank depth D=1.
2. **Bonus ALWAYS committed.** After the verifier scores a window it samples target bonus `b`.
   `x_g1==b` → keep run-ahead; else discard + re-draft from `b`. Committed anchor is always `b`.
3. **RAGGED VERIFY.** One batched forward scores particles at different per-particle
   seq_lens/positions. The CUDA graph tolerates ragged positions/seq_lens as buffer DATA
   (`kv_indptr=cumsum(seq_lens)` rebuilt per replay) as long as **batch size** and **per-step
   token count (γ+1)** stay uniform. *(GO — review 1.)*
4. **RAGGED-NOW resample.** ESS-triggered on a cadence over each particle's accumulated
   `interval_weights`. **Decision-point subtlety (review 2, finding b):** because S5 *drains all
   in-flight StepResps before `collect`*, every active particle has committed exactly the same
   number of windows at the resample instant → **frontiers are equalized at the decision point**,
   so there is **no staggered-window-count bias**. Raggedness lives *between* resamples (the
   overlap), not *at* the collect. The earlier "ESS over different window counts" framing is
   incorrect and is dropped.
5. **COPY-AHEAD clone.** A resampled particle copies the survivor's full current state *including*
   banked windows (tokens + KV) — SBP-style refcount-share reuse. Diversity coupling is bounded
   by D=1 and decorrelates after one independent drafter `multinomial` draw.
6. **RESET-TO-ZERO.** After resampling, reset all active particles' `interval_weights` to 0. The
   fused kernel already zeros *resampled* rows' `interval_weights` AND `log_weights`
   (`fused_collect.py:194-195`); the design extends the `interval_weights` zero to non-resampled
   active rows. **`finalize_score` (net-new) is never reset** — see §4 / B1 repair.
7. **K orthogonal.** ESS-check cadence (`SMCSD_RESAMPLE_INTERVAL`, accuracy knob) is separate from
   bank depth D (throughput knob).

**Rejected alternative (kept rejected):** "resample on a common synchronized frontier + rebase the
weight" — rejected in favor of ragged-now + reset-to-zero (simpler, more responsive, decorrelates
clones faster). Note (review 2): the *implementation* drains-before-collect, so the decision point
is in fact synchronized in window-count; the win over the rejected alternative is the **overlap
between resamples** and the **reset vs rebase** choice, not literally collecting over ragged
window counts.

---

## 2. File-level implementation spec

### 2a. Scheduler-side run-ahead state (net-new, `async_scheduler.py`)

A per-particle generalization of `SpecState` (`async_scheduler.py:57`), slot-indexed:

```python
@dataclass
class RunAhead:
    pending: PendingDecodeStep   # the in-flight run-ahead StepReq's verify snapshot
    bet_anchor: int              # x_g1 the run-ahead window was seeded from
    tag: int
    epoch: int
```

Held as `self._bank: dict[int, RunAhead]` keyed by survivor slot id, plus `self._inflight: int`
(the FIFO-safety outstanding-StepResp counter). The slot tensors (`seq_lens`,
`kv_allocated_lens`, `all_token_ids`, `req_pool_indices`) already carry the per-particle frontier
— **no new per-slot tensor.**

### 2b. Protocol — one field change, wire-compatible (`io_struct.py`, `worker.py`, `draft_server.py`)

`DraftStepReq.rollback: int → Optional[List[int]]` (default `None`). **REPAIRED PRIORITY
(B2 / review 2 finding 3c-i):** this is **required in S4**, not deferred to an optional S6. The
moment the mismatched subset S is padded/merged to a captured bs (Hazard 2 fix), a single StepReq
mixes matched rows (rollback 0) and mismatched rows (rollback γ+1); the scalar broadcast at
`draft_server.py:316` would corrupt the matched rows' `seq_lens` and trip the `:320` assert.

When `rollback` is a list: `draft_server.py:316` → `self.seq_lens[active] -= torch.tensor(msg.rollback, ...)`;
`send_step_req` (`worker.py:299`) accepts `rollback: int | List[int]`. The drafter element-wise
assert (`:320`) is unchanged. Scalar `int` remains accepted for the existing modes.

### 2c. Weight accumulators (net-new, `req_state.py`)

```python
def reset_interval_weights(self, active):
    self.interval_weights[active] = 0.0
```

Called after every ragged-now resample, **after** `dispatch_resample_batch`, over the
**post-rebuild** active set (`rebuild_active_slots` flips membership when finished ancestors are
cloned in — `scheduler.py:216-218`).

**B1 repair — net-new lifetime finalize accumulator.** The original spec claimed
`log_weights` is a never-zeroed lifetime weight; **this is FALSE against the code.**
`_fused_collect_kernel` zeros **both** `iw_ptr` (interval_weights) and `lw_ptr` (log_weights) at
every resampled row (`fused_collect.py:194-195`), and `finalize_group` selects
`max(log_weights, visible_output_len)` (`req_state.py:714-720`). So `log_weights` already means
"weight since this slot last resampled," in **all** modes today. Under reset-to-zero at staggered
frontiers, comparing `log_weights` across particles that resampled a different number of windows
ago is meaningless and silently favors the longest-un-resampled particle.

Repair: add a net-new tensor `self.finalize_score`, accumulated alongside `log_weights` in
`process_batch_result` (`req_state.py:609-610`: `self.finalize_score[active] += d`) but **never
touched by the collect kernel or by reset**, and select finalize on it:

```python
best_slot = max(slots, key=lambda s: (float(self.finalize_score[s].item()),
                                       visible_output_len(s)))
```

The collect kernel is **not** changed (no new pointer argument); `finalize_score` lives only in
the Python `process_batch_result` accumulation, parallel to `log_weights`, so the kernel never
zeros it. *(Decision: keep a true-lifetime importance-weight argmax for finalize. The alternative
— "uniform-among-survivors / max-output-len," argued by review 2 as the SMC-honest post-reset
selector — is recorded in §4 as the fallback if `finalize_score` empirically regresses.)*

---

## 3. The new async loop (`_run_async_bonus_train`, replaces `_run_decode_train` when flag on)

Branched at `_event_loop` (~`async_scheduler.py:196`). Steady-state, no K-window barrier:

```
PER ITERATION (one batched window over the current active set A):
1. recv current-window StepResp (one batched reply); _inflight -= 1.
   finish_decode asserts tag/epoch (worker.py:371-380). Epoch is pinned to the ESS-interval
   counter, NOT per-StepReq (review 2, 3a-ii): every StepReq between two ESS checks shares one
   epoch, so the merged verify shares one pending.epoch.
2. FIRE RUN-AHEAD (depth-1 bank) for the WHOLE active set off x_g1 = resp.tokens[:, gamma],
   seq_lens = slot_state.seq_lens[A]. One batched StepReq, tag T_next, epoch E. _inflight += 1.
   Record RunAhead per slot. (async_scheduler.py:263-269 fired UNCONDITIONALLY, per-particle.)
3. VERIFY current window: finish_decode(pending, resp, drop_bonus=False)  ← ALWAYS bonus.
   Commits b; returns logprob_diff over n_weighted=gamma columns; b = result.verified_id.
4. WRITEBACK: _writeback_window(result, A) → interval_weights[A] += d, log_weights[A] += d,
   finalize_score[A] += d. Frontier (seq_lens[A]) advanced by gamma+1.
5. MATCH/SPLIT (per-particle): x_g1 = resp.tokens[:, gamma]; match = (x_g1 == b) (EXACT id
   equality, worker.py:303). MATCHED slots: banked run-ahead valid; committed frontier now equals
   the seq_lens the run-ahead was fired at → bank consistent, no re-draft. MISMATCHED slots S:
   bank seeded by wrong anchor → discard.
6. DISCARD + subset re-draft (S4 shape, REPAIRED — review 2, B2):
   - recv the run-ahead StepResp fired in step 2 (drain its reply, _inflight -= 1). VALIDATE its
     (tag,epoch) against the recorded RunAhead before discarding (port async_scheduler.py:317-322).
   - Keep matched rows' run-ahead tokens as the next window's resp.
   - Re-fire a StepReq over only S, PADDED UP TO THE NEAREST CAPTURED cuda_graph_bs with dummy
     rows (outputs discarded), with verified_ids=b[S], seq_lens=frontier[S], rollback=[gamma+1]*|S|
     (per-slot list; dummies rollback 0). _inflight += 1.
   - MERGE: the next verify ctx gathers matched-run-ahead rows + re-drafted rows back into original
     slot order (a net-new responsibility of _build_ragged_ctx — see §4). Both sub-batches verify
     at uniform gamma+1 query tokens; KV history is ragged (the supported case).
7. ESS CHECK on cadence (every K check-points, K = SMCSD_RESAMPLE_INTERVAL):
   - DRAIN all _inflight StepResps first (FIFO-safety; validate each tag/epoch). _inflight == 0.
   - collect_resample_jobs_batch(slot_state)  ← ESS over interval_weights per-slot.
   - If n_jobs>0: dispatch_resample_batch (copy-ahead clone), send_commit, propagate bank
     inheritance to clones, reset_interval_weights(post-rebuild active).
8. Assemble next pending(s) from the matched run-ahead resp + the re-drafted resp via the merge
   gather (step 6).
```

The SBP drain-guard at `_event_loop:170` is generalized to a `_commit_async_bonus_standalone`
(analog of `_commit_spec_standalone` at `:494`) that **loops** on `_inflight` (≥2 possible:
run-ahead + re-draft) before any prefill/pause/idle (review 2, 3d-ii).

---

## 4. Ragged mechanics + resample + clone (the enablers)

**Ragged verify + ragged draft — no kernel/forward change (review 1: GO).**
- Verify: `finish_decode` builds `cache_locs` via `assign_smc_cache_locs_kernel` over
  `ctx.orig_seq_lens` (`worker.py:362-369`, per-row), positions via
  `orig_seq_lens.unsqueeze(1)+step_offsets` (`info.py:194`), `verify_batch.seq_lens=orig_seq_lens`
  (`info.py:209`). The linear-target-verify replay (`triton_backend.py:828-865`) splits the
  constraint exactly right: `qo_indptr=cumsum(full(bs,γ+1))` is **uniform** (the only uniform
  requirement), while `kv_indptr=cumsum(seq_lens)` and `kv_indices` are built **per-row**. With
  `skip_attn_backend_init=True` (`worker.py:402`) the kv metadata is written by the explicit
  `replay_prepare` (`info.py:234`) and read from the backend's persistent buffers — replay only
  re-copies `input_ids`/`positions`. `can_run` gates only on `cuda_graph_bs <= max_bs`, never on
  seq_len uniformity. This is the already-shipping colocated path — proven, not speculative.
- Draft: `from_slot_gather` (`info.py:54`) + `prepare_for_draft`'s
  `all_positions = orig_seq_lens + step_offsets` (`info.py:149`) are per-row. The mirror assert
  (`draft_server.py:320`) passes for any ragged subset as long as the verifier sends matching
  seq_lens. Every StepReq drafts exactly γ+1 AR steps for every row — the one preserved uniform.
- The async-bonus loop builds the next pending's ctx by gathering each slot's own
  `slot_state.seq_lens[slot] - (γ+1)` into `orig_seq_lens` (the `_build_spec_a1:625-631` pattern).

**CUDA-graph bs (the make-or-break, REPAIRED — reviews 1 & 2).** Matched/mismatched sub-batches
and the run-ahead must hit a captured bs. `disable_cuda_graph_padding` defaults **False**
(`server_args.py:615`), so `can_run` accepts any `|S| <= max_bs` and `replay_prepare` pads up to
the nearest captured bs via `bisect_left`. **Subset re-draft IS graph-able.** The original spec's
"re-draft the full active set" was self-contradictory with its own throughput claim: a full-set
re-draft means everyone re-drafts → no overlap saved → ~109, not ~130. **Repair: S4 ships the
subset re-draft, padded to a captured bs (dummy rows, outputs discarded), with per-slot rollback.**
The genuine off-graph fallback (`use_multistep`, `draft_server.py:364-369`) is correct (eager) but
slower — used only if `|S| > max_bs` or padding disabled.

**ESS over current full weight.** `collect_resample_jobs_batch` (`scheduler.py:127`) reads
`interval_weights[slot]` per-slot, normalizes within a group, ESS-checks — frontier-agnostic, zero
kernel change. ESS normalizes over `interval_weights` (confirmed: launch
`batched_collect_fused(log_weights, interval_weights, …)` → kernel `iw_ptr=interval_weights` drives
LSE/ESS, `fused_collect.py:131-143,254-257`). Trigger on the K cadence counter, not at a barrier.

**Copy-ahead clone — VALID, bounded at D=1 (review 1 c).** `dispatch_resample_batch`
(`scheduler.py:201-206`) vector-copies `seq_lens, kv_allocated_lens, all_token_ids, token_counts`
dst←src; `batched_resample_kv` clones the block table. Since the bank is committed into the slot
tensors *before* resample (the drain in step 7 guarantees it), the clone carries the banked window
for free. KV growth is free: `req_to_token` is `(pool_size, max_context_len)`, so writing
`[dst, :src_len]` for any `src_len ≤ max_context_len` is always in-bounds — the clone is a
block-table index copy with refcount share, not physical KV.

> **dst-capacity KV guard (REPAIRED — reviews 1 c & 2 finding 3b).** `batched_resample_kv` frees
> over `dst_alloc_lens[dst]` (phase 1) but writes only `src_seq_len` entries (phase 2). The clone
> is in-bounds **iff `kv_allocated_lens[dst] >= seq_lens[src]`** *and* the dst's logical view is
> reset to `[0, src_seq_len)` by `kv_allocated_lens[dst] = kv_allocated_lens[src]`
> (`scheduler.py:202`) + `seq_lens[dst] = seq_lens[src]` (`:201`). This holds **only if** the
> bank's KV allocation bumped `kv_allocated_lens` at commit time (not just `seq_lens`). **S5 must
> assert `kv_allocated_lens[dst] >= seq_lens[src]` per resample job** (port the
> `async_scheduler.py:614-616` invariant to the dispatch path). The drafter `_handle_commit`
> (`draft_server.py:512,518`) clones the identical committed `seq_lens[src]` and bumps
> `kv_allocated_lens[dst]` to match — **consistent, no `_handle_commit` change needed in S5**,
> *because* the drain (step 7) transitively guarantees the drafter applied the run-ahead seq_len
> advance before it sees the commit (the drafter is a serial FIFO reactor; `_handle_commit` sends
> no reply, so a drained StepResp proves its `_handle_step` already ran).

**Bank inheritance.** After `dispatch_resample_batch`, propagate the survivor's `RunAhead` to its
clones using the `ancestor` array (`async_scheduler.py:570-572`):
`for dst, src in jobs: self._bank[dst] = self._bank.get(src)`. **S5 must assert `_bank[src]` is
still un-recv'd at inheritance time** (review 2, 3c-ii) — the inherited run-ahead must be the
*unverified* banked window, else adopting an already-consumed window double-advances the clone.

**Reset-to-zero.** `reset_interval_weights(post-rebuild active)` after dispatch. `finalize_score`
left intact (B1 repair). `log_weights` is zeroed by the kernel on resampled rows regardless (it is
no longer load-bearing for finalize).

> **Finished-rider exclusion (REPAIRED — review 2, 3d-i).** Under stagger, a group may have a
> particle that hit EOS at window `w` (finished rider) while a sibling is at `w-1`. The group is
> not finalizable, but the rider still sits in `row_in_use` for `collect`. With reset-to-zero its
> `interval_weights=0`, equal to just-reset live siblings → the ESS can pick the **dead** rider as
> a resample **src (survivor)**, cloning a finished particle's KV onto a live slot, which then
> decodes past EOS. The synchronous path avoids this because finished particles carry their full
> *accumulated* (non-reset) weight at the barrier. **Repair: exclude finished riders from the
> resample candidate set** (mask them out of `row_in_use` before `collect`), OR freeze (do not
> reset) a finished rider's `interval_weights` at its pre-finish value. The exclusion approach is
> chosen (simpler; finalize already reads `finalize_score`, untouched). S5 asserts no resample job
> has a finished `src`.

---

## 5. Ordered, independently-shippable slices

Each compiles, has a smoke test and a GSM8K acceptance. Order de-risks: **isolate weight/finalize
bookkeeping in the SYNC path → prove ragged verify as a no-op → always-bonus accuracy → run-ahead
+ bet → async resample.** Baselines (1000q seed0): no-bonus async **66.3% @ 135** (tokens
**191522**); Mode A **71.1% @ 109**; lockstep+bonus **73.5% @ 107**. 200q no-bonus: **65.0% /
tokens 37964**.

### S0 — Flag scaffolding + mutual exclusion + drafter `DROP_BONUS` invariant (no-op)
- **Files:** `async_scheduler.py:81-86` (relax the hard `SMCSD_DROP_BONUS` raise **on the verifier
  scheduler only**, behind the flag), `:122-136` (add `SMCSD_ASYNC_BONUS` to the `_on` exclusivity
  list), `_event_loop:196` (branch to `_run_async_bonus_train` stub → falls through to
  `_run_decode_train` when flag off).
- **REPAIRED (review 2, secondary):** the drafter **must keep `n_emit=γ+1`** (it stays in no-bonus
  emit mode even though the verifier commits bonus), because the bet reads `tokens[:, gamma]`
  (`async_scheduler.py:303`) — column γ only exists when `n_emit=γ+1` (`draft_server.py:374`).
  **`SMCSD_DROP_BONUS=1` MUST remain set on the drafter process.** S0 relaxes only the verifier
  scheduler's `__init__` raise; it does **not** unset the env on the drafter. The validation
  command below sets `SMCSD_DROP_BONUS=1` (it governs the drafter emit width); the verifier
  ignores it under the flag.
- **Test:** import; construct with each flag pair → exclusivity raise fires; flag-off path
  unchanged; stub raises `NotImplementedError`.
- **GSM8K:** flag OFF → **bit-identical** to no-bonus async: 1000q **66.3% @ 135, tokens 191522**;
  200q **65.0% / 37964**.

### S1 — `finalize_score` + RESET-TO-ZERO bookkeeping, proven in the SYNC barrier path
- **Files:** `req_state.py` (add `finalize_score` accumulation + `reset_interval_weights`; finalize
  selects on `finalize_score`), `async_scheduler.py:_barrier_resample` (call `reset_interval_weights`
  after `dispatch_resample_batch`, gated by `_async_bonus` or a `SMCSD_RESET_ZERO` debug flag).
- **B1 repair baked in:** this slice validates the **finalize_score / interval split**, not just
  reset. The original "keep log_weights for finalize" plan is void (kernel zeros log_weights);
  finalize now reads the never-zeroed `finalize_score`.
- **Test (TIGHTENED — B1):** unit — accumulate known weights, resample, assert all-active rows'
  `interval_weights==0`, `finalize_score` **untouched**, `log_weights` zeroed only on resampled
  rows (kernel behavior). **AND** assert finalize selection matches a reference run that selects on
  the true lifetime score — accuracy-within-noise alone will NOT catch a finalize bias.
- **GSM8K:** no-bonus async + reset flag, 200q. Reset after a full-weight-consumed resample is
  ~neutral; expect within noise of **65.0% / 37964**. If it regresses, the reset/finalize semantics
  are wrong — caught here, cheaply.

### S2 — Ragged verify proven as a NO-OP (uniform-frontier input)
- **Files:** `async_scheduler.py` — a `_build_ragged_ctx(active)` helper gathering per-slot
  `seq_lens[active]-(γ+1)` into a hand-built `SMCDecodeContext` (clone of `_build_spec_a1:625-631`),
  routing the existing train's **verify ctx** through it when frontiers are uniform.
- **REPAIRED (review 2, secondary):** S2 swaps **only the verify-ctx construction**. It must
  **keep `prepare_for_decode`'s KV allocation** — `from_slot_gather` allocates (`info.py:92-100`);
  a gather-only ctx does not. For the uniform no-run-ahead case allocation already happened in
  `prepare_for_decode`, so the hand-built verify ctx is allocation-free and correct, but only if
  decode-batch prep is untouched.
- **Test:** assert `_build_ragged_ctx` over uniform active produces `orig_seq_lens` identical to
  `from_slot_gather`'s; verify forward emits `bs*(γ+1)` rows (`worker.py:408`); **`kv_allocated_lens`
  advances identically to baseline** (allocation parity).
- **GSM8K:** flag on, **no run-ahead**, 1000q → must reproduce no-bonus async **66.3% @ 135**
  (**tokens 191522 bit-identically**). Ragged machinery on uniform data = no-op. **This is the
  mandatory empirical conversion of the static GO into an empirical one (review 1).**

### S3 — ALWAYS-bonus commit (synchronous, no run-ahead) — accuracy lever, isolated
- **Files:** `async_scheduler.py` — in `_run_async_bonus_train`, call
  `finish_decode(..., drop_bonus=False)` every window (reuse `worker.py:463-474` verbatim); commit
  `b` as the next anchor; keep the train **synchronous** (re-draft the whole set from `b` every
  window, like `BET_DISCARD`'s full re-fire `:325-328`, unconditionally — no overlap yet).
- **Test:** every committed anchor == the bonus sample (not `tokens[:, gamma]`); `n_weighted=γ`
  (`worker.py:417`).
- **GSM8K:** 1000q → must reach **~71% (Mode A accuracy)**, throughput **~107-109** (serial
  re-draft). **Proves the accuracy target is reachable** in the new loop before optimizing
  throughput. `BET_DISCARD` already proved 71.1% @ 109; S3 must match.

### S4 — Per-particle run-ahead + bet + SUBSET re-draft (the throughput lever) — ragged frontier appears
- **Files:** `async_scheduler.py` (`RunAhead` dict, `_inflight`, fire run-ahead off `x_g1`
  unconditionally, match/split, drain+validate, subset re-draft, merge gather, epoch pinned to ESS
  counter, `_async_bonus_debug` KV-invariant guard every window). **`io_struct.py:88`**
  (`rollback: Optional[List[int]]`), **`draft_server.py:316`** (tensor subtract),
  **`worker.py:299`/`send_step_req`** (list-accept) — **pulled from S6 (B2 repair).**
- **REPAIRED (review 2, B2):** S4 ships the **subset** re-draft (only S), padded to a captured bs
  with dummy rows, with **per-slot `rollback: List[int]`**. The original full-set re-draft is
  inconsistent with the throughput claim. Add a **drafter-side** OOB guard (`cache_locs[:, step] <
  req_to_token.shape[1]`, `all_positions[:, -1] < kv_allocated` before the AR loop at
  `draft_server.py:379`), not just the verifier KV invariant.
- **Test:** the KV-invariant debug guard never fires across a matched/mismatched mix; the
  drafter-side OOB guard never fires; per-slot rollback element-wise assert (`:320`) passes on a
  mixed StepReq; padded subset re-draft hits a captured graph (no `use_multistep` fallthrough);
  `_inflight` returns to a known value each barrier; tag/epoch asserts never trip.
- **GSM8K:** 1000q → accuracy holds at **~71%** (anchor always clean `b`), throughput climbs toward
  **~125-130** (run-ahead overlaps verify; only the mismatched minority re-drafts). The
  make-or-break throughput slice.
- **PRE-FLIGHT (review 1, sharp risk):** before the 1000q run, use the shipped `SMCSD_BONUS_AGREE`
  diagnostic (`worker.py:434-462`) to measure the empirical `x_g1==b` rate. `x_g1` is sampled at
  `SMCSD_ANCHOR_TEMP=0.3` (drafter) while `b` is the target sample at `smc_target_temperature` —
  these are different distributions, so the match is a coincidence-of-samples, not a coupling. **If
  the match rate is the ~16% implied by 0.84/token, the "mismatched minority" assumption is wrong —
  it is a majority re-draft and the throughput target collapses.** This is a design-validity gate
  for S4, not just a tuning detail.

### S5 — Ragged-NOW ESS resample + copy-ahead + reset-to-zero (replace the barrier)
- **Files:** `async_scheduler.py` — replace `_barrier_resample` with ESS-on-cadence (step 7):
  drain+validate `_inflight`, `collect_resample_jobs_batch`, `dispatch_resample_batch`,
  `send_commit`, bank inheritance via `ancestor`, `reset_interval_weights(post-rebuild active)`.
  Add `_commit_async_bonus_standalone` (loops on `_inflight`).
- **REPAIRED (reviews 1 & 2):** (1) finished-rider exclusion from the resample candidate set before
  `collect` (3d-i); (2) dst-capacity KV guard `kv_allocated_lens[dst] >= seq_lens[src]` per job
  (1c/3b); (3) `_bank[src]` asserted un-recv'd at inheritance time (3c-ii); (4) the first StepReq
  after any commit reads `seq_lens` **post-dispatch** and `send_commit` is ordered **before** that
  StepReq (3a-i); (5) `reset_interval_weights` over the **post-rebuild** active set.
- **Test:** clone inherits survivor's `RunAhead`; drafter `_handle_commit` and verifier `dispatch`
  clone identical (banked) `seq_lens`; the seq_lens-divergence assert (`draft_server.py:320`) never
  fires after a resample; `_inflight==0` before every `send_commit`; no resample job has a finished
  `src`; dst-capacity invariant holds for every job.
- **GSM8K:** 1000q → **~71% @ ~130-135** (the full target). Reset-to-zero + independent re-draft of
  the mismatched minority breaks the mᴺ amplification while ragged overlap keeps throughput at the
  no-bonus floor.

### S6 (optional) — throughput polish
- **Files:** further `_inflight`-per-tag ledger if subset re-drafts make reply counts depend on
  `|S|` (B1/review 2: **do not ship subset variations beyond S4's padded-to-captured-bs shape
  without a per-tag ledger** — the scalar counter is insufficient once reply counts vary).
- **GSM8K:** 1000q → accuracy unchanged (~71%); throughput at or above 135 if the polish saves
  work. **Ship only if S5 leaves throughput on the table.**
- *(Note: per-slot rollback and bs-padded subset re-draft, originally S6, are now in S4 — they are
  prerequisites for S4's throughput, not polish.)*

---

## 6. Correctness verdicts (the four required + repairs)

### (a) Reset-to-zero unbiasedness — **VALID for the target; CONDITIONAL on the finalize fix (B1)**
Resampling is unbiased for any proper weights given (i) resample ancestors ∝ normalized weights and
(ii) reset resampled particles to uniform. The kernel does both for resampled rows
(`fused_collect.py:138,150-194`). Extending the `interval_weights` zero to **non-resampled** active
rows is a **variance/diversity operation, never a bias on the SMC target marginal** — resampling
and reset change variance/diversity, not the target. For a resampled group the extension is a
*no-op* (the kernel already zeroed all N slots, `:194`); it only changes non-resampled groups,
making them behave as if they had resampled-at-uniform without the KV shuffle. **CONDITIONAL on B1:**
finalize must select on the never-reset `finalize_score`, not on `log_weights` (which the kernel
zeros). S1's pass criterion includes a finalize-selection assertion, not just accuracy-within-noise.

### (b) Ragged-verify go/no-go + staggered-decision bias — **GO (verify); bias ZERO at the decision point**
Ragged verify: **GO** (review 1, §4 here). Staggered bias: the original "ESS over different window
counts" framing is **wrong**. Because S5 **drains all `_inflight` before `collect`**, every active
particle has committed the same number of windows at the resample instant → frontiers are equalized
→ **the staggered bias is exactly zero at the decision point.** Raggedness exists only *between*
resamples (the overlap that buys throughput). The "accepted staggered approximation" hand-wave is
**rejected and removed**; the design states explicitly that collect sees **equal window counts**
(drain-first). If a future slice ever collected *without* draining, the bias would be real and
**anti-ahead** (matched particles carry one extra typically-negative `logprob_diff` → preferentially
killed) and would need a per-particle window-count rebase — that path is **not taken.**

### (c) Copy-ahead coupling bound — **VALID, bounded at D=1; CONDITIONAL on the dst-capacity guard**
Cloning a survivor's full state (banked windows included) and resetting both to uniform is the
standard resample step (a duplicated ancestor). Diversity loss is the depth-D shared-window
coupling, **bounded by D=1** (depth-1 run-ahead) and decorrelated after one independent drafter
`multinomial` draw (`draft_server.py:401`). KV growth is free (block-table refcount share into
full-width `req_to_token` rows). **CONDITIONAL on** the dst-capacity KV guard
`kv_allocated_lens[dst] >= seq_lens[src]` (S5 test) and bank-commit bumping `kv_allocated_lens`, not
just `seq_lens` — guaranteed by the drain-first discipline on the drafter's FIFO.

### (d) Importance weight under "seed on `x_g1`, commit `b`" — **INVALID as originally written; REWRITTEN**
The original formula `Σ log p_target(·|b) − log q_draft(·|x_g1)` is **NOT a valid importance weight**
— it scores the target at a prefix `b` the proposal never conditioned on. The drafter anchor is the
**conditioning prefix x0 for the whole window** (`draft_server.py:375`: `current_ids = verified`;
the verify scores `p_target(x_j | a, x_<j)`), not a weighted token. Correct two-branch reality
(**there is no third case**):

- **MATCHED (`x_g1 == b`):** the prefix the drafter conditioned on (`x_g1`) **equals** the committed
  prefix (`b`) token-for-token. The banked window's verify scores
  `Σ_j [log p_target(y_j | b=x_g1, y_<j) − log q(y_j | x_g1, y_<j)]` — **same prefix on both sides →
  a standard, exact SIS increment** (exactly what `finish_decode` computes at the banked frontier,
  `worker.py:427`). **VALID.**
- **MISMATCHED (`x_g1 != b`):** the banked window is **discarded** and re-drafted from `b`; the new
  window is scored against `q(·|b)`, prefix `b` on both sides — **VALID**. The mismatched bank
  contributes **zero** to the weight.

So the original mixed-prefix formula **never actually occurs** (and would be wrong if it did).
**CONDITIONAL on:** (1) the match test being exact token-id equality (it is — `worker.py:303`,
`bet_miss = x_g1 != b_cpu`); (2) the discard path never letting a mismatched bank's logprobs leak
into `logprob_diff`; (3) the verify ctx for a banked window being built at the banked frontier whose
x0 is the survivor's own committed bonus. The S4 BONUS_AGREE pre-flight (above) measures the match
rate — it governs the **discard fraction (throughput)**, not correctness; the weight is valid in
both branches.

---

## 7. Open engineering TODOs

1. **`finalize_score` semantics decision (B1).** Ship `finalize_score` (true-lifetime importance
   argmax) as primary. If S1/S5 show it regresses vs the no-bonus baseline, switch finalize to the
   SMC-honest post-reset selector ("uniform-among-survivors / max visible_output_len") — the
   alternative recorded in §4. Do **not** revert to `log_weights` argmax (kernel-zeroed).
2. **BONUS_AGREE pre-flight gate (review 1).** Before S4's 1000q run, confirm the empirical
   `x_g1==b` rate is high enough that the re-drafted set S is a minority. If it is ~16%, the
   throughput target needs a coupling fix (e.g. sample the bet at the target temperature, or
   couple `x_g1` to `b` via shared randomness) — a design change, gated here.
3. **`_inflight`-per-tag ledger (B1/review 2).** The scalar `_inflight` is sufficient only while
   each ESS interval has a fixed reply count (S4's padded-to-captured-bs subset). Any further
   subset-shape variation (S6) needs a per-tag ledger.
4. **`_commit_async_bonus_standalone` loop semantics.** Must drain **both** the run-ahead and the
   re-draft (≥2 in-flight) at every prefill/pause/idle guard point — loop on `_inflight`, not pop
   once (3d-ii).
5. **Drafter `n_emit=γ+1` enforcement.** Keep `SMCSD_DROP_BONUS=1` on the drafter (or add an
   explicit drafter flag forcing `n_emit=γ+1`). Footgun: the env is inherited by the drafter
   process (`draft_server.py:85`); unsetting it makes `tokens[:, gamma]` OOB.
6. **Merge-gather (`_build_ragged_ctx`) for two-StepResp windows.** Net-new: gather matched-run-ahead
   rows + re-drafted rows back into original slot order into one verify ctx (S4). Keep
   `prepare_for_decode`'s allocation; swap only verify-ctx construction (S2 contract).
7. **Finished-rider exclusion plumbing.** Mask finished riders out of `row_in_use` before `collect`
   (S5); confirm no resample job picks a finished `src`.
8. **Off-graph fallback measurement.** If any subset re-draft lands off-graph (`|S| > max_bs`),
   confirm `use_multistep` is correct and measure its throughput cost; padding to a captured bs is
   the default.

---

## 8. Validation protocol

`scripts/accuracy_test_gsm8k.py` with `.venv/bin/python`. GPUs via `CUDA_VISIBLE_DEVICES` pair
(0,1)/(2,3). **`SMCSD_DROP_BONUS=1` is kept** (it governs the drafter emit width — do NOT drop it;
see §7.5).

```
CUDA_VISIBLE_DEVICES=0,1 SMCSD_ASYNC_BONUS=1 SMCSD_DROP_BONUS=1 \
  SMCSD_ANCHOR_TEMP=0.3 SMCSD_RESAMPLE_INTERVAL=2 SMCSD_DRAFT_CUDA_GRAPH=1 SMCSD_TIMING=1 \
  .venv/bin/python scripts/accuracy_test_gsm8k.py --mode smc_async \
  --particles 12 --gamma 8 --temperature 0.7 --attention-backend triton \
  --num-questions {200|1000} --mem-fraction-static 0.6 --seed 0
```

(`--drop-bonus` is **not** passed as a CLI arg — the verifier ignores bonus-dropping under the flag,
S0 relaxes the verifier `__init__` raise — but `SMCSD_DROP_BONUS=1` stays in the **env** so the
drafter emits γ+1 columns. This corrects the earlier "drop SMCSD_DROP_BONUS" note, which was a
footgun.)

**Baselines (1000q seed0):** no-bonus async **66.3% @ 135** (tokens **191522**); Mode A
**71.1% @ 109**; lockstep+bonus **73.5% @ 107**. 200q no-bonus: **65.0% / tokens 37964**.

**Targets / gates per slice:**
- **Bit-identity slices** (S0 flag-off, S2): must reproduce tokens **191522** (1000q) / **37964**
  (200q) **exactly**, and accuracy 66.3% / 65.0%.
- **Accuracy slice** (S3): **~71%** (Mode A), throughput ~107-109.
- **Throughput slices** (S4-S5): accuracy holds **~71%**, throughput **~130-135** tok/s.
- **Final target:** **~71% @ ~130-135** tok/s (S5).

> **Mandatory empirical next step (review 1):** run **S2** (`SMCSD_ASYNC_BONUS=1`, no run-ahead,
> 200q then 1000q) and confirm it reproduces no-bonus async tokens **37964 / 191522** bit-identically.
> That single run converts the static CUDA-graph GO into an empirical one — the graph machinery is
> not the gate; the staggered-resample statistics (S4/S5) are.

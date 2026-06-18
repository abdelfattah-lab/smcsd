# Handoff: fix the `SMCSD_COPYAHEAD_REDRAW` accuracy bug

**Status:** ✅ FIXED + VALIDATED. The shared-KV corruption (clones re-drafting into the survivor's refcount-shared `[S,S+G)` cells) was fixed by giving redraw clones **private** cells on both sides: verifier `_make_copyahead_redraw_kv_private`, and a drafter `truncate_kv` StepReq field → `draft_server._truncate_step_kv_if_requested` + a real free/realloc in `mem_cache/allocator.py`. Validated 100q (GPUs 0,1): **K=1 = 73.0%, K=2 = 74.0%** (was 38%/59% when buggy — the K-dependence signature is gone), and Mode A K=1 unchanged at 70% (no regression). 

**Outcome:** the variant is correct and is the **highest-accuracy async/overlap mode** (73–74%, lockstep+bonus class), confirming the design hypothesis (resample-every-verify + re-draw = max diversity = max accuracy). But throughput is **~64–68 tok/s** (below Mode A's 91): recv-wait ~66% ≈ 2 drafter passes/window — the uniform-catchup clone re-draft is a second pass every resample. So it's an accuracy-favoring point, not a throughput win. Open: head-to-head vs lockstep+bonus on this box; and a ragged-lag variant (no catch-up pass) is the only untried way to also win throughput. Historical bug diagnosis retained below for reference.

This doc is self-contained — a fresh agent should be able to work from it + the code without the originating conversation.

---

## 1. Environment / repro

- Repo: `/home/cc2869/smcsd`, branch `smc-async-draft`. Python: `.venv/bin/python`.
- GPUs: this box currently has **only GPUs 0,1** (decoupled SMC: target/verifier on GPU0, drafter on GPU1). Use `CUDA_VISIBLE_DEVICES=0,1`.
- Eval entrypoint: `scripts/accuracy_test_gsm8k.py`.
- All the code under review is **uncommitted** in the working tree (in `smcsd/decoupled/async_scheduler.py`).

**Repro the bug (100q, ~5 min):**
```bash
cd /home/cc2869/smcsd
env CUDA_VISIBLE_DEVICES=0,1 SMCSD_COPYAHEAD_REDRAW=1 SMCSD_DROP_BONUS=1 SMCSD_ANCHOR_TEMP=0.3 \
  SMCSD_RESAMPLE_INTERVAL=1 SMCSD_DRAFT_CUDA_GRAPH=1 SMCSD_TIMING=1 .venv/bin/python \
  scripts/accuracy_test_gsm8k.py --mode smc_async --particles 12 --gamma 8 --temperature 0.7 \
  --attention-backend triton --num-questions 100 --mem-fraction-static 0.6 --seed 0
```
Expected (correct): **~70%** accuracy (same as the re-draw baseline below). Actual (buggy): **38%**.

There is an invariant-assert debug mode: `SMCSD_ASYNC_BONUS_DEBUG=1` (and `SMCSD_SPEC_BARRIER_DEBUG=1`) — turn these on while debugging; they assert KV / seq_len invariants and may catch the corruption closer to its source. The bug runs *clean* (no crash) at 5q, so it's silent corruption, not an assert trip — the asserts may be too weak; consider strengthening them.

---

## 2. The diagnostic signal (why this is a bug, and where)

Measured (100q, GPUs 0,1, gamma=8, 12 particles, anchor-temp 0.3):

| design | resample cadence | clone behavior | accuracy | tok/s |
|---|---|---|---|---|
| `SMCSD_BET_DISCARD` (Mode A) | K=1 | re-draw (drain) | **70%** | 91 |
| `SMCSD_BET_DISCARD` (Mode A) | K=2 | re-draw (drain) | 69% | 92 |
| `SMCSD_COPYAHEAD_RESAMPLE` (inherit) | K=1 | inherit in-flight | 61% | 67 |
| **`SMCSD_COPYAHEAD_REDRAW`** | **K=1** | **re-draw (clone-subset)** | **38%** ⚠ | 48 |
| `SMCSD_COPYAHEAD_REDRAW` | K=2 | re-draw (clone-subset) | 59% | 53 |

**Why it must be a bug:** re-drawing fresh clones can only *preserve* diversity vs. inheriting — so REDRAW should be **>= the inherit number (61%)**, ideally ~70% (like Mode A re-draw). Instead it's **38%, below inherit**. Impossible if correct.

**Where the bug is (the K-dependence pins it):** accuracy goes **38% (K=1) → 59% (K=2)** — it *improves* as you resample (and therefore fire the clone-re-draft) **less** often. So **each clone-re-draft is corrupting state**; the more often it fires, the worse. The bug is in the clone-re-draft path (`_fire_copyahead_redraw` + the consume splice), NOT in the shared overlap/resample machinery (which the inherit variant exercises without this failure).

---

## 3. What this mode is supposed to do (design)

Streaming, decoupled SMC speculative decode. N=12 particles, importance reweighting, resample-on-low-ESS, KV reshuffle via a clone kernel. A "window" = the drafter emits `gamma+1 = 9` tokens per particle in one pass. `K = SMCSD_RESAMPLE_INTERVAL` = windows between ESS resamples (K=1 = resample every window).

Common to all bonus modes: each window fires the next window's run-ahead off the drafter's bet `x_g1` (overlap), verifies the current window committing the **exact target bonus `b`** (`db=False`, "full bonus"), and on a bet miss (`x_g1 != b`) re-drafts that window from `b`.

**`SMCSD_COPYAHEAD_RESAMPLE` (inherit, the 61% variant — works, kept as reference):** the next run-ahead StepReq is sent **before** `_barrier_resample`'s `send_commit` (DraftCommitResample). The drafter `event_loop` (`draft_server.py:186`) is single-thread FIFO, so it processes the StepReq first (advances its mirror `seq_lens[src]` S→S+G, writes the in-flight KV), then `_handle_commit` clones `seq_lens[dst]=seq_lens[src]` and copies KV `[0:seq_lens[src]]` src→dst. So a clone **inherits** the survivor's in-flight window. `_build_spec_a1` then gathers each post-rebuild A1 row's reply columns from its A0 ancestor (ancestor map `a[dst]=src`).

**`SMCSD_COPYAHEAD_REDRAW` (the BROKEN variant):** identical, except the clone must NOT inherit — it re-draws its own fresh window (preserving diversity). Implementation chosen = "uniform-catchup": survivors keep their run-ahead (frontier S+G); each clone is re-drafted from the committed frontier S up to S+G so the next verify is a uniform S+G batch. Mechanism:
- `_fire_copyahead_redraw(plan)` (≈`async_scheduler.py:1489`): after `send_commit`, fires a **clone-subset** StepReq over `plan.dst_slots`, seeded by each clone's committed bonus `b` (= `verified_ids[dst]`), with `seq_lens = committed = seq_lens[dst] - (gamma+1)` and `rollback = gamma+1`. Stashes the clone slots + tag on `self._spec` (`SpecState.redraw_clone_slots`, `redraw_tag`).
- `_consume_copyahead_window0` (≈`async_scheduler.py:1527`): after `_build_spec_a1`'s ancestor gather, if `spec.redraw_tag is not None`, recv the clone-re-draft reply and **splice** each clone A1 row's `tokens`/`logprobs` from it (replacing the inherited survivor window). Survivors keep their carried run-ahead.
- `_commit_spec_standalone`: must also drain + splice the in-flight clone reply at prefill/pause/idle boundaries.

So the redraw does the most dangerous KV/seq_len manipulation in the codebase: **re-draft into the `[S, S+G)` slack in place + rollback on both verifier pool and drafter mirror + a row-aligned splice into the verify ctx.**

---

## 4. Prime suspects (the three areas to hammer — these are where to look first)

1. **Row alignment of the splice.** `_fire_copyahead_redraw` fires a StepReq over `plan.dst_slots` (a *subset* in dst-slot order). The reply rows are in that subset order. At consume, each clone's **A1 row** must be spliced from the matching reply row. If the mapping `dst_slot -> reply_row -> A1_row` is misaligned (e.g. A1 is rebuilt/compacted in a different order than `plan.dst_slots`, or the ancestor map vs. the redraw subset disagree), clones verify *another clone's* window → garbage. **This is the most likely bug.** Check `_build_spec_a1` row order vs. `plan.dst_slots` order vs. the post-`rebuild_active_slots` A1 order.

2. **KV / seq_len of the in-place re-draft.** The clone's KV was copied as `[0, S+G)` (inherited window included). The re-draft rolls the drafter mirror S+G→S (`rollback=gamma+1`) and re-drafts `[S, S+G)` overwriting the slack. Verify: (a) `kv_allocated_lens[dst] >= S+G` so the slack cells are allocated (dst-capacity); (b) the verifier pool's `seq_lens`/`kv_allocated_lens` for the clone match the drafter mirror after the rollback; (c) no stale-KV read of the old inherited `[S, S+G)` cells before they're overwritten; (d) the committed-frontier `S = seq_lens[dst] - (gamma+1)` is the *right* S (is `seq_lens[dst]` S+G at that point, or already advanced?).

3. **Committed seq_len consistency across the FIFO sequence.** The drafter processes, in order: (survivor run-ahead StepReq) → (DraftCommitResample clone) → (clone-subset redraw StepReq). Confirm the drafter mirror `seq_lens` is exactly consistent with the verifier's expectation at each step, that `rollback=gamma+1` lands the clone mirror at S (not S±something), and that the verifier-side `seq_lens`/`kv_allocated` are rolled back identically. Watch the every-window (K=1) case specially — there are no interior windows, so window-0 is always the consume path. Also the **finished-group drain**: a survivor or clone whose group finishes mid-flight (see `_commit_spec_standalone`, `_drain_finished_groups`).

---

## 5. Files / methods (exact)

- `smcsd/decoupled/async_scheduler.py`:
  - `_run_copyahead_train` (~1260), `_fire_copyahead_spec` (~1413), **`_fire_copyahead_redraw` (~1489)**, **`_consume_copyahead_window0` (~1527)** — the redraw splice is in here.
  - `_prepare_decode_batch_fixed` (~1757), `_writeback_window` (~1820), `_prealloc_train_kv` (~528).
  - `_barrier_resample` (~1842), `_build_spec_a1` (~1898) — the SBP clone + ancestor map (shared, trusted by the inherit variant).
  - `__init__` flag setup (~105-160); the `_on` mutual-exclusivity list.
- `smcsd/decoupled/draft_server.py`: `_handle_step` (~308, the drafter AR + `rollback` mirror), `_handle_commit` (~509, the `batched_resample_kv` clone).
- `smcsd/core/kernels/fused_resample_kv.py`: `batched_resample_kv` (~66) — the dst<-src KV copy kernel. **Check its guarantees vs. how copy-ahead uses it.**
- `smcsd/core/req_state.py`: `slot_state` `seq_lens`/`kv_allocated_lens` (~111), `allocate_slots` (~211), `free_group_slots` (~298), `rebuild_active_slots` (~343), `from_slot_gather` (the KV alloc, ~390-411), `process_batch_result` (~508).
- `smcsd/decoupled/io_struct.py`: `DraftStepReq.rollback: Union[int, List[int]]`, `DraftCommitResample`.

## 6. What is TRUSTED vs SUSPECT

- **Trusted** (works, don't rewrite): the overlap + cross-barrier spec fire + `_barrier_resample` clone + `_build_spec_a1` ancestor map are all exercised by the **inherit** variant (`SMCSD_COPYAHEAD_RESAMPLE`, 61%) without this failure. Mode A (`SMCSD_BET_DISCARD`) drain+re-draw is the **correct reference** for what re-draw accuracy should look like (~70%).
- **Suspect** (the bug is here): everything that only the REDRAW path touches — `_fire_copyahead_redraw`, the consume splice, the redraw drain in `_commit_spec_standalone`, and the `SpecState.redraw_*` plumbing.

## 7. Acceptance / how to verify a fix

- `SMCSD_COPYAHEAD_REDRAW=1` at K=1, 100q → accuracy **~70%** (within noise of Mode A re-draw's 70%), and **monotone-or-flat** vs K (NOT improving as K rises — that signature must disappear). Throughput is secondary (the design question is whether overlap+re-draw beats the no-overlap 91; but first make it *correct*).
- Flag-off (`SMCSD_COPYAHEAD_REDRAW` unset) must stay byte-identical; `py_compile` clean; keep it in the `_on` mutual-exclusivity list.
- Don't disturb `SMCSD_COPYAHEAD_RESAMPLE`, `SMCSD_BONUS_WINDOWS`, `SMCSD_BET_DISCARD`/`KEEP`, `SMCSD_SPEC_BARRIER`, `SMCSD_BARRIER_BONUS`, or the no-bonus async path.

## 8. Broader context (why we're building this)

This is one variant in a study of "can full-bonus accuracy be had at better-than-Mode-A throughput." The full-bonus frontier is: no-bonus async 66%@135 (ceiling) · Mode A 71%@~107 (full-bonus optimum so far) · SBP 62%@177 (fast, coupling) · lockstep+bonus 73.5%@107. The REDRAW variant tests "overlap (drafter busy through the resample) + re-draw diversity (no coupling)" at per-window resampling — the hypothesis being it lands near 70% accuracy with overlap throughput. **It can't be evaluated until the bug above is fixed.** Design background: `docs/smc/async_bonus_design.md`.

## 9. ROOT CAUSE (Codex gpt-5.5:high, confirmed) — shared-KV corruption

**The clone re-draft overwrites the *survivor's* in-flight KV, because the clone reuses refcount-SHARED cells instead of getting private ones.**

Exact mechanism:
1. At resample, `batched_resample_kv` (`fused_resample_kv.py:56-63`) copies `src`→`dst` block-table entries and **refcount-shares** the inherited window cells `[S, S+G)`. So `dst[S:S+G)` and `src[S:S+G)` are the **same physical KV cells** (NOT a private copy — the comment at `async_scheduler.py:1501-1505` calling them "slack/private" is WRONG; they are shared inherited cells).
2. `_fire_copyahead_redraw` sends `rollback=gamma+1` with `seq_lens=S`, **but `kv_allocated_lens[dst]` is still `S+G`** → `from_slot_gather` allocates **0 new cells** → the clone re-draft writes its fresh window into `dst[S:S+G)` = the **shared** cells.
3. That **overwrites the survivor's carried drafter window**. The survivor's next StepReq then conditions on corrupted KV → garbage tokens. (Drafter side: `async_scheduler.py:1516` + `draft_server.py:322-345` + `fused_resample_kv.py:56-63`.)
4. **Same bug on the verifier side** (`async_scheduler.py:1597-1605`): the splice replaces clone tokens/logprobs but the clone rows still use the cloned req-to-token table where `dst[S:S+G)` aliases `src[S:S+G)`, so `finish_decode` writes survivor and clone verify-KV into the **same cache cells** → race/corruption.

So suspect **#2 (KV-cache)** from §4 is THE bug — **not** the row-alignment (#1) or the FIFO ordering. The K-dependence matches: more resamples → more clone-re-drafts → more survivor-KV corruption.

**Fix direction:** the clone re-draft needs **private / copy-on-write `[S, S+G)` cells**, on BOTH the drafter mirror and the verifier pool. Concretely: before the re-draft, also roll back `kv_allocated_lens[dst]` to `S` (so `from_slot_gather` allocates fresh private cells for `[S, S+G)` instead of reusing the shared ones) — and make sure the drafter's `_handle_step` rollback + the verifier's `from_slot_gather`/req-to-token both point the clone's `[S, S+G)` at newly-allocated cells, not the survivor's. Verify the survivor's `[S, S+G)` is never written by any clone afterward.

**Caveat Codex flagged:** it could not independently confirm the drafter event-loop FIFO ordering (the dispatch loop wasn't in the bundle). It's almost certainly fine (single serialized `recv_pyobj` queue, `draft_server.py:186`), but a fixer should sanity-check that `_handle_step` for the survivor run-ahead always runs before `_handle_commit` for the clone.

Implication for the inherit variant (`SMCSD_COPYAHEAD_RESAMPLE`, 61%): inherit clones **read** the shared window and never re-draft it, so they don't corrupt it — the 61% is a *real* (coupling-limited) number, not this bug.

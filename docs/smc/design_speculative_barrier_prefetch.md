# Speculative Barrier Prefetch (SBP) for the async decoupled SMC engine

> **Status (2026-06-14): IMPLEMENTED & MEASURED — see `report_sbp_prototype.md`.** Result:
> **+32% throughput but −4.4 pt GSM8K accuracy at 1000q** (NOT the accuracy-neutral free
> lunch this doc claims — the "benign one-window coupling" §Accuracy is underestimated; a
> no-resample A/B confirms the machinery is correct and the cost is purely the resample
> adoption-coupling). The architecture below is correct, but three
> mechanisms in the pseudocode were wrong on GPU and are **superseded** by the
> implementation: (1) score the spec window over the *post-rebuild survivors A1*, not the
> drafted set A0 (`_remap_spec_resp` → `_build_spec_a1`/`_spec_a1_source_rows`) — scoring A0
> reads stale KV (illegal memory access) for slots retired onto pre-train-finished
> particles; (2) the event-loop guard *commits* the spec, it does not discard it (discarding
> desyncs survivors), and the finalize-drain is not deferred / never runs mid-train; (3) the
> consume fires window 1 *before* verifying window 0 (else no overlap). The report's
> "diverged from the design doc" section has the full rationale.

**Thesis.** At K=2 the async decoupled SMC scheduler spends *half* its decode windows
in a **barrier stall**: when the verifier reaches a resample barrier it stops
prefetching (resampling reshuffles particle KV, so the next StepReq cannot be fired
until the plan is known), and the drafter — the system bottleneck at ~42 ms/window —
sits idle through the verifier's ~30 ms verify plus the ~1–10 ms resample. SBP fills
that stall by firing the **next train's window-0 DraftStepReq speculatively across the
barrier** (assume no resample) so the drafter computes it *while* the verifier verifies
and resamples. Resampling keeps most particles alive; survivors keep their own
speculative draft, and the existing `batched_resample_kv` frontier-clone copies a
survivor's full KV — *including its speculative window* — into each retired slot for
free (refcounted share, no redraft). A single verifier-side **ancestor remap** rewrites
the received speculative drafts (tokens, logprobs, **and the anchor**) so a retired slot
is scored against the survivor it adopted. SBP is **throughput-only and
accuracy-neutral**: it does not change the resample interval (stays K=2), so it targets
the same ~+11–12% tok/s that larger-K buys (133 → ~148 tok/s) **without** large-K's
2–5 pt accuracy loss — it keeps K=2's 66% @200q / 68.1% @1000q.

---

## Background — the barrier-stall problem

The async engine (`smcsd/decoupled/async_scheduler.py`,
`AsyncDecoupledSMCScheduler._run_decode_train`) runs **trains of K windows**
(K = `SMCSD_RESAMPLE_INTERVAL`, default 2). Per window the drafter (separate process,
GPU1) produces γ+1 draft tokens/particle; the verifier (GPU0) scores them with the
target model and accumulates SMC importance weights (`logprob_diff`). Two mechanisms
give the overlap:

- **Prefetch.** In no-bonus mode the next-window anchor is the drafter's *own*
  (γ+1)-th token (`resp.tokens[:, gamma]`, async_scheduler.py:162), so the verifier
  fires the next window's `DraftStepReq` (`worker.send_step_req`, raw lists) the instant
  it receives the current `StepResp` (async_scheduler.py:159–167), **then** verifies
  locally while the drafter computes the next window.
- **Barrier.** Resampling reshuffles particle KV, so it runs only at the **end** of a
  K-train where the pipeline is **drained** (no StepReq in flight). At the barrier the
  verifier does **not** prefetch (async_scheduler.py:191–196 — the loop exits, then
  `_barrier_resample` fires with nothing in flight). So the **barrier window stalls**:
  the drafter is idle through verify + resample.

**Measured (GSM8K 200q, async K=2, anchor 0.3, N=12 γ=8 temp 0.7 triton + drafter
CUDA graphs, batch 1, 2×A6000)** — from `docs/smc/async_smc_design.md`:

| side | component | per-window |
|---|---|---:|
| drafter | AR loop (9 cuda-graph steps) | **41.6 ms** |
| drafter | total | **42.2 ms** |
| verifier | **recv (drafter-wait)** | **28.0 ms (48%)** |
| verifier | verify + writeback | 30.3 ms (51%) |

With *perfect* overlap, recv would floor at `draft − verify` ≈ 42.2 − 30.3 ≈ **12.5 ms**.
It measures **28 ms** because at K=2 *half* the windows are resample barriers with no
prefetch — they wait the full ~42 ms draft. Averaging overlap (~14 ms) with barrier
(~42 ms) ≈ 28 ms, matching the measurement. The cuda-graph K-sweep confirms the
diagnosis (and bounds the prize):

| K | accuracy | tok/s |
|---:|---:|---:|
| **2** | **66.0%** | **133.1** |
| 4 | 61.0% | 148.6 |
| 8 | 63.5% | 147.3 |

Larger K removes barriers and lifts throughput **133 → 148.6 (+12%, saturating at
K≥4)** — but costs **2–5 pt accuracy** (more resampling delay). SBP's goal: capture that
~+11–12% *barrier-removal* ceiling while holding K=2's resample cadence and 66%
accuracy. (The drafter remains the residual bottleneck — draft 42 ms > verify 30 ms — so
~+12% is the honest ceiling for single-depth speculation; deeper W>1 speculation is
explicitly out of scope, see *Risks*.)

---

## The design

SBP fires the next train's window-0 StepReq across the barrier, overlapping the drafter
computing it with the verifier's verify + collect + dispatch + commit, then consumes and
remaps it as the next train's window-0. **The drafter stays a pure reactor**; all logic
is verifier-side in `async_scheduler.py`, plus a one-line `epoch` echo in
`draft_server.py` and an `epoch` field on the wire. The mechanism is gated behind a new
flag **`SMCSD_SPEC_BARRIER`** (default off) so A/B is clean.

### State carried across trains

```python
# AsyncDecoupledSMCScheduler.__init__
self._epoch = itertools.count(1)
self._spec  = None   # Optional[SpecState]

@dataclass
class SpecState:
    pending: PendingDecodeStep     # the spec window's snapshot (ctx.orig_seq_lens = S+KG)
    active_list_T: List[int]       # pre-resample active slot ids (drives the remap)
    active_t_T: torch.Tensor       # same as a device tensor (writeback target)
    tag: int                       # FIFO tag of the spec StepReq
    epoch: int                     # train counter (fail-fast fence)
    ancestor: Optional[np.ndarray] # a(i): identity for survivors, src for retired; None on no-resample
```

### Step-by-step at a barrier (train T)

Notation: `S` = `orig_seq_lens` at train T start (drained frontier from prior barrier);
`G` = γ+1; `K` = `barrier_k`. `V` = verifier `slot_state.seq_lens[active]`; `D` = drafter
mirror `seq_lens[active]`. At train T's barrier both are drained to `S+KG`.

1. **K-window loop runs unchanged** (async_scheduler.py:146–189), including the in-loop
   prefetch for `w < K−1`. Capture `last_resp` = the loop's final recv'd `StepResp`.

2. **SBP injection** (new, after the loop, *before* `_barrier_resample`):
   - `spec_anchor = torch.from_numpy(last_resp.tokens)[:, gamma].tolist()` — the
     drafter-known (γ+1)-th token, identical to what baseline window-0 would use.
   - `spec_seq_lens = self.slot_state.seq_lens[active_t].tolist()` — read **after** the
     last writeback; equals the drained frontier `S+KG` and the drafter mirror `D`.
   - `spec_batch = self._prepare_decode_batch_fixed(active_t, active_list)` — advances
     `V` `S+KG → S+(K+1)G` uniformly **and** allocates the verify-side target KV for the
     spec window at each row's *own* `req_pool_index`. **This snapshot is load-bearing:**
     it freezes `ctx.orig_seq_lens = S+KG`, which `finish_decode` later uses to place
     verify cache_locs. Never rebuild it post-resample.
   - `spec_tag = next(self._tag)`;
     `worker.send_step_req(active_list, spec_anchor, spec_seq_lens, tag=spec_tag, epoch=E0)`.
   - Stash `self._spec = SpecState(pending=PendingDecodeStep(spec_batch,
     spec_batch.spec_info.decode_ctx, spec_tag, epoch=E0), active_list_T=active_list,
     active_t_T=active_t, tag=spec_tag, epoch=E0, ancestor=None)`.

3. **Resample, concurrent with the spec draft** (`_barrier_resample`, unchanged except
   for ancestor capture). Because `send_step_req(spec)` was emitted **before**
   `send_commit` on the single FIFO ZMQ channel, the drafter dequeues `_handle_step(spec)`
   *strictly before* `_handle_commit(resample)`: it drafts the spec window (advancing
   `D` to `S+(K+1)G` and writing spec KV into **every** active slot, including
   future-retired ones), replies, *then* the commit's `batched_resample_kv` copies the
   survivor src KV `[0:seq_lens[src]=S+(K+1)G]` (which **includes** the survivor's spec
   window) into each retired dst and sets `seq_lens[dst]=seq_lens[src]=S+(K+1)G`.
   - **Ancestor map** (built locally from the plan the scheduler already holds — no
     `core/scheduler.py` change): `a = np.arange(max_slots)`; `a[plan.dst_slots] =
     plan.src_slots`; `self._spec.ancestor = a`. On `n_jobs == 0`, leave `ancestor=None`
     (identity). Survivors get `a(i)=i`; retired slot i gets `a(i)=j` = the survivor it
     cloned.

4. **Train T+1 consume + remap** (new branch at the top of `_run_decode_train`): the
   next train's window-0 StepReq is already in flight, so do **not** re-prepare/re-send
   window-0. Instead:
   1. `resp = client.recv_step_resp()`.
   2. Assert `resp.tag == spec.tag` **and** `resp.epoch == spec.epoch` (FIFO + fence).
   3. **Remap** via `_remap_spec_resp` (below): rewrite `resp.tokens`, `resp.logprobs`,
      **and** the spec batch's anchor `spec.pending.batch.spec_info.verified_id` by the
      same ancestor gather. Identity when `ancestor is None`.
   4. `result = worker.finish_decode(spec.pending, remapped_resp)`;
      `self._writeback_window(result, spec.active_t_T)`.
   5. Fire the in-loop prefetch for `w=1` off `remapped_resp.tokens[:, gamma]`.
   6. `self._spec = None`; continue the K-loop from `w=1` over the post-rebuild active
      set (`_prepare_decode_batch_fixed` exactly as today).

### Ancestor-adoption frontier-clone (the key trick)

Resampling kills only low-weight particles; each retired slot `i` adopts survivor `j`'s
*full* KV through the drafter's current frontier via the existing
`batched_resample_kv`. Because the drafter drafted the speculative window *before*
applying the commit, survivor `j`'s KV already includes its speculative window — so
cloning `j → i` copies `j`'s speculative window into `i` **for free**. No redraft.

### Why the verifier remap is mandatory (the load-bearing correction)

`finish_decode` (worker.py:314–421) scores `all_tokens = [x0] + resp.tokens[:, 0..γ−1]`
where **`x0 = draft_input.verified_id`** (worker.py:347) — the anchor frozen in
`spec.pending.batch.spec_info` at SBP-prep time, read at each row's *own* pre-resample
`req_pool_index` / cache_locs (worker.py:327–334). For a **survivor** (`a(i)=i`) its own
spec tokens are correct against its own KV — identity remap, a no-op. For a **retired**
slot `i` (`a(i)=j`): the resample clone already copied `verified_ids/all_token_ids/
finished_mask/token_counts` `dst←src` (scheduler.py:201–206) **and** `batched_resample_kv`
copied `j`'s KV (incl. spec window) into `i`'s target block table — so `i`'s committed
history and verify context are now `j`'s. But `resp.tokens[row_of(i)]` is still `i`'s own
*garbage* draft, and `x0` is still `i`'s own anchor. We must gather `j`'s drafted tokens,
`j`'s draft logprobs, **and `j`'s anchor** so that the target re-score (of `j`'s tokens at
slot `i`'s now-`j` KV) and the draft logprob form a *matched* proposal/target pair over
`j`'s trajectory. The remap is a single numpy/torch advanced-index on the `(bs, γ+1)`
token/logprob arrays and the `(bs,)` anchor, fenced by epoch, identity when no resample.

> **Correctness fix applied (verify verdict, holds=false on "remap correctness" /
> "resample-fires-vs-not").** The earlier sketch remapped only `tokens` and `logprobs`
> and left `x0 = verified_id` unremapped. Because the anchor is *sampled* per particle at
> `anchor_temperature` (draft_server.py:392–396; gate config temp 0.7 / anchor 0.3),
> `A_i ≠ A_j` w.h.p. after a γ=8 divergent window, and `x0` is re-fed every window at a
> position **outside** the cloned prefix (`assign_smc_cache_locs_kernel`, worker.py:327),
> so it is load-bearing in the score (row-0 predicts `tokens[:,0]` *conditioned on* `x0`).
> A retired slot scored as `[A_i, j_tok0, j_tok1, …]` over `j`'s KV yields a mismatched
> score/draft pair, corrupting the first post-resample weight increment and poisoning the
> resample distribution — a silent accuracy-gate violation. **`verified_id` MUST be
> remapped with the same `source_rows` gather.** See `_remap_spec_resp` below.

### `_remap_spec_resp` (new helper)

```python
def _remap_spec_resp(self, resp, spec):
    if spec.ancestor is None:           # n_jobs == 0 → identity, no-op
        return resp, spec.pending.batch.spec_info.verified_id
    pos_of = {slot: r for r, slot in enumerate(spec.active_list_T)}
    source_rows = np.fromiter(
        (pos_of[spec.ancestor[slot]] for slot in spec.active_list_T),
        dtype=np.int64, count=len(spec.active_list_T),
    )
    resp.tokens   = resp.tokens[source_rows]      # survivors: identity; retired: ancestor's row
    resp.logprobs = resp.logprobs[source_rows]
    src_t = torch.as_tensor(source_rows, device=self.device)
    remapped_anchor = spec.pending.batch.spec_info.verified_id[src_t]   # x0 ← ancestor's anchor
    spec.pending.batch.spec_info.verified_id = remapped_anchor
    return resp, remapped_anchor
```

`spec.ancestor[slot]` is `slot` for survivors (identity gather) and the survivor's slot
for retired; `pos_of` maps a slot to its row in the active list. A survivor `j` is always
present in `active_list_T` (finished slots are dropped only at the post-scoring
`rebuild_active_slots`, async_scheduler.py:195), so the gather never misses an ancestor.
Multi-dst-one-src (two retired slots cloning one survivor) both gather the same survivor
row — correct SMC duplicate semantics.

### ASCII timeline — current barrier stall vs SBP overlap (K=2, one train→train boundary)

```
LEGEND  D = drafter (GPU1)   V = verifier (GPU0)   ░ = idle/stall

CURRENT (barrier window stalls; drafter idle through verify+resample):
  D: [draft w0 ~42ms][draft w1 ~42ms]░░░░░░░░░░░░░░░░░░░░░░░░░░░[draft T+1 w0 ~42ms]
  V:        [verify w0 ~30ms][verify w1 ~30ms][collect+dispatch+commit ~1-10ms]
                              ^prefetch w1            ^NO prefetch — drafter waits
  barrier window recv-wait ≈ 42ms (full draft, zero overlap)

SBP (spec StepReq fired across the barrier; drafter draws T+1 w0 during verify+resample):
  D: [draft w0 ~42ms][draft w1 ~42ms][draft T+1 w0 (SPEC) ~42ms][draft T+1 w1 ~42ms]
  V:        [verify w0][verify w1][collect+dispatch+commit][consume+remap+verify T+1 w0]
                       ^prefetch  ^SPEC StepReq fired   ^ancestor map     ^writeback
                        w1         (before send_commit)  built post-dispatch
  barrier window recv-wait → overlap floor (~12-14ms): spec draft hidden behind V's
  ~30ms verify + ~1-10ms resample.
```

---

## Components to modify

| file:symbol | current | change | est LOC |
|---|---|---|---:|
| `io_struct.py` : `DraftStepReq` / `DraftStepResp` | `tag: int = 0` only (lines 78, 87) | add `epoch: int = 0` after `tag` on both. Pure fail-fast fence (train counter); default 0 keeps lockstep + pipelined callers wire-compatible. `DraftCommitResample` unchanged (dst/src already encode ancestry 1:1). | 2 |
| `draft_server.py` : `SMCDraftServer._handle_step` | returns `DraftStepResp(..., tag=msg.tag)` (line 428) | echo the epoch: `DraftStepResp(..., tag=msg.tag, epoch=msg.epoch)`. **The only drafter change** — it stays a pure reactor that cannot distinguish speculative from committed. `_handle_commit` byte-for-byte unchanged. | 1 |
| `worker.py` : `DraftEngineClient.send_step` / `DecoupledSMCWorker.send_step_req` / `start_decode` / `PendingDecodeStep` / `finish_decode` | tag threaded, no epoch (lines 96–108, 271–305, 144–152, 314) | thread optional `epoch: int = 0` through `send_step` (→ `DraftStepReq(..., epoch=epoch)`), `send_step_req`, `start_decode`; add `epoch: int = 0` to `PendingDecodeStep`; in `finish_decode`, after the tag assert (line 336–340) add `if resp.epoch != pending.epoch: raise RuntimeError(...)`. **No change to the verify math.** | 15 |
| `async_scheduler.py` : `__init__` | `self._tag = itertools.count(1)` (line 67) | add `self._epoch = itertools.count(1)`, `self._spec = None`, the `SpecState` dataclass, and read `SMCSD_SPEC_BARRIER`. | 14 |
| `async_scheduler.py` : `_run_decode_train` | cold window-0 prep + K-loop + barrier (lines 123–214) | **entry branch:** if `self._spec` set, recv + tag/epoch-assert + `_remap_spec_resp` the carried resp, adopt `spec.pending` as window-0, `finish_decode` + `_writeback_window` over `spec.active_t_T`, fire `w=1` prefetch off remapped `tokens[:,gamma]`, clear `self._spec`, continue from `w=1`; else today's cold path with `epoch=E0`. Capture `last_resp`. **After the K-loop, before `_barrier_resample`:** SBP injection (spec_anchor / spec_seq_lens / spec_batch / `send_step_req(..., epoch=E0)` / stash `self._spec`). Gate the whole spec path on `SMCSD_SPEC_BARRIER`. | 55 |
| `async_scheduler.py` : `_barrier_resample` | collect → dispatch → `send_commit` (lines 247–256) | after `dispatch_resample_batch`, build and store the ancestor map onto `self._spec`: `a = np.arange(max_slots, dtype=int64); a[plan.dst_slots]=plan.src_slots; self._spec.ancestor=a`. On `n_jobs==0` leave `ancestor=None`. collect/dispatch/`send_commit` byte-for-byte identical. | 8 |
| `async_scheduler.py` : `_remap_spec_resp` (new helper) | — | gather `resp.tokens`, `resp.logprobs`, **and** `spec.pending.batch.spec_info.verified_id` by `source_rows[r]=pos_of[ancestor[active_list_T[r]]]`; identity / no-op when `ancestor is None`. numpy advanced index + one torch gather. | 18 |
| `async_scheduler.py` : `_event_loop` | admits prefill / pauses / idles between trains (lines 84–109) | **PREFILL-ADMISSION / MEMBERSHIP DRAIN GUARD** (see *Risks* — load-bearing). At the top of each iteration, *after* `process_input_requests` and *before* the `_engine_paused` check, the prefill block, and the `is_empty` branch: if `self._spec` is set and (engine paused, or `waiting_groups`, or `slot_state.is_empty()`, or the about-to-run train's pre-rebuild active set ≠ `self._spec.active_list_T`), drain the in-flight spec resp via `recv_step_resp()` + tag/epoch-assert and set `self._spec = None`. | 14 |
| `core/scheduler.py` : `dispatch_resample_batch` | builds plan from `plan.dst_slots`/`src_slots` (lines 176–206) | **no change.** The async scheduler already holds the plan locally and derives `a(i)` itself — keeps the surgical diff + per-cohort isolation. | 0 |
| `core/kernels/fused_resample_kv.py` : `batched_resample_kv` | two-phase dec/inc-ref KV clone | **no change.** Document that with `dst_alloc_lens = kv_allocated_lens[dst] = S+(K+1)G`, Phase 1 dec-refs the retired slot's *own* spec window (→ `to_free`) and Phase 2 inc-refs the survivor's spec window into the retired row. (See *Refcount* §.) | 0 |
| `core/kernels/fused_collect.py` : `batched_collect_fused` | per-group ESS + systematic resample → flat plan | **no change.** The spec StepReq mutates no weight state before collect, so the plan (dst/src/mask) is bit-identical to baseline. | 0 |

**Net wire cost:** one `epoch: int = 0` field (~8 bytes/round), negligible vs the
`(bs, γ+1)` arrays already sent. Default `epoch=0` keeps every existing call site
(lockstep `_forward_decode` at worker.py:307, the pipelined scheduler) untouched — only
the async scheduler ever sets a nonzero epoch.

---

## seq_lens consistency

**Invariant** (draft_server.py:315): at the instant *any* `DraftStepReq` is dequeued by
`_handle_step`, `D == req.seq_lens`. **Key fact confirmed in code:** `prepare_for_decode`
is the *only* place `V` advances; `process_batch_result` / `_writeback_window` do **not**
touch `seq_lens` (async_scheduler.py:225–245). `DraftStepReq` carries
`ctx.orig_seq_lens_cpu` — the **pre-advance** value (worker.py:302).

**Train T, K=2** (`S` = frontier at train start, `G` = γ+1):

| step | action | V | D |
|---|---|---|---|
| t0 | drained barrier entry | `S` | `S` |
| w0 prep | `_prepare_decode_batch`→`prepare_for_decode` advances V; `start_decode` sends `req.seq_lens=S`, tag t_a, epoch E0 | `S+G` | `S` |
| drafter w0 | assert `D(S)==S` OK; `D→S+G`; reply | `S+G` | `S+G` |
| w0 recv (not last) | in-loop prefetch reads `V=S+G`, sends `req.seq_lens=S+G`, tag t_b | `S+G` | `S+G` |
| drafter w1 | assert `D(S+G)==S+G` OK; `D→S+2G`; reply | `S+G` | `S+2G` |
| w0 finish+writeback | seq_lens untouched; `_prepare_decode_batch_fixed` advances V `S+G→S+2G` | `S+2G` | `S+2G` |
| w1 recv (LAST/barrier) | consume resp(t_b); **no in-loop prefetch**; finish+writeback (seq_lens untouched) | `S+2G=S+KG` | `S+2G=S+KG` |

→ **drained, both agree at `S+KG`.**

**SBP injection (new):**

| step | action | V | D |
|---|---|---|---|
| spec prep | `spec_anchor=last_resp.tokens[:,γ]`; `spec_seq_lens=V=S+KG`; `_prepare_decode_batch_fixed` advances V `S+KG→S+(K+1)G` for ALL active slots + allocs verify-side spec KV; `send_step_req(req.seq_lens=S+KG, tag t_c, epoch E0)` | `S+(K+1)G` | `S+KG` |
| drafter spec | assert `D(S+KG)==S+KG` OK; `D→S+(K+1)G` on **every** active slot (survivors AND future-retired); reply(t_c) | `S+(K+1)G` | `S+(K+1)G` |

**Resample (concurrent with the spec draft):**

- `dispatch_resample_batch` reads dst/src from `V`. `V[dst]=V[src]`: every active `V`
  slot is `S+(K+1)G`, so `S+(K+1)G ← S+(K+1)G` — a **no-op on V** (consistent).
  `kv_allocated_lens` copied likewise. `send_commit`.
- drafter `_handle_commit` (FIFO: **after** spec `_handle_step`): `batched_resample_kv`
  copies src block table `[0:seq_lens[src]=S+(K+1)G]` (incl. survivor spec window at
  `[S+KG..S+(K+1)G)`) into dst; then `D[dst]=D[src]=S+(K+1)G`. Survivors already
  `S+(K+1)G`; retired set to src's `S+(K+1)G`. → **`V=D=S+(K+1)G` everywhere; mirror
  consistent.**

**Train T+1:**

- window-0 is the carried spec window (not re-prepared, not re-sent). recv spec
  resp(t_c, E0) — tag+epoch assert **pass**. Remap by `a(i)`. `finish_decode` reads
  `ctx.orig_seq_lens=S+KG` from `spec_batch` → verify cache_locs at `[S+KG..S+(K+1)G)`,
  exactly where the drafter wrote spec KV. Writeback does not move `V`.
- `w=1` of T+1 in-loop prefetch reads `V=S+(K+1)G`, sends `req.seq_lens=S+(K+1)G`;
  drafter asserts `D(S+(K+1)G)==S+(K+1)G` **pass**. Cycle continues with `S' = S+(K+1)G`.

**Two tripwires hold at every point a StepReq is actually sent:** draft_server.py:315
(`D == req.seq_lens`) and the verify reading `ctx.orig_seq_lens` from the SBP-prep
`spec_batch` for cache_locs. The only `V≠D` moment is the no-StepReq gap between
spec-fire and T+1 window-0, closed before any new StepReq fires.

> **Verdict (holds=true) — "seq_lens parity at the SBP barrier".** No divergence found.
> **Critical guard:** `spec_batch` MUST be the `_prepare_decode_batch_fixed` result
> captured at SBP-prep (`ctx.orig=S+KG`); never rebuild it post-resample (a wrong `orig`
> would silently mis-place verify cache_locs). Add a debug assert
> `ctx.orig_seq_lens == spec_seq_lens` at consume time. The frontier-mismatch hazard
> (drafter-T+1 at `S+(K+1)G` while verifier `V` stays `S+KG` until SBP-prep) is closed by
> ordering: `V` is advanced to `S+(K+1)G` by `_prepare_decode_batch_fixed` **before**
> `dispatch_resample_batch` reads `V[dst]/V[src]`, so `V[dst]=V[src]` is a no-op at
> `S+(K+1)G`, not a corrupting `S+KG` copy.

---

## Refcount / KV-sharing correctness

SBP **leverages** the existing refcounted sharing (`SMCRefCountedTokenAllocator`
`inc_ref`/`dec_ref_and_free`; `copy_block_table`) and the two-phase
`batched_resample_kv` — **zero new refcount code**. For job `(dst=i, src=j)`:

1. During the spec `_handle_step`, **every** active slot (incl. future-retired) allocs
   its *own* spec-window KV: `SMCDecodeContext.from_slot_gather` allocs the new G blocks
   (refcount=1) and `kv_allocated_lens[i] = S+(K+1)G`. So retired slot `i` holds a
   block table of length `S+(K+1)G` whose tail G blocks are `i`'s own garbage, each
   refcount 1 (owned only by `i`).
2. **Phase 1** (`fused_resample_kv.py`): reads `req_to_token[i, 0:dst_alloc]` where
   `dst_alloc = kv_allocated_lens[dst] = S+(K+1)G`, `atomic_add(refcount, -1)` over ALL
   of `i`'s KV **including `i`'s spec tail**. Those tail blocks drop 1→0, captured in
   `dec_out`, returned in `to_free`, freed by `allocator.free`. **Retired slot's garbage
   spec KV reclaimed — no leak.**
3. **Phase 2**: reads `req_to_token[j, 0:src_seq_len = seq_lens[src] = S+(K+1)G]`
   (survivor's, **including** `j`'s spec window), writes into `i`'s table,
   `atomic_add(refcount, +1)` on each. `j`'s spec blocks get +1 per retired adopter. Net
   for a survivor spec block shared by `j` and M adopters: refcount = 1(`j`) + M = M+1
   live owners. Triton `atomic_add` is race-safe under multiple jobs sharing one src
   (each `(d_i, j)` is a separate program; concurrent +1 on one slot is atomic).
4. **Shared-prefix blocks** `[0:S+KG]` that `i` and `j` already co-owned: dec'd once in
   Phase 1 (`i` releasing) and re-inc'd once in Phase 2 (`i` adopting `j`'s identical
   handle) — **net zero, refcount conserved.**

**No double-free / no transient zero:** `to_free` is computed in Python *after*
`cuda.synchronize` from the **final** `refcount[dec_out]==0`, then `torch.unique`. Because
`atomic_add` is commutative, the final refcount is independent of inter-job race order;
a block ends at 0 only if captured in `dec_out` and not re-inc'd. `torch.unique` prevents
double-free when the same dying block is captured by two retiring slots. The single
invariant the kernel relies on — **dst distinct, dst disjoint from src** — is asserted at
scheduler.py:179–182 and guaranteed by the cumsum-deterministic systematic-resample
collect.

**Verifier side is symmetric:** `dispatch_resample_batch`'s `batched_resample_kv`
(scheduler.py:189–196) runs over `slot_state` with `dst_alloc_lens = kv_allocated_lens[dst]
= S+(K+1)G` (SBP-prep advanced the verifier's `kv_allocated_lens`, req_state.py:398) and
`src_seq_lens = seq_lens[src] = S+(K+1)G`, so the verifier frees the retired slot's own
verify spec KV and adopts the survivor's symmetrically. **Mirror (verifier free-set ==
drafter free-set) holds** because `dst_alloc` and `src_len` derive from the same
uniformly-advanced `S+(K+1)G` on both sides. Crash safety: `_handle_close`
(draft_server.py:513–540) dec-refs every active slot's `kv_allocated_len` on shutdown, so
an elevated survivor refcount is reclaimed — no permanent leak.

> **Verdict (holds=true) — "refcount safety at the SBP speculative frontier".** The real
> Triton kernel was run on CUDA against the SBP barrier state, including the multi-clone
> case (one survivor → two retired): prefix returns to its original count, survivor spec
> blocks end at M+1, each retired slot's own spec blocks hit 0 and appear in `to_free`
> exactly once. 400 randomized valid-plan trials matched a pure-Python reference on final
> refcounts and free-sets, zero failures. **No leak, no double-free, no underflow.** The
> only breaking config is an invalid plan (dst∩src≠∅ or non-distinct dst) — impossible
> here (asserted scheduler.py:179–182).

---

## Accuracy argument

**Claim:** adopting the survivor's run-ahead is **distribution-equivalent** to redrafting
it from a fresh clone of the survivor — so SBP matches async-K=2 accuracy within noise.

The SMC weight increment per window is `logprob_diff_i = Σ_j (log p_target(x_j) −
log q_draft(x_j))` (worker.py:384), accumulated into `interval_weights/log_weights`. At
the barrier, collect normalizes, computes ESS, does systematic resampling
(`E[n_i]=N·W_i`), emits `(dst,src)`, and **zeros the resampled rows' weights**. dispatch
clones the survivor's full state into the retired slot. The drafter's window draft is a
pure function of `(anchor, KV prefix)` plus its multinomial draws:
`x^win ~ q(· | anchor, KV)`.

**Two orderings, same law:**

- **Baseline async-K=2 = "resample THEN propagate":** at frontier `F = S+K·G`, select
  ancestors on `interval_weights`, clone states at `F`, then T+1 window-0 draws a *fresh
  independent* `x ~ q(· | src_anchor, src_KV)` for every offspring.
- **SBP = "propagate THEN resample":** every slot drafts the spec window `x^spec ~
  q(· | state at F)` **before** resample; ancestor selection runs on the **same**
  `interval_weights` at `F` (the spec window's `logprob_diff` is computed in T+1 and added
  **after** resample, so it is **not** in the resample weights); each retired offspring
  then **adopts** the single `x^spec` of its ancestor via the KV clone + the verifier-side
  ancestor remap that gathers src's tokens, draft logprobs, **and anchor**, so its paired
  weight is the correctly matched pair for that exact draw.

**Why unbiased:** the resample weights are `F`-measurable — identical `interval_weights`
at `F` in both schemes, the same single collect call per barrier, the same `step_counter`
RNG stream (SBP adds/removes no collect call). The spec draft's RNG lives on the
drafter's GPU1 (`torch.multinomial`) while the resample CDF draw is `tl.rand` on the
verifier's GPU0, so **the spec draft cannot perturb the resample plan**. Hence the
ancestor multiplicities `n_i` are drawn from exactly the baseline distribution, and
"propagate-then-resample" is a textbook-valid SMC ordering. The marginal law of each
offspring's window-0 state is, in both schemes, a single draw from `q(· | ancestor state
at F)`: for a future-survivor the anchor + KV prefix the spec window conditioned on are
**bit-identical** to what a post-resample fresh redraft would condition on (resample is a
no-op for a survivor, `a(src)=src`). So every per-particle expectation — and thus the SMC
estimator's expectation and the finalize argmax-by-log-weight marginal — is preserved.

> **Verdict (holds=true) — "accuracy-neutrality / adopt-vs-redraft equivalence".** No
> bias-producing scenario found. **One honest correction:** the schemes are *not*
> identical in **joint** law — baseline offspring of a common ancestor are conditionally
> independent from window-0; SBP offspring are *perfectly correlated* (literally the same
> tokens) through the adopted window, diverging only at T+1 window-1. This is a benign
> one-window diversity/coupling effect ("copy now, perturb later"), **not** a
> distributional bias. With anchor_temperature=0.3 already mode-seeking and the gate at
> n=200 (σ≈3.4 pt), it is far below detection. On **no-resample** barriers (n_jobs==0, the
> common case) the spec window draw equals what baseline window-0 would draw at the
> identical RNG offset and un-resampled state → **token streams are byte-identical** (a
> clean deterministic check). The only way to turn this into a *real* bias is to get the
> matched-pair remap wrong — which is exactly why **the anchor `x0` must be remapped along
> with tokens+logprobs** (applied above).

**Residual risk:** under a *pathological* regime (very aggressive resampling retiring most
particles every barrier **and** very low draft/anchor temperature so the shared run-ahead
does not re-diversify), the one-window coupling could shrink effective sample size enough
to nudge accuracy. The gate operating point (N=12, γ=8, draft temp 0.7, anchor 0.3, K=2,
default ESS threshold, ≲50% of barriers resampling) is far from this; the forced-high-
resample test below bounds it empirically.

---

## Testing plan

All tests behind the new flag **`SMCSD_SPEC_BARRIER`** (default off) for a clean A/B. The
shared eval recipe (no-bonus async, anchor 0.3, K=2, N=12 γ=8 temp 0.7 triton + drafter
CUDA graphs). **Replace the run/eval entrypoints below with the repo's actual eval script
and flags** (`smcsd/eval/…` / the `--mode smc_async` harness referenced in
`docs/smc/async_smc_design.md`) — discover them first with
`rg -n "smc_async|--drop-bonus|SMCSD_ANCHOR_TEMP" --type py` rather than guessing.

Common env:

```bash
export SMCSD_DROP_BONUS=1
export SMCSD_ANCHOR_TEMP=0.3
export SMCSD_RESAMPLE_INTERVAL=2
export SMCSD_DRAFT_CUDA_GRAPH=1
# baseline: SMCSD_SPEC_BARRIER=0   |   SBP: SMCSD_SPEC_BARRIER=1
```

**(a) GSM8K accuracy gate (the GATE).** Baseline vs SBP must match within noise.

```bash
# Baseline (must reproduce 66% @200q)
SMCSD_SPEC_BARRIER=0 python -m smcsd.eval.gsm8k --mode smc_async \
  --num-questions 200 --num-particles 12 --gamma 8 --temperature 0.7 \
  --drop-bonus --anchor-temp 0.3 --attn triton 2>&1 | tee /tmp/sbp_base_200.log
# SBP ON (must match within σ≈3.4pt of 66%)
SMCSD_SPEC_BARRIER=1 python -m smcsd.eval.gsm8k --mode smc_async \
  --num-questions 200 --num-particles 12 --gamma 8 --temperature 0.7 \
  --drop-bonus --anchor-temp 0.3 --attn triton 2>&1 | tee /tmp/sbp_on_200.log
# Pass: |acc_on - acc_base| <= ~3.4pt. Then 1000q must hold ~68.1%:
SMCSD_SPEC_BARRIER=1 python -m smcsd.eval.gsm8k --mode smc_async \
  --num-questions 1000 --num-particles 12 --gamma 8 --temperature 0.7 \
  --drop-bonus --anchor-temp 0.3 --attn triton 2>&1 | tee /tmp/sbp_on_1000.log
```

**(b) seq_lens-assertion-never-fires smoke.** Run 50q with SBP on and `grep` for the
drafter divergence error (draft_server.py:316) and the tag/epoch errors — must be zero.

```bash
SMCSD_SPEC_BARRIER=1 python -m smcsd.eval.gsm8k --mode smc_async --num-questions 50 \
  --num-particles 12 --gamma 8 --temperature 0.7 --drop-bonus --anchor-temp 0.3 \
  --attn triton 2>&1 | tee /tmp/sbp_smoke.log
! grep -E "seq_lens divergence|tag mismatch|epoch|Expected DraftPrefillResp" /tmp/sbp_smoke.log
```

**(c) No-resample-barrier == plain overlap (byte-identical determinism).** Force ESS so
high that *no* barrier resamples (`n_jobs==0` always → `ancestor=None` → identity remap):
seed fixed, raise the ESS threshold above N so collect never fires. Assert the SBP token
stream is **byte-identical** to the K=2 baseline (same RNG offset, same un-resampled
state). Unit-test `_remap_spec_resp` in isolation first: synthetic `resp` + known
`(dst,src)` plan + `active_list_T`; assert (i) survivors gather identity, (ii) retired
gather survivor row, (iii) `ancestor=None` is a no-op, (iv) multi-dst-one-src both map to
the survivor, (v) **the anchor `verified_id` is gathered by the same `source_rows`**.

```bash
python -m pytest smcsd/decoupled/tests/test_spec_barrier_remap.py -q   # new unit test
# determinism: seeded, ESS threshold > N so no resample fires
SMCSD_SPEC_BARRIER=0 SMCSD_SEED=0 SMCSD_ESS_THRESHOLD=999 python -m smcsd.eval.gsm8k \
  --mode smc_async --num-questions 20 --dump-tokens /tmp/tok_base.jsonl ...
SMCSD_SPEC_BARRIER=1 SMCSD_SEED=0 SMCSD_ESS_THRESHOLD=999 python -m smcsd.eval.gsm8k \
  --mode smc_async --num-questions 20 --dump-tokens /tmp/tok_sbp.jsonl ...
diff /tmp/tok_base.jsonl /tmp/tok_sbp.jsonl && echo "BYTE-IDENTICAL OK"
```

**(d) Resample-fires check.** Force a high resample rate (low ESS threshold) so most
barriers retire particles; assert (i) zero seq_lens/tag/epoch crashes, (ii) retired slots
produce valid weights (no NaN/inf in `logprob_diff`), (iii) accuracy stays within noise.
Add a regression assert: for a retired slot `i` adopting `j`, the post-remap scored pair
equals a synthetic fresh-clone of `j` drafting w0 (same `logprob_diff`).

```bash
SMCSD_SPEC_BARRIER=1 SMCSD_ESS_THRESHOLD=0.95 python -m smcsd.eval.gsm8k \
  --mode smc_async --num-questions 200 --num-particles 12 --gamma 8 --temperature 0.7 \
  --drop-bonus --anchor-temp 0.3 --attn triton 2>&1 | tee /tmp/sbp_highresample.log
! grep -E "nan|inf|divergence|mismatch" /tmp/sbp_highresample.log
```

**(e) KV mem-leak check.** The scheduler's idle self-checks
(`_check_radix_cache_memory` / `_check_req_pool`, run in `self_check_during_idle`,
async_scheduler.py:109) must stay clean after a full SBP run — every spec-window KV slot
adopted or freed, refcounts back to baseline at idle. Run to completion and assert no
leak warning/error and the req pool returns to full.

```bash
SMCSD_SPEC_BARRIER=1 python -m smcsd.eval.gsm8k --mode smc_async --num-questions 200 \
  ...  2>&1 | tee /tmp/sbp_leak.log
! grep -Ei "leak|req pool|radix.*mismatch|refcount" /tmp/sbp_leak.log
# Optional: diff verifier free-set vs drafter free-set on a forced-resample barrier
#   (add a debug dump under SMCSD_SPEC_BARRIER_DEBUG=1) — must be identical.
```

**(f) Throughput measurement (133 → ~148 target).** Compare tok/s baseline vs SBP at
200q with **`SMCSD_TIMING=0`** (TIMING perturbs the drafter and inflates recv). Expect
~+8–12% (≈143–148 tok/s). Use `SMCSD_TIMING=1` only as a diagnostic to confirm per-window
recv-wait drops from ~28 ms toward ~14 ms.

```bash
for f in 0 1; do
  SMCSD_TIMING=0 SMCSD_SPEC_BARRIER=$f python -m smcsd.eval.gsm8k --mode smc_async \
    --num-questions 200 --num-particles 12 --gamma 8 --temperature 0.7 \
    --drop-bonus --anchor-temp 0.3 --attn triton 2>&1 | tee /tmp/sbp_tput_$f.log
done
grep -E "tok/s|throughput" /tmp/sbp_tput_0.log /tmp/sbp_tput_1.log
```

**(g) Determinism / seed check (adopt-vs-redraft equivalence).** Across ≥3 seeds, SBP
accuracy ≈ baseline accuracy (mean within noise) — confirms adoption is statistically
equivalent to redraft-from-clone, not just on one seed.

```bash
for s in 0 1 2; do
  SMCSD_SEED=$s SMCSD_SPEC_BARRIER=1 python -m smcsd.eval.gsm8k --mode smc_async \
    --num-questions 200 ... 2>&1 | tee /tmp/sbp_seed_$s.log
  SMCSD_SEED=$s SMCSD_SPEC_BARRIER=0 python -m smcsd.eval.gsm8k --mode smc_async \
    --num-questions 200 ... 2>&1 | tee /tmp/base_seed_$s.log
done   # compare per-seed deltas; mean delta within σ
```

---

## Implementation order

Flag **off by default** the whole way; each step is independently verifiable.

1. **Epoch fence as pure plumbing (no behavior change).** Add `epoch` to
   `DraftStepReq`/`DraftStepResp`; echo at draft_server.py:428; thread through
   `send_step`/`send_step_req`/`start_decode` + `PendingDecodeStep`; assert in
   `finish_decode` after the tag assert. Add `self._epoch` and emit a per-train epoch.
   **Verify:** existing async K=2 GSM8K 200q still reads 66%, zero epoch/tag asserts fire
   (fence inert). Smoke (b).
2. **`SpecState` + `self._spec=None` + `_remap_spec_resp` in isolation.** Unit-test the
   remap (test c-unit): survivors identity, retired→survivor row, `None` no-op,
   multi-dst-one-src, **anchor gathered**.
3. **Ancestor capture in `_barrier_resample`** (`a=arange; a[dst]=src`; store on
   `self._spec`; `None` on `n_jobs==0`). Assert `a[src]==src` and dst∩src=∅ (already
   asserted scheduler.py:180).
4. **Spec FIRE only, no consumption.** At the next train top, recv + tag/epoch-assert +
   **discard** the spec resp and fall back to today's cold window-0. Add the `_event_loop`
   drain guard. This isolates the hardest invariants (FIFO ordering: spec StepReq before
   `send_commit`; seq_lens parity). **Verify:** 200q accuracy **unchanged** (spec work
   discarded = pure overhead), zero seq_lens/tag/epoch crashes, and the prefill-admission
   guard fires cleanly under short `max_new_tokens` (test b, d crash-free path).
5. **Wire ADOPTION — n_jobs==0 fast path** (identity remap, most barriers): consume the
   carried spec resp as window-0, continue from `w=1`. **Verify:** 200q accuracy returns
   to baseline (spec work now used), tok/s rises; **byte-identical token diff** vs K=2
   baseline on no-resample trains (test c).
6. **Enable n_jobs>0 remap** (retired slots adopt survivor's spec columns + anchor).
   **Verify** under forced-high-resample (test d) that retired slots produce valid weights
   and accuracy is neutral; exercise the membership-change guard with back-to-back groups.
7. **FULL GATE.** GSM8K 200q matches async K=2 (66% within ~3.4 pt) and 1000q holds
   ~68.1% (test a); tok/s ~143–148 with `SMCSD_TIMING=0` (test f); multi-seed equivalence
   (test g); leak check clean (test e). Then default-on (or keep flag-gated per repo
   policy).

---

## Risks & alternatives rejected

**R1 — Prefill-admission / group-churn crash (load-bearing guard).** *Verdict
(holds=false) on "FIFO + tag ordering".* After the barrier fires the spec StepReq and
stores `self._spec`, control returns to `_event_loop`. If `recv_requests()` admits prefill
(`waiting_groups` truthy, async_scheduler.py:94), the verifier calls `prefill()` →
`send_pyobj(DraftPrefillReq)` then `self._recv(DraftPrefillResp)` (worker.py:65–94). The
drafter already pushed the spec `DraftStepResp(tag=t_c)` ahead of any `DraftPrefillResp`,
so `_recv` pops the spec resp, fails its `isinstance` check, and raises *"Expected
DraftPrefillResp from drafter, got DraftStepResp"* — a hard crash that **bypasses** the
tag/epoch fence (it never reaches `finish_decode`). Same root cause on a group finishing
at the barrier (`send_close`) or `_engine_paused` short-circuit (line 86) or the
all-finished `is_empty` path. **Fix (applied):** the `_event_loop` drain guard at the top
of each iteration, *before any other draft-channel send*, pops the in-flight spec resp via
its **correct** recv site (`recv_step_resp` + tag/epoch-assert) and clears `self._spec`
whenever the engine is paused, prefill will be admitted, a group finished, the slot set is
empty, or the about-to-run train's pre-rebuild active set ≠ `self._spec.active_list_T`.
With the guard the narrow FIFO/tag claim holds; without it SBP is not crash-safe across
prefill admission, pause, or full-group completion. **This guard is mandatory, not
optional.**

**R2 — Anchor remap omission (silent accuracy bug).** *Verdicts (holds=false) on "remap
correctness" and "resample-fires-vs-not".* Already folded into the design: `x0 =
verified_id` is the same ancestor-dependent quantity as the tokens and **must** be
remapped by the same `source_rows` gather. The `n_jobs==0` path is unaffected (identity).
Regression guard: assert the remapped `verified_id[retired_row]` equals
`slot_state.verified_ids[i]` after dispatch (both = survivor `j`'s anchor).

**R3 — KV pressure at the barrier.** SBP-prep allocates one *extra* window of target +
draft KV per active slot at the barrier (vs baseline allocating it next train). Same risk
class as the in-loop prefetch but now also at the barrier — size the pool for K+1 windows
of headroom; the leak check (test e) and an `alloc`-failure smoke near `max_ctx_len`
cover it.

**R4 — Residual joint-law coupling.** SBP offspring of a common ancestor are perfectly
correlated through one adopted window (benign diversity effect, *not* bias). Immeasurable
at the gate operating point; bounded empirically by the forced-high-resample test (d) and
multi-seed equivalence (g). Public claim softened to **"unbiased / accuracy-neutral in
expectation."**

**Alternatives rejected:**

- **Discard KV-sharing and re-prefill retired slots' spec window.** Correct but throws
  away the whole point: the frontier-clone already copies the survivor's spec window for
  *free* (refcounted share, zero compute). Re-prefilling each retired slot a full window
  every barrier would add `n_retired × G` draft/prefill steps, erasing the throughput win
  and *adding* latency to the barrier. **We keep sharing** — it is the mechanism, not an
  optimization.
- **Full-redraft of retired slots from the survivor commit.** Distribution-equivalent to
  adoption (the accuracy argument shows adopt == redraft-from-clone) but strictly slower
  (an extra draft pass) and needs a second drafter round-trip across the barrier — exactly
  what SBP exists to avoid. Adoption is the same distribution at zero extra draft.
- **Larger K (fewer barriers).** Buys the same ~+12% tok/s but costs **2–5 pt accuracy**
  (K=4→61%, K=8→63.5% vs K=2→66%). SBP captures the throughput **without** the accuracy
  loss by overlapping the barrier instead of reducing barrier count.
- **Depth-W>1 speculation across the barrier.** Rejected as premature: the drafter is the
  hard bottleneck (draft 42 ms > verify 30 ms), so keeping W>1 StepReqs in flight cannot
  beat the draft rate — it only adds `W·G` blocks of transient KV pressure per slot, a
  more complex inflight/poll path, and multi-generation spec lineage under resample.
  Single-depth SBP captures the full barrier-removal ceiling. Revisit only after the
  drafter itself is sped up.
- **Exporting the ancestor map from `dispatch_resample_batch`.** Unnecessary — the async
  scheduler already holds `plan.dst_slots`/`plan.src_slots` locally and builds `a(i)` with
  no `core/scheduler.py` change, preserving the surgical diff and per-cohort isolation.
- **Remapping inside `finish_decode` (worker change).** Rejected in favor of remapping the
  numpy arrays + anchor in the scheduler **before** `finish_decode`, which then receives an
  already-correct `DraftStepResp` and stays a pure verify path (minimal worker diff — only
  the epoch plumbing).
- **Polling the spec response during resample (non-blocking recv refactor).** Rejected for
  the first cut: the blocking recv at T+1's top is correctness-neutral and bounded (the
  drafter is computing the spec window we want anyway). Add a poll+defer path later only if
  profiling shows the verifier idling at the T+1 recv when resample ran long.

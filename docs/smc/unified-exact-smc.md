# Unified Exact/SMC Speculative Decoding — Design

Goal: one engine, one decode-cycle primitive, two (switchable) verification
semantics:

- **SMC mode** (today's smcsd): all `γ+1` drafted tokens accepted, divergence
  from the target absorbed into importance weights, ESS-triggered resampling.
  Fast (fixed `γ+1` tokens/cycle), *approximate* w.r.t. the target
  distribution at finite N (consistent as N→∞; the tempered/power target is
  intentionally biased toward high-likelihood sequences).
- **Exact mode** (new): multi-draft rejection sampling over the same N
  particles — the accepted prefix is distributed *exactly* according to the
  target model. Variable tokens/cycle (`E[accepted]+1 ≤ γ+1`), used for
  accuracy-sensitive spans.

Switching is allowed **mid-sequence and per-request**: a prefix generated
under either mode is simply conditioning; the continuation under exact mode is
exactly `p(· | prefix)`, and under SMC mode is the usual SMC approximation of
`p^α_T(· | prefix)`.

---

## 1. Background: the three algorithms on the table

### 1.1 SMC-SD (this repo)

Per decode cycle, per group of N particles: draft AR × γ (+bonus), one
`TARGET_VERIFY` pass over all `bs·(γ+1)` drafted tokens, per-position
log-weight increments `α·log p_T(x_t|·) − log q(x_t|·)`, bonus sampled from
`p_T^α`, ESS check + fused systematic resample (KV block-table copy under
refcounts). No rejection, no rollback, ever.

### 1.2 JetSpec (arXiv:2606.18394)

Two separable contributions:

1. **Causal parallel draft head** — a lightweight head on frozen target hidden
   states that emits an entire *candidate tree* in a single forward pass,
   with a tree-causal attention mask so each node conditions on its ancestors
   (unlike branch-agnostic parallel drafters). Best-first expansion under a
   node budget B, scored by accumulated draft log-prob.
2. **Lossless tree verification** — one target pass over the tree (tree
   attention mask), then the *standard* speculative acceptance rule
   `α_t = min(1, p_t/q_t)` walked along tree paths; longest accepted prefix
   wins, plus one correction token from the target. Exactness is inherited
   from vanilla speculative sampling.

Note: JetSpec's verification is path-wise standard acceptance. The fully
general "several sibling candidates at one position, exact via residual
updates" scheme is the multi-draft speculative sampling of
SpecTr (k-SEQ) / SpecInfer / MDSD. That is the scheme we adopt for exact
mode, because N i.i.d. particle chains are exactly its input format.

### 1.3 Why the two fit in one framework

The user-visible intuition is right: **JetSpec's draft tree and SMC's particle
set are the same object** — a set of candidate continuations sharing a prefix.
smcsd's N particles, drafted i.i.d. from a common (post-collapse) prefix, form
a depth-γ tree that branches only at the root fan-out. Both algorithms then
run **one target pass over all candidates** and differ *only* in the operator
applied to the (draft-token, draft-logprob, target-logprob) triple:

| | SMC mode | Exact mode |
|---|---|---|
| verify operator | importance weight `+= α·log p − log q` | sequential accept/reject with residual correction |
| particle update | keep all N paths; ESS-resample | collapse all N onto accepted prefix (one-hot "resample") |
| tokens committed / cycle | always `γ+1` | `accepted + 1` (residual or bonus token) |
| distribution | SMC approx of `p_T^α` | exactly `p` |

Everything else — slot state, refcounted KV lineage, draft AR loop,
`TARGET_VERIFY`, fused sampling, the resample-plan format and
`batched_resample_kv`, finalize — is shared. Exact mode is, mechanically,
*"a different weight kernel followed by a forced one-hot resample plus a
seq-len rollback."* That is the entire unification.

---

## 2. Exact mode: the algorithm

State entering a cycle in exact mode: all N particles of the group are
**identical** (same tokens, same `seq_len = L`, KV pages shared by refcount —
this is the post-collapse invariant; on mode entry from SMC it is established
by one forced resample from the posterior-sampled particle, see §4).

**Draft.** Each particle drafts `d_1..d_γ` i.i.d. from `q(·|prefix, own
tokens)` — the *unchanged* draft AR path. This yields N i.i.d. chains from
the same root: a root-fan-out tree. Additionally retain the **full draft
distributions** `q_t^{(i)}` (bs, γ, V) — needed for residuals (fp16 is fine;
N=8, γ=8, V=128k ≈ 16 MB/group). Today only the chosen-token logprob is kept.

**Verify.** Unchanged `TARGET_VERIFY`: full target logits for all
`bs·(γ+1)` rows (already materialized as `score_logits`).

**Accept (new operator, per group).** Multi-draft sequential rejection
sampling (SpecTr k-SEQ / SpecInfer with sampled candidates):

```
viable ← {1..N}                    # chains matching the accepted prefix
for t = 1..γ:
    p ← p_T(· | accepted prefix)   # target row of any viable chain at depth t
    for each i in viable (any fixed order):
        c ← d_t^(i)                # that chain's depth-t token
        accept c with prob min(1, p(c) / q_t(c))
        if accepted:
            commit c; viable ← {j ∈ viable : d_t^(j) = c}; break to t+1
        else:
            p ← normalize(max(p − q_t, 0))     # residual update
    if nothing accepted at depth t:
        commit one token ~ p (final residual); END CYCLE (accepted = t-1)
if all γ depths accepted:
    commit bonus ~ p_T(· | full chain)         # existing bonus position
```

Correctness: given the prefix, viable chains' depth-t tokens are i.i.d. draws
from the *same* `q_t` (their conditioning contexts are equal by the viability
invariant), which is precisely the setting in which sequential
rejection-with-residual yields an exactly `p`-distributed token
(SpecTr Thm; SpecInfer App. proof). By induction the whole committed prefix is
exact. Duplicate candidates need no special casing: after token v is rejected
the residual has `p(v)=0`, so duplicates auto-reject.

Notes:
- Temperatures: exact mode targets plain `p` at the *user's* sampling
  temperature. `smc_target_temperature` applies; `power_alpha` must be 1 in
  exact spans (α≠1 has no exact-sampling interpretation) — enforce/ignore.
- The greedy special case (temperature 0) degenerates to token-matching
  verification, as usual.

**Collapse (reuses resample machinery).** After acceptance, emit a resample
plan `dst = all slots ≠ winner, src = winner` (winner = any chain in the final
viable set; it matches the accepted prefix by construction) in the *existing*
`BatchedResampleResult` format, dispatched through the *existing*
`batched_resample_kv` (block-table copy + refcounts + lineage tensors). Then
roll back lengths: `seq_lens[slots] = L + accepted + 1`,
`token_counts` likewise; `verified_ids[slots] = committed last token`.

**Rollback preserves the `kv_alloc == seq` invariant (audit outcome).** The
originally-sketched "leave `kv_allocated_len` at `L+γ+1` and reuse stale
pages in place" approach is *rejected*: the fast prepare path
(`fused_prepare_decode`) and the host-side page-count arithmetic both bake in
the invariant `kv_allocated_lens == seq_lens` at prepare time (exactly γ+1
fresh pages per row, the allocation *is* the cache-locs table), and
`batched_resample_kv`'s Phase-2 copy (`:seq_lens[src]`) vs Phase-3 length
copy (`kv_allocated_lens[src]`) would go inconsistent — refcount leaks /
double-frees at the boundary. Instead the collapse maintains the invariant,
with **zero kernel changes** (implemented in
`ScheduleBatchSMC.collapse_exact`):

1. dec_ref-and-free the *winner's* stale tail pages `[L+a+1, L+γ+1)`
   (rejected drafts' KV; exclusively owned this cycle, refcount 1→0);
2. roll the winner's `seq_lens` **and** `kv_allocated_lens` back to `L+a+1`
   *before* dispatching the one-hot plan — the unchanged resample kernel then
   Phase-1 dec_refs each loser's full old span (their fresh pages hit 0 and
   land in the freed buffer), Phase-2 copies exactly the committed span with
   inc_ref, and Phase-3 propagates the rolled-back lengths to the losers;
3. next cycle's `fused_prepare_decode` allocates the usual γ+1 fresh pages
   per row — nothing downstream ever observes `kv_alloc ≠ seq`.

The CPU shadow `seq_lens_host` is rolled back from a synced `accept_len`
(one (G,)-tensor `.cpu()` per cycle — exact mode runs the sequential loop in
phase 1, see below).

**Deferred-bonus interaction.** `prev_last_draft_id` must be set to the last
*drafted* token of the winner chain at the committed boundary. On a rejection
at depth t the winner's `d_t` was rejected, so the "prev" seed is the
committed token itself — simplest is to disable the deferred-bonus schedule
in exact-mode cycles initially (the flag machinery for per-config fallback
already exists) and revisit.

---

## 3. What changes where (implementation map)

| Piece | Change |
|---|---|
| `ServerArgs` / `SMCEngine` / http server | `smc_mode: "smc" \| "exact" \| "auto"` (engine default), per-request override via sampling params; `smc_exact_*` knobs. |
| `SequenceGroup` (`core/scheduler.py`) | carries current mode; mode is *per-group* so mixed batches are natural (all kernels are already per-row gated). |
| `ScheduleBatchSMC` (`core/req_state.py`) | `mode` per-row tensor; write-back must mask token scatter by per-row `accept_len` instead of assuming `γ+1` (same masking pattern as the existing post-EOS masking in `write_back_gpu`); `token_counts += accept_len+1`. |
| `SMCWorker._forward_decode` (`core/worker.py`) | keep full draft logits when row is exact-mode; after verify, branch: SMC weight math (existing) vs exact-accept operator. Both consume the same `score_logits_3d` / draft tensors. |
| New kernel `core/kernels/fused_mdsd_accept.py` | per-group program: sequential accept/reject over (N chains × γ depths) with residual updates over V in blocks; outputs `accept_len`, committed tokens, winner slot, residual/bonus sample, and a one-hot resample plan in `BatchedResampleResult` format. Fixed shape ⇒ capturable in the cycle graph. Philox seeding identical to `fused_sampling.py`. |
| `SMCCoordinator` | in exact mode skip ESS collect; feed the accept-kernel's plan straight to `dispatch_resample_batch`. |
| Cycle graph runners | exact-mode capture variant (the accept kernel replaces weight-diff+bonus in the recording). Eager path first; graph in phase 2. |
| Finalize | unchanged (weights all zero ⇒ uniform pick among identical particles). `log_Z_hat` reported only for SMC-mode segments. |

Shared and untouched: slot/row lifetime invariants, refcounted KV allocator,
draft AR loop + multistep backends, `TARGET_VERIFY` + split-KV verify kernel,
fused sampling, overlap loop, hybrid (Mamba) state commit (exact-mode Mamba
rollback needs the target's depth-indexed intermediate state — commit at
`accepted_steps = accept_len` instead of `γ`, which the existing
`update_mamba_state_after_mtp_verify` API already parameterizes).

---

## 4. Switching semantics and policy

**SMC → exact** (entering an accuracy-sensitive span): finalize-style
posterior sample over the group's weights picks the ancestor; forced one-hot
resample collapses all particles onto it; zero the weights; set mode=exact.
This is exactly the existing finalize selection + existing resample dispatch,
run mid-flight. The exact continuation is then exactly `p(·|prefix)`.

**Exact → SMC**: trivial — particles are identical; weights are zero; set
mode=smc. Diversity re-emerges from the first SMC cycle's i.i.d. drafts.

**Policy hooks (`"auto"` mode)**, evaluated per group per cycle on signals we
already have on-device:

- *Span triggers*: token-class sets (digits, tool-call/JSON delimiters, code
  fences), or explicit control markers in the prompt/template.
- *Target entropy / margin* at recent verify rows: low-entropy regions are
  cheap for exact mode (acceptance ≈ 1, throughput ≈ SMC) — prefer exact.
- *Draft–target disagreement* (mean `log p − log q` over the last cycles):
  high disagreement makes exact mode stall (low acceptance) while SMC still
  commits `γ+1`/cycle — prefer SMC unless the span is flagged sensitive.
- *ESS trajectory*: persistent ESS collapse in SMC mode means the particle
  population is effectively degenerate anyway — exact mode costs little.

The policy only ever changes *future* cycles' operator; it never invalidates
committed state, so it composes with the overlap loop (mode flips take effect
next cycle, one-step-late is fine).

---

## 5. What JetSpec contributes beyond the exact operator (later phases)

1. **Tree dedup of the root fan-out.** N i.i.d. chains re-read the shared
   prefix N× in the draft (mitigated separately by cascade decode, PR #31)
   and duplicate tokens across chains carry no extra information. A
   JetSpec-style *tree* draft (branch top-W at shallow depths, best-first
   under budget N·γ) raises expected accepted length per verified token.
   Caveat: deterministic top-W candidates are no longer i.i.d. draws from q,
   so exactness needs the distinct-candidate MDSD variant (SpecHub/greedy
   MDSD literature) instead of the i.i.d. scheme — a contained change inside
   the accept kernel. SMC mode with tree drafts corresponds to particles
   with shared partial histories (already representable via the refcount
   lineage).
2. **Single-pass causal draft head.** Replace the γ-step AR draft with a
   JetSpec/EAGLE-style head on target hidden states emitting all depths in
   one forward (tree-causal mask). Orthogonal to the verify operator: SMC
   mode absorbs head-quality loss into weights; exact mode sees it as lower
   acceptance. This subsumes the roadmap's "EAGLE support" item and is the
   biggest draft-side speed lever (γ forwards → 1).

---

## 6. Phasing / status

- **Phase 0+1 — IMPLEMENTED** (engine-level `mode="exact"`, eager exact
  path, this branch):
  - `smcsd/core/exact_accept.py` — the accept operator: an audited per-group
    reference (`mdsd_accept_reference`) and a group-vectorized torch
    implementation (`mdsd_accept_batched`) used by the worker.
  - `SMCWorker` — `smc_mode` attr; the legacy draft loop additionally
    retains full proposal dists `q_t` (γ × (bs, V), incl. the greedy
    one-hot case); `_exact_verify_outputs` replaces weight-diff + bonus
    with the accept operator on the same `score_logits_3d`; per-group
    results ride `SMCDraftInput.exact_{accept_len,tokens,winner}`.
    Exact mode forces `power_alpha == 1`, disables deferred-bonus and the
    cycle graph (draft per-step decode graphs and the TARGET_VERIFY graph
    still replay), and rejects hybrid (Mamba/GDN) pairs until the
    variable-depth recurrent commit is wired.
  - `ScheduleBatchSMC.write_back_exact` (variable-length commit, EOS scan
    masked to committed columns, weights untouched), `collapse_exact`
    (one-hot plan in `BatchedResampleResult` format + invariant-preserving
    rollback, see §2), `rollback_seq_lens_host`.
  - `SMCScheduler._exact_commit` replaces the `_resample` body per cycle;
    exact mode pins the sequential (non-overlap) event loop.
  - `SMCEngine(mode="smc"|"exact")`.
  - Tests: `tests/test_exact_accept.py` (CPU: structural invariants,
    batched≡reference walk equivalence, statistical exactness of the
    committed stream vs the target's AR factorization for N∈{1,2,4,8});
    `tests/test_exact_commit_path.py` (CUDA: write-back/collapse/rollback
    through the real dispatch kernel — refcounts, freed-page capture,
    block-table mirroring, finish state).
- **Phase 2 (switching + mixed batches) — IMPLEMENTED** (this branch):
  - `SMCEngine(mode="mixed")`: per-request initial mode via
    `sampling_params.custom_params["smc_mode"]`, and mid-sequence switch
    points via `custom_params["smc_mode_plan"] = [[token_threshold, mode],
    …]` (applied in decode postprocessing, so a switch cleanly changes the
    next cycle's operator).
  - Per-group mode state: `ScheduleBatchSMC.group_mode` /
    `row_exact` (gates the fused ESS collect OFF for exact rows) /
    `set_group_mode` / cached `exact_partition()` (CPU-built, sync-free).
  - Mixed-mode batches: the worker computes the SMC tail for all rows and
    overwrites the exact groups' rows from the accept operator
    (`_run_exact_accept` on the row subset); the scheduler's
    `_mixed_commit` runs `write_back_gpu(rows=smc_rows)` + ESS resample on
    the SMC partition and the exact commit (subset write-back, collapse,
    rollback) on the exact partition — two dispatches over disjoint slot
    sets.
  - SMC→exact switching: `force_collapse_group` — finalize-style posterior
    ancestor sample, one-hot plan through the existing dispatch kernel,
    weights zeroed.  exact→SMC is a gate flip (diversity re-emerges from
    the next cycle's i.i.d. drafts).
  - Telemetry: per-request `smc_mode_stats` (smc/exact cycle counts,
    exact accepted tokens, accept rate) on the result dict — the raw
    signal for a future "auto" policy.
  - Tests: mixed-partition isolation + ESS gating + forced-collapse CUDA
    tests; mixed e2e smoke (one engine: SMC request, exact request, and a
    plan-switching request in one batch).
- **Cycle schedules + first perf pass — IMPLEMENTED** (this branch):
  - **Cycle schedule** (the pre-"auto" control knob): a cyclic
    per-cycle mode schedule, e.g. *4 cycles exact, 2 cycles SMC,
    repeat* — engine-wide via `SMCEngine(mode_cycles=[("exact", 4),
    ("smc", 2)])` (implies `mode="mixed"`), or per request via
    `custom_params={"smc_mode_cycles": [["exact", 4], ["smc", 2]]}`.
    Evaluated in decode postprocessing next to the token-threshold
    `smc_mode_plan`; each boundary goes through the standard switch
    (posterior collapse on SMC→exact, gate flip on exact→SMC).
  - **`fused_mdsd_accept` Triton kernel** (`core/kernels/`): the whole
    multi-draft rejection walk — viability bitmask, per-candidate accept
    draws (Philox), residual updates and CDF-scan residual/bonus sampling
    over V — in ONE launch per cycle, replacing the torch operator's
    ~N·γ·6 small launches.  Statistically verified against the same
    exactness harness (GPU, 60k trials).  `SMC_FUSED_ACCEPT=0` falls back
    to the torch operator for A/B.
  - **Cycle CUDA graph in mixed mode**: the full-cycle graph is captured
    at init (deferred bonus stays off) and replayed for cycles whose
    batch carries no exact groups; exact-carrying cycles fall back to the
    eager path via a sync-free per-batch gate.  SMC cycles inside a
    schedule now run at graph speed.
  - **Sync-slimming in the exact commit**: token scatter and collapse-plan
    construction rewritten without boolean advanced indexing (data-
    dependent shapes force host syncs); the winner-tail page release goes
    through a new `dec_ref_tail_pages` kernel into the standard freed-page
    buffer (was `dec_ref_and_free` with two syncs).  One deliberate sync
    remains per exact cycle: the `accept_len` host read that feeds the
    seq-len shadow, telemetry, and schedules.

  Measured (single stream, Qwen2.5-1.5B target + 0.5B draft, N=8 γ=8,
  512 tokens, B200):

  | config | tok/s | tok/cycle | ms/cycle |
  |---|---|---|---|
  | SMC (full perf path) | ~884 | 9.0 | 10.2 |
  | exact, torch operator | ~234 | ~4.9 | 21.1 |
  | exact, fused kernel | ~334 | ~4.9 | 13.8 |
  | schedule exact4/smc2 | ~470–580 | ~6–7 | ~13 |

  Exact mode's remaining gap vs SMC is (a) inherent: ~4.9 committed
  tokens/cycle at ≈0.48 acceptance vs SMC's fixed 9 — a draft-quality
  property, addressed by better drafts (phase 4), and (b) ~3.6 ms/cycle of
  eager-path overhead — addressed by capturing the exact cycle
  (draft AR + verify + fused accept + collapse are all fixed-shape now).

- **GSM8K sweep (2026-07-18)** — 200 questions, Llama-3.1-8B-Instruct
  target + Llama-3.2-1B-Instruct draft, γ=8, temp 0.7 (draft & target),
  batch 1, B200, `scripts/accuracy_test_gsm8k.py --smc-mode ...`:

  | config | accuracy | tok/s |
  |---|---|---|
  | SMC N=8 | 71.5% | 597 |
  | exact N=8 (multi-draft lossless) | **84.0%** | 384 |
  | mixed exact4/smc2 N=8 | 74.0% | 452 |
  | exact N=1 (vanilla speculative sampling) | 81.5% | 365 |

  **Frontier sweep (2026-07-18)** — same harness, mapping the
  speed/accuracy Pareto frontier (figure: `assets/pareto-gsm8k.png`):

  Llama-3.1-8B + 3.2-1B draft (γ=8 unless noted):

  | config | accuracy | tok/s | on frontier |
  |---|---|---|---|
  | SMC | 71.5% | 597 | ✓ (fast end) |
  | mixed exact4/smc2 | 74.0% | 452 | ✓ |
  | mixed exact4/smc1 | 79.0% | 435 | ✓ |
  | mixed exact8/smc1 | 81.0% | 423 | ✓ |
  | exact | 84.0% | 384 | ✓ (lossless end) |
  | exact γ=12 | 81.5% | 389 | dominated |
  | vanilla SD (exact N=1) | 81.5% | 365 | dominated |
  | SMC threshold=0 | 63.0% | 392 | dominated |

  Qwen3-8B + Qwen3-0.6B draft (γ=8, thinking disabled):

  | config | accuracy | tok/s |
  |---|---|---|
  | SMC | 68.5% | 509 |
  | mixed exact4/smc1 | 90.5% | 364 |
  | mixed exact8/smc1 | 94.0% | 354 |
  | exact | 94.5% | 342 |

  Frontier reading: (1) mixed schedules trace a clean interior frontier;
  accuracy tracks the SMC *token share* (SMC cycles commit γ+1 vs exact's
  ~a+1, so exact4/smc2 is ~45% SMC tokens — near-SMC accuracy — while
  exact8/smc1 is ~17% — near-exact accuracy).  The "barely slower than
  exact, no accuracy loss" point exists (exact8/smc1); a "barely slower
  than SMC at exact accuracy" point does NOT exist with blanket periodic
  schedules — bias in SMC segments propagates through the chain.  Closing
  that corner needs (a) SMC accuracy work (the defaults lose 12–26 pts to
  lossless on these pairs — threshold=0 is worse, so it is not the
  resampling knob alone), (b) targeted spans / the phase-3 auto policy,
  or (c) faster exact (higher acceptance via the phase-4 draft head, exact
  cycle graph, overlap).  (2) Multi-particle exact strictly dominates
  vanilla speculative sampling (N=1): equal-or-better accuracy at +5–16%
  speed — sibling candidates lengthen the accepted prefix at bs=1 for
  nearly free.  (3) exact matches the target model's reference quality on
  both pairs (84% Llama, 94.5% Qwen3), as losslessness requires.

  Reading: exact N=8 matches the target model's reference GSM8K quality
  (Llama-3.1-8B ≈ 84%) — as it must, being distribution-lossless — and
  multi-draft N=8 beats vanilla N=1 on BOTH axes (more sibling candidates
  ⇒ longer accepted prefixes; extra particles are nearly free at bs=1
  where decode is weight-bound).  The periodic 4/2 interleave interpolates
  speed but accuracy stays near the SMC end: a sequence is a chain, so
  bias injected in SMC segments propagates — blanket cycle ratios buy
  speed, while accuracy-targeted spans (or higher exact shares) are what
  move quality.  N.B. the SMC accuracy point is at default knobs
  (threshold 0.5, α=1); tuned SMC configs may close part of the gap.
  EXACT cycle (all pieces are single fixed-shape launches now; the
  accept kernel + collapse plan + dec_ref_tail are capture-friendly),
  hybrid (Mamba) rollback commit
  (`update_mamba_state_after_mtp_verify` already takes per-row
  `accepted_steps`), overlap-loop compatibility (accept_len via the pinned
  snapshot instead of a sync).
- **Phase 3** — `"auto"` policy signals + API for span markers; benchmarks:
  GSM8K accuracy/exactness vs tok/s across mode schedules.
- **Phase 4** — JetSpec-side: tree-dedup drafting (distinct-candidate MDSD
  variant); single-pass causal draft head.

## 7. Risks / open questions

- ~~`batched_resample_kv` copy-length semantics for `kv_alloc > seq_len`~~
  — resolved: the collapse maintains `kv_alloc == seq` (§2), the kernel is
  unchanged.
- Variable `accept_len` under the overlap loop: next cycle's
  `prepare_for_decode` reads `seq_lens` written by this cycle's collapse —
  ordering is already stream-correct (collapse is enqueued before the next
  prepare), but the *reserved-slot admission* accounting (#33 fix) should be
  re-audited with variable-length commits.
- Exact-mode CUDA-graph capture: the accept kernel is data-dependent in
  *content* but fixed in *shape* — same trick as EOS masking, should
  capture; the eager fallback must remain for A/B.
- Per-request `power_alpha≠1` + exact mode: define as "α applies only to SMC
  segments" and document.

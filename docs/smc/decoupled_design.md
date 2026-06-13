# Decoupled SMC-SD — Design (prototype)

Goal: target (verifier) engine and draft (drafter) engine in **separate processes
on separate GPUs**. Verifier receives user requests; per decode round it asks the
drafter for gamma draft tokens + per-position draft logprobs, verifies (logprobs
only, no rejection), reweights, resamples, and drives the drafter's state.

Upstream reference: SGLang decoupled spec PR #22520 (see `lesson/`). We keep its
*shape* — engine roles, explicit wire protocol, sync/commit/close lifecycle —
but run **lockstep**, because SMC cannot legally run ahead (see below).

## Why lockstep (answers lesson/08 open questions)

- **Q1 (no rejections)**: SMC accepts all gamma draft tokens + 1 target-sampled
  bonus every round (`accept_lens == gamma+1` always, `smcsd/core/worker.py`).
  Commits never truncate → upstream's KV-rollback/echo-back machinery has no
  trigger in lockstep. Omitted, not just cold.
- **Q2 (particles)**: the drafter must hold all N particle KV states (each
  particle drafts from its own context), but holds no weights or resample logic.
- **Q3 (separate draft model)**: yes — independent TpModelWorker (Llama-3.2-1B),
  so "two engines, two GPUs" maps cleanly.
- **Why no run-ahead**: round t+1's anchor x0 is the *bonus token sampled by the
  target* at round t, and resampling can replace any particle's entire state at
  round boundaries. A drafter that runs ahead must speculate on both → upstream's
  DraftTailBuffer/run-ahead design solves an asynchrony SMC's algorithm forbids.
  Future parallelism = overlap draft(group A) with verify(group B), not run-ahead.

## Topology

```
┌────────── verifier process (GPU0) ──────────┐   ZMQ ipc:// (pickled dataclasses)
│ DecoupledSMCEngine (user API, tokenizer)    │
│  └─ DecoupledSMCScheduler (SMCScheduler)    │   requests:  PUSH ──────────────▶ PULL
│      ├─ target SMCTpModelWorker (8B)        │   replies:   PULL ◀────────────── PUSH
│      ├─ ScheduleBatchSMC + SMCCoordinator   │
│      └─ DecoupledSMCWorker ── DraftClient ──┼──▶ ┌────── drafter process (GPU1) ──────┐
└─────────────────────────────────────────────┘    │ SMCDraftServer                     │
                                                    │  ├─ SMCTpModelWorker (1B, own      │
                                                    │  │   refcounted allocator+pools)   │
                                                    │  └─ slot mirror: req_pool_indices, │
                                                    │      seq_lens, kv_allocated_lens   │
                                                    └────────────────────────────────────┘
```

## Consistency invariant (SMC analog of upstream's committed-prefix invariant)

```
∀ slot s:  mirror.seq_lens[s] == verifier.slot_state.seq_lens[s]
           mirror block-table[s] holds draft KV for exactly the token sequence
           the verifier attributes to slot s
```

Maintained by replaying the same membership ops in the same order on both sides
(single FIFO ZMQ channel = total order). The drafter **asserts** seq_lens match
on every step request — fail-fast tripwire instead of upstream's reconciliation.

## Wire protocol (`smcsd/decoupled/io_struct.py`) — CPU data only

| Message (verifier→drafter)      | When                          | Drafter action |
|---|---|---|
| `DraftPrefillReq(group_ids, input_ids, sampling)` | `_forward_extend` (blocking) | create parent Req, extend forward (writes prompt draft-KV), sample x0 → `DraftPrefillResp(next_token_ids)` |
| `DraftMaterializeGroup(group_id, slots, shared_seq_len)` | after verifier materialize succeeds | alloc N pool rows, `copy_block_table` parent→particles (+refcount), release parent, register slots |
| `DraftStepReq(slots, verified_ids, seq_lens)` | `_forward_decode` (blocking) | assert seq_lens; alloc gamma+1 KV (`SMCDecodeContext.from_slot_gather`); AR loop gamma+1 steps → `DraftStepResp(tokens (bs,γ), logprobs (bs,γ))` |
| `DraftCommitResample(dst_slots, src_slots)` | after a resample dispatch | `batched_resample_kv` on own pools; copy seq/alloc dst←src |
| `DraftCloseGroup(group_id)` | finalize / abort / finished-at-prefill | free slots or pending parent (idempotent) |
| `DraftShutdown` | engine shutdown | exit |

Per round the verifier worker sends one `DraftStepReq` and blocks on the resp;
the scheduler then sends `DraftCommitResample` only when resampling fired.
Round t's commit lands before round t+1's step (FIFO).

Note x_{γ+1} (the draft's last sample) is discarded as in colocated code; the
(γ+1)-th AR step exists to write x_γ's draft KV.

## Process/files

- `smcsd/decoupled/io_struct.py` — dataclasses above.
- `smcsd/decoupled/draft_server.py` — `run_smc_draft_server_process` + `SMCDraftServer`
  (TpModelWorker(draft model) with SMC refcounted allocator, ChunkCache, slot mirror,
  prefill via `ScheduleBatch.init_new`+`prepare_for_extend`, AR loop = simple
  per-step path of `SMCWorker._forward_decode` with cuda-graph off).
- `smcsd/decoupled/worker.py` — `DecoupledSMCWorker`: SMCWorker minus local draft
  model; draft AR replaced by RPC; cache_locs computed directly via
  `assign_smc_cache_locs_kernel`; verify/weights/bonus identical to colocated.
- `smcsd/decoupled/scheduler.py` — `DecoupledSMCScheduler(SMCScheduler)`: wires the
  client/worker; sends Materialize/CommitResample/CloseGroup at the existing hooks;
  + `run_decoupled_smc_scheduler_process`.
- `smcsd/decoupled/engine.py` — `DecoupledSMCEngine(SMCEngine)`: spawns the drafter
  process on `draft_gpu_id` (default 1) before the scheduler; IPC names via
  `SMCSD_DRAFT_IPC` env.
- `smcsd/engine.py` — minimal hook: `scheduler_process_func` class attribute.
- `scripts/accuracy_test_gsm8k.py` — `--mode smc_decoupled`.

## Verification plan

1. Smoke: 2 prompts, N=4 γ=4, both engines on 2 GPUs → sane text, no mem-leak warnings.
2. GSM8K 100q (N=12, γ=8, temp 0.7, triton) → expect ≈70%+.
3. **Gate**: GSM8K 1000q → 70–78% (matches colocated reference behavior).

## Phase 2 — cross-group pipelining (experimental)

Lockstep leaves one GPU idle at a time. SMC forbids run-ahead *within* a group
(bonus anchor + resampling), but groups are independent — so overlap **draft of
cohort B with verify of cohort A**:

```
GPU1 (1B):  [draft A,t]   [draft B,t]   [draft A,t+1] ...
GPU0 (8B):  ...........   [verify A,t]  [verify B,t]  ...
```

Design (`smcsd/decoupled/pipeline_scheduler.py`):
- Groups are partitioned into `SMCSD_PIPELINE_COHORTS` cohorts (default 2) at
  materialize time (least-loaded). Per-cohort state machine: IDLE → DRAFTING
  (StepReq in flight) → READY (StepResp held) → verify → IDLE.
- `DecoupledSMCWorker._forward_decode` split into `start_decode` (cache locs +
  tagged StepReq) and `finish_decode` (verify/weights/bonus). Lockstep path =
  start + blocking recv + finish (behavior unchanged).
- StepReq/Resp gain a `tag` echoed by the drafter; replies matched FIFO and
  asserted by tag.
- Per-cohort verify writes back through `process_batch_result(active=…)` and a
  **row-masked** resample collect (only the verified cohort's group rows), so a
  cohort mid-draft can never be resampled — that would desync the drafter mirror.
- Per-cohort FIFO ordering on the wire is preserved: Commit_A (sent during A's
  verify) precedes StepReq_A(t+1); B's in-flight StepReq ordering is unaffected
  (disjoint slots).
- Prefill is a sync point: in-flight StepResps are drained (held as READY)
  before the blocking PrefillReq so the FIFO reply channel stays unambiguous.

Core touches are minimal + defaulted: optional `active` overrides on
`prepare_for_decode` / `build_model_worker_batch` / `process_batch_result`, and
an optional `row_mask` on `SMCCoordinator.collect_resample_jobs_batch`.

Entry points: `PipelinedDecoupledSMCEngine`, eval `--mode smc_pipelined`.

### Measured (GSM8K 80q, N=12, γ=8, temp 0.7, triton, 2×A6000, mem 0.6)

| Mode | batch | Accuracy | tok/s | Wall |
|---|---:|---:|---:|---:|
| decoupled lockstep | 1 | 73.0% (1000q gate) | 78 | — |
| colocated | 4 | 71.2% | 202 | 78.5s |
| decoupled lockstep | 4 | 71.2% | 161 | 100.0s |
| decoupled **pipelined** | 4 | 68.8% | **208** | 78.9s |
| colocated | 8 | 68.8% | 253 | 61.6s |
| decoupled lockstep | 8 | 67.5% | 214 | 75.0s |
| decoupled **pipelined** | 8 | 67.5% | **258** | 62.3s |
| colocated | 12 (96q) | 68.8% | 264 | 73.4s |
| decoupled **pipelined** | 12 (96q) | 70.8% | **291** | 67.4s |

Pipelining = +29% over lockstep at batch 4, +20% at batch 8; accuracy within
noise of lockstep at every point (n=80 → σ≈5pts). All decoupled accuracy gated
against the colocated control on the same questions.

### Bug found during batch-size validation (fixed)

Upstream force-disables piecewise CUDA graphs whenever `speculative_algorithm`
is set — colocated SMC therefore never drafts through `PiecewiseCudaGraphRunner`.
The drafter process (no spec algorithm in its ServerArgs) had piecewise ENABLED,
and its hand-mutated per-step AR ForwardBatch replayed through compiled
static-buffer graphs → silently degraded drafts on ragged multi-group batches
(62.5% @ b4 vs 71.2% colocated control; batch-1 was unaffected). Fix:
`disable_piecewise_cuda_graph=True` in the drafter ServerArgs. See
`tasks/lesson.md` ("A 'plain' sglang server is NOT a safe drop-in draft engine").

## Acceptance gate command

```
python scripts/accuracy_test_gsm8k.py --mode smc_decoupled \
  --particles 12 --gamma 8 --temperature 0.7 \
  --attention-backend triton --num-questions 1000
```

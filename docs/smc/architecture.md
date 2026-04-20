# SMC Speculative Decoding — Architecture Overview

Entry point for the Sequential-Monte-Carlo speculative-decoding implementation.
This doc is the map; two companions go deeper:

- [`state.md`](./state.md) — data structures: `SequenceGroup`,
  `ScheduleBatchSMC`, `StackedGroupState`, the refcounted KV allocator.
- [`pipeline.md`](./pipeline.md) — the end-to-end flow: admit → prefill →
  materialize → decode cycle → resample → finalize.

All code lives under the `smcsd/` package. There is no legacy variant; this
is the implementation.

---

## 1. What SMC does

For every user request (a **parent** `Req`), SMC materializes **N particle
`Req`s** that share the prompt prefix. Every decode step advances each
particle by `γ+1` tokens:

```
        ┌────────── draft model ──────────┐      ┌── target model ──┐
x₀ ───► x₁ ───► x₂ ───► … ───► x_γ ───► x_{γ+1}    verify x₁…x_γ
   (autoregressive γ+1 steps)             (bonus)  in one batched pass
```

Then the target model is run once (`TARGET_VERIFY`) over the drafted tokens.
Per particle we compute a log-weight update `Σ (score_logprob − draft_logprob)`.
Every cycle we test each group's ESS; if it dips below threshold we
**resample** — low-weight particles are overwritten by copies of high-weight
siblings, both KV and metadata. At the end, the highest-weighted particle's
output is copied onto the parent `Req`.

No rejection loop. All `γ+1` drafted tokens are always accepted; divergence
from the target is absorbed into the log-weight.

---

## 2. System hierarchy

Top-down: one scheduler process drives two models and a stack of fused
kernels. The scheduler owns all SMC-specific orchestration; workers are
thin forward-pass executors.

```
┌──────────────────────────────────────────────────────────────────────┐
│  SMCEngine  (in-process)                                             │
│    • tokenizer + offline generate() API                              │
│    • forks the scheduler subprocess                                  │
│    • ZMQ IPC ◄─────────────────────────────────────────┐             │
└────────────────────────────────────────────────────────┼─────────────┘
                                                         │
                                                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Scheduler subprocess                                                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  SMCSchedulerV2                                                │  │
│  │    waiting_groups  ┐                                           │  │
│  │    prefill_groups  ├─ List[SequenceGroup]                      │  │
│  │    running_groups  ┘                                           │  │
│  │                                                                │  │
│  │    slot_state : ScheduleBatchSMC        ← slot-major GPU state │  │
│  │       └── stacked : StackedGroupState   ← group-major GPU state│  │
│  │    coordinator : SMCCoordinatorV2       ← ESS + resample       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│             │                                   │                    │
│             ▼                                   ▼                    │
│  ┌────────────────────────┐    ┌───────────────────────────────────┐ │
│  │  SMCTpModelWorker      │    │  SMCWorkerV2  (BaseSpecWorker)    │ │
│  │   target / score model │    │    owns a plain TpModelWorker     │ │
│  │    → SMCModelRunner    │    │    for the draft model and a ref  │ │
│  │    → SMCRefCounted-    │    │    to the target worker.          │ │
│  │       TokenAllocator   │    │                                   │ │
│  │    → SMCCudaGraph-     │    │    _forward_decode:               │ │
│  │       Runner (emits    │    │      draft AR × (γ+1)             │ │
│  │       SMCVerifyInput   │    │      → TARGET_VERIFY              │ │
│  │       during capture)  │    │      → logprob_diff + bonus       │ │
│  └────────────────────────┘    └───────────────────────────────────┘ │
│                     │                             │                  │
│                     └────────┬────────────────────┘                  │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Fused Triton kernels  (v2/kernels/)                           │  │
│  │    fused_collect      — mask/normalize/ESS/systematic/compact  │  │
│  │    fused_resample_kv  — batched block-table dec / copy / inc   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Two models live in one process. The **score (target)** model owns the
refcounted KV pool; the **draft** model shares that pool via its own
`TpModelWorker` attached to `SMCWorkerV2`.

---

## 3. Module map

| Area            | Module                                | Role                                                              |
|-----------------|---------------------------------------|-------------------------------------------------------------------|
| Entry           | `engine`                              | `SMCEngine` — tokenization, ZMQ, fork scheduler.                  |
| Scheduler       | `v2/scheduler`                        | `SMCSchedulerV2`, `SequenceGroup`, `SMCCoordinatorV2`.            |
| Slot state      | `v2/req_state`                        | `ScheduleBatchSMC` — slot-major persistent decode batch.          |
| Group state     | `v2/stacked_state`                    | `StackedGroupState` — group-major `(max_G, N)` tensors.           |
| Spec IO         | `v2/info`                             | `SMCDecodeContext` (cycle) + `SMCDraftInputV2` (batch.spec_info). |
| Worker          | `v2/worker`                           | `SMCWorkerV2` — draft AR, verify, logprob diff, bonus.            |
| Kernels         | `v2/kernels/fused_collect`            | Fused normalize + ESS + systematic resample + compaction.         |
| Kernels         | `v2/kernels/fused_resample_kv`        | Fused block-table dec_ref / copy / inc_ref over many jobs.        |
| KV memory       | `mem_cache/allocator`                 | `SMCRefCountedTokenAllocator`, `copy_block_table`.                |
| TP worker       | `managers/smc_tp_worker`              | Installs refcounted allocator on the target runner.               |
| Target runner   | `model_executor/smc_model_runner`     | Allocator swap, spec-info shape for warmup.                       |
| Graph runner    | `model_executor/smc_cuda_graph_runner`| Emits `SMCVerifyInput` during graph capture.                      |
| Common          | `common/utils`                        | Particle clone, shared-prefix, normalize / ESS / systematic.      |
| Common          | `common/verify`                       | `SMCVerifyInput`, per-particle cache-loc kernel.                  |

---

## 4. Lifecycle of a user request

Five stages. Each step in the cycle is unpacked in
[`pipeline.md`](./pipeline.md); the state it touches is cataloged in
[`state.md`](./state.md).

```
  request ──► [ ADMIT ] ──► [ PREFILL ] ──► [ MATERIALIZE ] ──► [ DECODE CYCLE ] ╶╮
                                                                                 │
                                              ◄───────── repeat until drained ◄──╯
                                                                │
                                                                ▼
                                                         [ FINALIZE ]
                                                                │
                                                                ▼
                                                           response
```

**ADMIT.** A `SequenceGroup(parent_req, n_particles)` is wrapped and pushed
onto `waiting_groups`. Validation rejects parents that request logprobs,
grammar, multimodal, etc. — these don't compose with N-particle decoding.

**PREFILL.** Admitted parents are extended through a standard `ScheduleBatch`
pass on the score model. This samples `x₀` and leaves the committed KV prefix
on the parent.

**MATERIALIZE.** The one-shot fan-out from 1 parent to N particles:

```
                parent Req
             kv_committed_len = L
                    │
                    │  clone_req_for_smc_particle × N
                    ▼
          ┌──────────┬──────────┬─────┬──────────┐
          │ part₀    │ part₁    │ …   │ part_{N−1}
          │ pidx = 0 │ pidx = 1 │     │ pidx = N−1
          └──────────┴──────────┴─────┴──────────┘
                    │
                    │  copy_block_table(parent → part_i, L)
                    │     (inc_ref on each of the L shared KV slots)
                    │  _release_smc_parent_req(parent)
                    │     (dec_ref_and_free the parent's block table)
                    │
                    ▼
         slot_state.allocate_slots(...)
             • N free slots claimed
             • StackedGroupState.register_group → a row of (max_G, N)
             • views: group_log_weights[gid]  →  log_weights[row, :N]
                      group_interval_weights  →  interval_weights[row, :N]
```

After this, the group is in `running_groups` and its N particles live at
fixed slots for the rest of the group's life.

**DECODE CYCLE.** Runs as long as any group has at least one active particle.
Each cycle does:

```
  prepare_for_decode       sparse slots → gather(active) → vectorised KV alloc
       │                   for γ+1 tokens → scatter new lens back to slots
       ▼
  build_model_worker_batch gather active slot tensors → dense ModelWorkerBatch
       │
       ▼
  SMCWorkerV2              draft AR × (γ+1) → TARGET_VERIFY → logprob_diff + bonus
       │
       ▼
  process_batch_result     scatter accepted tokens back into per-slot
       │                   all_token_ids, accumulate weights into the
       │                   stacked row, mark newly finished slots.
       ▼
  coordinator.collect      ESS check per group; either a per-group Python
       │                   pass (slow) or one fused kernel over every
       │                   stacked row (fast).
       ▼
  coordinator.dispatch     copy KV block table + slot tensors + Req
       │                   metadata from src → dst for every resample pair.
       ▼
  rebuild_active_slots     at most once per cycle, only if membership moved.
```

See [`pipeline.md`](./pipeline.md) for each substage in detail.

**FINALIZE.** When a group runs out of active slots, the best particle is
picked by `argmax(log_weight, output_length)`, its `output_ids` /
`finished_reason` are copied onto the parent `Req`, the group's slots are
freed (allocator refcounts drop), the stacked row is released, and the
parent is streamed out.

---

## 5. Core invariants

These thread through every piece of code in this package. The rest of the
docs lean on them.

1. **Slots are for life.** A particle's slot index is chosen at materialize
   and freed at finalize. Between those, the only thing that changes from
   cycle to cycle is which slots are *active* — expressed by the
   `active_slots` gather vector, not by rebuilding the batch.

2. **Rows are for life.** Each `SequenceGroup` owns one row of
   `StackedGroupState` from materialize to finalize. The dict-style
   `group_log_weights[gid]` and `group_interval_weights[gid]` are thin
   slices into that row — writes land in the stacked tensors, which are
   the ground truth consumed by the fused collect kernel.

3. **`smc_particle_idx` is the column.** The particle index a clone gets
   at `clone_req_for_smc_particle` time *is* its column in the stacked
   row, stable forever.

4. **`active_cell_mask` is about allocation, not finish.** It flips True
   at `register_group`, False at `unregister_group`. Finished particles
   remain in the candidate set for resampling; the `finished_mask` bit
   is then copy-propagated through `resample_copy_slot` / the fused KV
   kernel — so a particle that finishes with high weight can still act
   as a resample ancestor for a dying sibling.

5. **KV is multi-owner via refcount.** Both `copy_block_table` (parent
   fan-out) and the fused KV kernel (resample) `inc_ref` on slots they
   duplicate. `dec_ref_and_free` only returns a slot to the free pool
   when the last owner releases it. This is the *only* mechanism that
   lets N particles safely share a prefix.

6. **Rebuild is deferred.** `process_batch_result` and
   `dispatch_resample_batch` both run with `rebuild_active=False`; the
   scheduler rebuilds `active_slots` once, at the end of the cycle,
   and only if membership changed.

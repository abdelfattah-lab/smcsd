# SMC State Model

This doc catalogs every piece of state the SMC scheduler touches. Three tiers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — Request layer (CPU orchestration)                             │
│    parent Req  ──  SequenceGroup  ──  N particle Reqs                   │
│    waiting_groups / prefill_groups / running_groups                     │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │  materialize
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 2 — Batch state (GPU, two views of the same particles)            │
│    ScheduleBatchSMC        — slot-major, [max_slots]     (sparse)       │
│    StackedGroupState       — group-major, (max_G, N)     (dense)        │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │  forward pass reads / writes KV
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 3 — KV memory                                                     │
│    ReqToTokenPool.req_to_token : [pool_size, max_ctx]  (block table)    │
│    SMCRefCountedTokenAllocator.slot_ref_count : [kv_pool_size]          │
└─────────────────────────────────────────────────────────────────────────┘
```

Each tier is covered below. Lifecycle (who allocates / mutates / frees) is
in [`pipeline.md`](./pipeline.md); this doc is about layout and invariants.

---

## 1. Requests and `SequenceGroup`

A user request starts life as a single parent `Req` (upstream sglang type,
unchanged). SMC-specific fields are `smc_particle_idx` (set on particles)
and `kv_committed_len` (prefix length shared with particles).

A `SequenceGroup` is a thin scheduler-side wrapper. It owns the parent
before prefill and the N particles afterwards:

```
SequenceGroup
├── parent_req              : Req                     # before materialize
├── n_particles             : int                     # = N
├── particle_temperature    : float
└── particle_reqs           : {smc_particle_idx -> Req}    # after materialize
                              (empty before materialize)
```

Particles are built by `clone_req_for_smc_particle`:

```
                  parent Req
                       │
                       │   copy: input_ids, tokenizer, eos_token_ids,
                       │         sampling_params (temperature clamped to ≥1e-5),
                       │         priority, routing_key, extra_key, surr_offset,
                       │         read_offset, lora_id
                       │
                       │   reset: return_logprob, top_logprobs_num,
                       │          return_hidden_states, session, grammar,
                       │          disagg_*, http_worker_ipc, metrics_collector,
                       │          time_stats, bootstrap_*
                       │
                       │   assign: smc_particle_idx = i
                       │           rid = f"{parent.rid}_smc_p{i}_particle"
                       ▼
            ┌──────────┬──────────┬─────┬──────────┐
            │ part₀    │ part₁    │ …   │ part_{N-1}
            │ pidx = 0 │ pidx = 1 │     │ pidx = N-1
            └──────────┴──────────┴─────┴──────────┘
```

The **shared prefix length** handed to `copy_block_table` and to the
`TARGET_VERIFY` positions is `parent.kv_committed_len`. This equals
`visible_seq_len - 1` whenever the parent has already sampled its `x₀` bonus
during prefill, which is the case by the time the group reaches materialize.

Validation (`validate_smc_parent_req`) rejects parents whose request config
doesn't compose with N-particle decoding: multimodal inputs, input_embeds,
grammars, returned logprobs / hidden states, stop strings, regex stop.

---

## 2. `ScheduleBatchSMC` — slot-major GPU state

Persistent batch state. One slot per particle, claimed at materialize,
freed at finalize. All GPU tensors are of shape `[max_slots]` (or
`[max_slots, ...]`) and sparse — each iteration we gather the active ones
into a dense `ModelWorkerBatch` and scatter results back.

### 2.1 Per-slot tensors

```
                  dtype     shape                    meaning
─────────────────────────────────────────────────────────────────────────
req_pool_indices  int64     [max_slots]             Req → row in req_to_token
seq_lens          int64     [max_slots]             committed tokens in slot
kv_allocated_lens int64     [max_slots]             KV physically alloc'd
verified_ids      int32     [max_slots]             last accepted token (x₀ bonus feed)
token_counts      int32     [max_slots]             #tokens written to all_token_ids
group_indices     int32     [max_slots]             scheduler-side group number
particle_indices  int32     [max_slots]             smc_particle_idx (= stacked column)
finished_mask     bool      [max_slots]             is this particle done
ignore_eos_t      bool      [max_slots]             from sampling_params
max_new_tokens_t  int32     [max_slots]             hard length cap
eos_token_ids_t   int64     [max_slots, 8]          EOS ids (padded with -1)

all_token_ids     int32     [max_slots, max_output_len]   complete history

temperatures      float32   [max_slots, 1]          sampling params
top_ps            float32   [max_slots]             (static after alloc —
top_ks            int32     [max_slots]              SMC worker does its
min_ps            float32   [max_slots]              own temp adjust)
```

`EMPTY_SLOT` (= -1) marks a free slot in `req_pool_indices`,
`group_indices`, `particle_indices`.

### 2.2 Per-slot CPU bookkeeping

```
free_slots          : deque[int]               available slot ids
slot_to_req         : {slot -> Req}            back-pointer for req-side
                                               state (output_ids,
                                               finished_reason, etc.)
slot_to_group_id    : {slot -> group_id}

group_slot_lists    : {group_id -> [slot, ...]}      every particle of
group_n_particles   : {group_id -> int}              a group, sorted by
                                                     smc_particle_idx

# Thin VIEWS into stacked.{log,interval}_weights — see §3
group_log_weights       : {group_id -> tensor[:N]}
group_interval_weights  : {group_id -> tensor[:N]}
```

### 2.3 `active_slots` — the sparse → dense idx mapping

The hot loop never filters or merges batches. Instead:

```
            sparse slot space                  dense batch space
            ─────────────────                  ─────────────────
slot  0  ┌────────────┐                        ┌────────────┐
slot  1  │ group A p0 │    ┌─► batch idx 0 ──► │ group A p0 │
slot  2  │ group A p1 │────┘      1     ───►   │ group A p1 │
slot  3  │   EMPTY    │           2     ───►   │ group A p3 │
slot  4  │ group A p3 │────┐      3     ───►   │ group B p0 │
slot  5  │ group B p0 │    │      4     ───►   │ group B p1 │
slot  6  │ group B p1 │    │                   └────────────┘
slot  7  │ group B p2  │   │
         │ (finished) │    │    active_slots = [1, 2, 4, 5, 6]
         └────────────┘    │    group_active_indptr = [0, 3, 5]
                           │                        │  └── group B: slots 5..6
                           │                        └───── group A: slots 1..4
                           │
                           └── the gather vector passed to GPU ops
```

`active_slots` is a length-`num_active` int64 GPU tensor; `group_active_indptr`
is its CSR-style group boundary on the CPU. Sorting by group lets per-group
slices of `logprob_diff` land in the right stacked row without any dict
building on the hot path.

`rebuild_active_slots` — which produces this tensor — is **O(G·N)** and runs
**at most once per decode cycle**, only if membership changed. Both
`process_batch_result` and `dispatch_resample_batch` take a `rebuild_active`
flag that defaults to False in the scheduler.

### 2.4 Slot lifecycle

```
  free_slots.popleft()
         │
         ▼
  [CLAIMED]  ── allocate_slots ─── tensors filled, particle_idx set,
         │                          stacked row registered, views bound
         ▼
  [ACTIVE]   ── per cycle: gather → forward → scatter → weight update
         │                           │
         │                           └─ possibly: finished_mask[slot] = True
         │                              → dropped from next active_slots
         ▼
  [FINISHED] ── held for resample ancestry (see §3) but excluded from gather
         │
         ▼
  free_group_slots(gid)
         │       • req_to_token_pool.free(Req)
         │       • dec_ref_and_free on allocated KV slice
         │       • slot tensors reset to EMPTY_SLOT / 0 / False
         │       • slot pushed back onto free_slots
         ▼
  [FREE]
```

---

## 3. `StackedGroupState` — group-major GPU state

Primary storage for per-group SMC state. Shape `(max_G, N)`, one row per
group, one column per particle index:

```
              col 0   col 1   col 2   col 3   col 4   col 5   col 6   col 7
            ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
   row 0    │  w₀   │  w₁   │  w₂   │   0   │   0   │   0   │   0   │   0   │
            │  s₃   │  s₄   │  s₅   │  -1   │  -1   │  -1   │  -1   │  -1   │
            │   T   │   T   │   T   │   F   │   F   │   F   │   F   │   F   │   n_active=3, in_use
            ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
   row 1    │   —    free row, on the _free_rows stack   —                  │   n_active=0, not in_use
            ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
   row 2    │  w₀   │  w₁   │  w₂   │  w₃   │  w₄   │   0   │   0   │   0   │
            │  s₉   │ s₁₀   │ s₁₁   │ s₁₂   │ s₁₃   │  -1   │  -1   │  -1   │
            │   T   │   T   │   T   │   T   │   T   │   F   │   F   │   F   │   n_active=5, in_use
            ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
   row 3    │   —   free —                                                  │
            └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

            log_weights     (max_G, N)  float64     ─── cumulative log-weight
            interval_weights(max_G, N)  float64     ─── resets on resample
            particle_to_slot(max_G, N)  int32       ─── column → slot index
            active_cell_mask(max_G, N)  bool        ─── allocated, not finished
            n_active        (max_G,)    int32       ─── number of used columns
            row_in_use      (max_G,)    bool        ─── row is claimed
```

### 3.1 Column semantics

The column of a particle is its `smc_particle_idx` — assigned at
`clone_req_for_smc_particle`, fixed for life. This is the same integer
stored in `ScheduleBatchSMC.particle_indices[slot]`. So:

```
  stacked.log_weights[row, slot_state.particle_indices[slot]]
  == this particle's cumulative log-weight
```

### 3.2 `active_cell_mask` is **allocation**, not **finish**

Flipped True at `register_group`, False at `unregister_group`. It does
**not** flip when a particle finishes. This is deliberate — the resample
candidate set is "all allocated particles", including finished ones.
`finished_mask` is then copy-propagated by `resample_copy_slot` / the fused
KV kernel, so a finished-but-high-weight particle can still act as a
resample ancestor for a dying sibling.

### 3.3 Views: the dict-style API is a thin slice

```
  slot_state.group_log_weights[gid]
     └── is a VIEW of:  stacked.log_weights[row, :n_particles]

  slot_state.group_interval_weights[gid]
     └── is a VIEW of:  stacked.interval_weights[row, :n_particles]
```

Writes through these dict entries land in the stacked tensors. The stacked
tensors are the single source of truth consumed by `fused_collect`; the
dicts exist only because the slow-path Python coordinator reads them.

### 3.4 Scratch buffers for the fused kernel

One `StackedGroupState` also owns persistent scratch, sized to the worst
case `max_G * N`. Reused across every decode step — no per-step allocation:

```
scratch_dst_flat        int32  [max_G * N]    dead-slot destinations
scratch_src_flat        int32  [max_G * N]    surviving-slot sources
scratch_row_of_job      int32  [max_G * N]    which row each (dst,src) came from
scratch_counter         int32  [1]            atomic add target for job count
scratch_resample_mask   int32  [max_G]        per-row "did we resample" flag
```

One fused collect kernel launch writes all five, atomically packing
`(dst, src, row)` triples into the flat arrays in completion order. The
only boundary sync is a single `.item()` to read `scratch_counter` and
slice the outputs down to `n_jobs`.

---

## 4. `SMCDecodeContext` — per-cycle scratch

Not persistent. Built once per decode cycle by
`ScheduleBatchSMC.prepare_for_decode` from gathered slot tensors, consumed
by `SMCWorkerV2._forward_decode`, thrown away at cycle end.

```
SMCDecodeContext
├── orig_seq_lens        (bs,) int64   seq_lens BEFORE advancing by γ+1
├── orig_seq_lens_cpu    (bs,) int64   CPU copy
├── orig_seq_lens_sum    int           scalar sum
├── new_seq_lens         (bs,) int64   orig + γ+1
└── gamma                int           γ (not γ+1)
```

The factory `from_slot_gather` is where the vectorised KV allocation
happens: one `alloc_token_slots` call for the total `γ+1·bs` new tokens,
one `assign_req_to_token_pool_func` kernel to write the new slots into
each particle's block table. That replaces what used to be a per-request
Python loop.

`SMCDraftInputV2` is the pure-data carrier on `batch.spec_info` — just
`verified_id`, `logprob_diff`, `num_tokens_per_req`, and the
`SMCDecodeContext` attached by `prepare_for_decode`. It has no prepare
methods; those live on the context.

---

## 5. KV memory: `SMCRefCountedTokenAllocator`

Wraps the upstream token-level KV allocator and adds one tensor:

```
slot_ref_count : int32 [kv_pool_size + 1]
```

### 5.1 API

```
alloc(n)              → super().alloc(n); slot_ref_count[picked] = 1
free(indices)         → slot_ref_count[indices] = 0; super().free(indices)
inc_ref(indices)      → slot_ref_count[indices] += 1
dec_ref_and_free(ix)  → slot_ref_count[ix] -= 1
                        free(ix[ slot_ref_count[ix] == 0 ])
```

`inc_ref` / `dec_ref_and_free` are used wherever a KV slot gains or loses
an owner without a fresh allocation: parent → particle fan-out
(`copy_block_table`) and resample src → dst copies (slow path and
`fused_resample_kv` alike).

### 5.2 `copy_block_table`

Used once per particle at materialize. Clones `seq_len` block-table
entries from the parent Req's row of `req_to_token` into the particle's
row, and `inc_ref`s every cloned slot:

```
             parent.req_pool_idx                   partᵢ.req_pool_idx
  req_to_token[parent, :L]   ── clone ──►   req_to_token[partᵢ, :L]
                                                 │
                                                 └─ inc_ref(those slots)
```

After N particles have copied, every shared slot has refcount `N+1`
(parent + N particles). `_release_smc_parent_req` then `dec_ref_and_free`s
the parent's block table, leaving each shared slot at refcount `N` — owned
by the N particles, not yet free.

### 5.3 Refcount state machine (one slot's life)

```
            UNALLOC  ── alloc(n) ──►  OWNED (rc=1)
                                          │
                     ┌────────────────────┼──────────────────┐
                     │                    │                  │
            copy_block_table    fused KV kernel    resample slow path
              (parent fanout)   (src bump, dst    (inc_ref src)
                     │           dec_ref)                    │
                     ▼                    │                  ▼
                 SHARED (rc≥2) ◄──────────┘ ◄────────────── …
                     │
                     │  dec_ref_and_free
                     │    (parent release /
                     │     loser resample /
                     │     finalize)
                     ▼
               rc == 0 ? ── no ──► still SHARED (another owner remains)
                     │
                     yes
                     ▼
                   FREED  ── super().free() ──►  UNALLOC
```

Key invariants:

- **Shared-prefix slots are always multi-owner** between materialize and
  finalize. They never dip below `N` owners while the group is alive.
- **Resample hands off ownership.** On each (dst, src) pair: the dst's old
  KV slots are dec_ref'd (free'd if they hit 0); the src's current KV slots
  are inc_ref'd (now co-owned by dst).
- **`free_group_begin/end`** (upstream API) brackets the slow-path resample
  dispatch so the allocator can coalesce frees; the fast path batches frees
  inside the kernel wrapper's return `to_free`.

### 5.4 Block-table layout, worked example

Group of N=2 particles, shared prefix L=3, each having extended by 2 more
tokens since materialize:

```
req_to_token
───────────────────────────────────────────────────────────────
parent row      (freed)

part₀ row       [ K₀ , K₁ , K₂ , K₃ , K₄ , – , … ]
                 └──shared──┘  └── part₀-only ──┘
part₁ row       [ K₀ , K₁ , K₂ , K₅ , K₆ , – , … ]
                 └──shared──┘  └── part₁-only ──┘

slot_ref_count  K₀:2  K₁:2  K₂:2     (shared, owned by both particles)
                K₃:1  K₄:1  K₅:1  K₆:1    (single-owner tails)
```

On a resample where part₁ copies from part₀:

```
before:   part₁ owns K₅, K₆   (rc 1 each)
          part₀ owns K₃, K₄   (rc 1 each)

dec_ref(K₅, K₆)   → rc 0 each → freed to pool
inc_ref(K₃, K₄)   → rc 2 each → now co-owned

after:    req_to_token[part₁, :5] = [K₀, K₁, K₂, K₃, K₄]
          slot_ref_count: K₀:2 K₁:2 K₂:2 K₃:2 K₄:2
                          K₅: free   K₆: free
```

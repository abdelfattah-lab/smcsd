# SMC Particle-Group Aware VLLM Backend — Draft Model Design

**Scope**: within a single `execute_model` call — draft model runs γ+1
autoregressive decode steps over N particles, then the target model immediately
scores those tokens.  Both draft and target run on the same GPU.
Resampling is **out of scope** for this doc.

---

## 1. Overview

vllm's `GPUModelRunner` manages requests as flat, independent rows in
`RequestState` and `BlockTables`.  It currently has no concept of SMC **SequenceGroup**:
one parent request -> N particles that share a prompt prefix and are jointly advanced each decode cycle.

We want to make the gpu model runner **aware of the SequenceGroup concept**. 

---

## 2. Scheduler-side changes (SMCVLLMScheduler)

The scheduler owns block allocation and CPU-side request state.

1. Track particle groups at the CPU level
2. Fork a parent request into N particles when its prefill completes
3. Pre-allocate KV blocks for all N particles (shared prefix + per-particle
   decode slots for γ+1 tokens)
4. Return `SMCSchedulerOutput` with particle group data; `EngineCore` passes it
   to `Worker` → `SMCGPUModelRunner.execute_model`, which dispatches `smc_draft_cycle`
5. Update CPU-side state after each draft cycle

Injected via the existing plugin point:
`vllm_config.scheduler_config.scheduler_cls` -> picked up by `engine/core.py:130`.

### 2.1 CPU-side particle group state

```python
@dataclass
class SMCParticleGroupState:
    parent_req: Request                              
    n_particles: int
    num_computed_tokens: int                         # parent prompt length at fork
    seed_token_ids: list[int]                        # per particle; updated after each cycle
    # Allocated at fork:
    shared_prefix_block_ids: tuple[list[int], ...]   # [kv_group][block]  — read-only
    decode_block_ids: list[tuple[list[int], ...]]    # [particle][kv_group][block]
```

`SMCVLLMScheduler` adds:

```python
self.smc_groups: dict[str, SMCParticleGroupState] = {}  # parent_req_id -> state
```

### 2.2 `fork_blocks` on SingleTypeKVCacheManager

**`SingleTypeKVCacheManager.fork_blocks`**:

```python
def fork_blocks(
    self,
    parent_req_id: str,
    particle_req_ids: list[str],
    n_decode_blocks: int,          # fresh decode blocks to allocate per particle
) -> tuple[list[int], list[list[int]]]:
    prefix_blocks = list(self.req_to_blocks.get(parent_req_id, []))

    # Touch prefix blocks once per particle: ref_cnt 1 -> 1+N.
    for _ in particle_req_ids:
        self.block_pool.touch(prefix_blocks)

    # Allocate decode blocks and register each particle.
    decode_block_ids: list[list[int]] = []
    for p_id in particle_req_ids:
        dec_blocks = self.block_pool.get_new_blocks(n_decode_blocks)
        self.req_to_blocks[p_id] = prefix_blocks + dec_blocks   # dict assignment
        decode_block_ids.append([b.block_id for b in dec_blocks])

    # Free parent: ref_cnt of prefix blocks 1+N -> N; parent entry removed.
    self.free(parent_req_id)

    return [b.block_id for b in prefix_blocks], decode_block_ids
```

`KVCacheManager.fork_blocks` (public entry point) iterates
`self.coordinator.single_type_managers` and delegates to each manager's
`fork_blocks`, following the same pattern as `take_new_block_ids`.

As each particle finishes, `kv_cache_manager.free(particle_req_id)` decrements
`ref_cnt` on the shared prefix blocks; the last particle drives `ref_cnt` to 0.

### 2.3 Fork: parent request → N particles

Called once when the parent finishes prefill.

```
fork_to_particles(parent_req_id, n_particles, seed_token_id):
    1. Compute n_decode_blocks = ceil((gamma + 1) / block_size)
    2. Call kv_cache_manager.fork_blocks(
           parent_req_id, particle_req_ids, n_decode_blocks)
       → prefix blocks: ref_cnt 1 → N (parent relinquishes ownership)
       → each particle registered with prefix_blocks + fresh decode_blocks
       → returns prefix_block_ids, decode_block_ids[particle]
    3. For each particle p:
       compute kv_slots[p, :] = physical slot IDs for positions
       [num_computed_tokens … num_computed_tokens + gamma]
       (see §2.4 for slot formula)
    4. Build NewParticleGroupData(group_id, prefix_block_ids, decode_block_ids,
                                   seed_token_ids, kv_slots, seq_lens, ...)
       and add to SMCSchedulerOutput.new_particle_groups
       → runner calls add_particle_group in execute_model (see §3.3)
    5. Store SMCParticleGroupState(group_id=parent_req_id, ...) in self.smc_groups
```

### 2.4 KV slot computation

The scheduler converts block IDs to flat physical slot IDs before sending them
to the runner.  For each particle `p`, kv-group `g`, and draft step `s`:

```
position     = num_computed_tokens + s
block_index  = position // block_size
block_offset = position % block_size
slot_id[g,p,s] = decode_block_ids[p][g][block_index] * block_size + block_offset
```

`kv_slots` is shaped `[num_kv_groups, P, γ+1]`.  For standard single-attention-group
models this collapses to `[1, P, γ+1]`.  `build_slot_mappings_by_layer` in the draft
loop receives this tensor and expands it to per-layer slot mappings as needed.

### 2.5 SMCSchedulerOutput

Mirrors vllm's separation of `scheduled_new_reqs` vs `scheduled_running_reqs`.

```python
@dataclass
class NewParticleGroupData:
    """Carries block IDs for the fork step — runner calls add_particle_group."""
    group_id: str
    particle_req_ids: list[str]
    prompt_token_ids: list[int]
    num_computed_tokens: int
    prefix_block_ids: tuple[list[int], ...]        # [kv_group][block]
    decode_block_ids: list[tuple[list[int], ...]]  # [particle][kv_group][block]
    # first draft cycle data (fork and first draft run in the same execute_model):
    seed_token_ids: torch.Tensor   # [P]
    kv_slots: torch.Tensor         # [num_kv_groups, P, γ+1]
    seq_lens: torch.Tensor         # [P]
    gamma: int
    temperature: float

@dataclass
class SMCGroupBatch:
    """Ongoing draft cycle — group already registered, no block IDs needed."""
    group_id: str
    seed_token_ids: torch.Tensor   # [P]
    kv_slots: torch.Tensor         # [num_kv_groups, P, γ+1]
    seq_lens: torch.Tensor         # [P]
    gamma: int
    temperature: float

@dataclass
class SMCSchedulerOutput(SchedulerOutput):
    new_particle_groups: list[NewParticleGroupData] = field(default_factory=list) # new group where we need to create row idx
    smc_groups: list[SMCGroupBatch] = field(default_factory=list) # ongoing draft groups
```

---

## 3. SMCGPUModelRunner

### 3.1. ParticleGroup (new, lives on SMCGPUModelRunner)

```python
@dataclass
class ParticleGroup:
    group_id: str             # parent request_id
    particle_rows: list[int]  # row indices into RequestState / BlockTables
    n_particles: int
```

`SMCGPUModelRunner` adds:

```python
self.particle_groups: dict[str, ParticleGroup] = {}  # group_id → ParticleGroup
```

**Row index assignment and ownership.**

`particle_rows` contains the persistent row indices that identify each particle
throughout its lifetime.  These are the same indices used to address every
`[max_num_reqs, ...]` tensor in `RequestState` and `BlockTables` (the same
index space shared by all normal requests).

Row indices are assigned and ownwed by the **runner**.

### 3.2 RequestState rows (reuse existing, one row per particle)

Each particle occupies one row index in the existing tensors:

| tensor | shape | meaning per particle row |
|---|---|---|
| `all_token_ids` | `[max_num_reqs, max_model_len]` UVA | full token sequence |
| `num_computed_tokens` | `[max_num_reqs]` UVA | tokens with KV computed; **frozen during draft** |
| `total_len` | `[max_num_reqs]` GPU | current sequence length |
| `last_sampled_tokens` | `[max_num_reqs, 1]` GPU | last accepted/sampled token |

All particles in a group start with the same `num_computed_tokens` = parent's
prompt length.  During the draft cycle these values do not change; they are
updated in bulk after verify+accept.

### 3.3 BlockTables rows (reuse existing, one row per particle)

Each particle row in `block_tables[g][r, :]` stores:
- **Prefix blocks** (copied from parent at fork): read-only during draft
- **Decode blocks** (pre-allocated by scheduler): written during draft

```
particle row r: [pfx_blk0, pfx_blk1, ..., dec_blk0, dec_blk1, ...]
```

### 3.4 `SMCGPUModelRunner` API

```python
class SMCGPUModelRunner(GPUModelRunner):
    """Extends GPUModelRunner with particle-group awareness."""

    particle_groups: dict[str, ParticleGroup]

    # lifecycle
    def add_particle_group(
        self,
        group_id: str,
        particle_req_ids: list[str],
        prompt_token_ids: list[int],
        num_computed_tokens: int,
        prefix_block_ids: tuple[list[int], ...],   # per kv-group, shared prefix
        decode_block_ids: list[tuple[list[int], ...]],  # [particle_idx][kv_group]
    ) -> None:
        """Register N particles, assigning row indices internally."""
        ...

    def remove_particle_group(self, group_id: str) -> None:
        """Free all particle rows.  Called after verify+resample committed."""
        ...

    # Draft + verify cycle

    def smc_draft_cycle(
        self,
        particle_rows: torch.Tensor,    # [P] int32 — row indices
        seed_token_ids: torch.Tensor,   # [P] int32 — last accepted token
        kv_slots: torch.Tensor,         # [num_kv_groups, P, γ+1] int64 — pre-alloc'd write slots
        seq_lens: torch.Tensor,         # [P] int32 — num_computed_tokens + already-written
        gamma: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run γ+1 draft decode steps for all P particles.

        Returns:
            draft_token_ids  [P, γ+1]    — tokens sampled at each step (step 0 = seed)
            draft_log_probs  [P, γ+1, V] — log-softmax'd logits at each step
        """
        ...

    def smc_target_score(
        self,
        particle_rows: torch.Tensor,   # [P]
        draft_token_ids: torch.Tensor, # [P, γ+1]
        seq_lens: torch.Tensor,        # [P] — num_computed_tokens at draft start
        kv_slots: torch.Tensor,        # [num_kv_groups, P, γ+1] — same slots as draft
    ) -> torch.Tensor:
        """Score all draft tokens with the target model in one batched pass.

        Runs the target model on the P×(γ+1) draft tokens attending to the
        prefix KV already in cache.  On single GPU the target model is the
        same model used for drafting; the draft KV written during smc_draft_cycle
        is reused here.

        Returns:
            target_log_probs  [P, γ+1, V]
        """
        ...
```

### 3.4 `smc_draft_cycle`
`_gather_block_tables(particle_rows)` is called **once before the loop**.
It takes sparse row indices and produces a dense tensor
where position `i` holds the block table for `particle_rows[i]`. 

``` python
P = len(particle_rows)
block_tables     = self._gather_block_tables(particle_rows)          # [num_kv_groups, P, max_blocks]; once before loop
draft_token_ids  = torch.zeros(P, gamma + 1, dtype=torch.int32,  device=self.device)
draft_log_probs  = torch.zeros(P, gamma + 1, self.vocab_size, dtype=torch.float32, device=self.device)
draft_token_ids[:, 0] = seed_token_ids  # last accepted token; starting point for step 0
seq_lens_cur = seq_lens.clone()          # tracks KV sequence length as steps advance

for step in range(gamma + 1):
    input_ids = draft_token_ids[:, step]   # [P]
    positions  = seq_lens_cur              # [P]
    slot_ids   = kv_slots[:, :, step]      # [num_kv_groups, P]

    # attn_metadata and slot_mappings_by_layer are built directly from our particle tensors 
    attn_metadata = self._build_draft_attn_metadata(
        block_tables, seq_lens_cur + 1, slot_ids
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_ids, self.kv_cache_config   # slot_ids expanded per kv-group
    )

    with set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens              = P,
        slot_mapping            = slot_mappings_by_layer,
        cudagraph_runtime_mode  = CUDAGraphMode.NONE,   # eager, no graph for draft
        batch_descriptor        = BatchDescriptor(num_tokens=P, has_lora=False),
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=None,   
            inputs_embeds=None,        
        )  # [P, hidden_dim]

    logits = self.model.compute_logits(hidden_states)  # [P, vocab_size]

    log_probs = torch.log_softmax(logits / temperature, dim=-1)   # [P, V]
    draft_log_probs[:, step] = log_probs

    if step < gamma:
        next_tokens = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
        draft_token_ids[:, step + 1] = next_tokens

    seq_lens_cur += 1

return draft_token_ids, draft_log_probs
```

### 3.5 `_gather_block_tables` for particles
Block tables are needed by the attention backend. 

```python
def _gather_block_tables(self, particle_rows: torch.Tensor) -> tuple[Tensor, ...]:
    # Use existing BlockTables.gather_block_tables() with particle_rows as idx_mapping.
    return self.block_tables.gather_block_tables(particle_rows, num_reqs_padded=len(particle_rows))
```

### 3.6 `execute_model` override

```python
def execute_model(
    self, scheduler_output: SMCSchedulerOutput
) -> ModelRunnerOutput:
    # Prefill: parent requests still prefilling, delegated to base runner.
    output = super().execute_model(scheduler_output)

    # Newly forked groups: register rows in RequestState + BlockTables,
    # then immediately run their first draft cycle.
    for new_group in scheduler_output.new_particle_groups:
        self.add_particle_group(
            group_id             = new_group.group_id,
            particle_req_ids     = new_group.particle_req_ids,
            prompt_token_ids     = new_group.prompt_token_ids,
            num_computed_tokens  = new_group.num_computed_tokens,
            prefix_block_ids     = new_group.prefix_block_ids,
            decode_block_ids     = new_group.decode_block_ids,
        )
        particle_rows = self.particle_groups[new_group.group_id].particle_rows
        draft_token_ids, draft_log_probs = self.smc_draft_cycle(
            particle_rows, new_group.seed_token_ids,
            new_group.kv_slots, new_group.seq_lens,
            new_group.gamma, new_group.temperature,
        )
        # target verify

    # Ongoing groups: already registered
    for group_batch in scheduler_output.smc_groups:
        particle_rows = self.particle_groups[group_batch.group_id].particle_rows
        draft_token_ids, draft_log_probs = self.smc_draft_cycle(
            particle_rows, group_batch.seed_token_ids,
            group_batch.kv_slots, group_batch.seq_lens,
            group_batch.gamma, group_batch.temperature,
        )
        # target verify

    return output
```

---

## 5. Flow summary

```
# Fork step (parent prefill just completed):
Scheduler.schedule()
    → kv_cache_manager.fork_blocks(...)
        → prefix blocks: ref_cnt 1 → N; each particle gets prefix + decode blocks
    → return SMCSchedulerOutput(new_particle_groups=[NewParticleGroupData(...)], smc_groups=[])

SMCGPUModelRunner.execute_model()
    → super().execute_model()            # normal prefill/decode requests
    → for new_group in new_particle_groups:
          add_particle_group(...)        # assigns particle_rows, populates BlockTables
          smc_draft_cycle(...)           # first draft cycle runs immediately
          # target verify

# Subsequent cycles (particles already registered):
Scheduler.schedule()
    → return SMCSchedulerOutput(new_particle_groups=[], smc_groups=[SMCGroupBatch(...)])

SMCGPUModelRunner.execute_model()
    → super().execute_model()
    → for group_batch in smc_groups:
          particle_rows = self.particle_groups[group_batch.group_id].particle_rows
          smc_draft_cycle(...)           # lookup particle_rows locally, no block IDs needed
          # target verify

# ← resampling: out of scope for this doc
```

---

## 6. Notes
SMCScheduler

- Request is CPU state and lives at scheduler/engine level
- scheduler allocates/free block
- smc scheduler need to fork a parent request into N particles when its prefill completes
- smc scheduler need to pre-allocate kV blocks for all N particles
- SMC doesn’t change prefill: let prefill flow through the vllm base scheduler and base model runner
- smc scheduler scheduler() is responsible for populating running_req (for normal prefill), new_particle_groups (register rows) and smc_groups (ongoing draft) and return in scheduler output

SMCGPUModelRunner

- row idx contains the persistent row indices that identify each request through its lifetime. Row indices are assigned and owned by the runner
- smc gpu runner: add_particle_group: allocate N RequestState rows
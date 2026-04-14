# SMC Speculative Decoding Architecture

SMC runs **N particles** (parallel generation paths) per request. A draft model proposes tokens, the target model scores them, and particles are resampled by importance weight so compute focuses on promising paths. Unlike standard speculative decoding, **no tokens are rejected** -- low-weight particles are killed and replaced with clones of high-weight ones.

## Core Workflow

```
User Request
  |
  v
[1] PREFILL  (target + draft, normal path)
  |
  v
[2] CREATE GROUP  (spawn N particles, share parent KV via refcount)
  |
  v
[3] DECODE LOOP  --------+
  |                       |
  |  [a] Draft   (draft model: gamma+1 AR steps per particle)
  |  [b] Score   (target model: single extend pass over all drafted tokens)
  |  [c] Weight  (log_w += target_logprob - draft_logprob)
  |  [d] Accept  (all tokens accepted + bonus from target)
  |  [e] Resample (if ESS < threshold: clone winners, kill losers)
  |                       |
  +-------(until done)----+
  |
  v
[4] FINALIZE  (best particle by log_weight -> user output)
```

## Key Files

All under `python/sglang/srt/`.

| File | Role |
|------|------|
| `smc/smc_workers.py` | Draft + score forward passes, bonus token sampling |
| `smc/smc_info.py` | `SMCDraftInput` / `SMCVerifyInput`, KV allocation, seq_lens |
| `smc/smc_manager.py` | Group lifecycle: create, track, finalize |
| `smc/smc_resampler.py` | ESS check, ancestor sampling, KV row copy, stall/resume |
| `smc/smc_utils.py` | Particle cloning, `systematic_resample`, weight math |
| `managers/scheduler_output_processor_mixin.py` | Prefill/decode result hooks |
| `managers/schedule_batch.py` | `SMCGroupSpan`, per-request SMC fields |

---

## KV-Cache Architecture

### Shared Prefix (Group Creation)

All particles share the prompt's KV via **reference counting** -- no data copy.

```
Parent KV:  [page0][page1][page2]  (prompt prefix)
                |       |      |
Particle 0: [page0][page1][page2]   refcount += 1 per particle
Particle 1: [page0][page1][page2]
Particle 2: [page0][page1][page2]
Particle 3: [page0][page1][page2]
```

### Per-Cycle Allocation (Draft Phase)

Before each draft loop, **gamma+1 KV slots are pre-allocated** per particle:

```
alloc_token_slots(kv_allocated_len .. seq_len + gamma + 1)
```

A Triton kernel maps these into `req_to_token` rows so the draft AR loop can write KV at each step.

### Resampling (Index Rewrite, Not Data Copy)

Resampling is cheap: **O(seq_len) index copies**, not O(seq_len * hidden_dim) tensor copies.

For each eviction `(dst, src)` where dst is being overwritten by src:

```
1. dec_ref_and_free(req_to_token[dst, :dst_len])    # release loser's KV refs
2. src_indices = req_to_token[src, :src_len]         # snapshot winner's row
3. req_to_token[dst, :src_len] = src_indices         # point dst at winner's KV
4. inc_ref(src_indices)                              # bump refcounts
```

After resampling, dst and src share the same physical KV slots (like prefix sharing).

---

## seq_lens Lifecycle

The trickiest part of the system. One full decode cycle with `committed_len=100`, `gamma=3`:

```
                            seq_lens   kv_committed   kv_allocated
                            --------   ------------   ------------
Start of cycle:               100          100            100

prepare_for_decode():
  save _orig_seq_lens          (saved as 100)
  alloc gamma+1 KV slots                                  104
  advance on schedule stream   104          100            104

Draft AR loop:
  step 0 (fwd seq_len):       101*         100            104
  step 1:                     102*         100            104
  step 2:                     103*         100            104
  step 3:                     104*         100            104

Score/verify (fwd):            100**        100            104

Accept gamma+1 tokens:         104          104            104

Post-decode refresh:           104          104            104
```

`*` Per-step forward seq_lens = orig + step + 1.
`**` Verify resets to `orig_seq_lens`; draft tokens enter via extend-style causal mask.

### Key Invariants

- **Schedule stream advances early**: `seq_lens += gamma+1` before draft starts (safe tensor return without sync).
- **Draft uses per-step seq_lens**: each step sees `orig + step + 1`.
- **Verify uses orig_seq_lens**: prefix-only; draft tokens handled by extend attention.
- **Post-decode refresh**: `seq_lens` rebuilt from `kv_committed_len`.
- **kv_committed_len <= kv_allocated_len**: allocated is the watermark, committed tracks accepted tokens.

---

## Resampling

When all particles complete a step and are aligned:

1. Flush deferred `pending_diffs` to GPU log_weights
2. Normalize: `w = softmax(log_weights)`
3. ESS = `1 / sum(w^2)`
4. If `ESS < N * threshold` (default 0.5):

```
Before (weights [0.6, 0.3, 0.05, 0.05]):
  P0: "The answer is 42..."     w=0.6
  P1: "The result equals 42..." w=0.3
  P2: "I don't know..."         w=0.05
  P3: "Maybe 7..."              w=0.05

Ancestors = [0, 0, 1, 0]  (systematic resample)

After (weights reset to 0):
  P0: "The answer is 42..."     w=0  (kept)
  P1: "The answer is 42..."     w=0  (cloned from P0)
  P2: "The result equals 42..." w=0  (cloned from P1)
  P3: "The answer is 42..."     w=0  (cloned from P0)
```

During resampling, all group particles are removed from `running_batch`. After KV row copies and state restore, active particles re-enter the batch.

---

## Finalize

When all particles finish: flush weights, select `argmax(log_weight)` (tiebreak on length), copy best particle's output to the parent request, release all particle KV.

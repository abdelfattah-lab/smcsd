# SMC Decode Performance — the bs=1 Optimization Campaign

This doc explains the decode-throughput optimizations landed on
`perf/bs1-decode` (smcsd PR #29 + sglang fork PR #8), measured on B200 with
Llama-3.1-8B target + Llama-3.2-1B draft at the single-stream operating
point: **1 request, N=8 particles, γ=8**.  It has two halves: a
plain-language explainer of *why* each change works, then the reference
tables (results, flags, kernel inventory).  Companions:
[`architecture.md`](./architecture.md) for the system map,
[`pipeline.md`](./pipeline.md) for the per-stage data flow.

## Results first

| workload (1024-token outputs) | before (stock launch) | after (bare launch) |
|---|---|---|
| 256-token prompts  | 522 tok/s | **677 tok/s** (+30%) |
| 1024-token prompts | —         | **641 tok/s** |
| 4096-token prompts | ~280 tok/s | **522 tok/s** (+79%) |
| GSM8K 200q, 3-seed mean | 508 tok/s | **567 tok/s** (+11.6%) |

GSM8K accuracy is unchanged (70.2% vs 71.3% across seeds, within noise):
every change below computes the same math with less waste — nothing is
approximated, and the sampled distributions are identical.

---

## Part 1 — Plain-language explainer

### What one decode cycle does

With N=8 particles and γ=8, one cycle is:

1. The **draft** model (1B) runs 8 times back-to-back, each run guessing one
   more token for all 8 particles (a batch of 8 rows per run).
2. The **target** model (8B) runs **once**, scoring all 72 drafted tokens
   (8 particles × 9 positions) in a single verify pass.
3. Sampling + importance-weight math, state bookkeeping, repeat.

Each cycle emits 9 tokens per particle.  Before this campaign a cycle took
~16 ms; the whole game is asking where inside those 16 ms the time goes.
Profiling split it into two very different kinds of waste.

### Waste type 1: the GPU doing its work badly

A GPU is thousands of tiny workers next to a giant warehouse (device
memory, holding the weights and the KV cache).  Fetching from the warehouse
is the slow part; arithmetic is nearly free.  A kernel is efficient when
many workers fetch *different* data in parallel, and inefficient when
workers idle or fetch the *same* box repeatedly.

**Offender #1: verify attention (41% of all decode GPU time at 4k
context).**  When the target checks the 72 drafted tokens, every token must
look back at the whole prompt — i.e. read the prompt's cached keys/values
out of the warehouse.  The stock triton extend kernel did this ~10–25×
slower than the hardware can read, for two reasons:

- *Redundant reads.*  Under GQA, the 32 query heads share K/V in groups
  of 4 — but the kernel assigned one worker-team per **query head**, so the
  same prompt K/V was hauled out 4 times.
- *No parallelism over the prompt.*  Each team walked the entire 4k prefix
  front-to-back alone — one team crawling a long aisle while the aisle
  could have been split.

**Fix — the split-KV, GQA-packed verify kernel**
(`smcsd/core/kernels/verify_attention.py`): one team holds the whole GQA
group's queries (9 tokens × 4 heads = 36 rows), and the prefix is chopped
into 7 chunks processed by different teams simultaneously, with the exact
running-softmax merge combining the partials (flash-decoding style).
319 µs → 64 µs per 8B layer at 4k.  This is the single biggest win
(+60% end-to-end at 4k) and also covers the deferred-bonus 2-token head.

**Offender #2: sampling (as expensive as a draft lm_head, 8× per
cycle).**  Sampling one token from 128k vocabulary scores ran as ~10
separate torch ops — noise, log, log, add, argmax, gather, logsumexp — and
*each op re-reads the 128k scores from the warehouse*, plus fixed per-op
overhead.

**Fix — fused Gumbel sampling** (`fused_sampling.py`): one kernel reads the
scores once and produces the sampled token, its log-probability, and (for
the power target) the log-normalizer in a single pass.  Ten passes → one.
The subtlety is randomness under CUDA graphs: a replayed recording must
still draw *fresh* noise each cycle, so the kernel reads a seed counter
that lives on the GPU and ticks up inside the recording, with disjoint
Philox counter ranges per launch so no two steps or rows share a stream.

### Waste type 2: the GPU doing nothing

The CPU is the manager: it issues kernels one at a time, a few
microseconds each.  A cycle has hundreds of kernels, many finishing faster
than the CPU can issue the next — so the GPU idles.  At the start of the
campaign decode was ~21% idle.

- **CUDA graphs** (the pre-existing `SMC_CYCLE_GRAPH` full-cycle capture)
  record the whole cycle's kernel sequence once; each cycle the CPU says
  one thing: "play the tape."
- **Fix — metadata captured in-graph**: the attention-metadata refresh
  (kv-indptr cumsums, kv-indices gathers, kv-split counts) and the verify
  staging were still issued eagerly before every replay.  All of it reads
  only staged device buffers or the persistent block table, so it was
  moved *inside* the recording.  A replay is now ~8 small copies + one
  launch.  Combined with the pre-existing overlap loop
  (`SMC_ENABLE_OVERLAP`: CPU prepares cycle t+1 while the GPU runs cycle
  t), decode idle went 21% → **3%**.  Outputs are byte-identical.

### Making the fast path the default

All of the above — plus the pre-existing deferred-bonus schedule that saves
one draft forward per cycle — sat behind opt-in flags, so a bare launch got
the slow engine.  `SMCEngine` and the http server now default the best
configuration on (`SMC_CYCLE_GRAPH`, `SMC_ENABLE_OVERLAP`,
`SMC_DEFER_BONUS` for non-hybrid drafts, `attention_backend=triton`,
`triton_attention_num_kv_splits=16`).  Explicit env settings — including
`=0` kill switches — always win; the flags exist to turn things *off* for
triage, ablations, or reproducing old RNG streams.

### The one that didn't pay (yet): cascade decode

SMC particles share their prompt KV pages *by construction* (refcounted
page_size=1 lineage), yet draft decoding re-reads the prompt once per
particle.  A cascade kernel that reads it once per group (one CTA holds
all N×G query rows) was built and measured during this campaign: a
**wash at 4k** — at that size the kernel is bound by scattered-read
latency, not traffic, so reading less doesn't help — but 1.9×/3×/4.2×
over stock at 8k/16k/32k prefixes.  It ships in its own follow-up PR
(#31, `perf/cascade-decode`) rather than here: it is not wired into the hot
path — dispatch cannot branch inside a captured graph, so enabling it
needs dual-graph capture keyed on prompt-length regime — and it deserves
its own review alongside that integration.

### What's left

Short-context decode is now ~97% GPU-busy; the residual is the actual
GEMMs (draft 40%, verify 26%) plus a swarm of tiny norm/rope/activation
kernels (31%) whose fixed overhead dwarfs their math at batch-of-8 scale.
Squeezing that means fusing whole model chunks into single kernels — the
megakernel direction — not more surgery of this kind.  Measured dead ends,
for the record: FP8 draft weights (±0%: draft GEMMs are latency-bound at
M=8, not weight-bandwidth-bound) and in-register K-transpose loads
(slower than the strided load).

---

## Part 2 — Reference

### Flags (all default ON; opt out via `SMCEngine` kwargs or env)

Resolution order per optimization: `SMC_*` env var if set (kill switch for
CLI harnesses) > `SMCEngine(...)` kwarg (`defer_bonus` / `cycle_graph` /
`enable_overlap`, threaded through server_args) > default ON.  Unsupported
configs downgrade with a warning instead of failing.

| flag | what it gates | off-switch use case |
|---|---|---|
| `SMC_CYCLE_GRAPH` | full-cycle CUDA graph (draft AR + verify + weights + bonus in one launch) | capture failures on new configs |
| `SMC_DEFER_BONUS` | deferred-bonus draft schedule (γ instead of γ+1 draft forwards) | hybrid/MLA drafts (auto-skipped by the launch default) |
| `SMC_ENABLE_OVERLAP` | overlapped scheduler loop (postprocess step t during step t+1) | one-step-late semantics debugging |
| `SMC_FAST_VERIFY` | split-KV GQA-packed verify kernel dispatch | widest input surface; first switch to try in triage |
| `SMC_FUSED_SAMPLING` | fused Gumbel sampling in the cycle graph | reproducing pre-change RNG streams |

Plus `--triton-attention-num-kv-splits 16` (kwarg-defaulted; neutral at
short context, +7% at 4k).

### Kernel inventory (all in `smcsd/core/kernels/`, tests in `tests/`)

| kernel | replaces | speedup at the SMC shape |
|---|---|---|
| `verify_attention.py` | triton extend `_fwd_kernel` on the linear TARGET_VERIFY path | 5× at 4k ctx (319→64 µs/8B layer) |
| `fused_sampling.py` | ~10-op torch Gumbel chain + verify score extraction | ~6× on the sampling chain |
| `cascade_decode.py` (follow-up PR #31) | grouped decode with shared-prefix reads | 1.9–4.2× at 8k–32k prefixes; wash at ≤4k |

### Why not FA4 / trtllm?

FA4 and trtllm-gen exist in the vendored sglang and would run verify near
roofline — but on B200 they force KV page sizes of 128 / 16–64, while the
entire SMC state machine (refcounted per-token lineage sharing, fused
prepare/resample/write-back, `req_state.py` hard-requires `page_size=1`).
Adopting them is a paged-KV redesign of resampling (copy-on-write at page
granularity), not a backend swap.  The kernels above capture most of the
win at bs=1 while keeping page_size=1.

# Benchmarking the SMC megakernel vs the current implementation

This answers: *how do you benchmark running smcsd as ONE kernel vs the current implementation, and
where does the speedup come from?*

## What the cycle is

One SMC decode cycle (bs=1, N particles, γ draft steps) does:

```
γ × draft forward (1B)  →  target verify forward (8B)  →  reweight → resample → bonus
```

- **Current implementation (the baseline the megakernel must beat):** runs this as *many* GPU operations.
  Each draft step, the target verify, and the resample are separate kernel launches / CUDA-graph replays,
  with host orchestration and activation+KV round-trips to HBM at every boundary. The shipped stack already
  graphs the cycle (+14–16%), and the roofline measurement showed the bs=1 cycle is **~65% bubble / ~34%
  roofline efficiency** — i.e. most of the time is *not* useful weight-streaming, it's launch/orchestration/
  boundary overhead.
- **Megakernel (`kernels/m7b_full_cycle.py`):** the **entire cycle in ONE persistent kernel launch**. No
  per-op launches, no host round-trips between stages, no grid teardown/relaunch, tokens flow draft→verify
  in-graph.

## How the benchmark works

Measure **per-cycle latency at bs=1**, same (draft model, target model, N, γ), megakernel ON vs OFF:

1. Warm up (first launch JIT-compiles / fills caches — exclude it).
2. Time only the launches with CUDA events, averaged over several iterations.
   - Megakernel: `cute.compile(...)` once, then time `comp(*tensors)` (zeroing the barrier `arr`/`sen`
     tensors each launch — the sense flag persists across launches or the next one deadlocks).
   - Baseline: replay the per-cycle CUDA graph the same number of times.
3. Report inter-token latency (= cycle latency / accepted-tokens-per-cycle) and the per-cycle latency.
4. **Correctness gate:** the megakernel must match the eager torch SMC estimator — drafted tokens, importance
   weights, resample ancestors, and the bonus token. `m7b_full_cycle.py` checks exactly this every run
   (draft 16/16, ancestors EXACT, bonus EXACT, weights within the bf16 floor).

The win, when it materializes, is the megakernel **closing the 65% bubble**: no launch dispatch, no HBM
activation round-trip at kernel boundaries, and (with software pipelining) weight-load of op N+1 overlapping
compute of op N — the levers a graph fundamentally cannot pull.

## Current numbers (B200, bs=1, N=4, γ=4, draft Llama-3.2-1B + target Llama-3.1-8B)

| version | per-cycle latency | notes |
|---|---|---|
| megakernel, naive kernels | 1733 ms | first correct fusion |
| megakernel, verify optimized | 602 ms | warp-GEMV + norm-once + weight-reuse on the 8B verify |
| **megakernel, draft+verify optimized** | **~165 ms** | both phases optimized (this is `m7b_full_cycle.py` today) |
| production smcsd cycle-graph | ~10 ms | (roofline run: 465 tok/s ≈ 10.75 ms/cycle) |

**Honest status:** the megakernel is **~16× slower than production in absolute terms today.** That is NOT a
fusion problem — it is a *kernel-quality* problem: the in-kernel GEMVs are hand-written **CUDA-core** kernels
(~12 TFLOP/s ≈ 15% of CUDA-core peak), while production uses **tensor-core** GEMMs (cuBLAS / FlashInfer) that
run an order of magnitude faster on the 8B verify. The megakernel has already proven the hard part — the whole
multi-model speculative cycle *fuses into one correct launch* — but to be *faster* than production it needs
tensor-core (tcgen05/WGMMA) GEMMs inside the persistent kernel. That is the remaining performance work, and
it is exactly the rung the fusion ladder was built for: once the in-kernel compute is competitive, the 65%
bubble the megakernel removes becomes the decisive margin.

## Isolating the fusion benefit (apples-to-apples)

Absolute "megakernel vs production" conflates two things: kernel quality (tensor vs CUDA cores) and fusion
(one launch vs many). To measure **just the fusion** — the megakernel's actual contribution — compare the same
kernels run two ways:

- **one launch** (the fused `m7b_full_cycle`), vs
- **multi-launch**: the same draft / verify / resample kernels launched separately with a host sync and an
  activation/KV re-read between each.

`T_multi − T_fused` = launch overhead + grid teardown + host orchestration + boundary HBM round-trips removed.
At bs=1 this gap grows as the kernels approach roofline (the fraction that is overhead rises) — which is why
the production stack, whose kernels are near-roofline, is 65% bubble and stands to gain the most from fusion.

## Integration into the worker (for the real A/B)

`smcsd/core/worker.py` runs the cycle behind the CUDA-graph path. To benchmark the megakernel in-system:

1. Add a flag (`SMC_FULL_MEGAKERNEL=1`) selecting the megakernel cycle instead of the graph replay.
2. Adapt the entry: production drafts only the γ decode steps over an **existing KV cache** (the target's
   prefill already populated it) — not the prompt prefill this prototype teacher-forces. Pass the live KV
   cache + start tokens in; free-run the draft (feed back its own sampled token) instead of teacher-forcing.
3. Port the **graph-safe Philox RNG** for the Gumbel/resample noise (this prototype passes precomputed
   uniform-noise tables for deterministic validation).
4. **Zero the barrier `arr`/`sen` tensors before every launch** (the sense flag persists → a 2nd launch on
   dirty state deadlocks).
5. Run the existing equivalence + GSM8K harness with the flag on/off (accuracy-neutral check) and the latency
   harness above (per-cycle latency, sweep N and γ).

## TL;DR

- The benchmark = per-cycle latency at bs=1, megakernel-one-launch vs production-cycle-graph, gated on
  estimator equivalence.
- The megakernel **already does the whole cycle in one correct launch** and is **10.3× faster than its own
  naive version** (1733 → 165 ms).
- It is **not yet faster than production** (CUDA-core vs tensor-core GEMMs); the remaining lever is tensor-core
  in-kernel GEMMs, after which the fusion's bubble-elimination is the win.

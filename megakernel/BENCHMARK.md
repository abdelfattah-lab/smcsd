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

## Isolating the fusion benefit (measured — `benchmarks/bench_fusion.py`)

Absolute "megakernel vs production" conflates kernel quality (tensor vs CUDA cores) with fusion (one launch vs
many). To isolate **just the fusion**, measure the orchestration unit on this B200:

| quantity | measured |
|---|---|
| per kernel-launch dispatch (no sync) | ~3.9 µs |
| per op-boundary (launch + host sync) | ~8.5 µs |

**Honest decomposition (this corrects a tempting overclaim):** launch+sync is *small*. Even a cycle with ~50 op
boundaries is ~0.4 ms of pure launch/sync — that **does not** explain the measured 65% bubble (~6.7 ms of a
10.75 ms cycle). The bubble is mostly **deeper** than launch overhead:

- **(a)** small-batch kernel inefficiency — the per-op GEMMs run far below peak at bs=1 (the tensor-core sweep
  above: only 0.6 TB/s at M=64);
- **(b)** sequential dependency stalls between the γ+2 forwards;
- **(c)** activation/KV round-trips to HBM at each op boundary;
- **(d)** **no overlap** of op N+1's weight-load with op N's compute — each separate kernel/graph node starts cold.

**What the megakernel uniquely removes:** the launch/sync (small), *and* — the thing a CUDA graph fundamentally
cannot express — **software-pipelining across ops**: prefetch op N+1's weights while op N computes. Since the
bs=1 cycle is *weight-read-bound*, that cross-op overlap is the **decisive** lever. The current prototype runs
ops sequentially with grid barriers (no pipelining yet), so it has proven the fusion *structurally* (one correct
launch, zero host bubble) but has **not yet realized the overlap** — that is the next real perf work, and it is
the one lever unavailable to the graph baseline.

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

## Do tensor-core GEMMs make it beat production? (measured — important)

Before investing in the large tensor-core (tcgen05/TMA) integration, I measured the **validated** Ampere
tensor-core GEMM reference (`ampere/tensorop_gemm.py`) at the verify's exact shapes and swept M (= N·Sv, the
particle-row batch), N=K=4096:

| M | time | TFLOP/s | weight-read BW |
|---|---|---|---|
| **64** (≈ our N=4,γ=4) | 55.8 µs | 39 | **0.60 TB/s** |
| 128 | 56.2 µs | 76 | 0.60 TB/s |
| 256 | 56.6 µs | 152 | 0.59 TB/s |
| 512 | 57.2 µs | 300 | 0.59 TB/s |
| 1024 | 102 µs | 335 | 0.33 TB/s |
| 4096 | 331 µs | 416 | — |

**Reading:** from M=64 to M=512 the GEMM takes the *same* ~56 µs — it is **weight-read-bound**, and even the
tensor-core kernel only hits **0.6 TB/s (7.5% of the 8 TB/s HBM peak)** at small M, because there aren't enough
tiles to fill 148 SMs. **Conclusion: at bs=1 / N=4 the verify's M≈40 is small-batch, weight-bandwidth-bound —
the tensor cores' compute is already hidden under the weight read.** Tensor cores would buy a *bandwidth* win
(better TMA/vectorized weight loads → my 0.12 TB/s up toward ~0.6 TB/s, ≈ 5× → verify ~20 ms), **not** the 50×
compute win one might expect — and that ~0.6 TB/s itself is an occupancy ceiling intrinsic to small M. Even at
that ceiling, verify ≈ 20 ms + draft (M=4, even more bandwidth-starved) ⇒ a full-tensor-core cycle would land
**~40–60 ms — still ~4–6× the ~10 ms production cycle.** Production wins at bs=1 not by bigger GEMMs but by a
deeply tuned small-batch path (FlashInfer/cuBLAS).

**Therefore:** the large tensor-core/TMA megakernel integration (flagged in the notes as "the riskiest part, a
large separate effort") is **high-effort, bounded-reward (~3× cycle), and still does not beat production at
bs=1/N=4** — because that regime is fundamentally small-batch memory-bound for *everyone*. The honest, defensible
contributions of this megakernel are therefore:
1. **It exists and is correct** — first single-launch realization of the whole multi-model SMC cycle.
2. **The fusion benefit** — closing the ~65% orchestration bubble (one-launch vs multi-launch, above), which is
   independent of GEMM throughput and is the paper's actual thesis.
3. Tensor cores pay off in the **larger-N / batched regime** (M grows → the table's right side, 300–416 TFLOP/s)
   — exactly where the paper already plans to *disaggregate*. So tensor-core GEMMs belong with the
   batched/disaggregated story, not the bs=1 single-kernel latency story.

Cheaper partial lever if pursuing bs=1 perf anyway: **128-bit vectorized weight loads + higher occupancy** in
the existing CUDA-core verify (chase the 0.12 → 0.6 TB/s bandwidth gap directly), which captures most of the
tensor-core upside without the TMA/tcgen05 rewrite.

## TL;DR

- The benchmark = per-cycle latency at bs=1, megakernel-one-launch vs production-cycle-graph, gated on
  estimator equivalence.
- The megakernel **already does the whole cycle in one correct launch** and is **10.3× faster than its own
  naive version** (1733 → 165 ms).
- It is **not yet faster than production** in absolute terms. Two measured findings reshape the perf strategy:
  1. **Tensor cores are not the bs=1 lever** — at M≈40 the verify is weight-read-bound and even the validated
     tensor-core ref only hits 0.6 TB/s (small-batch occupancy ceiling). ~5× bandwidth at best, still loses to
     production. Tensor cores belong to the larger-N / batched / disaggregated regime.
  2. **The 65% bubble is mostly NOT launch overhead** (~8.5 µs/boundary, small). It is small-batch inefficiency +
     dependency stalls + no cross-op weight-load/compute overlap. The megakernel's decisive, graph-impossible
     lever is **software-pipelining op N+1's weight-load under op N's compute** — proven *possible* by the fusion,
     not yet *realized* in the prototype. That is the next real perf work.

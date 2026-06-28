# SMC Draft-Phase Megakernel — Handoff

**Status (2026-06-28): M5d done (draft +13%). M6 DONE — HEADLINE: draft(1B)+verify(8B) FUSED in ONE kernel,
tokens flowing draft→verify in-graph. M7a DONE — in-kernel verify lm_head + tempered logprob. M7b DONE — the
full N-particle SMC cycle (batched draft / per-particle masked verify / reweight / systematic resample / bonus)
is built as megakernel stages, all validated, and the END-TO-END cycle matches the eager torch SMC estimator.
Remaining: fuse the N-particle stages into ONE launch (mechanism already proven by M6c) + worker integration.**

> **M7b update (2026-06-28): the full N-particle SMC cycle works (validated end-to-end vs eager torch SMC).**
> Built and validated as separate stages (the methodology), each on B200:
> - **M7b.1** `m7b_draft_N.py`: N-particle BATCHED draft (N AR streams, per-particle KV + Gumbel; layers go
>   "fat" over N rows). 4 particles draw DISTINCT sequences, **16/16** match torch Gumbel.
> - **M7b.2** `m7b2_verify_N.py`: N-particle verify with the **per-particle BLOCK-DIAGONAL causal mask** (the
>   one genuinely new kernel mechanic) — row (p,s) attends only keys (p,s'≤s), RoPE per-particle-position.
>   **top-1 100%** per particle vs HF-8B run independently; drafted logp within bf16 floor.
> - **M7b.3** `m7b3_resample.py`: in-kernel **reweight (α·target_logp−draft_logp) + softmax + SYSTEMATIC
>   resample** (cumsum+searchsorted), matching the SHIPPED `smcsd/common/utils.py` + `worker.py:774` math —
>   **200/200 trials, 0 ancestor mismatches**, max|ΔW|=5e-7.
> - **M7b.4** `m7b4_cycle_ref.py`: END-TO-END cycle with real 1B+8B — draft→verify→reweight→resample(kernel,
>   real logprobs)→bonus(Gumbel on tempered target) — **weights, ancestors, and bonus all match eager torch
>   SMC** (max|ΔW|=1e-8, ancestors exact). Example: weights [.031,.007,0,.962] → ancestors [3,3,3,3] → bonus
>   ' city'.
> **What remains = engineering, not science:** fuse the N-particle stages into a single launch (the draft+verify
> single-kernel fusion is already proven in `m6c`/`m7_cycle_mega.py`; M7b.1–.4 prove each N-particle stage works
> as a megakernel stage and the estimator is exact), then wire behind the worker flag (zero `arr`/`sen` per
> launch; port graph-safe Philox for the Gumbel noise; free-running feed-back instead of teacher-forcing).

> **M6 update (2026-06-28): the fusion headline works.** Three steps, all validated on B200:
> - **M6a** (`kernels/m6a_verify_ref.py`): 8B target verify forward + tempered-logprob via the per-op
>   kernels vs HF — top-1 **100%**, logp matches HF.
> - **M6b** (`kernels/m6b_verify_mega.py`): the 8B verify as ONE persistent kernel (m3b2 multi-token forward
>   on 8B) — top-1 **100%**; rigorously, **mine-vs-HF-fp32 logp = 1.1e-2, better than HF-bf16's own 5.7e-2
>   floor** (f32 accumulation throughout).
> - **M6c** (`kernels/m6c_cycle_mega.py`): **draft(1B 16L) + verify(8B 32L) in ONE kernel launch.** The
>   draft's Gumbel-sampled tokens are written to `gVtok` and read by the 8B verify embed IN-KERNEL (no host
>   round-trip — the fusion SMC's static shape uniquely permits). Validated: draft **4/4** vs 1B Gumbel ref;
>   verify top-1 **100%** vs HF-8B over `[prompt, drafted]`; logp within the bf16 floor (mine-vs-fp32 2.3e-2 <
>   HF-bf16's 7.7e-2). One launch ≈ 449 ms (naive verify kernels — correctness milestone, perf is later).
> What remains for the *complete* one-kernel cycle (M7): move lm_head + tempered-logprob in-kernel, then add
> reweight/resample/bonus. The hard structural part (two models, two weight/dim regimes, two-phase persistent
> kernel, in-graph token flow) is DONE.

> **M5d update (2026-06-28):** the documented "2.76 ms/token" was a *contention* measurement on a shared box.
> On an **idle** B200 the true clean baseline (M5) is **~1.70 ms/token**. **M5d = M5 + shared-memory activation
> staging** brings it to **~1.50 ms/token (clean +13%)** — within the 1–1.5 ms target. Best kernel is now
> **`kernels/m5d_draft_mega.py`**. Three other levers were tested on a clean GPU and **rejected**:
> warp-per-head attention (−10%, barrier-bound), 4-way FMA ILP (neutral — weight-bandwidth-bound, not
> latency-bound), and higher occupancy BLK=768/1024 (−/regress). Isolation harness: `benchmarks/bench_iso.py`
> (one flag-parametrized kernel, all variants timed interleaved on an idle GPU, preds verified identical).
> **ALWAYS benchmark on an idle GPU** (`nvidia-smi`; pick 0% util) — contention silently flips A/B verdicts.

This directory contains a from-scratch **persistent CUDA megakernel** (written in the CUTLASS CuTe Python
DSL) that runs the *entire* SMC **draft** decode cycle for Llama-3.2-1B — token embedding → 16 transformer
layers with a growing KV cache → lm_head → in-kernel Gumbel-max sampling → feed-back, looping γ steps —
**in a single kernel launch**. It reproduces the real model's decode and is the prototype for the
"megakernel" thesis of the planned MLSys paper (see `paper-outline.md`).

**SCOPE — read this:** the **end goal is a megakernel for the WHOLE SMC cycle** (γ draft forwards + **target
verify** + **reweight/resample/bonus**), all in one launch. What's built here is the **DRAFT phase — step 1
of 3.** The draft was first because it holds the hard new mechanics (persistent kernel, grid barrier, AR
loop, in-kernel sampling, KV cache) on the small 1B model; the remaining phases reuse that machinery. The
full roadmap (M5d → **M6 verify** → **M7 resample/bonus** → integrate → measure) is in §7. The verify+resample
fusion is the *novel* part — SMC's static no-rejection shape is what makes it possible (rejection-based SD
can't fuse verify); draft-only undersells the thesis.

If you're picking this up: read this file top to bottom, then run `kernels/m5_draft_mega.py` (the current
best). The immediate task is **M5d** (finish draft perf); the bigger arc is **M6/M7** (§7).

---

## 1. TL;DR / current numbers

- **What runs:** `kernels/m5_draft_mega.py` — the full SMC draft megakernel, one persistent launch.
- **Correctness:** 4/4 generated tokens match a torch Gumbel-max reference on the real model's logits;
  per-step logit cosine 0.9999. (Greedy variant: 100% token agreement vs `model.generate`.)
- **Speed:** **~1.50 ms/token** (M5d, B200, Llama-3.2-1B, single stream, *idle GPU*). M5 baseline ~1.70 ms;
  naive first version was 62 ms/token. (The old "2.76 ms" figure was a contended measurement — see the M5d
  note above; always measure idle.)
- **Where it stands vs roofline:** ~3× off the ~0.5 ms/token memory floor. A production CUDA-graphed
  1B decode is ~0.5–1 ms/token. (Do NOT compare to HF eager `model.generate` ≈ 35 ms/token — that is
  Python-overhead-bound, not a GPU baseline.)
- **Bottleneck (measured, §6):** the per-layer compute — specifically the fused-norm GEMVs re-reading the
  activation scalar per output, plus the thread-per-head attention. **NOT** barriers (~12%), **NOT** weight
  bandwidth (GEMVs already ~38% roofline; vectorization does not help).

---

## 2. Environment (REQUIRED — exact)

- **GPU:** NVIDIA B200, compute capability 10.0 (sm_100 / Blackwell). 148 SMs.
- **Conda env:** `test` (NOT base). `conda run -n test python ...`. The base env has the WRONG sglang.
- **Key pkgs in `test`:** `nvidia-cutlass-dsl 4.4.2`, torch 2.11+cu130, transformers, sglang (repo submodule,
  for the model only). No ThunderKittens.
- **Run a kernel:** `cd megakernel/kernels && CUDA_VISIBLE_DEVICES=<free_gpu> conda run -n test python m5_draft_mega.py`
- **Free GPUs:** check `nvidia-smi`; GPUs 4–7 were free during development.
- Each run loads the 1B model (~10s) + JIT-compiles the kernel (~10–60s first time). Be patient; a full run
  is ~1–2 min. For timing, the scripts already use `cute.compile` once + CUDA events.

---

## 3. The milestone ladder (file index)

Each kernel was validated against a reference before moving on. Run any of them standalone.
`m_kernels.py` is a shared module (validated per-op kernels + a `block()` helper) imported by several.

| File (`kernels/`) | Milestone | What it proves | Validation |
|---|---|---|---|
| `cute_smoke.py` | M0 | CuTe DSL JIT compiles+runs on sm_100 | exact vs torch |
| `m2a_gemv.py` | M2a | skinny GEMV (decode-shaped, M=4) | rel 1e-6 |
| `m2b_rmsnorm_proj.py` | M2b | fused RMSNorm+proj, intermediate on-chip | rel 3e-6 |
| `m2c_attn.py` / `m2c2_attn_rope.py` | M2c | GQA flash attention (+RoPE, +causal) | rel 1e-7 |
| `m2d_mlp.py` | M2d | SwiGLU MLP | rel 3e-6 |
| `m2e_capture_ref.py` → `m2e_block.py` | M2e | full decoder block vs **real HF layer-0** | rel 5e-3, cos 0.999996 |
| `m3a_fullmodel.py` | M3a | all 16 layers via these kernels = real model | **100% top-1 tokens** |
| `m3b1_barrier.py` | M3b.1 | hand-rolled **grid barrier** (sense-reversing, gpu-scope atomics) | 148 CTAs, no hang |
| `m3b2_mega.py` | M3b.2 | full 16-layer model in **ONE persistent kernel** | 100% tokens, cos 0.999995 |
| `m4a_gumbel.py` | M4a | in-kernel Gumbel-max sampling | exact vs torch |
| `m4_draft_mega.py` | M4 | **SMC draft megakernel** (AR loop + KV cache + Gumbel), naive | 4/4 tokens, cos 0.9999 |
| `m5_draft_mega.py` | M5 | + coalesced GEMVs + parallel argmax + occupancy | 4/4, ~1.70 ms/tok (idle) |
| `m5d_draft_mega.py` | **M5d (BEST)** | + **smem activation staging** (warp-attn reverted: regressed) | 4/4, **~1.50 ms/tok (idle)** |
| `m5c_draft_mega.py` | M5c | + 128-bit vectorized loads — **DEAD END (slower)** | 4/4 but no speedup |
| `m6a_verify_ref.py` | M6a | 8B target verify forward + tempered logprob (per-op kernels) | top1 100%, logp✓ |
| `m6b_verify_mega.py` | M6b | 8B verify as ONE persistent kernel (prefill-shaped) | top1 100%, <fp32 of HF-bf16 |
| `m6c_cycle_mega.py` | **M6c (HEADLINE)** | **draft(1B)+verify(8B) FUSED, one launch, in-graph token flow** | draft 4/4, verify top1 100% |
| `m7_cycle_mega.py` | M7a | M6c + **in-kernel** verify lm_head + tempered logprob (M5b vocab-reduce) | logp vs host 5e-3, vs fp32 3e-2 |
| `m7b_draft_N.py` | M7b.1 | **N-particle batched draft** (N AR streams, per-particle KV+Gumbel) | 16/16 vs torch, particles diverge |
| `m7b2_verify_N.py` | M7b.2 | N-particle verify, **per-particle block-diagonal causal mask** | top1 100%/particle vs HF-8B |
| `m7b3_resample.py` | M7b.3 | in-kernel **reweight + systematic resample** vs shipped SMC funcs | 200/200, 0 mismatches |
| `m7b4_cycle_ref.py` | **M7b.4** | **end-to-end N-particle cycle + bonus** vs eager torch SMC | weights/ancestors/bonus exact |

`m2e_block.py` needs `ref_block.pt`, produced by running `m2e_capture_ref.py` first (writes it next to the
script). All paths are script-relative.

`benchmarks/` holds the microbenchmarks used to find the bottleneck (barrier cost, scalar-vs-vec GEMV,
read bandwidth, NL sweep, etc.) and the API probes (`probe_*.py`).
`reference_cutlass/` holds the CUTLASS v4.4.2 CuTe-DSL example kernels (`dense_gemm`, `fmha`, `rmsnorm`, …)
— the templates for the vectorized/TMA machinery you'll need for M5d.

---

## 4. Architecture of the draft megakernel (`m5_draft_mega.py`)

One `@cute.kernel` (`draft_k`), launched on **B=148 blocks × BLK=512 threads** (1 block/SM so the grid
barrier can't deadlock). All state lives in **global memory** and persists across the AR loop:

- **AR loop** `for t in range(SP+NGEN-1)`: process token `gTok[t]` at position t, predict token t+1.
  Prompt tokens are teacher-forced (sequential prefill); then γ generated. (In production the draft only
  does the γ decode steps — the cache is pre-populated by the target's prefill.)
- **Per step:** embed (lookup `gEmb[tok]`) → **16 layers** → if `t+1>=SP`: lm_head + Gumbel sample → `gPred`.
- **Per layer, 5 stages, a grid barrier between each** (`gbar`, the M3b.1 barrier as a `@cute.jit` helper):
  1. **QKV**: RMSNorm(h) (computed once per warp) → Q/K/V projections; RoPE on K, write to KV cache.
  2. **Attention**: per head, online softmax over the cached K/V (RoPE on Q at attn time).
  3. **O proj + residual** → res1.
  4. **gate/up + SiLU** (RMSNorm(res1)) → act.
  5. **down proj + residual** → h (for next layer).
- **GEMV pattern (M5a):** *warp-per-output* — the 32 lanes split the K dim (coalesced reads), butterfly
  all-reduce (`wreduce`). Each warp grid-strides over outputs.
- **Sampling (M5b):** parallel Gumbel-argmax — each thread local-best over its vocab slice, warp
  argmax-reduce (carry value+index via shuffle), per-warp scratch `gWBV/gWBI`, then one thread reduces the
  NW warp-bests. (This replaced a single-threaded 128k-vocab log-loop that was the hidden bottleneck.)

**The grid barrier** (`m3b1_barrier.py` is the standalone validated version): per-block local `sense` flag
flips each call; thread 0 does `fence_acq_rel_gpu` → `atomic_add(arrive,1,acq_rel,gpu)`; the last arriver
`atomic_exch(arrive,0)` + `atomic_exch(sense,ls,release)`; others spin `while cur!=ls` on
`atomic_add(sense,0,acquire,gpu)`; then `fence_acq_rel_gpu` + `sync_threads`. Pointer for atomics =
`gTensor.iterator + idx`.

---

## 5. How to reproduce the key results

```bash
cd megakernel/kernels
G=4  # a free GPU

# flagship: validate + time the optimized draft megakernel
CUDA_VISIBLE_DEVICES=$G conda run -n test python m5_draft_mega.py
# -> "[M5a coalesced] ... match 4/4 / CORRECT: PASS / [M5a timing] ~2.7 ms/token"

# full model in one persistent kernel (M3b.2)
CUDA_VISIBLE_DEVICES=$G conda run -n test python m3b2_mega.py 16   # 100% tokens

# the grid barrier in isolation (M3b.1)
CUDA_VISIBLE_DEVICES=$G conda run -n test python m3b1_barrier.py

# attribution sweep (per-layer vs fixed cost)
CUDA_VISIBLE_DEVICES=$G conda run -n test python ../benchmarks/m5_nlsweep.py
```

---

## 6. Perf attribution (MEASURED — trust this, not intuition)

NL sweep (`benchmarks/m5_nlsweep.py`): 2/8/16 layers → 5144 / 13850 / 24680 µs per launch.
- **155 µs per layer per step** (1395 µs/launch ÷ 9 steps); fixed (embed+lm_head+sample) ~262 µs/token.
- 16 layers ≈ 2.5 ms = **~90% of the 2.76 ms/token**.
- Within a layer (~155 µs): **barriers ~12%** (5 barriers × ~3.6 µs), **compute ~88%**.

So the kernel is **compute-bound in the layer GEMVs**, NOT barrier-bound and NOT weight-bandwidth-bound:
- `benchmarks/m5_scalar_vs_vec.py`: the scalar warp-GEMV already hits **~38% roofline**; the 128-bit
  vectorized version is *slower* (fragment/copy overhead). → vectorization is a dead end here.
- `benchmarks/m5_barrier_cost.py`: a grid barrier is ~3.6 µs; 720 barriers/launch ≈ 2.6 ms = ~10% of the
  24.7 ms launch.

**Real culprit (original hypothesis):** the fused-norm GEMVs (Q, gate/up, lm_head) re-read the activation
(`gh`, `gg1`/`gnorm`) **scalar, per element, for every output** → redundant L2 traffic. Plus attention runs
on only 32 threads (one per head). See §7.

**M5d post-mortem (MEASURED on an idle GPU — `benchmarks/bench_iso.py`):**
- **Smem activation staging WORKS: +13%** (1.70 → 1.50 ms/tok). Staging the normed/weighted GEMV input
  (`gh*inv*g1`, `gR*inv2*g2`, `gh*invf*gnorm`, `gA`, `gAct`) into a per-block 32 KB F32 smem buffer once,
  then reading it from smem in every warp-per-output GEMV, is the one real win. Implemented in `m5d_draft_mega.py`.
- **Attention was NOT the problem.** Warp-per-head attention (32 lanes split D, online softmax) measured
  **−10%** — attention is *barrier-bound*, not thread-bound: the grid barrier waits on all 148 blocks no matter
  how many threads do attention, and the per-key warp-reduce only adds critical-path latency. Reverted.
- **GEMVs are weight-bandwidth-bound, not FMA-latency-bound.** 4-way accumulator ILP gave **0%** (the warp's
  32 lanes already provide enough memory-level parallelism). Consistent with the 128-bit-vectorization dead end.
- **Not occupancy-bound either.** BLK 512→768→1024 (25%→50% occ, still 1 block/SM) *regressed* (1.50→1.55→1.90).
  BLK=512 is optimal. The remaining gap to roofline is structural: 5 grid-barrier-serialized stages/layer +
  raw weight bandwidth. Closing it needs cross-stage weight-load↔compute pipelining (the M6 megakernel lever,
  a rewrite) or tensor cores — not another GEMV micro-opt.

---

## 7. ROADMAP — toward the FULL-CYCLE megakernel

**The end goal is ONE persistent kernel for the WHOLE SMC cycle:** γ draft forwards + **target verify** +
**reweight/resample/bonus**. What's built so far (M0–M5) is the **DRAFT phase — step 1 of 3.** The draft was
deliberately first: it holds all the hard new mechanics (persistent kernel, grid barrier, AR loop, in-kernel
sampling, KV cache) on the small 1B model. The remaining phases (M6, M7) **reuse that machinery** and are what
make the paper novel: SMC's *static, no-rejection* shape is what lets the **verify** fuse into the same
kernel — rejection-based SD (EAGLE/Medusa) can't, because their accept-count is data-dependent. So fusing
verify+resample is the headline, not the draft.

Build order: **M5d (finish draft perf) → M6 (fuse verify) → M7 (resample/bonus) → integrate → measure.**

### M5d — finish optimizing the draft — **DONE (2026-06-28). Result: 1.50 ms/token, target met.**
1. **[DONE] Stage the (normed) activation in shared memory once per block** — GEMVs read it from smem instead
   of re-reading `gh`/`gg1` from global per output. **Clean +13%** (1.70→1.50 ms/tok). The mechanism: a per-block
   F32 smem buffer of size `I` (32 KB) via `cutlass.utils.SmemAllocator().allocate_tensor(F32, make_layout(I))`,
   filled cooperatively (`i=tx; while i<K: sY[i]=...; i+=BLK`) + `cute.arch.barrier()`, then read `sY[k]` in
   each GEMV. Launch needs `smem=I*4+256`. Smem mechanics probe: `benchmarks/probe_smem.py`. In `m5d_draft_mega.py`.
2. **[TESTED — REJECTED] Parallelize attention to warp-per-head.** Measured **−10%** on an idle GPU. Attention
   is barrier-bound, not thread-bound (the grid barrier waits on all 148 blocks regardless), and the per-key
   warp-reduce adds critical-path latency. Reverted to thread-per-head. See §6 post-mortem.
3. **[TESTED — REJECTED] ILP (4 accumulators) and higher occupancy (BLK 768/1024)** — 0% and a regression
   respectively; the GEMVs are weight-bandwidth-bound, not latency/occupancy-bound. See §6.
4. **Net: 1.50 ms/token, within the 1–1.5 ms target.** Draft micro-opt is exhausted; further draft speedup
   needs the cross-stage pipelining that M6 introduces, so **move on to M6**. (Optional future: fuse stages
   1+2 to drop one of the 5 barriers/layer — modest ~2%, not done.)

### M6 — fuse the TARGET VERIFY phase — **DONE (2026-06-28). The headline mechanic works.**
- **M6a** `m6a_verify_ref.py`: 8B verify forward + tempered logprob via per-op kernels vs HF → top1 **100%**.
- **M6b** `m6b_verify_mega.py`: 8B verify as ONE persistent kernel (m3b2 multi-token forward, dims→8B). top1
  **100%**; mine-vs-HF-fp32 logp **1.1e-2** < HF-bf16's own 5.7e-2 floor (f32 accum → beats HF bf16). ~492 ms.
- **M6c** `m6c_cycle_mega.py` (**HEADLINE**): **draft(1B 16L)+verify(8B 32L) in ONE launch.** Draft writes its
  Gumbel tokens to `gVtok[SP+ii]`; the 8B verify embed reads `gVtok` IN-KERNEL → tokens flow draft→verify with
  no host round-trip. One `@cute.kernel` with TWO dim regimes (suffix d=1B, v=8B) + two weight sets; one grid
  barrier between phases. Validated: draft **4/4**, verify top1 **100%**, logp within bf16 floor
  (mine-vs-fp32 2.3e-2 < HF-bf16 7.7e-2). ~449 ms (naive verify kernels; perf later). KEY: lm_head + tempered
  logprob are still on HOST (move in-kernel in M7); the verify phase outputs `ghv[S,hid8]` (final hidden).
- **Caveat carried forward:** N=1 particle so far (the draft prototype is single-stream). The full cycle
  batches N particles in the draft; the verify already handles the S-token block. Scaling draft to N>1 +
  in-kernel lm_head/logprob/resample is M7.

<details><summary>original M6 plan (for reference)</summary>

Run the **target/score model forward over the `N×(γ+1)` drafted tokens INSIDE the same persistent kernel**,
with the draft's tokens flowing to the verify input **in-graph** (no host round-trip — this is the fusion
that the static shape uniquely permits).
- **Mechanically this is the transformer forward you already have** — reuse the layer/barrier/warp-GEMV
  machinery from `m_kernels.py` / `m3b2_mega.py`. Differences vs the draft: (a) a **bigger model** (8B), (b)
  it's a **prefill-shaped batched forward** over the static `N×(γ+1)`-token block (not 1 token/step), (c) **no
  sampling** — instead extract the **tempered target logprob** per drafted position: `logp = (logits/T)[tok]
  − logsumexp(logits/T)`.
- The draft and verify weights are different models → the kernel streams both weight sets (two residency
  regimes in one kernel; draft is small/reused, target streams). See `paper-outline.md` §5.
- Start by validating the verify forward standalone (8B logprobs over a drafted block) vs HF, like M2e/M3a
  did for the 1B draft; then fuse it after the draft phase in the persistent kernel (one barrier between
  draft-done and verify-start).
- **Large-target caveat:** for 70B + tensor parallelism the single-kernel property breaks (needs in-kernel
  collectives). There the design is the **hybrid: draft-megakernel + target-CUDA-graph, disaggregated.** So
  "everything in one kernel" is the **single-GPU / 8B-target latency regime**; the large-target case is the
  hybrid. The paper presents both as the "fuse vs disaggregate" planner — don't try to force 70B/TP into one
  kernel.
</details>

### M7 — reweight / resample / bonus (closes the cycle)
- **M7a DONE (2026-06-28)** `m7_cycle_mega.py`: extends M6c with **in-kernel verify lm_head + tempered
  logprob**. After the fused draft→verify forward, a final in-kernel stage computes, per drafted position:
  GEMV `ghv·LM8`→logits (warp-per-output, smem-staged) then a vocab `logsumexp` via the M5b reduction
  (parallel max → grid-reduce via `gWBV`+`gRed`, parallel sum-exp → grid-reduce) → `logp=(logit/T)[tok]−lse`.
  Output `gVlp[NGEN]`. Validated: in-kernel logp matches the host lm_head path to **5e-3** and HF-fp32 to
  **2.9e-2** (bf16 floor). So the verify's actual OUTPUT (the logprobs reweight needs) is now in-kernel. ~449 ms.
- **M7b — NEXT (the cross-particle SMC stages). REQUIRES N>1 particles** — the prototype is N=1 (single draft
  stream), so reweight/resample are degenerate. The real build:
  1. **Batch the draft to N particles** (N independent AR streams, each its own KV-cache slice + Gumbel noise;
     the draft GEMVs become fat over N — better arithmetic intensity, as the paper argues). The verify already
     handles an S-token block; for N particles the block is N×(γ+1) tokens with a **per-particle (block-diagonal)
     causal mask** (each particle attends only to its own tokens + shared prefix) — the one real new kernel
     mechanic for M7b.
  2. **Draft logprob in-kernel** too: the reweight needs the draft's tempered logprob of its OWN sampled token
     = `(gLogd[ii,tok]/T)−logsumexp(gLogd[ii]/T)` — add the same reduction in the draft phase (gLogd already exists).
  3. **Reweight**: per particle, `w_i ∝ exp(α·Σ_pos(target_logp − draft_logp))`; **normalize across N**
     (cross-particle reduction); **resample** N indices ∝ w (inclusive-scan + uniform draws); **gather** states.
  4. **Bonus**: Gumbel-max on the tempered target at the last position (logits at `S-1`, predicting the bonus
     token) — reuse the draft's M5b Gumbel-argmax verbatim.
  Cross-check the whole estimator against the eager torch SMC path (it computes the same `logprob_diff`/α-weights).
  <br>Original M7 notes:
The SMC importance-weighting + resampling, fused as the final stages:
- **per-position α-weighted logprob diff** → importance weights, shape `(bs, N)` (draft logprobs from M4 vs
  verify logprobs from M6).
- **normalize across the N particles** (cross-particle reduction), **resample** particle indices, **gather**
  the surviving states.
- **bonus draw**: Gumbel-max on the tempered target at the last position.
- These are small cross-particle reductions — a handful of final stages + a couple of barriers.
- After M7: **one launch covers the entire worker cycle** = the headline contribution.
- Cross-check the estimator against the eager torch SMC path (the shipped code computes the same
  `logprob_diff` / α-weighting — match it).

### Then: integration into the SMC worker
- Wire the megakernel behind a flag (e.g. `SMC_FULL_MEGAKERNEL=1`, with `SMC_DRAFT_MEGAKERNEL=1` for the
  draft-only intermediate) as an alternative to the per-cycle CUDA graph in `smcsd/core/worker.py`.
  **IMPORTANT:** the barrier `arrive`/`sense` tensors must be ZEROED before each launch (the `sense` flag
  persists across launches → a 2nd launch on dirty state DEADLOCKS).
- In production the draft only does γ decode steps (cache pre-populated by the target prefill), not the
  sequential prefill this prototype does. Adapt the entry to take an existing KV cache + start token.
- Port the **graph-safe Philox RNG** from the shipped SMC work for the Gumbel noise (this prototype passes a
  precomputed uniform-noise table for deterministic validation).

### Then: the paper measurement
- Draft milestone: **draft-megakernel vs the draft-phase CUDA graph** at bs=1.
- Headline: **full-cycle megakernel vs the full per-cycle CUDA graph** at bs=1 — the "bubble-closing" number
  (recall the roofline: ~65% of the bs=1 cycle was bubble). See `paper-outline.md` (megakernel-as-centerpiece,
  static-shape thesis, fuse-vs-disaggregate planner).

---

## 8. CuTe DSL gotchas (hard-won — will save you hours)

- **Call `@cute.jit` fns directly** to JIT+run; `cute.compile(fn, *args)` bakes `Constexpr` args and returns
  a callable taking ONLY the runtime tensors.
- **TIMING:** `cute.compile` ONCE + `from_dlpack` the tensors ONCE, then time only `compiled(...)` launches
  with CUDA events. Calling `from_dlpack`/the jit fn inside the timed loop adds ~8 ms/call of Python overhead
  and will make kernels look ~1000× slower than they are. (This bit me — twice.) Also: per-LAUNCH vs
  per-TOKEN units — don't mix them.
- A **device helper with control flow must be `@cute.jit`** (e.g. the barrier) or its `if`/`while` aren't
  AST-preprocessed → "Unable to convert dynamic Boolean value to bool" error.
- **Variables assigned only inside an `if`/`else` must be pre-initialized** before the branch, else
  "name not defined".
- Loop-carried scalar accumulation works with `cutlass.range(K)` (a dynamic loop). `range_dynamic` is
  deprecated. Native `while` works (used for spin-waits / grid-strides).
- Casts: `x.to(cutlass.Float32)`. `cute.rsqrt`, `cute.exp`, `cute.log` are TOP-LEVEL (not `cute.arch`).
  Scalar max: `cutlass.max(a,b)`. `cute.where` is TENSOR-only.
- Atomics: `cute.arch.atomic_add/exch/cas(ptr, val, sem='acq_rel'|'release'|'acquire'|'relaxed', scope='gpu')`.
  Device fence: `cute.arch.fence_acq_rel_gpu()`. Pointer to a tensor element: `gT.iterator + idx`.
- Warp ops: `cute.arch.shuffle_sync_bfly(val, offset)` (all-reduce), `shuffle_sync_down` (lane0 gets result).
- Store F32 into a bf16 scratch tensor: cast `acc.to(gT.element_type)`.
- Vectorized 128-bit load: `make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=128)` +
  `cute.local_tile` + `make_fragment` + `cute.copy` + `fragment.load()`; needs `from_dlpack(t, assumed_align=16)`
  or you get a "src ptr alignment (16 bits) does not meet requirement (128 bits)" verifier error.
- `cute.make_fragment(n, dtype)` for a register array (deprecation warning, still works).
- The reference example kernels live in CUTLASS repo `examples/python/CuTeDSL/blackwell/` (tag v4.4.2 to
  match 4.4.2). Copies are in `reference_cutlass/`.

---

## 9. Numerical-correctness notes

- To match HF token-for-token, **round res1 to bf16 before the post-attention RMSNorm** (HF keeps hidden
  states bf16). Skipping this flips occasional near-tie argmaxes (cost us 83%→100% in M3b.2).
- **KV-cache write must store all D dims of K (RoPE'd) and V.** An early bug guarded with `if d<DH` and
  silently dropped the upper half of every V vector → logit cosine fell to 0.44; fixing it → 0.9999.
- Store **post-RoPE K** in the cache (RoPE K at its own position t before writing); only RoPE Q at attn time.
- Validate via **teacher-forcing** (feed HF's reference sequence, predict to a side buffer) to separate
  per-step correctness from free-running near-tie compounding; check per-step logit cosine, not just argmax.

---

## 10. Pointers

- `paper-outline.md` — the MLSys paper plan (megakernel centerpiece; static-shape thesis; eval plan).
- Agent memory (if available to you): `smcsd-mlsys-paper.md` has the running log of every milestone, number,
  and lesson (including the two benchmarking-units mistakes and their corrections).
- The shipped SMC perf work (CUDA-graph stack, +14–16%) is the production baseline the megakernel must beat;
  it lives on the `mau/fable-expt` branch / PR #12.

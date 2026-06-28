# SMC Draft-Phase Megakernel — Handoff

**Status (2026-06-28): BUILT, VALIDATED, OPTIMIZED 22.4×. Correctness done; perf has one clear lever left.**

This directory contains a from-scratch **persistent CUDA megakernel** (written in the CUTLASS CuTe Python
DSL) that runs the *entire* SMC draft decode cycle for Llama-3.2-1B — token embedding → 16 transformer
layers with a growing KV cache → lm_head → in-kernel Gumbel-max sampling → feed-back, looping γ steps —
**in a single kernel launch**. It reproduces the real model's decode and is the prototype for the
"megakernel" thesis of the planned MLSys paper (see `paper-outline.md`).

If you're picking this up: read this file top to bottom, then run `kernels/m5_draft_mega.py` (the current
best). The next concrete task is **M5d** (§7).

---

## 1. TL;DR / current numbers

- **What runs:** `kernels/m5_draft_mega.py` — the full SMC draft megakernel, one persistent launch.
- **Correctness:** 4/4 generated tokens match a torch Gumbel-max reference on the real model's logits;
  per-step logit cosine 0.9999. (Greedy variant: 100% token agreement vs `model.generate`.)
- **Speed:** **2.76 ms/token** (B200, Llama-3.2-1B, single stream). Naive first version was 62 ms/token →
  **22.4× optimized**.
- **Where it stands vs roofline:** ~5.5× off the ~0.5 ms/token memory floor. A production CUDA-graphed
  1B decode is ~0.5–1 ms/token. (Do NOT compare to HF eager `model.generate` ≈ 40 ms/token — that is
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
| `m5_draft_mega.py` | **M5 (BEST)** | + coalesced GEMVs + parallel argmax + occupancy | 4/4, **2.76 ms/tok** |
| `m5c_draft_mega.py` | M5c | + 128-bit vectorized loads — **DEAD END (slower)** | 4/4 but no speedup |

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

**Real culprit:** the fused-norm GEMVs (Q, gate/up, lm_head) re-read the activation (`gh`, `gg1`/`gnorm`)
**scalar, per element, for every output** → redundant L2 traffic. Plus attention runs on only 32 threads
(one per head). See §7.

---

## 7. NEXT STEPS (in priority order)

### M5d — the next perf lever (start here)
1. **Stage the (normed) activation in shared memory once per block**, then GEMVs read it from smem instead
   of re-reading `gh`/`gg1` from global per output. This removes the dominant redundant traffic. This is
   **new DSL territory** (no smem used yet here) — study `reference_cutlass/rmsnorm.py` (it stages X in smem
   via `cp.async` + a `SmemAllocator`) and `cutlass.utils.SmemAllocator` / `cute.arch.alloc_smem`.
2. **Parallelize attention** — currently thread-per-head (32 threads). Make it warp-per-head (or
   warp-per-(head,key-block)) so the grid isn't 116/148 idle during the attention stage.
3. Consider **fewer barriers/layer**: if each warp does its heads' QKV *and* attention end-to-end (head-local),
   you can fuse stages 1+2 and drop one barrier (5→4). Modest.
4. Re-measure. Plausible target ~1–1.5 ms/token.

### Then: integration into the SMC worker
- Wire the megakernel behind a flag (e.g. `SMC_DRAFT_MEGAKERNEL=1`) as an alternative to the draft-phase
  CUDA graph in `smcsd/core/worker.py`. **IMPORTANT:** the barrier `arrive`/`sense` tensors must be ZEROED
  before each launch (the `sense` flag persists across launches → a 2nd launch on dirty state DEADLOCKS).
- In production the draft only does γ decode steps (cache pre-populated by the target prefill), not the
  sequential prefill this prototype does. Adapt the entry to take an existing KV cache + start token.
- Port the **graph-safe Philox RNG** from the shipped SMC work for the Gumbel noise (this prototype passes a
  precomputed uniform-noise table for deterministic validation).

### Then: the paper measurement
- The headline comparison: **megakernel vs the draft-phase CUDA graph in the SAME config** at bs=1 — the
  "bubble-closing" number. See `paper-outline.md` (megakernel-as-centerpiece, static-shape thesis).

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

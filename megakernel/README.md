# megakernel/ — SMC draft-phase megakernel (CuTe DSL, Blackwell/sm_100)

A from-scratch persistent CUDA megakernel (CUTLASS CuTe Python DSL) that runs the **entire SMC draft
decode cycle for Llama-3.2-1B in one kernel launch** — embed → 16 layers + KV cache → lm_head →
in-kernel Gumbel sampling → AR loop. Built and validated end-to-end; optimized 22.4× (62 → 2.76 ms/token).

**👉 Read [`HANDOFF.md`](HANDOFF.md) first.** It has the milestone ladder, how to run, the measured
bottleneck, the next step (M5d), and the CuTe DSL gotchas.

Quick start:
```bash
cd kernels && CUDA_VISIBLE_DEVICES=<free_gpu> conda run -n test python m5_draft_mega.py
```

- `kernels/`            validated milestone kernels (M0→M5); `m5_draft_mega.py` is the current best.
- `benchmarks/`         microbenchmarks used for the perf attribution + API probes.
- `reference_cutlass/`  CUTLASS v4.4.2 CuTe-DSL example kernels (templates for vectorized/TMA work).
- `paper-outline.md`    the MLSys paper plan (megakernel centerpiece).

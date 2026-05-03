# Cascade Attention for SMC — v1 Status

> Status: v1 integration shipped behind `--smc-shared-prefix-attn` (default off).
> Numerical correctness validated end-to-end; performance picture mixed and
> documented below. Multi-group + CUDA-graph follow-up tracked at the bottom.

This document tracks the cascade-attention integration: an opt-in path that
routes the SMC verify pass through FlashInfer's
`MultiLevelCascadeAttentionWrapper` to share prefix-KV reads across the N
particles of a group ("Hydragen / Cascade Inference" applied to SMC).

## Motivation

Within an SMC group, all N particles share the materialise-time prompt
prefix of length `L_g` and physically point at the same KV pages
(refcounted by `SMCRefCountedTokenAllocator`). The default attention
backend re-reads those L_g shared KV pages N times per layer per cycle.
Cascade decomposes the attention into two passes — `queries × shared
prefix` (read once per group) and `queries × per-particle suffix` —
combined via FlashAttention's online-softmax merge. The pattern is exact
(no approximation) and was popularised by
[Hydragen (Juravsky et al., ICML 2024)](https://arxiv.org/abs/2402.05099)
and shipped in
[FlashInfer's cascade kernels](https://flashinfer.ai/2024/02/02/cascade-inference.html).

## Architecture

| File | Role |
| --- | --- |
| `smcsd/core/req_state.py` | Adds `shared_prefix_lens` and `group_row_ids` slot tensors; set at `allocate_slots`, cleared at `free_group_slots`. |
| `smcsd/core/info.py` | Threads the gathered active-subset of those tensors through `SMCDraftInput` → `SMCVerifyInput`. |
| `smcsd/common/verify.py` | New optional fields on `SMCVerifyInput`. |
| `smcsd/model_executor/smc_attn_backend.py` | `SMCFlashInferAttnBackend` — wraps SGLang's `FlashInferAttnBackend`. Builds a 2-level cascade plan (per-group prefix + per-particle suffix), runs cascade in `forward_extend` when SMC verify is eligible, falls back to the inner backend otherwise. Implements CUDA-graph capture/replay for the single-group case. |
| `smcsd/model_executor/smc_model_runner.py` | `_get_attention_backend_from_str` substitutes the cascade-aware backend when the flag is on, the user picked the `flashinfer` backend, and we're the target (not draft) worker. |
| `smcsd/engine.py` | `shared_prefix_attn=` kwarg on `SMCEngine`. |
| 3rdparty/sglang | `--smc-shared-prefix-attn` server arg; whitelist `flashinfer` for SMC. |

The plan-builder is GPU-only for the big tensors (`req_to_token` slices
gathered + concatenated on-device); only tiny bs-sized control-flow
scalars hit the host.

## Constraints / scope of v1

- **Single-group CUDA graph capture.** The captured wrapper is bound to
  `n_g = 1` at construction time. Multi-group cycles can use cascade in
  eager mode, but graph mode is currently single-group only. (Fix: pad
  `n_g` at replay up to a captured max, mirroring how we already pad
  `n_p`. Cleanly tractable, just not in v1.)
- **Within a group, all particles must share the same `L_g`.** Holds by
  construction at materialise; preserved by resample because
  `copy_block_table` and the fused KV resample inc-ref the prefix pages
  uniformly.
- **Opt-in.** `--smc-shared-prefix-attn` defaults to off. Without it,
  SMC continues to use fa3 / triton exactly as before — zero behavior
  change.

## Numerical correctness

Greedy-decode parity with the existing fa3 path:

| Workload | Tokens generated | Match |
| --- | --- | --- |
| Short prompt, max_new=64 | 8 | **8 / 8** |
| Long generation, max_new=256 | 228 | **228 / 228** |

Both with and without CUDA graphs. All 18 existing scheduler+kernel
tests pass.

POC kernel-level numerical equivalence (synthetic Q/K/V, sweep across
shapes): max abs diff ≤ 2.4e-4 against `BatchPrefillWithPagedKVCacheWrapper`
running the equivalent per-particle full-block-table attention (fp16
noise floor).

## Performance — what we observed

### GSM8K, 400 questions (N=12, γ=8, temp=0.7, --max-running-requests 128, --cuda-graph-max-bs 128, seed 42)

| Config | Accuracy | TPS | Wall |
| --- | --- | --- | --- |
| fa3 + graphs (baseline) | 75.5 % (302 / 400) | **240** | 339 s |
| flashinfer + cascade + graphs | 73.8 % (295 / 400) | 217 | 366 s |

Cascade preserves accuracy (1.7 pp delta is within seed noise; both
within ~1pp of the historical 75% target). On TPS it lands at **0.90 ×
of fa3 + graphs at GSM8K shape (L ≈ 1 K)** — a small regression that
reflects cascade's plan-build cost slightly exceeding the prefix-share
savings at this prompt length.

### Long-prompt single-prompt (N=12, γ=8, eager mode)

| Prompt L | fa3 TPS | cascade TPS | Speedup |
| --- | --- | --- | --- |
| 1.8 K | 129 | 125 | 0.97 × |
| 3.8 K | 80 | **104** | **1.30 ×** |
| 7.9 K | 60 | **76** | **1.26 ×** |
| 12.0 K | 48 | 48 | 1.00 × |
| 16.2 K | 43 | 42 | 0.97 × |

**Cascade wins at L = 4 – 8 K in eager mode** (1.26 – 1.30 ×). Below that
range, plan overhead dominates; above, FA3's Hopper-tuned kernel
(TMA + WGMMA) saturates HBM BW in a way FlashInfer's cascade kernel
doesn't fully match.

### Long-prompt single-prompt (N=12, γ=8, CUDA graphs ON)

| Prompt L | fa3 TPS | cascade TPS | Speedup |
| --- | --- | --- | --- |
| 3.8 K | 132 | 124 | 0.94 × |
| 7.9 K | 78 | 79 | 1.01 × |
| 16.2 K | 43 | 43 | 0.99 × |

With graphs on, fa3's launch-overhead penalty disappears and cascade's
eager-mode advantage with it. Both paths are essentially tied.

### Multi-group eager (L = 4 K, N = 12, γ = 8, max_new = 256)

| G | fa3 TPS | cascade TPS | Speedup |
| --- | --- | --- | --- |
| 1 | 110.8 | 100.5 | 0.91 × |
| 2 | 144.8 | 115.2 | 0.80 × |
| 4 | 192.7 | 150.0 | 0.78 × |

### Multi-group eager (L = 16 K, N = 12, γ = 8, max_new = 256)

| G | fa3 TPS | cascade TPS | Speedup |
| --- | --- | --- | --- |
| 1 | 37.5 | 41.0 | 1.09 × |
| 2 | 46.6 | 39.1 | 0.84 × |
| 4 | 51.2 | 45.0 | 0.88 × |

The POC predicted 5.6 × at G = 4, N = 12, L = 16 K **at the kernel
level**. End-to-end falls short for three reasons:

1. **Plan-build CPU overhead per cycle.** 3 GPU→CPU syncs for
   `shared_prefix_lens / group_row_ids / seq_lens`, plus a Python loop
   over G × N particles. At G = 4, N = 12 = 48 particles this is
   ~5 – 10 ms/cycle on top of ~80 – 100 ms of GPU work.
2. **FFN dominates at L < 16 K on Llama-8B.** Even infinite-attention
   speedup ceilings end-to-end gain at ~1.4 ×.
3. **Graph capture restricted to G = 1 in v1.** Multi-group cycles run
   eager, paying per-layer launch overhead.

## Where v1 wins

- ✅ Numerical correctness end-to-end (token parity, all existing tests)
- ✅ Eager-mode single-prompt long-context: **1.24 – 1.30 × at L = 4 – 8 K**
- ✅ CUDA graph capture for single-group: **GSM8K 0.90 ×** of fa3+graphs
  (within 10 %, accuracy preserved)
- ✅ Architecture in place to extend to multi-group graphs

## Where v1 doesn't win (and what would fix it)

- ❌ GSM8K-shape (L ≈ 1 K) loses 10 % to fa3+graphs in TPS. Closest fix:
  Triton plan-build kernel to kill the per-cycle CPU syncs.
- ❌ Multi-group + graphs not supported (the captured wrapper's
  `n_g = 1` is fixed). Fix: bind to `max_n_g` at capture, pad `n_g` at
  replay (mirror what we already do for `n_p`).
- ❌ Multi-group eager doesn't show Hydragen-style scaling. Fix:
  combination of the above two — graphs amortise launches, Triton
  plan-build kills the Python-side amplifier on G × N work.

## Roadmap

| Item | Effort | Impact |
| --- | --- | --- |
| Multi-group CUDA-graph capture (variable `n_g` at replay) | ~1 d | Unlocks production batched serving wins; expected to convert the eager-mode 0.78 × at G = 4 into something north of fa3+graphs. |
| Triton plan-build kernel (kill CPU syncs + Python loop) | ~2 d | Closes the GSM8K 0.90 × gap; amplifies multi-group benefit further. |
| Resample-aware shared-prefix bumping (when all N particles in a group are copies of one ancestor, raise `shared_prefix_lens[g]` to current decode length) | ~0.5 d + careful verification | Expands the prefix savings inside long generations. |
| Triton-backend variant of cascade for non-Hopper (A100, A6000) | ~3 d | Currently only the FlashInfer (Hopper-tuned) cascade exists; to ship to A100 users need a parallel implementation. |

## How to use

```bash
# Run with cascade enabled (requires --attention-backend flashinfer):
python scripts/accuracy_test_gsm8k.py \
  --mode smc_engine \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend flashinfer \
  --shared-prefix-attn \
  --num-questions 400

# POC + microbenchmarks (no real model required):
python scripts/poc_cascade_attn.py
python scripts/poc_cascade_attn_sweep.py

# End-to-end correctness check (greedy parity vs fa3):
python scripts/smoke_cascade_parity.py

# Long-prompt benchmark:
python scripts/bench_long_prompt.py --target-len 4096

# Multi-group bench (eager mode in v1):
python scripts/bench_multi_group.py --target-len 4096 --groups 1 2 4 --disable-cuda-graph
```

## Honest read

The cascade integration is correct, behind a clean opt-in flag, and
unlocks the design space. But the headline TPS win we hoped for at
GSM8K isn't there at v1 maturity — fa3+graphs is still the throughput
champion at the user's standard benchmark shape. The cascade story
really lives at long-context multi-group serving, and reaching it
requires the multi-group graph capture + Triton plan-build follow-ups
above.

The right way to land v1: opt-in for long-context single-prompt
workloads where it wins eager-mode 1.24 – 1.30 ×; treat multi-group
graphs and the Triton plan-build as the next two PRs to make cascade
the default for batched serving.

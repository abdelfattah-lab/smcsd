"""Fused Gumbel-max sampling + logprob kernels for the SMC decode cycle.

The in-graph sampling chain per draft step is ~10 torch kernels over the
(bs, V) logits — rand, log, log, neg, add, argmax (a 29 us reduce at
V=128k), gather, logsumexp (two more passes) — measured ~60-70 us per
step at bs=8, i.e. as expensive as the draft lm_head GEMM itself, times
gamma steps + the bonus draw per cycle.

Two fused replacements, each a two-stage split reduction over V:

* ``fused_gumbel_sample`` — draw ``idx ~ softmax(alpha * logits / T)`` via
  Gumbel-max and return the chosen token's logprob under the scaled
  distribution plus (optionally, alpha != 1) the power-target normalizer
  ``log Z = lse(alpha*base) - alpha*lse(base)``.  One pass over logits.
* ``fused_chosen_logprob`` — given already-chosen tokens, return
  ``(logits/T)[token] - lse(logits/T)`` per row (the verify-side score
  extraction).  One pass over logits.

Graph-safe RNG: stage 1 reads a seed from a device buffer
(``tl.load(Seed)``) and offsets it by ``(row, col)``; the caller bumps the
buffer with a captured ``seed_buf += 1`` so every graph replay draws fresh
noise, deterministic from the initial seed.  This mirrors the graph-safe
Philox property of ``torch.rand_like`` under capture.

The sampled distribution is identical to the torch chain; the RNG stream
differs (triton Philox vs torch Philox), so token-level outputs are not
bit-comparable with the old path — only distributionally equivalent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Split count for the V-dimension reduction; SP2 = SPLITS padded to pow2
# for the stage-2 arange.
SPLITS = 16
BLOCK_V = 2048


@triton.jit
def _gumbel_sample_stage1(
    Logits,          # (R, V) fp32/bf16
    Seed,            # (1,) int64 device scalar
    P_zmax, P_zidx,  # (R, S) partial gumbel max / argmax
    P_m, P_l,        # (R, S) partial max / sumexp of scaled
    P_mb, P_lb,      # (R, S) partial max / sumexp of base (alpha != 1)
    stride_lr,
    V,
    inv_t,           # 1 / temperature
    alpha,
    row_offset,      # disjoint counter ranges across launches sharing a seed
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_BASE: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)

    chunk = (V + S - 1) // S
    lo = split * chunk
    hi = tl.minimum(lo + chunk, V)

    seed = tl.load(Seed).to(tl.int32)

    zmax = float("-inf")
    zidx = 0
    m = float("-inf")
    l = 0.0
    mb = float("-inf")
    lb = 0.0

    for start in range(lo, hi, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hi
        x = tl.load(Logits + row * stride_lr + offs, mask=mask, other=float("-inf"))
        base = x.to(tl.float32) * inv_t
        scaled = base * alpha

        # Gumbel noise: one Philox seed per cycle, disjoint counter range per
        # (launch, row): callers sharing a seed across several launches (the
        # in-graph draft steps + bonus) pass a distinct row_offset per launch.
        u = tl.rand(seed, (row_offset + row) * V + offs)
        tiny = 1.1754944e-38
        g = -tl.log(-tl.log(tl.maximum(u, tiny)))
        z = tl.where(mask, scaled + g, float("-inf"))

        b_zmax = tl.max(z, 0)
        b_zarg = tl.argmax(z, 0)
        if b_zmax > zmax:
            zmax = b_zmax
            zidx = start + b_zarg

        b_m = tl.max(tl.where(mask, scaled, float("-inf")), 0)
        new_m = tl.maximum(m, b_m)
        # Guard the empty-chunk case: keep -inf accumulators inert.
        rescale = tl.where(new_m == float("-inf"), 1.0, tl.exp(m - new_m))
        l = l * rescale + tl.sum(tl.where(mask, tl.exp(scaled - new_m), 0.0), 0)
        m = new_m

        if NEED_BASE:
            b_mb = tl.max(tl.where(mask, base, float("-inf")), 0)
            new_mb = tl.maximum(mb, b_mb)
            rescale_b = tl.where(new_mb == float("-inf"), 1.0, tl.exp(mb - new_mb))
            lb = lb * rescale_b + tl.sum(tl.where(mask, tl.exp(base - new_mb), 0.0), 0)
            mb = new_mb

    off = row * S + split
    tl.store(P_zmax + off, zmax)
    tl.store(P_zidx + off, zidx)
    tl.store(P_m + off, m)
    tl.store(P_l + off, l)
    if NEED_BASE:
        tl.store(P_mb + off, mb)
        tl.store(P_lb + off, lb)


@triton.jit
def _gumbel_sample_stage2(
    Logits,
    P_zmax, P_zidx, P_m, P_l, P_mb, P_lb,
    Out_idx,         # (R,) int64/int32
    Out_logp,        # (R,) fp32: scaled[idx] - lse(scaled)
    Out_logz,        # (R,) fp32: lse(scaled) - alpha * lse(base)
    stride_lr,
    inv_t,
    alpha,
    S: tl.constexpr,
    SP2: tl.constexpr,
    NEED_BASE: tl.constexpr,
    NEED_LOGP: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, SP2)
    mask = offs < S

    zmax = tl.load(P_zmax + row * S + offs, mask=mask, other=float("-inf"))
    win = tl.argmax(zmax, 0)
    idx = tl.load(P_zidx + row * S + win)
    tl.store(Out_idx + row, idx)

    m = tl.load(P_m + row * S + offs, mask=mask, other=float("-inf"))
    l = tl.load(P_l + row * S + offs, mask=mask, other=0.0)
    m_max = tl.max(m, 0)
    lse = m_max + tl.log(tl.sum(l * tl.exp(m - m_max), 0))

    if NEED_LOGP:
        chosen = tl.load(Logits + row * stride_lr + idx).to(tl.float32) * inv_t * alpha
        tl.store(Out_logp + row, chosen - lse)

    if NEED_BASE:
        mb = tl.load(P_mb + row * S + offs, mask=mask, other=float("-inf"))
        lb = tl.load(P_lb + row * S + offs, mask=mask, other=0.0)
        mb_max = tl.max(mb, 0)
        lse_b = mb_max + tl.log(tl.sum(lb * tl.exp(mb - mb_max), 0))
        tl.store(Out_logz + row, lse - alpha * lse_b)


@triton.jit
def _chosen_logprob_stage1(
    Logits,          # (R, V)
    P_m, P_l,        # (R, S)
    stride_lr,
    V,
    inv_t,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    chunk = (V + S - 1) // S
    lo = split * chunk
    hi = tl.minimum(lo + chunk, V)

    m = float("-inf")
    l = 0.0
    for start in range(lo, hi, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < hi
        x = tl.load(Logits + row * stride_lr + offs, mask=mask, other=float("-inf"))
        scaled = x.to(tl.float32) * inv_t
        b_m = tl.max(tl.where(mask, scaled, float("-inf")), 0)
        new_m = tl.maximum(m, b_m)
        rescale = tl.where(new_m == float("-inf"), 1.0, tl.exp(m - new_m))
        l = l * rescale + tl.sum(tl.where(mask, tl.exp(scaled - new_m), 0.0), 0)
        m = new_m

    off = row * S + split
    tl.store(P_m + off, m)
    tl.store(P_l + off, l)


@triton.jit
def _chosen_logprob_stage2(
    Logits,
    Tokens,          # (R,) chosen token per row
    P_m, P_l,
    Out_logp,        # (R,)
    stride_lr,
    inv_t,
    S: tl.constexpr,
    SP2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, SP2)
    mask = offs < S
    m = tl.load(P_m + row * S + offs, mask=mask, other=float("-inf"))
    l = tl.load(P_l + row * S + offs, mask=mask, other=0.0)
    m_max = tl.max(m, 0)
    lse = m_max + tl.log(tl.sum(l * tl.exp(m - m_max), 0))
    tok = tl.load(Tokens + row)
    chosen = tl.load(Logits + row * stride_lr + tok).to(tl.float32) * inv_t
    tl.store(Out_logp + row, chosen - lse)


# Shape-keyed persistent partial buffers, shared module-level state.  Safe
# under the current execution model — one scheduler subprocess, one CUDA
# stream, and cycle-graph capture serializes buffer use — but NOT safe for
# concurrent launches on multiple streams sharing a shape key.
_BUF_CACHE: dict = {}


def _bufs(r: int, device, need_base: bool):
    key = (r, device, need_base)
    b = _BUF_CACHE.get(key)
    if b is None:
        zmax = torch.empty((r, SPLITS), dtype=torch.float32, device=device)
        zidx = torch.empty((r, SPLITS), dtype=torch.int64, device=device)
        m = torch.empty((r, SPLITS), dtype=torch.float32, device=device)
        l = torch.empty((r, SPLITS), dtype=torch.float32, device=device)
        if need_base:
            mb = torch.empty((r, SPLITS), dtype=torch.float32, device=device)
            lb = torch.empty((r, SPLITS), dtype=torch.float32, device=device)
        else:
            mb = m  # unused placeholders (never read/written)
            lb = l
        b = (zmax, zidx, m, l, mb, lb)
        _BUF_CACHE[key] = b
    return b


def fused_gumbel_sample(
    logits: torch.Tensor,      # (R, V)
    temperature: float,
    seed_buf: torch.Tensor,    # (1,) int64, caller bumps per launch/replay
    alpha: float = 1.0,
    need_logp: bool = True,
    need_logz: bool = False,
    row_offset: int = 0,
):
    """Returns (idx (R,) int64, logp (R,) fp32 | None, logz (R,) fp32 | None).

    idx ~ softmax(alpha * logits / temperature) via Gumbel-max;
    logp = scaled[idx] - lse(scaled); logz = lse(scaled) - alpha*lse(base).
    """
    r, v = logits.shape
    # Philox counters are computed in int32 inside the kernel:
    # (row_offset + row) * V + col.  Overflow does NOT crash — it silently
    # wraps and reuses another launch's noise stream, which would correlate
    # SMC particle proposals.  Unreachable at the bs<=max_bs operating
    # points this was built for; fail loudly rather than silently if a
    # future large-batch caller crosses the line.
    assert (row_offset + r) * v < 2**31, (
        f"fused_gumbel_sample Philox counter overflow: "
        f"(row_offset={row_offset} + rows={r}) * V={v} exceeds int32; "
        "reduce row_offset stride or widen the kernel's counter math."
    )
    need_base = need_logz and alpha != 1.0
    zmax, zidx, m, l, mb, lb = _bufs(r, logits.device, need_base)
    out_idx = torch.empty((r,), dtype=torch.int64, device=logits.device)
    out_logp = (
        torch.empty((r,), dtype=torch.float32, device=logits.device)
        if need_logp else None
    )
    out_logz = (
        torch.zeros((r,), dtype=torch.float32, device=logits.device)
        if need_logz else None
    )

    inv_t = 1.0 / temperature
    _gumbel_sample_stage1[(r, SPLITS)](
        logits, seed_buf,
        zmax, zidx, m, l, mb, lb,
        logits.stride(0), v, inv_t, alpha, row_offset,
        S=SPLITS, BLOCK=BLOCK_V, NEED_BASE=need_base,
        num_warps=8, num_stages=2,
    )
    _gumbel_sample_stage2[(r,)](
        logits,
        zmax, zidx, m, l, mb, lb,
        out_idx,
        out_logp if out_logp is not None else zmax,  # dummy ptr when unused
        out_logz if out_logz is not None else zmax,
        logits.stride(0), inv_t, alpha,
        S=SPLITS, SP2=triton.next_power_of_2(SPLITS),
        NEED_BASE=need_base, NEED_LOGP=need_logp,
        num_warps=1, num_stages=1,
    )
    # logz is identically 0 at alpha == 1 (out_logz pre-zeroed, kernel skips).
    return out_idx, out_logp, out_logz


def fused_chosen_logprob(
    logits: torch.Tensor,      # (R, V)
    tokens: torch.Tensor,      # (R,)
    temperature: float,
) -> torch.Tensor:
    """logp[r] = (logits[r]/T)[tokens[r]] - lse(logits[r]/T), fp32 (R,)."""
    r, v = logits.shape
    _, _, m, l, _, _ = _bufs(r, logits.device, False)
    out = torch.empty((r,), dtype=torch.float32, device=logits.device)
    inv_t = 1.0 / temperature
    _chosen_logprob_stage1[(r, SPLITS)](
        logits, m, l, logits.stride(0), v, inv_t,
        S=SPLITS, BLOCK=BLOCK_V, num_warps=8, num_stages=2,
    )
    _chosen_logprob_stage2[(r,)](
        logits, tokens, m, l, out, logits.stride(0), inv_t,
        S=SPLITS, SP2=triton.next_power_of_2(SPLITS),
        num_warps=1, num_stages=1,
    )
    return out

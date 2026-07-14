"""Split-KV, GQA-packed attention for the SMC linear TARGET_VERIFY path.

The stock triton extend kernel (``extend_attention_fwd``) is a poor fit for
SMC verify shapes — per request only gamma+1 query tokens (9) against a long
shared prefix, with GQA group 4:

* grid ``(seq, q_head, 1)``: the 4 q-heads of a GQA group each stream the
  SAME prefix K/V (4x redundant traffic), and there is no KV-split
  parallelism — each CTA serially walks the whole prefix with
  ``num_stages=1``.  Measured on B200 at 4k context: ~256 us/layer,
  10-25x off the KV-read roofline, 41% of all decode GPU time.

This kernel packs the whole GQA group's queries into one CTA tile
(``E x G`` rows, e.g. 9 x 4 = 36) and parallelizes over prefix chunks:

    grid = (bs, num_kv_heads, NUM_SPLITS + 1)

Splits ``0..NUM_SPLITS-1`` cover disjoint prefix chunks (flash-decoding
style partial softmax); split ``NUM_SPLITS`` handles the causal
extend-x-extend triangle from the contiguous ``k_extend``/``v_extend``
tensors.  A second tiny kernel merges the partials.  Prefix K/V is thus
read exactly once per kv-head, fully parallel across chunks.

Constraints (all satisfied by SMC verify + the deferred-bonus head):
uniform extend length per row, no custom mask, no sliding window, no
sinks, no logit cap, pow-2 head dims, page_size=1 kv indices.

CUDA-graph-safe: grid shape depends only on (bs, num_kv_heads, NUM_SPLITS);
all sequence lengths are read from device tensors inside the kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fixed split count keeps the launch grid static under CUDA graph capture.
# 7 prefix chunks + 1 extend-triangle = 8 partials, so the stage-2 merge's
# tl.arange over partials is a power of 2.
NUM_SPLITS = 7


@triton.jit
def _verify_attn_stage1(
    Q_Extend,      # (total_tokens, H_q, Lq)
    K_Extend,      # (total_tokens, H_kv, Lq) contiguous new K
    V_Extend,      # (total_tokens, H_kv, Lv) contiguous new V
    K_Buffer,      # KV pool key buffer
    V_Buffer,      # KV pool value buffer
    kv_indptr,     # (bs+1,) prefix lengths cumsum
    kv_indices,    # flat prefix kv locations
    Part_Acc,      # (bs, H_kv, S1, EG, Lv) fp32
    Part_M,        # (bs, H_kv, S1, EG) fp32
    Part_L,        # (bs, H_kv, S1, EG) fp32
    sm_scale,
    stride_qbs, stride_qh,
    stride_kebs, stride_keh,
    stride_vebs, stride_veh,
    stride_buf_kbs, stride_buf_kh,
    stride_buf_vbs, stride_buf_vh,
    stride_pa_b, stride_pa_h, stride_pa_s, stride_pa_r,
    stride_pm_b, stride_pm_h, stride_pm_s,
    E: tl.constexpr,           # extend tokens per row (gamma+1 / 2)
    G: tl.constexpr,           # kv_group_num
    EG: tl.constexpr,          # E * G
    SPLITS: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_M: tl.constexpr,     # >= EG, pow2
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_split = tl.program_id(2)

    prefix_start = tl.load(kv_indptr + cur_seq).to(tl.int32)
    prefix_len = tl.load(kv_indptr + cur_seq + 1).to(tl.int32) - prefix_start

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, Lq)
    offs_dv = tl.arange(0, Lv)
    mask_m = offs_m < EG

    # Row r <-> (token e, group head g): e = r // G, g = r % G.
    row_e = offs_m // G
    row_g = offs_m % G

    # Load the group's Q tile: token (seq*E + e), head (kv_head*G + g).
    q_ptrs = (
        Q_Extend
        + (cur_seq * E + row_e)[:, None] * stride_qbs
        + (cur_kv_head * G + row_g)[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, Lv], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    offs_n = tl.arange(0, BLOCK_N)

    if cur_split < SPLITS:
        # ── Prefix chunk [chunk_lo, chunk_hi) via kv_indices gather ──
        chunk = (prefix_len + SPLITS - 1) // SPLITS
        # Round to BLOCK_N so inner tiles are aligned.
        chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        chunk_lo = cur_split * chunk
        chunk_hi = tl.minimum(chunk_lo + chunk, prefix_len)

        for start_n in range(chunk_lo, chunk_hi, BLOCK_N):
            mask_n = (start_n + offs_n) < chunk_hi
            kv_loc = tl.load(
                kv_indices + prefix_start + start_n + offs_n,
                mask=mask_n,
                other=0,
            )
            # Coalesced (n, d) load — each token's head row is contiguous —
            # then an in-register transpose for the dot.  The stock kernel's
            # strided (d, n) global load is what kills its bandwidth.
            k_ptrs = (
                K_Buffer
                + kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q.to(k.dtype), tl.trans(k)) * sm_scale
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            new_m = tl.maximum(m_i, row_max_fixed)
            re_scale = tl.exp(m_i - new_m)
            p = tl.exp(qk - new_m[:, None])
            l_i = l_i * re_scale + tl.sum(p, 1)

            v_ptrs = (
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = new_m
    else:
        # ── Causal extend x extend triangle from contiguous K/V_Extend ──
        # Query token e attends extend kv positions j <= e.  E is tiny
        # (<= gamma+1), one BLOCK_N tile suffices when BLOCK_N >= E.
        for start_e in range(0, E, BLOCK_N):
            ext_n = start_e + offs_n
            mask_n = ext_n < E
            k_ptrs = (
                K_Extend
                + (cur_seq * E + ext_n)[:, None] * stride_kebs
                + cur_kv_head * stride_keh
                + offs_d[None, :]
            )
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            qk = tl.dot(q.to(k.dtype), tl.trans(k)) * sm_scale
            causal = row_e[:, None] >= ext_n[None, :]
            qk = tl.where(
                causal & mask_m[:, None] & mask_n[None, :], qk, float("-inf")
            )

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            new_m = tl.maximum(m_i, row_max_fixed)
            re_scale = tl.exp(m_i - new_m)
            p = tl.exp(qk - new_m[:, None])
            l_i = l_i * re_scale + tl.sum(p, 1)

            v_ptrs = (
                V_Extend
                + (cur_seq * E + ext_n)[:, None] * stride_vebs
                + cur_kv_head * stride_veh
                + offs_dv[None, :]
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = new_m

    # ── Store partial (acc unnormalized, m, l) ──
    pa_ptrs = (
        Part_Acc
        + cur_seq * stride_pa_b
        + cur_kv_head * stride_pa_h
        + cur_split * stride_pa_s
        + offs_m[:, None] * stride_pa_r
        + offs_dv[None, :]
    )
    tl.store(pa_ptrs, acc, mask=mask_m[:, None])
    pm_off = (
        cur_seq * stride_pm_b
        + cur_kv_head * stride_pm_h
        + cur_split * stride_pm_s
        + offs_m
    )
    tl.store(Part_M + pm_off, m_i, mask=mask_m)
    tl.store(Part_L + pm_off, l_i, mask=mask_m)


@triton.jit
def _verify_attn_stage2(
    Part_Acc,     # (bs, H_kv, S1, EG, Lv) fp32
    Part_M,       # (bs, H_kv, S1, EG)
    Part_L,       # (bs, H_kv, S1, EG)
    O_Extend,     # (total_tokens, H_q, Lv)
    stride_pa_b, stride_pa_h, stride_pa_s, stride_pa_r,
    stride_pm_b, stride_pm_h, stride_pm_s,
    stride_obs, stride_oh,
    E: tl.constexpr,
    G: tl.constexpr,
    S1: tl.constexpr,          # NUM_SPLITS + 1
    Lv: tl.constexpr,
):
    """One program per (seq, q_head, extend token): merge S1 partials."""
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_e = tl.program_id(2)

    cur_kv_head = cur_head // G
    cur_g = cur_head % G
    row = cur_e * G + cur_g

    offs_dv = tl.arange(0, Lv)
    offs_s = tl.arange(0, S1)

    pm_off = (
        cur_seq * stride_pm_b
        + cur_kv_head * stride_pm_h
        + offs_s * stride_pm_s
        + row
    )
    m_s = tl.load(Part_M + pm_off)          # (S1,)
    l_s = tl.load(Part_L + pm_off)          # (S1,)

    m_max = tl.max(m_s, 0)
    scale = tl.exp(m_s - m_max)             # empty splits: exp(-inf)=0
    l_total = tl.sum(l_s * scale, 0)

    pa_ptrs = (
        Part_Acc
        + cur_seq * stride_pa_b
        + cur_kv_head * stride_pa_h
        + offs_s[:, None] * stride_pa_s
        + row * stride_pa_r
        + offs_dv[None, :]
    )
    acc_s = tl.load(pa_ptrs)                # (S1, Lv)
    out = tl.sum(acc_s * scale[:, None], 0) / l_total

    o_ptrs = (
        O_Extend
        + (cur_seq * E + cur_e) * stride_obs
        + cur_head * stride_oh
        + offs_dv
    )
    tl.store(o_ptrs, out.to(O_Extend.dtype.element_ty))


_PARTIAL_CACHE: dict = {}


def _get_partials(bs: int, h_kv: int, eg: int, lv: int, device):
    key = (bs, h_kv, eg, lv, device)
    bufs = _PARTIAL_CACHE.get(key)
    if bufs is None:
        s1 = NUM_SPLITS + 1
        acc = torch.empty((bs, h_kv, s1, eg, lv), dtype=torch.float32, device=device)
        m = torch.empty((bs, h_kv, s1, eg), dtype=torch.float32, device=device)
        l = torch.empty((bs, h_kv, s1, eg), dtype=torch.float32, device=device)
        bufs = (acc, m, l)
        _PARTIAL_CACHE[key] = bufs
    return bufs


def verify_attention_fwd(
    q_extend: torch.Tensor,   # (total_tokens, H_q, Lq)
    k_extend: torch.Tensor,   # (total_tokens, H_kv, Lq)
    v_extend: torch.Tensor,   # (total_tokens, H_kv, Lv)
    o_extend: torch.Tensor,   # (total_tokens, H_q, Lv)
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,  # (bs+1,)
    kv_indices: torch.Tensor,
    extend_token_num: int,    # E, uniform per row
    sm_scale: float,
    block_n: int = 64,
    num_warps: int | None = None,
    num_stages: int = 3,
) -> None:
    E = extend_token_num
    total_tokens, h_q, lq = q_extend.shape
    h_kv = k_extend.shape[1]
    lv = v_extend.shape[-1]
    bs = total_tokens // E
    G = h_q // h_kv
    EG = E * G

    part_acc, part_m, part_l = _get_partials(bs, h_kv, EG, lv, q_extend.device)

    BLOCK_M = max(16, triton.next_power_of_2(EG))
    BLOCK_N = block_n
    if num_warps is None:
        # B200-swept: BLOCK_N=64 / 8 warps / 3 stages ≈ 2.1 TB/s at 4k ctx,
        # 5x over the stock extend kernel (which runs ~0.4 TB/s here).
        num_warps = 8

    grid = (bs, h_kv, NUM_SPLITS + 1)
    _verify_attn_stage1[grid](
        q_extend, k_extend, v_extend,
        k_buffer, v_buffer,
        kv_indptr, kv_indices,
        part_acc, part_m, part_l,
        sm_scale,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        E=E, G=G, EG=EG,
        SPLITS=NUM_SPLITS,
        Lq=lq, Lv=lv,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid2 = (bs, h_q, E)
    _verify_attn_stage2[grid2](
        part_acc, part_m, part_l,
        o_extend,
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        o_extend.stride(0), o_extend.stride(1),
        E=E, G=G, S1=NUM_SPLITS + 1, Lv=lv,
        num_warps=1, num_stages=1,
    )

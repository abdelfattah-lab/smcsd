"""Group-shared-prefix (cascade) decode attention for SMC draft steps.

SMC particles of one group share their prompt KV pages by construction
(refcounted page_size=1 lineage; the shared prefix length is fixed at
group materialization).  The stock grouped decode kernel launches one CTA
chain per (particle, kv_head) and therefore reads the shared prefix N
times per group.  At 4k context the draft loop's decode attention is the
largest attention cost in the cycle (~2 TB/s effective, N-fold redundant).

This kernel exploits the group structure:

* stage 1 (shared): grid ``(n_groups, H_kv, SPLITS)`` — one CTA holds the
  Q rows of ALL particles x group-heads of a group (N*G rows, e.g.
  8*4=32) and streams a chunk of the group's shared prefix KV once.
  N-fold traffic cut on the shared range.
* suffix: grid slot ``SPLITS`` handles each particle's private suffix
  ``[L0, S_p)`` per (particle, kv_head) CTA (grid (bs, H_kv)).
* stage 2: standard flash-decoding partial merge.

Assumptions (guaranteed by ScheduleBatchSMC's global-N invariant):
particles of a group are contiguous in batch order, every group has
exactly N particles, and ``shared_lens`` is uniform within a group.
Pad rows/groups carry shared_len 0 and tiny suffix lengths — they only
produce garbage that callers slice off.

CUDA-graph-safe: grids depend only on (bs, N, H_kv); lengths are read
from device tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CASCADE_SPLITS = 7  # +1 suffix slot = 8 partials (pow2 for the merge)


@triton.jit
def _cascade_stage1_shared(
    Q,               # (bs, H_q, Lq)
    K_Buffer, V_Buffer,
    kv_indptr,       # (bs+1,)
    kv_indices,
    shared_lens,     # (bs,) L0 per particle (uniform within group)
    Part_Acc,        # (bs*G rows layout: (n_groups, H_kv, S1, N*G, Lv))
    Part_M, Part_L,
    sm_scale,
    stride_qb, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_pa_g, stride_pa_h, stride_pa_s, stride_pa_r,
    stride_pm_g, stride_pm_h, stride_pm_s,
    N: tl.constexpr,           # particles per group
    G: tl.constexpr,           # kv group num (q heads per kv head)
    NG: tl.constexpr,          # N * G
    SPLITS: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_M: tl.constexpr,     # >= NG, pow2
    BLOCK_N: tl.constexpr,
):
    group = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    p0 = group * N  # first particle of the group
    # Shared range bounds from the first particle (uniform in group).
    l0 = tl.load(shared_lens + p0).to(tl.int32)
    kv_start = tl.load(kv_indptr + p0).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, Lq)
    offs_dv = tl.arange(0, Lv)
    mask_m = offs_m < NG

    # Row r <-> (particle n, group head g): n = r // G, g = r % G.
    row_n = offs_m // G
    row_g = offs_m % G

    q_ptrs = (
        Q
        + (p0 + row_n)[:, None] * stride_qb
        + (kv_head * G + row_g)[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, Lv], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    offs_n = tl.arange(0, BLOCK_N)

    chunk = (l0 + SPLITS - 1) // SPLITS
    chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    lo = split * chunk
    hi = tl.minimum(lo + chunk, l0)

    for start_n in range(lo, hi, BLOCK_N):
        mask_n = (start_n + offs_n) < hi
        kv_loc = tl.load(
            kv_indices + kv_start + start_n + offs_n, mask=mask_n, other=0
        )
        k_ptrs = (
            K_Buffer
            + kv_loc[None, :] * stride_kbs
            + kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q.to(k.dtype), k) * sm_scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        new_m = tl.maximum(m_i, row_max_fixed)
        re_scale = tl.exp(m_i - new_m)
        p = tl.exp(qk - new_m[:, None])
        l_i = l_i * re_scale + tl.sum(p, 1)

        v_ptrs = (
            V_Buffer
            + kv_loc[:, None] * stride_vbs
            + kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = new_m

    pa_ptrs = (
        Part_Acc
        + group * stride_pa_g
        + kv_head * stride_pa_h
        + split * stride_pa_s
        + offs_m[:, None] * stride_pa_r
        + offs_dv[None, :]
    )
    tl.store(pa_ptrs, acc, mask=mask_m[:, None])
    pm_off = (
        group * stride_pm_g
        + kv_head * stride_pm_h
        + split * stride_pm_s
        + offs_m
    )
    tl.store(Part_M + pm_off, m_i, mask=mask_m)
    tl.store(Part_L + pm_off, l_i, mask=mask_m)


@triton.jit
def _cascade_stage1_suffix(
    Q,
    K_Buffer, V_Buffer,
    kv_indptr, kv_indices, shared_lens,
    Part_Acc, Part_M, Part_L,   # same layout; written at split == SPLITS
    sm_scale,
    stride_qb, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_pa_g, stride_pa_h, stride_pa_s, stride_pa_r,
    stride_pm_g, stride_pm_h, stride_pm_s,
    N: tl.constexpr,
    G: tl.constexpr,
    SPLITS: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_G: tl.constexpr,      # >= G, pow2, >= 16 for tl.dot
    BLOCK_N: tl.constexpr,
):
    """Per-particle private suffix [L0, S_p): grid (bs, H_kv)."""
    p = tl.program_id(0)
    kv_head = tl.program_id(1)

    l0 = tl.load(shared_lens + p).to(tl.int32)
    kv_start = tl.load(kv_indptr + p).to(tl.int32)
    kv_len = tl.load(kv_indptr + p + 1).to(tl.int32) - kv_start

    offs_m = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, Lq)
    offs_dv = tl.arange(0, Lv)
    mask_m = offs_m < G

    q_ptrs = (
        Q
        + p * stride_qb
        + (kv_head * G + offs_m)[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_G, Lv], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_G], dtype=tl.float32) - float("inf")
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(l0, kv_len, BLOCK_N):
        mask_n = (start_n + offs_n) < kv_len
        kv_loc = tl.load(
            kv_indices + kv_start + start_n + offs_n, mask=mask_n, other=0
        )
        k_ptrs = (
            K_Buffer
            + kv_loc[None, :] * stride_kbs
            + kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q.to(k.dtype), k) * sm_scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        new_m = tl.maximum(m_i, row_max_fixed)
        re_scale = tl.exp(m_i - new_m)
        pr = tl.exp(qk - new_m[:, None])
        l_i = l_i * re_scale + tl.sum(pr, 1)

        v_ptrs = (
            V_Buffer
            + kv_loc[:, None] * stride_vbs
            + kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc = acc * re_scale[:, None] + tl.dot(pr.to(v.dtype), v)
        m_i = new_m

    # Store into the group-layout partials at split == SPLITS, rows
    # (p_in_group * G + g).
    group = p // N
    p_in_g = p % N
    rows = p_in_g * G + offs_m
    pa_ptrs = (
        Part_Acc
        + group * stride_pa_g
        + kv_head * stride_pa_h
        + SPLITS * stride_pa_s
        + rows[:, None] * stride_pa_r
        + offs_dv[None, :]
    )
    tl.store(pa_ptrs, acc, mask=mask_m[:, None])
    pm_off = (
        group * stride_pm_g
        + kv_head * stride_pm_h
        + SPLITS * stride_pm_s
        + rows
    )
    tl.store(Part_M + pm_off, m_i, mask=mask_m)
    tl.store(Part_L + pm_off, l_i, mask=mask_m)


@triton.jit
def _cascade_stage2(
    Part_Acc, Part_M, Part_L,
    Out,             # (bs, H_q, Lv)
    stride_pa_g, stride_pa_h, stride_pa_s, stride_pa_r,
    stride_pm_g, stride_pm_h, stride_pm_s,
    stride_ob, stride_oh,
    N: tl.constexpr,
    G: tl.constexpr,
    S1: tl.constexpr,
    Lv: tl.constexpr,
):
    p = tl.program_id(0)
    head = tl.program_id(1)

    kv_head = head // G
    g = head % G
    group = p // N
    row = (p % N) * G + g

    offs_s = tl.arange(0, S1)
    offs_dv = tl.arange(0, Lv)

    pm_off = (
        group * stride_pm_g
        + kv_head * stride_pm_h
        + offs_s * stride_pm_s
        + row
    )
    m_s = tl.load(Part_M + pm_off)
    l_s = tl.load(Part_L + pm_off)
    m_max = tl.max(m_s, 0)
    scale = tl.exp(m_s - m_max)
    l_total = tl.sum(l_s * scale, 0)

    pa_ptrs = (
        Part_Acc
        + group * stride_pa_g
        + kv_head * stride_pa_h
        + offs_s[:, None] * stride_pa_s
        + row * stride_pa_r
        + offs_dv[None, :]
    )
    acc_s = tl.load(pa_ptrs)
    out = tl.sum(acc_s * scale[:, None], 0) / l_total
    tl.store(
        Out + p * stride_ob + head * stride_oh + offs_dv,
        out.to(Out.dtype.element_ty),
    )


_PART_CACHE: dict = {}


def _parts(n_groups: int, h_kv: int, ng: int, lv: int, device):
    key = (n_groups, h_kv, ng, lv, device)
    b = _PART_CACHE.get(key)
    if b is None:
        s1 = CASCADE_SPLITS + 1
        acc = torch.empty(
            (n_groups, h_kv, s1, ng, lv), dtype=torch.float32, device=device
        )
        m = torch.empty((n_groups, h_kv, s1, ng), dtype=torch.float32, device=device)
        l = torch.empty((n_groups, h_kv, s1, ng), dtype=torch.float32, device=device)
        b = (acc, m, l)
        _PART_CACHE[key] = b
    return b


def cascade_decode_fwd(
    q: torch.Tensor,            # (bs, H_q, Lq)
    o: torch.Tensor,            # (bs, H_q, Lv)
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,    # (bs+1,)
    kv_indices: torch.Tensor,
    shared_lens: torch.Tensor,  # (bs,)
    group_size: int,            # N particles per group
    sm_scale: float,
) -> None:
    bs, h_q, lq = q.shape
    h_kv = k_buffer.shape[1]
    lv = v_buffer.shape[-1]
    G = h_q // h_kv
    N = group_size
    assert bs % N == 0, f"batch {bs} not group-aligned (N={N})"
    n_groups = bs // N
    NG = N * G

    part_acc, part_m, part_l = _parts(n_groups, h_kv, NG, lv, q.device)

    BLOCK_M = max(16, triton.next_power_of_2(NG))
    BLOCK_G = max(16, triton.next_power_of_2(G))

    _cascade_stage1_shared[(n_groups, h_kv, CASCADE_SPLITS)](
        q, k_buffer, v_buffer,
        kv_indptr, kv_indices, shared_lens,
        part_acc, part_m, part_l,
        sm_scale,
        q.stride(0), q.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        N=N, G=G, NG=NG, SPLITS=CASCADE_SPLITS,
        Lq=lq, Lv=lv, BLOCK_M=BLOCK_M, BLOCK_N=64,
        num_warps=8, num_stages=3,
    )
    _cascade_stage1_suffix[(bs, h_kv)](
        q, k_buffer, v_buffer,
        kv_indptr, kv_indices, shared_lens,
        part_acc, part_m, part_l,
        sm_scale,
        q.stride(0), q.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        N=N, G=G, SPLITS=CASCADE_SPLITS,
        Lq=lq, Lv=lv, BLOCK_G=BLOCK_G, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )
    _cascade_stage2[(bs, h_q)](
        part_acc, part_m, part_l,
        o,
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        o.stride(0), o.stride(1),
        N=N, G=G, S1=CASCADE_SPLITS + 1, Lv=lv,
        num_warps=1, num_stages=1,
    )

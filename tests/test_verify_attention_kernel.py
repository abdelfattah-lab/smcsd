"""Equivalence tests for the SMC fast verify-attention kernel.

Compares ``verify_attention_fwd`` against the stock sglang triton extend
kernel (the current linear TARGET_VERIFY implementation) and a pure-torch
reference, over the shapes SMC actually produces: uniform extend length
(gamma+1 for verify, 2 for the deferred-bonus head), GQA, page_size=1
scattered kv indices.
"""

import pytest
import torch

from smcsd.core.kernels.verify_attention import verify_attention_fwd


def _torch_reference(q, ke, ve, k_buf, v_buf, kv_indptr, kv_indices, E, sm_scale):
    """Naive per-row attention over [prefix ++ causal extend]."""
    total, h_q, d = q.shape
    h_kv = ke.shape[1]
    g = h_q // h_kv
    lv = ve.shape[-1]
    bs = total // E
    out = torch.empty((total, h_q, lv), dtype=torch.float32, device=q.device)
    for i in range(bs):
        lo, hi = int(kv_indptr[i]), int(kv_indptr[i + 1])
        locs = kv_indices[lo:hi].long()
        for e in range(E):
            row = i * E + e
            for h in range(h_q):
                kv_h = h // g
                k_pre = k_buf[locs, kv_h].float()          # (P, d)
                v_pre = v_buf[locs, kv_h].float()
                k_ext = ke[i * E : i * E + e + 1, kv_h].float()
                v_ext = ve[i * E : i * E + e + 1, kv_h].float()
                k_all = torch.cat([k_pre, k_ext], 0)
                v_all = torch.cat([v_pre, v_ext], 0)
                s = (q[row, h].float() @ k_all.T) * sm_scale
                p = torch.softmax(s, dim=-1)
                out[row, h] = p @ v_all
    return out


@pytest.mark.parametrize(
    "bs,E,h_q,h_kv,d,prefix_lo,prefix_hi",
    [
        (8, 9, 32, 8, 128, 300, 5000),   # target verify (8B-like), long ctx
        (8, 9, 32, 8, 128, 1, 40),       # short / near-empty prefix
        (8, 2, 32, 8, 64, 200, 3000),    # deferred-bonus head (1B-like)
        (3, 9, 32, 8, 128, 500, 2000),   # non-pow2 bs
        (8, 9, 32, 32, 128, 300, 1500),  # MHA (group=1)
    ],
)
@torch.inference_mode()
def test_verify_attention_matches_reference(bs, E, h_q, h_kv, d, prefix_lo, prefix_hi):
    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.bfloat16
    total = bs * E
    pool = 1 << 16

    prefix_lens = torch.randint(prefix_lo, prefix_hi + 1, (bs,), device=device)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens, 0)
    n_pre = int(kv_indptr[-1])
    # page_size=1: scattered, unique locations
    kv_indices = torch.randperm(pool, device=device)[:n_pre]

    q = torch.randn(total, h_q, d, dtype=dtype, device=device)
    ke = torch.randn(total, h_kv, d, dtype=dtype, device=device)
    ve = torch.randn(total, h_kv, d, dtype=dtype, device=device)
    k_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    v_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    sm_scale = d ** -0.5

    o_fast = torch.empty_like(q)
    verify_attention_fwd(
        q, ke, ve, o_fast, k_buf, v_buf, kv_indptr, kv_indices, E, sm_scale
    )

    ref = _torch_reference(
        q, ke, ve, k_buf, v_buf, kv_indptr, kv_indices, E, sm_scale
    )
    diff = (o_fast.float() - ref).abs()
    rel = diff.max() / ref.abs().max()
    assert rel < 2e-2, f"max abs diff {diff.max():.4f}, rel {rel:.4f}"


@torch.inference_mode()
def test_verify_attention_matches_stock_extend_kernel():
    """Bit-comparable path check vs the kernel it replaces."""
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd,
    )

    torch.manual_seed(11)
    device = "cuda"
    dtype = torch.bfloat16
    bs, E, h_q, h_kv, d = 8, 9, 32, 8, 128
    total = bs * E
    pool = 1 << 16

    prefix_lens = torch.randint(700, 4200, (bs,), device=device)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens, 0)
    kv_indices = torch.randperm(pool, device=device)[: int(kv_indptr[-1])]

    qo_indptr = torch.arange(0, (bs + 1) * E, E, dtype=torch.int64, device=device)

    q = torch.randn(total, h_q, d, dtype=dtype, device=device)
    ke = torch.randn(total, h_kv, d, dtype=dtype, device=device)
    ve = torch.randn(total, h_kv, d, dtype=dtype, device=device)
    k_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    v_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    sm_scale = d ** -0.5

    o_ref = torch.empty_like(q)
    extend_attention_fwd(
        q, ke, ve, o_ref, k_buf, v_buf,
        qo_indptr, kv_indptr, kv_indices,
        None, True, None, E, 1.0, 1.0, sm_scale,
    )

    o_fast = torch.empty_like(q)
    verify_attention_fwd(
        q, ke, ve, o_fast, k_buf, v_buf, kv_indptr, kv_indices, E, sm_scale
    )

    diff = (o_fast.float() - o_ref.float()).abs()
    rel = diff.max() / o_ref.float().abs().max()
    assert rel < 2e-2, f"max abs diff {diff.max():.4f}, rel {rel:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

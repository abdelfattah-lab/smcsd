"""Equivalence test for the SMC linear-verify split-KV attention path.

Compares linear_verify_split_attention (decode grouped split-KV stage 1
over per-token prefix rows + fused in-flight/merge stage 2) against:
  1. a pure-PyTorch fp32 reference (primary oracle), and
  2. the stock extend_attention_fwd linear-verify path (the kernel it
     replaces), on the same inputs.

Covers ragged prefix lengths (including prefix=1), GQA grouping, and both
ext=gamma+1 (target verify) and ext=2 (deferred-bonus draft head) shapes.
"""

import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
)
from sglang.srt.layers.attention.triton_ops.linear_verify_split import (
    linear_verify_split_attention,
)

DEVICE = "cuda"


def _torch_reference(q, k_inflight, v_inflight, k_buffer, v_buffer,
                     prefix_locs_per_req, ext, sm_scale):
    """fp32 reference: per request r, token i attends [prefix_r, inflight_0..i]."""
    bs = len(prefix_locs_per_req)
    num_heads = q.shape[1]
    kv_heads = k_inflight.shape[1]
    group = num_heads // kv_heads
    out = torch.empty(
        (bs * ext, num_heads, v_buffer.shape[-1]), dtype=torch.float32, device=q.device
    )
    for r in range(bs):
        locs = prefix_locs_per_req[r]
        for i in range(ext):
            row = r * ext + i
            for h in range(num_heads):
                kvh = h // group
                k_pref = k_buffer[locs, kvh].float()           # (S, D)
                v_pref = v_buffer[locs, kvh].float()
                k_inf = k_inflight[r * ext : r * ext + i + 1, kvh].float()
                v_inf = v_inflight[r * ext : r * ext + i + 1, kvh].float()
                keys = torch.cat([k_pref, k_inf], dim=0)
                vals = torch.cat([v_pref, v_inf], dim=0)
                scores = (keys @ q[row, h].float()) * sm_scale
                probs = torch.softmax(scores, dim=0)
                out[row, h] = probs @ vals
    return out


@pytest.mark.parametrize("ext,num_heads,kv_heads,head_dim", [
    (5, 32, 8, 128),   # target verify shape (Llama-8B-like)
    (2, 32, 8, 64),    # deferred-bonus draft head shape (Llama-1B-like)
    (9, 16, 2, 128),   # gamma=8, extreme GQA grouping
])
def test_linear_verify_split_matches_references(ext, num_heads, kv_heads, head_dim):
    torch.manual_seed(0)
    bs = 4
    prefix_lens = [1, 173, 700, 2111][:bs]
    max_kv_splits = 8
    dtype = torch.bfloat16

    total_prefix = sum(prefix_lens)
    pool_size = total_prefix + 64
    # Random (non-contiguous) cache placement for each request's prefix.
    perm = torch.randperm(pool_size, device=DEVICE)
    prefix_locs_per_req = []
    ofs = 0
    for sl in prefix_lens:
        prefix_locs_per_req.append(perm[ofs : ofs + sl].to(torch.int64))
        ofs += sl

    k_buffer = torch.randn(pool_size, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_buffer = torch.randn(pool_size, kv_heads, head_dim, dtype=dtype, device=DEVICE)

    num_tokens = bs * ext
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k_inflight = torch.randn(num_tokens, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_inflight = torch.randn(num_tokens, kv_heads, head_dim, dtype=dtype, device=DEVICE)
    sm_scale = head_dim ** -0.5

    # ---- split-KV path under test ----
    rep_prefix = torch.tensor(prefix_lens, device=DEVICE).repeat_interleave(ext)
    token_kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=DEVICE)
    token_kv_indptr[1:] = torch.cumsum(rep_prefix, dim=0)
    token_kv_indices = torch.cat(
        [prefix_locs_per_req[r] for r in range(bs) for _ in range(ext)]
    )
    token_num_kv_splits = torch.full(
        (num_tokens,), max_kv_splits, dtype=torch.int32, device=DEVICE
    )
    attn_logits = torch.empty(
        (num_tokens, num_heads, max_kv_splits, head_dim),
        dtype=torch.float32,
        device=DEVICE,
    )
    attn_lse = torch.empty(
        (num_tokens, num_heads, max_kv_splits), dtype=torch.float32, device=DEVICE
    )
    o_split = torch.empty_like(q)
    linear_verify_split_attention(
        q,
        k_inflight,
        v_inflight,
        o_split,
        k_buffer,
        v_buffer,
        token_kv_indptr,
        token_kv_indices,
        token_num_kv_splits,
        attn_logits,
        attn_lse,
        max_kv_splits,
        ext,
        sm_scale,
    )

    # ---- pure torch fp32 oracle ----
    o_ref = _torch_reference(
        q, k_inflight, v_inflight, k_buffer, v_buffer,
        prefix_locs_per_req, ext, sm_scale,
    )
    diff = (o_split.float() - o_ref).abs().max().item()
    assert diff < 2.5e-2, f"split path vs torch reference: max abs diff {diff}"

    # ---- stock extend kernel (the path being replaced) ----
    req_kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    req_kv_indptr[1:] = torch.cumsum(
        torch.tensor(prefix_lens, device=DEVICE), dim=0
    )
    req_kv_indices = torch.cat(prefix_locs_per_req)
    qo_indptr = torch.arange(
        0, (bs + 1) * ext, ext, dtype=torch.int32, device=DEVICE
    )
    o_extend = torch.empty_like(q)
    extend_attention_fwd(
        q,
        k_inflight,
        v_inflight,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        req_kv_indptr,
        req_kv_indices,
        None,            # custom_mask
        True,            # causal
        None,            # mask_indptr
        ext,             # max_len_extend
        1.0,             # k_scale
        1.0,             # v_scale
        sm_scale=sm_scale,
    )
    diff2 = (o_split.float() - o_extend.float()).abs().max().item()
    assert diff2 < 2.5e-2, f"split path vs extend kernel: max abs diff {diff2}"

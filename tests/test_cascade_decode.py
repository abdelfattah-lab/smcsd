"""Equivalence test for the group-shared-prefix cascade decode kernel."""

import pytest
import torch

from smcsd.core.kernels.cascade_decode import cascade_decode_fwd


def _reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale):
    bs, h_q, d = q.shape
    h_kv = k_buf.shape[1]
    g = h_q // h_kv
    lv = v_buf.shape[-1]
    out = torch.empty((bs, h_q, lv), dtype=torch.float32, device=q.device)
    for p in range(bs):
        lo, hi = int(kv_indptr[p]), int(kv_indptr[p + 1])
        locs = kv_indices[lo:hi].long()
        for h in range(h_q):
            kv_h = h // g
            k = k_buf[locs, kv_h].float()
            v = v_buf[locs, kv_h].float()
            s = (q[p, h].float() @ k.T) * sm_scale
            out[p, h] = torch.softmax(s, -1) @ v
    return out


@pytest.mark.parametrize(
    "n_groups,N,h_q,h_kv,d,L0,suffix_max",
    [
        (1, 8, 32, 8, 64, 4096, 700),    # draft 1B shape, bs=1 groups
        (2, 8, 32, 8, 64, 1500, 300),    # two groups, different suffixes
        (1, 8, 32, 8, 128, 2048, 400),   # target-like head dim
        (1, 4, 32, 8, 64, 900, 100),     # N=4
        (1, 8, 32, 32, 64, 1000, 200),   # MHA G=1
        (1, 8, 32, 8, 64, 5, 3),         # tiny shared prefix
    ],
)
@torch.inference_mode()
def test_cascade_decode_matches_reference(n_groups, N, h_q, h_kv, d, L0, suffix_max):
    torch.manual_seed(9)
    device = "cuda"
    dtype = torch.bfloat16
    bs = n_groups * N
    pool = 1 << 17

    # Group-shared prefix: same L0 pages for every particle in a group,
    # private suffix pages per particle (varying lengths).
    perm = torch.randperm(pool, device=device)
    cursor = 0
    indices_rows = []
    shared_lens = torch.zeros(bs, dtype=torch.int64, device=device)
    for gi in range(n_groups):
        shared = perm[cursor : cursor + L0]; cursor += L0
        for pi in range(N):
            p = gi * N + pi
            slen = int(torch.randint(1, suffix_max + 1, (1,)))
            suf = perm[cursor : cursor + slen]; cursor += slen
            indices_rows.append(torch.cat([shared, suf]))
            shared_lens[p] = L0
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int64, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor([len(r) for r in indices_rows], device=device), 0
    )
    kv_indices = torch.cat(indices_rows)

    q = torch.randn(bs, h_q, d, dtype=dtype, device=device)
    k_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    v_buf = torch.randn(pool, h_kv, d, dtype=dtype, device=device)
    o = torch.empty_like(q)
    sm_scale = d ** -0.5

    cascade_decode_fwd(
        q, o, k_buf, v_buf, kv_indptr, kv_indices, shared_lens, N, sm_scale
    )
    ref = _reference(q, k_buf, v_buf, kv_indptr, kv_indices, sm_scale)
    diff = (o.float() - ref).abs()
    rel = diff.max() / ref.abs().max()
    assert rel < 2e-2, f"max abs {diff.max():.4f} rel {rel:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

"""Equivalence tests for the device-driven Mamba resample-copy kernel.

The kernel replaces the torch advanced-indexing ``copy_from`` path in
``copy_smc_resampled_hybrid_state``; these tests check it against a plain
per-job torch reference over the same duck-typed pool shape the real
HybridReqToTokenPool exposes (``mamba_pool.mamba_cache.conv`` /
``.temporal`` + ``req_index_to_mamba_index_mapping``).
"""

import types

import pytest
import torch

from smcsd.core.kernels.fused_resample_mamba import fused_mamba_resample_copy


def _make_pool(num_layers=4, num_rows=32, conv_shape=(8, 64), temp_shape=(8, 16, 16),
               device="cuda", seed=0):
    torch.manual_seed(seed)
    conv = [
        torch.randn(num_layers, num_rows, *conv_shape, device=device),
        torch.randn(num_layers, num_rows, *conv_shape, device=device),
    ]
    temporal = torch.randn(num_layers, num_rows, *temp_shape, device=device)
    cache = types.SimpleNamespace(conv=conv, temporal=temporal)
    mamba_pool = types.SimpleNamespace(mamba_cache=cache)
    # identity req->mamba mapping over num_rows entries
    mapping = torch.arange(num_rows, dtype=torch.int32, device=device)
    return types.SimpleNamespace(
        mamba_pool=mamba_pool, req_index_to_mamba_index_mapping=mapping
    )


def _reference_copy(pool, req_pool_indices, dst_slots, src_slots, n_jobs):
    mapping = pool.req_index_to_mamba_index_mapping.long()
    tensors = list(pool.mamba_pool.mamba_cache.conv) + [
        pool.mamba_pool.mamba_cache.temporal
    ]
    for j in range(n_jobs):
        d_req = int(req_pool_indices[int(dst_slots[j])])
        s_req = int(req_pool_indices[int(src_slots[j])])
        if not (0 <= d_req < mapping.numel() and 0 <= s_req < mapping.numel()):
            continue
        d_row, s_row = int(mapping[d_req]), int(mapping[s_req])
        for t in tensors:
            t[:, d_row] = t[:, s_row]


@torch.inference_mode()
def test_matches_reference_and_respects_counter():
    device = "cuda"
    max_slots, max_jobs, n_jobs = 24, 8, 3
    pool_a = _make_pool(device=device, seed=1)
    pool_b = _make_pool(device=device, seed=1)  # identical contents

    req_pool_indices = torch.randperm(32, device=device)[:max_slots].long()
    dst = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.int32, device=device)
    src = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.int32, device=device)
    counter = torch.tensor([n_jobs], dtype=torch.int32, device=device)

    fused_mamba_resample_copy(pool_a, req_pool_indices, dst, src, counter, max_jobs)
    _reference_copy(pool_b, req_pool_indices, dst, src, n_jobs)

    for ta, tb in zip(
        list(pool_a.mamba_pool.mamba_cache.conv) + [pool_a.mamba_pool.mamba_cache.temporal],
        list(pool_b.mamba_pool.mamba_cache.conv) + [pool_b.mamba_pool.mamba_cache.temporal],
    ):
        assert torch.equal(ta, tb)


@torch.inference_mode()
def test_empty_plan_is_noop_and_empty_slot_guarded():
    device = "cuda"
    pool = _make_pool(device=device, seed=2)
    before = [t.clone() for t in pool.mamba_pool.mamba_cache.conv] + [
        pool.mamba_pool.mamba_cache.temporal.clone()
    ]
    req_pool_indices = torch.arange(24, device=device).long()
    # slot 3 maps to EMPTY_SLOT (-1) req: job 1 must be skipped
    req_pool_indices[3] = -1
    dst = torch.tensor([0, 3], dtype=torch.int32, device=device)
    src = torch.tensor([1, 5], dtype=torch.int32, device=device)

    # counter = 0 -> nothing happens even though the plan has entries
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    fused_mamba_resample_copy(pool, req_pool_indices, dst, src, counter, 2)
    after = list(pool.mamba_pool.mamba_cache.conv) + [pool.mamba_pool.mamba_cache.temporal]
    for a, b in zip(after, before):
        assert torch.equal(a, b)

    # counter = 2 -> job 0 copies, job 1 (EMPTY_SLOT dst) is guarded
    counter[0] = 2
    fused_mamba_resample_copy(pool, req_pool_indices, dst, src, counter, 2)
    t = pool.mamba_pool.mamba_cache.temporal
    assert torch.equal(t[:, 0], t[:, 1])           # job 0 applied
    assert torch.equal(t[:, 3], before[-1][:, 3])  # job 1 skipped


@torch.inference_mode()
def test_no_mamba_pool_is_noop():
    device = "cuda"
    pool = types.SimpleNamespace()  # no mamba_pool attr
    fused_mamba_resample_copy(
        pool,
        torch.arange(4, device=device).long(),
        torch.zeros(2, dtype=torch.int32, device=device),
        torch.ones(2, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        2,
    )  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

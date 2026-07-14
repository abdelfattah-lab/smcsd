"""Tests for the fused Gumbel-max sampling / logprob kernels."""

import math

import pytest
import torch

from smcsd.core.kernels.fused_sampling import (
    fused_chosen_logprob,
    fused_gumbel_sample,
)


@torch.inference_mode()
def test_chosen_logprob_matches_torch():
    torch.manual_seed(3)
    r, v, t = 72, 128256, 0.7
    logits = torch.randn(r, v, dtype=torch.float32, device="cuda") * 4
    tokens = torch.randint(0, v, (r,), device="cuda")
    got = fused_chosen_logprob(logits, tokens, t)
    scaled = logits / t
    want = scaled.gather(1, tokens.unsqueeze(1)).squeeze(1) - torch.logsumexp(
        scaled, dim=-1
    )
    assert torch.allclose(got, want, atol=1e-3, rtol=1e-4), (
        (got - want).abs().max()
    )


@torch.inference_mode()
def test_gumbel_sample_logp_and_logz():
    """logp/logz of whatever token was sampled must match torch math."""
    torch.manual_seed(4)
    r, v, t, alpha = 8, 128256, 0.7, 1.7
    logits = torch.randn(r, v, dtype=torch.float32, device="cuda") * 3
    seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64, device="cuda")
    idx, logp, logz = fused_gumbel_sample(
        logits, t, seed, alpha=alpha, need_logp=True, need_logz=True
    )
    base = logits / t
    scaled = alpha * base
    want_logp = scaled.gather(1, idx.unsqueeze(1)).squeeze(1) - torch.logsumexp(
        scaled, dim=-1
    )
    want_logz = torch.logsumexp(scaled, -1) - alpha * torch.logsumexp(base, -1)
    assert torch.allclose(logp, want_logp, atol=1e-3, rtol=1e-4)
    assert torch.allclose(logz, want_logz, atol=1e-3, rtol=1e-4)


@torch.inference_mode()
def test_gumbel_sample_logz_zero_at_alpha_one():
    torch.manual_seed(5)
    logits = torch.randn(4, 4096, dtype=torch.float32, device="cuda")
    seed = torch.zeros(1, dtype=torch.int64, device="cuda")
    _, _, logz = fused_gumbel_sample(
        logits, 0.7, seed, alpha=1.0, need_logp=False, need_logz=True
    )
    assert torch.all(logz == 0)


@torch.inference_mode()
def test_gumbel_sample_distribution():
    """Empirical sampling frequency must match softmax(logits/T)."""
    torch.manual_seed(6)
    v, t = 512, 0.9
    logits = (torch.randn(1, v, device="cuda") * 2).float()
    probs = torch.softmax(logits / t, dim=-1).squeeze(0)

    n = 20000
    counts = torch.zeros(v, device="cuda")
    seed = torch.zeros(1, dtype=torch.int64, device="cuda")
    big = logits.expand(200, v).contiguous()
    for _ in range(n // 200):
        idx, _, _ = fused_gumbel_sample(big, t, seed, need_logp=False)
        counts += torch.bincount(idx, minlength=v).float()
        seed += 1
    emp = counts / counts.sum()
    # Total-variation distance small for 20k draws over 512 bins.
    tv = 0.5 * (emp - probs).abs().sum().item()
    assert tv < 0.08, f"TV distance {tv:.3f}"
    # Seeds actually vary the draws.
    assert (counts > 0).sum() > 50


@torch.inference_mode()
def test_gumbel_sample_seed_replay_determinism():
    torch.manual_seed(7)
    logits = torch.randn(8, 32000, device="cuda").float()
    s1 = torch.full((1,), 42, dtype=torch.int64, device="cuda")
    a, _, _ = fused_gumbel_sample(logits, 0.7, s1, need_logp=False)
    b, _, _ = fused_gumbel_sample(logits, 0.7, s1, need_logp=False)
    s2 = torch.full((1,), 43, dtype=torch.int64, device="cuda")
    c, _, _ = fused_gumbel_sample(logits, 0.7, s2, need_logp=False)
    assert torch.equal(a, b), "same seed must reproduce"
    assert not torch.equal(a, c), "different seed must differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])

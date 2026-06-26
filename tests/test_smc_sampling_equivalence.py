"""Equivalence tests for the fused sampling / logprob path in ``SMCWorker``.

These guard the two optimizations in ``smcsd/core/worker.py::_forward_decode``:

* Gumbel-max sampling replaces ``log_softmax → exp → multinomial`` for both
  the draft loop and the bonus token.
* The per-token logprob is computed as ``logit/T - logsumexp(logit/T)``
  instead of materializing a full-vocab ``log_softmax`` and gathering.

Both are *mathematical identities* (up to float rounding for the logprob, and
up to sampling noise for the draw), so the tests assert exactly that — no GPU,
no sglang, no model required. They are intentionally CPU-only so they run in
any CI; the worker uses the same torch ops on device.
"""

import unittest

import torch


def fused_logprob(logits: torch.Tensor, idx: torch.Tensor, temperature: float):
    """The new path: logprob of ``idx`` under softmax(logits / T)."""
    scaled = logits / temperature
    chosen = scaled.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(scaled, dim=-1)


def reference_logprob(logits: torch.Tensor, idx: torch.Tensor, temperature: float):
    """The old path: full log_softmax then gather."""
    lp = torch.log_softmax(logits / temperature, dim=-1)
    return lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def gumbel_argmax(logits: torch.Tensor, temperature: float, generator=None):
    """The new sampling path: argmax(logits/T + Gumbel(0,1))."""
    scaled = logits / temperature
    u = torch.rand(scaled.shape, generator=generator).clamp_min_(
        torch.finfo(scaled.dtype).tiny
    )
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(scaled + gumbel, dim=-1)


class TestLogprobIdentity(unittest.TestCase):
    def test_matches_log_softmax_gather(self):
        torch.manual_seed(0)
        for temperature in (0.5, 0.7, 1.0, 1.3):
            logits = torch.randn(6, 8, 32000) * 3.0
            idx = torch.randint(0, 32000, (6, 8))
            new = fused_logprob(logits, idx, temperature)
            old = reference_logprob(logits, idx, temperature)
            self.assertLess(
                (new - old).abs().max().item(),
                1e-4,
                f"logprob mismatch at T={temperature}",
            )

    def test_handles_extreme_logits(self):
        # Large-magnitude logits are where a naive softmax would overflow;
        # logsumexp is stable, so the identity must still hold.
        torch.manual_seed(1)
        logits = torch.randn(4, 4, 1000) * 50.0
        idx = torch.randint(0, 1000, (4, 4))
        new = fused_logprob(logits, idx, 0.7)
        old = reference_logprob(logits, idx, 0.7)
        self.assertTrue(torch.isfinite(new).all())
        self.assertLess((new - old).abs().max().item(), 1e-3)


class TestGumbelSampling(unittest.TestCase):
    def test_distribution_matches_multinomial(self):
        # Gumbel-max draws from softmax(logits/T) exactly. With finite samples
        # its deviation from the true pmf should be within Monte-Carlo noise of
        # what multinomial itself achieves.
        torch.manual_seed(2)
        vocab = 50
        logits = torch.randn(vocab) * 2.0
        temperature = 0.8
        probs = torch.softmax(logits / temperature, dim=-1)
        n = 1_000_000

        mn = torch.multinomial(probs, n, replacement=True)
        mn_freq = torch.bincount(mn, minlength=vocab).float() / n

        gm = gumbel_argmax(logits.expand(n, vocab), temperature)
        gm_freq = torch.bincount(gm, minlength=vocab).float() / n

        # Total-variation distance between the two empirical pmfs should be
        # tiny — both are sampling the same distribution.
        tv = 0.5 * (gm_freq - mn_freq).abs().sum().item()
        self.assertLess(tv, 0.01, f"TV distance too large: {tv}")

        # And Gumbel-max's own error from the true pmf is the same order as
        # multinomial's.
        self.assertLess((gm_freq - probs).abs().max().item(), 0.01)

    def test_greedy_limit(self):
        # As T → 0 the draw must concentrate on the argmax logit.
        torch.manual_seed(3)
        logits = torch.randn(16, 200)
        expected = logits.argmax(dim=-1)
        got = gumbel_argmax(logits, temperature=1e-4)
        self.assertTrue((got == expected).all())


if __name__ == "__main__":
    unittest.main()

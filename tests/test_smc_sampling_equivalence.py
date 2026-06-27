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


def bonus_logz(logits: torch.Tensor, alpha: float, temperature: float):
    """Per-row bonus normalizer ``log Z = logsumexp(alpha*ℓ/T) - alpha*logsumexp(ℓ/T)``.

    Mirrors ``SMCWorker._sample_target_power`` /
    ``smc_draft_phase_graph_runner._verify_in_graph``.
    """
    base = logits / temperature
    return torch.logsumexp(alpha * base, dim=-1) - alpha * torch.logsumexp(
        base, dim=-1
    )


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


class TestBonusLogZ(unittest.TestCase):
    """The bonus token's incremental importance weight under the joint-power
    target is ``Z = sum_x p_T(x)^alpha`` (it is sampled from p_T^alpha / Z), i.e.
    ``log Z = logsumexp(alpha*ℓ/T) - alpha*logsumexp(ℓ/T)``."""

    def test_zero_at_alpha_one(self):
        # At alpha=1, Z = sum_x p_T(x) = 1, so log Z == 0 exactly. This is what
        # makes the fix a no-op for the default (non-power) path.
        torch.manual_seed(4)
        for temperature in (0.5, 0.7, 1.0, 1.3):
            logits = torch.randn(8, 32000) * 3.0
            lz = bonus_logz(logits, alpha=1.0, temperature=temperature)
            self.assertLess(lz.abs().max().item(), 1e-4)

    def test_matches_explicit_normalizer(self):
        # log Z must equal log(sum_x p_T(x)^alpha) computed from the normalized
        # tempered probabilities directly.
        torch.manual_seed(5)
        for alpha in (0.5, 2.0, 3.0):
            for temperature in (0.7, 1.0):
                logits = torch.randn(6, 5000) * 2.0
                p = torch.softmax(logits / temperature, dim=-1)
                explicit = torch.log((p**alpha).sum(dim=-1))
                lz = bonus_logz(logits, alpha=alpha, temperature=temperature)
                self.assertLess((lz - explicit).abs().max().item(), 1e-3)

    def test_sign(self):
        # For alpha>1 the power distribution sharpens: sum p^alpha <= 1 => logZ<=0.
        # For alpha<1 it flattens: sum p^alpha >= 1 => logZ>=0.
        torch.manual_seed(6)
        logits = torch.randn(64, 1000) * 2.0
        self.assertTrue((bonus_logz(logits, 2.0, 1.0) <= 1e-6).all())
        self.assertTrue((bonus_logz(logits, 0.5, 1.0) >= -1e-6).all())


def _weight_increment(logprob_diff, bonus_logz, first_eos, eos_hit,
                      length_hit, prev_fin, gamma):
    """Reimplements ``write_back_gpu``'s per-particle weight increment d,
    including the bonus-normalizer gating. Pure CPU, mirrors req_state.py."""
    cutoff = torch.full((logprob_diff.shape[0],), gamma - 1, dtype=torch.int64)
    newly = (length_hit | eos_hit) & ~prev_fin
    eos_cut = newly & eos_hit
    cutoff = torch.where(eos_cut, first_eos.clamp(max=gamma - 1), cutoff)
    cutoff = torch.where(prev_fin, torch.full_like(cutoff, -1), cutoff)
    cols = torch.arange(gamma).unsqueeze(0)
    keep = cols <= cutoff.unsqueeze(1)
    d = (logprob_diff.to(torch.float64) * keep).sum(dim=1)
    eos_in_draft = eos_cut & (first_eos < gamma)
    add_bonus = (~prev_fin & ~eos_in_draft).to(torch.float64)
    d = d + bonus_logz.to(torch.float64) * add_bonus
    return d


class TestBonusWeightGating(unittest.TestCase):
    """The bonus normalizer is added iff the bonus token is actually emitted:
    not when the particle was already finished, and not when EOS terminated the
    sequence within the draft columns 0..gamma-1 (EOS in the bonus column still
    emits the bonus)."""

    def test_scenarios(self):
        gamma, stride = 3, 4  # logprob_diff cols = 3, block = 4 (incl. bonus)
        lpd = torch.tensor([1.0, 2.0, 4.0])  # distinct so cutoffs are visible
        BZ = 10.0
        draft_full = 1.0 + 2.0 + 4.0  # 7.0

        # (first_eos, eos_hit, length_hit, prev_fin) -> expected d
        cases = [
            # no EOS, alive            -> full draft + bonus
            ((stride, False, False, False), draft_full + BZ),
            # EOS in draft col 1       -> cols 0..1, no bonus
            ((1, True, False, False), 1.0 + 2.0),
            # EOS in LAST draft col 2  -> cols 0..2, bonus is post-EOS -> dropped
            ((2, True, False, False), draft_full),
            # EOS in bonus col (==gamma) -> full draft + bonus (bonus emitted)
            ((gamma, True, False, False), draft_full + BZ),
            # already finished         -> nothing
            ((stride, False, False, True), 0.0),
            # length-only finish       -> full draft + bonus
            ((stride, False, True, False), draft_full + BZ),
        ]
        for (fe, eh, lh, pf), expected in cases:
            d = _weight_increment(
                lpd.unsqueeze(0),
                torch.tensor([BZ]),
                torch.tensor([fe]),
                torch.tensor([eh]),
                torch.tensor([lh]),
                torch.tensor([pf]),
                gamma,
            )
            self.assertAlmostEqual(
                d.item(), expected, places=6,
                msg=f"first_eos={fe} eos={eh} len={lh} prev_fin={pf}",
            )

    def test_alpha_one_is_noop(self):
        # At alpha=1 the bonus normalizer is 0, so the increment equals the
        # pure draft-weight sum with no bonus contribution.
        torch.manual_seed(8)
        gamma, stride = 4, 5
        lpd = torch.randn(7, gamma)
        logits = torch.randn(7, 2000) * 2.0
        bz = bonus_logz(logits, alpha=1.0, temperature=0.9)
        d = _weight_increment(
            lpd, bz,
            torch.full((7,), stride),  # no EOS
            torch.zeros(7, dtype=torch.bool),
            torch.zeros(7, dtype=torch.bool),
            torch.zeros(7, dtype=torch.bool),
            gamma,
        )
        self.assertLess(
            (d - lpd.to(torch.float64).sum(dim=1)).abs().max().item(), 1e-4
        )


class TestSampleTargetPower(unittest.TestCase):
    """``_sample_target_power`` draws from softmax(alpha*logits/T)."""

    def test_distribution_matches_power_target(self):
        torch.manual_seed(7)
        vocab = 50
        logits = torch.randn(vocab) * 2.0
        alpha, temperature = 2.0, 0.8
        probs = torch.softmax(alpha * logits / temperature, dim=-1)
        n = 1_000_000

        # Gumbel-max over alpha*logits/T is exactly _sample_target_power's draw.
        draws = gumbel_argmax(logits.expand(n, vocab), temperature / alpha)
        freq = torch.bincount(draws, minlength=vocab).float() / n
        self.assertLess((freq - probs).abs().max().item(), 0.01)


if __name__ == "__main__":
    unittest.main()

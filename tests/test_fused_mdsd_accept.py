"""Tests for the fused Triton exact-accept kernel.

The kernel is distributionally equivalent to
``exact_accept.mdsd_accept_batched`` but uses Philox RNG, so equivalence is
asserted at two levels:

* deterministic structural invariants (independent of RNG): full acceptance
  when q == p with identical chains, winner-matches-prefix, committed
  prefix equals a drafted chain's prefix, accept_len bounds;
* statistical exactness on the synthetic context-dependent (p, q) harness
  from test_exact_accept — the committed stream must follow the target's
  AR factorization (position-0 marginal, position-1 conditionals).

Requires CUDA (Triton); skipped otherwise.
"""

import unittest

import torch

if torch.cuda.is_available():
    from smcsd.core.kernels.fused_mdsd_accept import fused_mdsd_accept

from tests.test_exact_accept import SyntheticModel, _rand_dist


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (Triton kernel)")
class TestFusedStructural(unittest.TestCase):
    def test_full_accept_when_q_equals_p_and_chains_agree(self):
        gen = torch.Generator().manual_seed(0)
        V, gamma, N, G = 4001, 5, 8, 3
        for trial in range(5):
            probs = _rand_dist((G, gamma + 1, V), gen)
            toks = torch.stack(
                [
                    torch.stack(
                        [
                            torch.multinomial(
                                probs[g, t], 1, generator=gen
                            ).squeeze()
                            for t in range(gamma)
                        ]
                    )
                    for g in range(G)
                ]
            )  # (G, gamma)
            tokens = toks.unsqueeze(1).expand(G, N, gamma).cuda()
            q = probs[:, :gamma].unsqueeze(1).expand(G, N, gamma, V).cuda()
            p = probs.unsqueeze(1).expand(G, N, gamma + 1, V).cuda()
            res = fused_mdsd_accept(tokens, q, p)
            self.assertEqual(res.accept_len.tolist(), [gamma] * G)
            self.assertTrue(
                torch.equal(res.tokens[:, :gamma].cpu(), toks)
            )

    def test_winner_matches_accepted_prefix(self):
        gen = torch.Generator().manual_seed(1)
        V, gamma, N, G = 517, 4, 6, 8
        for trial in range(10):
            tokens = torch.randint(0, V, (G, N, gamma), generator=gen).cuda()
            q = _rand_dist((G, N, gamma, V), gen).cuda()
            p = _rand_dist((G, N, gamma + 1, V), gen).cuda()
            res = fused_mdsd_accept(tokens, q, p)
            for g in range(G):
                a = int(res.accept_len[g])
                w = int(res.winner[g])
                self.assertTrue(0 <= a <= gamma)
                self.assertTrue(0 <= w < N)
                self.assertTrue(
                    torch.equal(res.tokens[g, :a], tokens[g, w, :a]),
                    "winner chain must match the accepted prefix",
                )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (Triton kernel)")
class TestFusedStatisticalExactness(unittest.TestCase):
    """Committed stream follows the target's AR factorization (GPU RNG)."""

    TRIALS = 20000
    VOCAB = 5
    GAMMA = 3
    TV_TOL = 0.04

    def _run(self, n_chains, seed):
        model = SyntheticModel(self.VOCAB, seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        pos0 = torch.zeros(self.VOCAB)
        pos1 = torch.zeros(self.VOCAB, self.VOCAB)
        B = 200
        for _ in range(self.TRIALS // B):
            toks, qs, ps = [], [], []
            for _ in range(B):
                t_, q_, p_ = model.draft_group(n_chains, self.GAMMA, gen)
                toks.append(t_); qs.append(q_); ps.append(p_)
            res = fused_mdsd_accept(
                torch.stack(toks).cuda(),
                torch.stack(qs).cuda(),
                torch.stack(ps).cuda(),
            )
            al = res.accept_len.cpu()
            tk = res.tokens.cpu()
            for g in range(B):
                c0 = int(tk[g, 0])
                pos0[c0] += 1
                if int(al[g]) >= 1:
                    pos1[c0, int(tk[g, 1])] += 1

        emp0 = pos0 / pos0.sum()
        tv0 = 0.5 * (emp0 - model.p([])).abs().sum()
        self.assertLess(
            float(tv0), self.TV_TOL,
            f"pos-0 marginal off: TV={float(tv0):.4f} (N={n_chains})",
        )
        for c in range(self.VOCAB):
            n_c = pos1[c].sum()
            if n_c < 1500:
                continue
            emp1 = pos1[c] / n_c
            tv1 = 0.5 * (emp1 - model.p([c])).abs().sum()
            self.assertLess(
                float(tv1), 2.5 * self.TV_TOL,
                f"pos-1 conditional off at c={c}: TV={float(tv1):.4f} "
                f"(N={n_chains}, n={int(n_c)})",
            )

    def test_exactness_n1(self):
        self._run(n_chains=1, seed=21)

    def test_exactness_n4(self):
        self._run(n_chains=4, seed=22)

    def test_exactness_n8(self):
        self._run(n_chains=8, seed=23)


def draft_tree_group(model, n_chains, gamma, fanout, generator):
    """Draft a coordinated tree: at branch depth t, each parent block draws
    fanout[t] DISTINCT tokens (sequential residual draws — the order the
    kernel's without-replacement correction assumes) and each child block
    adopts one; other depths draft i.i.d. per row."""
    V = model.vocab
    tokens = torch.zeros(n_chains, gamma, dtype=torch.int64)
    q_probs = torch.zeros(n_chains, gamma, V)
    p_probs = torch.zeros(n_chains, gamma + 1, V)
    ctxs = [[] for _ in range(n_chains)]
    cp = 1
    for t in range(gamma):
        f = fanout[t] if t < len(fanout) else 1
        for i in range(n_chains):
            q_probs[i, t] = model.q(ctxs[i])
            p_probs[i, t] = model.p(ctxs[i])
        if f > 1:
            parent = n_chains // cp
            child = parent // f
            for p0 in range(0, n_chains, parent):
                avail = model.q(ctxs[p0]).clone()
                for j in range(f):
                    tok = int(
                        torch.multinomial(
                            avail / avail.sum(), 1, generator=generator
                        )
                    )
                    for r in range(p0 + j * child, p0 + (j + 1) * child):
                        tokens[r, t] = tok
                    avail[tok] = 0.0
            cp *= f
        else:
            for i in range(n_chains):
                tokens[i, t] = int(
                    torch.multinomial(
                        q_probs[i, t], 1, generator=generator
                    )
                )
        for i in range(n_chains):
            ctxs[i].append(int(tokens[i, t]))
    for i in range(n_chains):
        p_probs[i, gamma] = model.p(ctxs[i])
    return tokens, q_probs, p_probs


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (Triton kernel)")
class TestTreeStatisticalExactness(unittest.TestCase):
    """Tree drafting (distinct-candidate branch depths) stays lossless."""

    TRIALS = 20000
    VOCAB = 5
    GAMMA = 3
    TV_TOL = 0.04

    def _run(self, n_chains, fanout, seed):
        model = SyntheticModel(self.VOCAB, seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        pos0 = torch.zeros(self.VOCAB)
        pos1 = torch.zeros(self.VOCAB, self.VOCAB)
        B = 200
        for _ in range(self.TRIALS // B):
            toks, qs, ps = [], [], []
            for _ in range(B):
                t_, q_, p_ = draft_tree_group(
                    model, n_chains, self.GAMMA, fanout, gen
                )
                toks.append(t_); qs.append(q_); ps.append(p_)
            res = fused_mdsd_accept(
                torch.stack(toks).cuda(),
                torch.stack(qs).cuda(),
                torch.stack(ps).cuda(),
                fanout=fanout,
            )
            al = res.accept_len.cpu()
            tk = res.tokens.cpu()
            for g in range(B):
                c0 = int(tk[g, 0])
                pos0[c0] += 1
                if int(al[g]) >= 1:
                    pos1[c0, int(tk[g, 1])] += 1

        emp0 = pos0 / pos0.sum()
        tv0 = 0.5 * (emp0 - model.p([])).abs().sum()
        self.assertLess(
            float(tv0), self.TV_TOL,
            f"pos-0 marginal off: TV={float(tv0):.4f} "
            f"(N={n_chains}, fanout={fanout})",
        )
        for c in range(self.VOCAB):
            n_c = pos1[c].sum()
            if n_c < 1500:
                continue
            emp1 = pos1[c] / n_c
            tv1 = 0.5 * (emp1 - model.p([c])).abs().sum()
            self.assertLess(
                float(tv1), 2.5 * self.TV_TOL,
                f"pos-1 conditional off at c={c}: TV={float(tv1):.4f} "
                f"(N={n_chains}, fanout={fanout}, n={int(n_c)})",
            )

    def test_tree_fan2(self):
        self._run(n_chains=4, fanout=[2], seed=31)

    def test_tree_fan22(self):
        self._run(n_chains=8, fanout=[2, 2], seed=32)

    def test_tree_fan4(self):
        self._run(n_chains=8, fanout=[4], seed=33)


if __name__ == "__main__":
    unittest.main()

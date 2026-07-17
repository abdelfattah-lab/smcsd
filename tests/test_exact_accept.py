"""Tests for the exact-mode multi-draft accept operator.

Covers ``smcsd/core/exact_accept.py`` — the SpecTr k-SEQ / SpecInfer-style
sequential rejection-with-residual scheme over N i.i.d. particle chains.

Three layers, all CPU-only (pure torch, no sglang, no GPU, no model):

1. Structural invariants: full acceptance when q == p and chains agree,
   duplicate auto-rejection, winner/viability consistency.
2. Reference vs batched equivalence on the deterministic part of the walk
   (accept_len, accepted prefix, winner) under injected accept draws.
3. Statistical exactness: on a synthetic context-dependent (p, q) pair, the
   committed tokens must follow the target's autoregressive factorization —
   position-0 marginal equals p(.|root); the position-1 conditional given
   the committed first token equals p(.|c).  This is the property that makes
   "exact mode" exact, tested for N in {1, 2, 4} (N=1 degenerates to vanilla
   speculative sampling).
"""

import unittest

import torch

from smcsd.core.exact_accept import (
    mdsd_accept_batched,
    mdsd_accept_reference,
)


def _rand_dist(shape, generator):
    """Random strictly-positive distributions over the last dim."""
    x = torch.rand(*shape, generator=generator).clamp_min(1e-3)
    return x / x.sum(-1, keepdim=True)


class SyntheticModel:
    """Context-dependent target p and proposal q over a small vocab.

    Distributions are lazily materialized per context tuple with a seeded
    generator, so p(.|ctx) is a fixed random function of the context.
    """

    def __init__(self, vocab: int, seed: int, sharpness: float = 1.0):
        self.vocab = vocab
        self.sharpness = sharpness
        self._gen_seed = seed
        self._p = {}
        self._q = {}

    def _make(self, table, ctx):
        if ctx not in table:
            # Per-context deterministic generator: seed mixes the context.
            g = torch.Generator().manual_seed(
                (self._gen_seed * 1000003 + hash(ctx)) % (2**63)
            )
            d = torch.rand(self.vocab, generator=g).clamp_min(1e-3)
            d = d**self.sharpness
            table[ctx] = d / d.sum()
        return table[ctx]

    def p(self, ctx):
        return self._make(self._p, ("p",) + tuple(ctx))

    def q(self, ctx):
        return self._make(self._q, ("q",) + tuple(ctx))

    def draft_group(self, n_chains, gamma, generator):
        """Draft N i.i.d. chains from q and assemble the operator's inputs."""
        V = self.vocab
        tokens = torch.zeros(n_chains, gamma, dtype=torch.int64)
        q_probs = torch.zeros(n_chains, gamma, V)
        p_probs = torch.zeros(n_chains, gamma + 1, V)
        for i in range(n_chains):
            ctx = []
            for t in range(gamma):
                q_t = self.q(ctx)
                p_probs[i, t] = self.p(ctx)
                q_probs[i, t] = q_t
                tok = int(torch.multinomial(q_t, 1, generator=generator))
                tokens[i, t] = tok
                ctx.append(tok)
            p_probs[i, gamma] = self.p(ctx)
        return tokens, q_probs, p_probs


class TestStructural(unittest.TestCase):
    def test_full_accept_when_q_equals_p_and_chains_agree(self):
        """q == p and identical chains => accept prob 1 at every depth."""
        gen = torch.Generator().manual_seed(0)
        V, gamma, N = 7, 5, 4
        for _ in range(20):
            probs = _rand_dist((gamma + 1, V), gen)
            tokens_1 = torch.stack(
                [torch.multinomial(probs[t], 1, generator=gen).squeeze()
                 for t in range(gamma)]
            )
            tokens = tokens_1.unsqueeze(0).expand(N, gamma)
            q = probs[:gamma].unsqueeze(0).expand(N, gamma, V)
            p = probs.unsqueeze(0).expand(N, gamma + 1, V)
            res = mdsd_accept_reference(
                tokens, q, p, generator=gen
            )
            self.assertEqual(int(res.accept_len), gamma)
            self.assertTrue(torch.equal(res.tokens[:gamma], tokens_1))

    def test_winner_matches_accepted_prefix(self):
        gen = torch.Generator().manual_seed(1)
        V, gamma, N = 6, 4, 4
        for _ in range(50):
            tokens = torch.randint(0, V, (N, gamma), generator=gen)
            q = _rand_dist((N, gamma, V), gen)
            p = _rand_dist((N, gamma + 1, V), gen)
            res = mdsd_accept_reference(tokens, q, p, generator=gen)
            a = int(res.accept_len)
            w = int(res.winner)
            self.assertTrue(
                torch.equal(res.tokens[:a], tokens[w, :a]),
                "winner chain must match the accepted prefix",
            )

    def test_duplicate_candidates_auto_reject(self):
        """After c is rejected, an identical candidate must also reject
        (residual has zero mass at c)."""
        V, gamma, N = 4, 1, 2
        tokens = torch.tensor([[2], [2]])
        q = torch.full((N, gamma, V), 0.25)
        # p(2) < q(2) so rejection of chain 0's token is possible.
        p_row = torch.tensor([0.4, 0.3, 0.1, 0.2])
        p = torch.zeros(N, gamma + 1, V)
        p[:, :, :] = p_row
        # Force: chain 0 rejects (u=0.99 > 0.1/0.25), chain 1's identical
        # token must auto-reject even with u=0 (residual mass at 2 is 0)...
        # u=0 <= 0 accepts on the boundary, so use a tiny positive u.
        uniforms = torch.tensor([[0.99, 1e-9]])
        gen = torch.Generator().manual_seed(3)
        res = mdsd_accept_reference(
            tokens, q, p, generator=gen, uniforms=uniforms
        )
        self.assertEqual(int(res.accept_len), 0)
        # Residual sample must avoid token 2 entirely.
        self.assertNotEqual(int(res.tokens[0]), 2)


class TestBatchedEquivalence(unittest.TestCase):
    def test_accept_walk_matches_reference(self):
        """With injected accept draws, the deterministic part of the walk
        (accept_len, accepted prefix, winner) is identical between the
        batched and reference implementations."""
        gen = torch.Generator().manual_seed(42)
        for trial in range(30):
            G = int(torch.randint(1, 5, (1,), generator=gen))
            N = int(torch.randint(1, 6, (1,), generator=gen))
            gamma = int(torch.randint(1, 6, (1,), generator=gen))
            V = int(torch.randint(3, 12, (1,), generator=gen))
            tokens = torch.randint(0, V, (G, N, gamma), generator=gen)
            q = _rand_dist((G, N, gamma, V), gen)
            p = _rand_dist((G, N, gamma + 1, V), gen)
            uniforms = torch.rand(G, gamma, N, generator=gen)

            bat = mdsd_accept_batched(
                tokens, q, p,
                generator=torch.Generator().manual_seed(7),
                uniforms=uniforms,
            )
            for g in range(G):
                ref = mdsd_accept_reference(
                    tokens[g], q[g], p[g],
                    generator=torch.Generator().manual_seed(7),
                    uniforms=uniforms[g],
                )
                a = int(ref.accept_len)
                self.assertEqual(int(bat.accept_len[g]), a)
                self.assertEqual(int(bat.winner[g]), int(ref.winner))
                self.assertTrue(
                    torch.equal(bat.tokens[g, :a], ref.tokens[:a])
                )


class TestStatisticalExactness(unittest.TestCase):
    """The committed block must follow the target's AR factorization."""

    TRIALS = 20000
    VOCAB = 5
    GAMMA = 3
    TV_TOL = 0.04  # ~5x the expected TV noise floor at these counts

    def _run(self, n_chains, seed, use_batched):
        model = SyntheticModel(self.VOCAB, seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        pos0 = torch.zeros(self.VOCAB)
        # counts of committed[1] given committed[0] = c (when len >= 2)
        pos1 = torch.zeros(self.VOCAB, self.VOCAB)
        if use_batched:
            # Batch many groups per operator call for speed.
            B = 100
            n_batches = self.TRIALS // B
            for _ in range(n_batches):
                toks, qs, ps = [], [], []
                for _ in range(B):
                    t_, q_, p_ = model.draft_group(
                        n_chains, self.GAMMA, gen
                    )
                    toks.append(t_); qs.append(q_); ps.append(p_)
                res = mdsd_accept_batched(
                    torch.stack(toks), torch.stack(qs), torch.stack(ps),
                    generator=gen,
                )
                for g in range(B):
                    c0 = int(res.tokens[g, 0])
                    pos0[c0] += 1
                    if int(res.accept_len[g]) >= 1:
                        pos1[c0, int(res.tokens[g, 1])] += 1
        else:
            for _ in range(self.TRIALS):
                t_, q_, p_ = model.draft_group(n_chains, self.GAMMA, gen)
                res = mdsd_accept_reference(t_, q_, p_, generator=gen)
                c0 = int(res.tokens[0])
                pos0[c0] += 1
                if int(res.accept_len) >= 1:
                    pos1[c0, int(res.tokens[1])] += 1

        # Position-0 marginal == p(.|root).
        emp0 = pos0 / pos0.sum()
        tv0 = 0.5 * (emp0 - model.p([])).abs().sum()
        self.assertLess(
            float(tv0), self.TV_TOL,
            f"pos-0 marginal off: TV={float(tv0):.4f} (N={n_chains})",
        )

        # Position-1 conditional == p(.|c) for well-sampled prefixes.
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

    def test_exactness_reference_n1(self):
        # N=1 degenerates to vanilla speculative sampling.
        self._run(n_chains=1, seed=11, use_batched=False)

    def test_exactness_reference_n2(self):
        self._run(n_chains=2, seed=12, use_batched=False)

    def test_exactness_batched_n4(self):
        self._run(n_chains=4, seed=13, use_batched=True)

    def test_exactness_batched_n8(self):
        self._run(n_chains=8, seed=14, use_batched=True)


if __name__ == "__main__":
    unittest.main()

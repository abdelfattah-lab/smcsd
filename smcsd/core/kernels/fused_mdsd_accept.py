"""Fused multi-draft rejection-sampling accept kernel (exact mode).

One Triton program per group replaces the torch implementation of
``smcsd.core.exact_accept.mdsd_accept_batched`` — which walks gamma depths x
N chains with ~6 small CUDA ops per candidate (~400 launches per decode
cycle at N=8, gamma=8) — with a single launch.

Semantics (identical to the torch operator, see exact_accept.py):

    per group, per depth t:
        beta <- p(. | committed prefix)   (the depth-t row of the first
                                           viable chain)
        for each viable chain i (ascending):
            c = d[i, t]
            accept c w.p. min(1, beta[c] / q_i[c])   -> commit, shrink
                                                        viability, next depth
            reject: beta <- max(beta - q_i, 0)  (renormalization is carried
                    implicitly: the running mass S divides the ratio)
        no acceptance: commit x ~ beta / S (CDF scan); STOP (accept_len = t)
    all gamma depths accepted: commit bonus ~ p(. | full chain)

RNG is Philox via ``tl.rand(seed, offset)`` with one unique offset per
(group, depth, candidate) accept draw and per (group, depth) residual/bonus
draw — a fresh host ``seed`` per call keeps cycles independent (same pattern
as ``fused_collect``).

The viability set is a bitmask (N <= 32).  ``beta`` lives in a (G, V) fp32
scratch buffer; the residual update and the CDF-scan sample are V-block
loops executed only when actually needed (scalar branches).  fp32 residual
math vs the torch operator's fp64 is statistically indistinguishable at the
test tolerances (and the accept draw itself dominates the noise).
"""

import torch
import triton
import triton.language as tl

from smcsd.core.exact_accept import ExactAcceptResult

_seed_counter = [12345]


@triton.jit
def _mdsd_accept_kernel(
    d_ptr,        # (G, N, GAMMA) int32 — drafted tokens
    q_ptr,        # (G, N, GAMMA, V) fp32 — normalized proposal rows
    p_ptr,        # (G, N, GAMMA+1, V) fp32 — normalized target rows
    beta_ptr,     # (G, V) fp32 scratch
    out_len_ptr,  # (G,) int32
    out_tok_ptr,  # (G, GAMMA+1) int32
    out_win_ptr,  # (G,) int32
    seed,         # int32 host seed for this call
    V,            # vocab size (runtime)
    N: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)

    d_g = d_ptr + g * N * GAMMA
    q_g = q_ptr + g.to(tl.int64) * N * GAMMA * V
    p_g = p_ptr + g.to(tl.int64) * N * (GAMMA + 1) * V
    beta_g = beta_ptr + g.to(tl.int64) * V

    viable = (1 << N) - 1
    stopped = 0
    accept_len = GAMMA
    winner = 0

    for t in range(GAMMA):
        if stopped == 0:
            # rep = first viable chain (lowest set bit index).
            rep = 0
            found_rep = 0
            for j in range(N):
                take = ((viable >> j) & 1) & (found_rep == 0)
                rep = tl.where(take != 0, j, rep)
                found_rep = found_rep | take

            # beta <- p[rep, t, :]; S = 1 (rows are normalized).  Copy is
            # deferred: while no rejection has happened, read beta straight
            # from p (beta_valid flags whether scratch holds the residual).
            beta_row = p_g + rep.to(tl.int64) * (GAMMA + 1) * V + t * V
            beta_valid = 0
            s_mass = 1.0

            accepted = 0
            tok = 0
            for i in range(N):
                do_i = ((viable >> i) & 1) & (accepted == 0)
                if do_i != 0:
                    c = tl.load(d_g + i * GAMMA + t)
                    q_row = q_g + i.to(tl.int64) * GAMMA * V + t * V
                    q_c = tl.load(q_row + c)
                    if beta_valid == 0:
                        beta_c = tl.load(beta_row + c)
                    else:
                        beta_c = tl.load(beta_g + c)
                    u = tl.rand(seed, (g * GAMMA + t) * (N + 1) + i)
                    ratio = (beta_c / s_mass) / tl.maximum(q_c, 1e-30)
                    if u <= tl.minimum(ratio, 1.0):
                        accepted = 1
                        tok = c
                    else:
                        # Residual update: beta <- max(beta - S*q, 0) over V
                        # (dividing by S is deferred into the ratio; keeping
                        # beta at the original scale means subtracting S*q).
                        new_s = 0.0
                        for vb in range(0, tl.cdiv(V, BLOCK)):
                            offs = vb * BLOCK + tl.arange(0, BLOCK)
                            m = offs < V
                            if beta_valid == 0:
                                b = tl.load(beta_row + offs, mask=m, other=0.0)
                            else:
                                b = tl.load(beta_g + offs, mask=m, other=0.0)
                            qv = tl.load(q_row + offs, mask=m, other=0.0)
                            b = tl.maximum(b - s_mass * qv, 0.0)
                            tl.store(beta_g + offs, b, mask=m)
                            new_s += tl.sum(b, axis=0)
                        beta_valid = 1
                        s_mass = tl.maximum(new_s, 1e-30)

            if accepted != 0:
                # Shrink viability to chains matching the committed token.
                new_viable = 0
                for j in range(N):
                    dj = tl.load(d_g + j * GAMMA + t)
                    keep = ((viable >> j) & 1) & (dj == tok)
                    new_viable = new_viable | (keep << j)
                viable = new_viable
                tl.store(out_tok_ptr + g * (GAMMA + 1) + t, tok)
            else:
                # Residual sample x ~ beta / S: CDF scan over V blocks.
                u2 = tl.rand(seed, (g * GAMMA + t) * (N + 1) + N)
                target = u2.to(tl.float32) * s_mass
                running = 0.0
                found = V
                for vb in range(0, tl.cdiv(V, BLOCK)):
                    offs = vb * BLOCK + tl.arange(0, BLOCK)
                    m = offs < V
                    if beta_valid == 0:
                        b = tl.load(beta_row + offs, mask=m, other=0.0)
                    else:
                        b = tl.load(beta_g + offs, mask=m, other=0.0)
                    pref = tl.cumsum(b, axis=0)
                    hit = m & (running + pref >= target)
                    idx = tl.min(tl.where(hit, offs, V), axis=0)
                    found = tl.minimum(found, tl.where(idx < V, idx, V))
                    running += tl.sum(b, axis=0)
                    # No early break: found freezes at the first hit since
                    # later blocks can only produce larger indices... they
                    # could also hit, but `found` keeps the minimum.
                x = tl.minimum(found, V - 1)
                tl.store(out_tok_ptr + g * (GAMMA + 1) + t, x)
                accept_len = t
                winner = rep
                stopped = 1

    if stopped == 0:
        # Full acceptance: bonus ~ p[rep, GAMMA, :] (normalized, S=1).
        rep = 0
        found_rep = 0
        for j in range(N):
            take = ((viable >> j) & 1) & (found_rep == 0)
            rep = tl.where(take != 0, j, rep)
            found_rep = found_rep | take
        winner = rep
        p_row = p_g + rep.to(tl.int64) * (GAMMA + 1) * V + GAMMA * V
        u2 = tl.rand(seed, (g * GAMMA + GAMMA - 1) * (N + 1) + N)
        target = u2.to(tl.float32)
        running = 0.0
        found = V
        for vb in range(0, tl.cdiv(V, BLOCK)):
            offs = vb * BLOCK + tl.arange(0, BLOCK)
            m = offs < V
            b = tl.load(p_row + offs, mask=m, other=0.0)
            pref = tl.cumsum(b, axis=0)
            hit = m & (running + pref >= target)
            idx = tl.min(tl.where(hit, offs, V), axis=0)
            found = tl.minimum(found, tl.where(idx < V, idx, V))
            running += tl.sum(b, axis=0)
        tl.store(
            out_tok_ptr + g * (GAMMA + 1) + GAMMA, tl.minimum(found, V - 1)
        )

    tl.store(out_len_ptr + g, accept_len)
    tl.store(out_win_ptr + g, winner)


def fused_mdsd_accept(
    draft_tokens: torch.Tensor,   # (G, N, gamma) int
    draft_probs: torch.Tensor,    # (G, N, gamma, V) float
    target_probs: torch.Tensor,   # (G, N, gamma+1, V) float
    *,
    seed: int = None,
) -> ExactAcceptResult:
    """One-launch exact accept over G groups (see module docstring).

    Distributionally equivalent to ``mdsd_accept_batched`` (same operator,
    Philox RNG instead of torch's generator).  CUDA only.
    """
    G, N, gamma = draft_tokens.shape
    V = draft_probs.shape[-1]
    device = draft_tokens.device
    assert N <= 32, "viability bitmask supports N <= 32"

    if seed is None:
        _seed_counter[0] = (_seed_counter[0] + 1) % (2**31 - 1)
        seed = _seed_counter[0]

    d = draft_tokens.to(torch.int32).contiguous()
    q = draft_probs.to(torch.float32).contiguous()
    p = target_probs.to(torch.float32).contiguous()
    beta = torch.empty(G, V, dtype=torch.float32, device=device)
    out_len = torch.empty(G, dtype=torch.int32, device=device)
    out_tok = torch.zeros(G, gamma + 1, dtype=torch.int32, device=device)
    out_win = torch.empty(G, dtype=torch.int32, device=device)

    _mdsd_accept_kernel[(G,)](
        d, q, p, beta, out_len, out_tok, out_win,
        seed, V,
        N=N, GAMMA=gamma, BLOCK=1024,
    )
    return ExactAcceptResult(
        accept_len=out_len.to(torch.int64),
        tokens=out_tok.to(torch.int64),
        winner=out_win.to(torch.int64),
    )

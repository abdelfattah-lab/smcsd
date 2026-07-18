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
draw.  The seed is loaded from a (1,) int32 DEVICE buffer that the caller
bumps before each launch (``seed_buf.add_(1)``) — device-side so the launch
is CUDA-graph-capturable (a replayed in-graph ``add_`` keeps every replay's
noise fresh, the same pattern as the cycle runner's ``sample_seed``).

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
    seed_ptr,     # (1,) int32 device seed (bumped by the caller per launch)
    V,            # vocab size (runtime)
    N: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK: tl.constexpr,
    # Tree fan-out per shallow depth (1 = no branch).  At a branch depth
    # the chains were drafted with COORDINATED DISTINCT tokens per child
    # block (sampling without replacement, block-leader order), so the
    # candidates are the viable child leaders and their proposal densities
    # carry the without-replacement correction q/(1-R).
    FAN0: tl.constexpr = 1,
    FAN1: tl.constexpr = 1,
    FAN2: tl.constexpr = 1,
    FAN3: tl.constexpr = 1,
):
    g = tl.program_id(0)
    seed = tl.load(seed_ptr)

    d_g = d_ptr + g * N * GAMMA
    q_g = q_ptr + g.to(tl.int64) * N * GAMMA * V
    p_g = p_ptr + g.to(tl.int64) * N * (GAMMA + 1) * V
    beta_g = beta_ptr + g.to(tl.int64) * V

    viable = (1 << N) - 1
    stopped = 0
    accept_len = GAMMA
    winner = 0
    cp = 1  # cumulative fan product (rows per subtree = N // cp)

    for t in range(GAMMA):
        fan_t = 1
        if t == 0:
            fan_t = FAN0
        elif t == 1:
            fan_t = FAN1
        elif t == 2:
            fan_t = FAN2
        elif t == 3:
            fan_t = FAN3
        # Child-block size: candidates at a branch depth are the viable
        # child leaders (row % blk == 0); at i.i.d. depths every viable
        # row is a candidate (blk = 1).
        blk = 1
        if fan_t > 1:
            blk = N // (cp * fan_t)
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
            r_mass = 0.0  # earlier-sibling proposal mass (branch depths)
            for i in range(N):
                is_cand = (viable >> i) & 1
                if blk > 1:
                    is_cand = is_cand & ((i % blk) == 0)
                do_i = is_cand & (accepted == 0)
                if do_i != 0:
                    c = tl.load(d_g + i * GAMMA + t)
                    q_row = q_g + i.to(tl.int64) * GAMMA * V + t * V
                    q_c = tl.load(q_row + c)
                    if beta_valid == 0:
                        beta_c = tl.load(beta_row + c)
                    else:
                        beta_c = tl.load(beta_g + c)
                    u = tl.rand(seed, (g * GAMMA + t) * (N + 1) + i)
                    # Without-replacement correction: candidate i was drawn
                    # from q / (1 - R_i) restricted off earlier siblings
                    # (R_i = their q-mass, in draft order = scan order).
                    # R stays 0 at i.i.d. depths.
                    inv = tl.maximum(1.0 - r_mass, 1e-30)
                    ratio = (beta_c * inv) / (
                        s_mass * tl.maximum(q_c, 1e-30)
                    )
                    if u <= tl.minimum(ratio, 1.0):
                        accepted = 1
                        tok = c
                    else:
                        # Residual update: beta <- max(beta - S*q_i, 0)
                        # with q_i = q/(1-R_i) off earlier siblings (their
                        # beta is already 0, so the clamp absorbs the
                        # unrestricted subtraction there).
                        scale = s_mass / inv
                        new_s = 0.0
                        for vb in range(0, tl.cdiv(V, BLOCK)):
                            offs = vb * BLOCK + tl.arange(0, BLOCK)
                            m = offs < V
                            if beta_valid == 0:
                                b = tl.load(beta_row + offs, mask=m, other=0.0)
                            else:
                                b = tl.load(beta_g + offs, mask=m, other=0.0)
                            qv = tl.load(q_row + offs, mask=m, other=0.0)
                            b = tl.maximum(b - scale * qv, 0.0)
                            tl.store(beta_g + offs, b, mask=m)
                            new_s += tl.sum(b, axis=0)
                        beta_valid = 1
                        s_mass = tl.maximum(new_s, 1e-30)
                        if fan_t > 1:
                            r_mass = r_mass + q_c

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
        cp = cp * fan_t

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


def fused_mdsd_accept_into(
    n_groups: int,
    n_particles: int,
    gamma: int,
    vocab: int,
    d_i32: torch.Tensor,      # (>=G*N rows viewed (G,N,gamma)) int32
    q_f32: torch.Tensor,      # (>=G*N, gamma, V) fp32, normalized
    p_f32: torch.Tensor,      # (>=G*N, gamma+1, V) fp32, normalized
    beta: torch.Tensor,       # (>=G, V) fp32 scratch
    out_len: torch.Tensor,    # (>=G,) int32
    out_tok: torch.Tensor,    # (>=G, gamma+1) int32
    out_win: torch.Tensor,    # (>=G,) int32
    seed_buf: torch.Tensor,   # (1,) int32 device seed (caller bumps it)
    fanout=None,              # tree fan-out per shallow depth, e.g. (2, 2)
) -> None:
    """Launch into caller-owned buffers — no allocations, no dtype
    conversion, seed from device memory: CUDA-graph-capturable."""
    assert n_particles <= 32, "viability bitmask supports N <= 32"
    fans = _norm_fanout(fanout, n_particles, gamma)
    _mdsd_accept_kernel[(n_groups,)](
        d_i32, q_f32, p_f32, beta, out_len, out_tok, out_win,
        seed_buf, vocab,
        N=n_particles, GAMMA=gamma, BLOCK=1024,
        FAN0=fans[0], FAN1=fans[1], FAN2=fans[2], FAN3=fans[3],
    )


def _norm_fanout(fanout, n_particles: int, gamma: int):
    """Validate + pad a tree fan-out schedule to the kernel's 4 slots."""
    if not fanout:
        return (1, 1, 1, 1)
    fans = [int(f) for f in fanout]
    assert 0 < len(fans) <= min(4, gamma), (
        f"tree fan-out supports 1..min(4, gamma) depths, got {fans}"
    )
    prod = 1
    for f in fans:
        assert f >= 1, f"fan-out entries must be >= 1, got {fans}"
        prod *= f
    assert n_particles % prod == 0, (
        f"fan-out product {prod} must divide n_particles {n_particles}"
    )
    return tuple(fans + [1] * (4 - len(fans)))


_seed_bufs = {}


def _device_seed_buf(device) -> torch.Tensor:
    buf = _seed_bufs.get(device)
    if buf is None:
        _seed_counter[0] = (_seed_counter[0] + 1) % (2**31 - 1)
        buf = torch.tensor(
            [_seed_counter[0]], dtype=torch.int32, device=device
        )
        _seed_bufs[device] = buf
    return buf


def fused_mdsd_accept(
    draft_tokens: torch.Tensor,   # (G, N, gamma) int
    draft_probs: torch.Tensor,    # (G, N, gamma, V) float
    target_probs: torch.Tensor,   # (G, N, gamma+1, V) float
    *,
    fanout=None,
) -> ExactAcceptResult:
    """One-launch exact accept over G groups (see module docstring).

    Distributionally equivalent to ``mdsd_accept_batched`` (same operator,
    Philox RNG instead of torch's generator).  CUDA only.
    """
    G, N, gamma = draft_tokens.shape
    V = draft_probs.shape[-1]
    device = draft_tokens.device

    d = draft_tokens.to(torch.int32).contiguous()
    q = draft_probs.to(torch.float32).contiguous()
    p = target_probs.to(torch.float32).contiguous()
    beta = torch.empty(G, V, dtype=torch.float32, device=device)
    out_len = torch.empty(G, dtype=torch.int32, device=device)
    out_tok = torch.zeros(G, gamma + 1, dtype=torch.int32, device=device)
    out_win = torch.empty(G, dtype=torch.int32, device=device)
    seed_buf = _device_seed_buf(device)
    seed_buf.add_(1)  # device-side: no H2D on the hot path

    fused_mdsd_accept_into(
        G, N, gamma, V, d, q, p, beta, out_len, out_tok, out_win, seed_buf,
        fanout=fanout,
    )
    return ExactAcceptResult(
        accept_len=out_len.to(torch.int64),
        tokens=out_tok.to(torch.int64),
        winner=out_win.to(torch.int64),
    )

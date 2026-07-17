"""Exact-mode accept operator: multi-draft sequential rejection sampling.

This is the "exact" verification operator of the unified exact/SMC design
(``docs/smc/unified-exact-smc.md``).  Given N particle chains drafted i.i.d.
from a common (post-collapse) prefix — smcsd's native particle format — it
walks the depth-major candidate structure with the multi-draft sequential
rejection-with-residual scheme (SpecTr k-SEQ / SpecInfer multi-round), so the
committed token block is distributed *exactly* according to the target model
``p``, per the standard multi-draft speculative-sampling correctness proof.

Scheme, per group of N chains and depth t = 0..gamma-1::

    viable <- chains whose tokens 0..t-1 equal the committed prefix
    beta   <- p(. | committed prefix)        # target row of any viable chain
    for i in viable (ascending chain order):
        c = d[i, t]                          # chain i's depth-t token
        accept c w.p. min(1, beta[c] / q[c]) # q = common proposal at depth t
        on accept: commit c; viable <- {j : d[j, t] == c}; next depth
        on reject: beta <- normalize(max(beta - q, 0))
    if no candidate accepted:
        commit x ~ beta (final residual); STOP (accept_len = t)
    if all gamma depths accepted:
        commit bonus ~ p(. | full committed chain)

Key invariant this relies on: *viable* chains share the exact conditioning
context at depth t, so their proposal distributions coincide — precisely the
i.i.d.-candidates setting of the multi-draft proof.  Rejected duplicate
candidates auto-reject (a rejection of token c implies ``beta[c] < q[c]``, so
the updated residual has ``beta[c] = 0``).

Two implementations:

* ``mdsd_accept_reference`` — per-group Python loops, unvectorized, written
  for auditability.  Used by the unit / statistical tests as ground truth.
* ``mdsd_accept_batched``  — vectorized over groups (loops only over the
  small N and gamma), used by the worker's eager exact path.  Same RNG
  consumption pattern per group as the reference when fed identical uniform
  draws (tests exploit this for equivalence checks).

Both are pure torch (CPU or CUDA) with no sglang dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "ExactAcceptResult",
    "mdsd_accept_reference",
    "mdsd_accept_batched",
]

# Residual mass below this is treated as numerically-zero (mathematically the
# residual after a rejection always has positive mass; see module docstring).
_EPS = 1e-12


@dataclass
class ExactAcceptResult:
    """Output of one exact-accept application over G groups.

    accept_len[g] in [0, gamma]: number of *drafted* tokens accepted.  The
    committed block is ``tokens[g, : accept_len[g] + 1]`` — the accepted
    prefix plus one extra token (the residual sample on early stop, or the
    bonus draw when all gamma depths accepted).  ``winner[g]`` is a chain
    index whose drafted tokens match the accepted prefix (its KV / history is
    the collapse ancestor).
    """

    accept_len: torch.Tensor  # (G,) int64
    tokens: torch.Tensor      # (G, gamma+1) int64, valid prefix accept_len+1
    winner: torch.Tensor      # (G,) int64 chain index in [0, N)


def _normalized_residual(beta: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """max(beta - q, 0), renormalized; guards numerically-empty residuals."""
    res = torch.clamp(beta - q, min=0)
    total = res.sum(dim=-1, keepdim=True)
    safe = torch.clamp(total, min=_EPS)
    res = res / safe
    # Mathematically total > 0 after any real rejection; if float rounding
    # produced ~0 mass, fall back to the un-updated beta (renormalized).
    fallback = beta / torch.clamp(beta.sum(dim=-1, keepdim=True), min=_EPS)
    return torch.where(total > _EPS, res, fallback)


def mdsd_accept_reference(
    draft_tokens: torch.Tensor,   # (N, gamma) int64
    draft_probs: torch.Tensor,    # (N, gamma, V) — q_t^{(i)}(.)
    target_probs: torch.Tensor,   # (N, gamma+1, V) — p rows from TARGET_VERIFY
    *,
    generator: Optional[torch.Generator] = None,
    uniforms: Optional[torch.Tensor] = None,   # (gamma, N) accept draws (tests)
) -> ExactAcceptResult:
    """Single-group reference implementation (see module docstring).

    ``target_probs[i, t]`` must be ``p(. | prefix, d_0^{(i)} .. d_{t-1}^{(i)})``
    (the verify row that scores chain i's depth-t token); row ``gamma`` is the
    bonus row conditioned on the chain's full gamma tokens.

    ``uniforms`` optionally injects the per-(depth, chain) accept draws so
    tests can force specific accept/reject paths; residual / bonus draws
    always come from ``generator``.
    """
    N, gamma = draft_tokens.shape
    device = draft_tokens.device
    f64 = torch.float64

    viable = list(range(N))
    committed = torch.zeros(gamma + 1, dtype=torch.int64, device=device)

    for t in range(gamma):
        rep = viable[0]
        beta = target_probs[rep, t].to(f64)
        beta = beta / torch.clamp(beta.sum(), min=_EPS)
        accepted = False
        for i in viable:
            c = int(draft_tokens[i, t])
            q = draft_probs[i, t].to(f64)
            q = q / torch.clamp(q.sum(), min=_EPS)
            if uniforms is not None:
                u = float(uniforms[t, i])
            else:
                u = float(torch.rand((), generator=generator, device=device))
            ratio = float(beta[c] / torch.clamp(q[c], min=_EPS))
            if u <= min(1.0, ratio):
                committed[t] = c
                viable = [j for j in viable if int(draft_tokens[j, t]) == c]
                accepted = True
                break
            beta = _normalized_residual(beta, q)
        if not accepted:
            x = int(
                torch.multinomial(beta, num_samples=1, generator=generator)
            )
            committed[t] = x
            return ExactAcceptResult(
                accept_len=torch.tensor(t, dtype=torch.int64, device=device),
                tokens=committed,
                winner=torch.tensor(rep, dtype=torch.int64, device=device),
            )

    rep = viable[0]
    bonus_p = target_probs[rep, gamma].to(f64)
    bonus_p = bonus_p / torch.clamp(bonus_p.sum(), min=_EPS)
    committed[gamma] = int(
        torch.multinomial(bonus_p, num_samples=1, generator=generator)
    )
    return ExactAcceptResult(
        accept_len=torch.tensor(gamma, dtype=torch.int64, device=device),
        tokens=committed,
        winner=torch.tensor(rep, dtype=torch.int64, device=device),
    )


def mdsd_accept_batched(
    draft_tokens: torch.Tensor,   # (G, N, gamma) int64
    draft_probs: torch.Tensor,    # (G, N, gamma, V)
    target_probs: torch.Tensor,   # (G, N, gamma+1, V)
    *,
    generator: Optional[torch.Generator] = None,
    uniforms: Optional[torch.Tensor] = None,   # (G, gamma, N) accept draws
) -> ExactAcceptResult:
    """Vectorized-over-groups exact accept (loops only over gamma and N).

    Semantics identical to running ``mdsd_accept_reference`` on each group
    independently.  When ``uniforms`` is provided (tests), the per-group
    accept-draw consumption matches the reference exactly; the residual and
    bonus draws are one ``torch.multinomial`` per depth / at the end, applied
    only to the groups that need them.
    """
    G, N, gamma = draft_tokens.shape
    V = draft_probs.shape[-1]
    device = draft_tokens.device
    f64 = torch.float64
    g_idx = torch.arange(G, device=device)

    viable = torch.ones(G, N, dtype=torch.bool, device=device)
    done = torch.zeros(G, dtype=torch.bool, device=device)
    accept_len = torch.full((G,), gamma, dtype=torch.int64, device=device)
    committed = torch.zeros(G, gamma + 1, dtype=torch.int64, device=device)
    winner = torch.zeros(G, dtype=torch.int64, device=device)

    def _first_viable(v: torch.Tensor) -> torch.Tensor:
        # v: (G, N) bool with >= 1 True per row (invariant).
        return torch.argmax(v.to(torch.int64), dim=1)

    for t in range(gamma):
        active = ~done
        rep = _first_viable(viable)                       # (G,)
        beta = target_probs[g_idx, rep, t].to(f64)        # (G, V)
        beta = beta / torch.clamp(beta.sum(-1, keepdim=True), min=_EPS)
        accepted_t = torch.zeros(G, dtype=torch.bool, device=device)

        for i in range(N):
            cand = active & viable[:, i] & ~accepted_t    # (G,)
            if not bool(cand.any()):
                continue
            c = draft_tokens[:, i, t]                     # (G,)
            q = draft_probs[:, i, t].to(f64)              # (G, V)
            q = q / torch.clamp(q.sum(-1, keepdim=True), min=_EPS)
            q_c = q.gather(1, c.unsqueeze(1)).squeeze(1)
            beta_c = beta.gather(1, c.unsqueeze(1)).squeeze(1)
            ratio = beta_c / torch.clamp(q_c, min=_EPS)
            if uniforms is not None:
                u = uniforms[:, t, i].to(f64)
            else:
                u = torch.rand(
                    G, generator=generator, device=device, dtype=f64
                )
            acc = cand & (u <= torch.clamp(ratio, max=1.0))
            rej = cand & ~acc

            # Commit + shrink viability on accepting groups.
            committed[:, t] = torch.where(acc, c, committed[:, t])
            accepted_t |= acc
            match = draft_tokens[:, :, t] == c.unsqueeze(1)   # (G, N)
            viable = torch.where(acc.unsqueeze(1), viable & match, viable)

            # Residual update on rejecting groups.
            if bool(rej.any()):
                beta = torch.where(
                    rej.unsqueeze(1), _normalized_residual(beta, q), beta
                )

        # Groups where every viable candidate was rejected: residual sample,
        # stop at accept_len = t.  The winner is the depth-t representative
        # (matches the committed prefix through t-1).
        stop = active & ~accepted_t
        if bool(stop.any()):
            x = torch.multinomial(beta, num_samples=1, generator=generator)
            x = x.squeeze(1)
            committed[:, t] = torch.where(stop, x, committed[:, t])
            accept_len = torch.where(
                stop, torch.full_like(accept_len, t), accept_len
            )
            winner = torch.where(stop, rep, winner)
            done |= stop

    # Groups that accepted all gamma depths: bonus from the winner's row.
    alive = ~done
    if bool(alive.any()):
        rep = _first_viable(viable)
        winner = torch.where(alive, rep, winner)
        bonus_p = target_probs[g_idx, rep, gamma].to(f64)
        bonus_p = bonus_p / torch.clamp(bonus_p.sum(-1, keepdim=True), min=_EPS)
        b = torch.multinomial(bonus_p, num_samples=1, generator=generator)
        committed[:, gamma] = torch.where(
            alive, b.squeeze(1), committed[:, gamma]
        )

    return ExactAcceptResult(
        accept_len=accept_len, tokens=committed, winner=winner
    )

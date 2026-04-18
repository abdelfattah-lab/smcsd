"""Block chi-squared + ESS harness for SMC-native draft evaluation.

Three metrics, all on held-out prefixes with N particles sampled from q_theta:
  * Block chi-squared at K: chi^2(p_K || q_K) = E_q[(p/q)^2] - 1
  * ESS/N per prefix: (sum w)^2 / (N * sum w^2)
  * Mean log importance ratio: E_q[log p - log q] (diagnostic)

Log-space throughout. Track mean, median, and tail quantiles because log_w is
heavy-tailed when q is far from p.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def sample_from_proposal(
    model,
    prefix_ids: list[list[int]],
    k_max: int,
    pad_id: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive sample k_max tokens per prefix, tracking log q at each step.

    Returns:
        gen_tokens: [B, k_max] int64
        log_q:      [B, k_max] float32
    """
    B = len(prefix_ids)
    plens = [len(p) for p in prefix_ids]
    pmax = max(plens)

    input_ids = torch.full((B, pmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, pmax), dtype=torch.long, device=device)
    for i, p in enumerate(prefix_ids):
        input_ids[i, pmax - len(p):] = torch.tensor(p, device=device)
        attn[i, pmax - len(p):] = 1

    gen = torch.zeros((B, k_max), dtype=torch.long, device=device)
    log_q = torch.zeros((B, k_max), dtype=torch.float32, device=device)

    past = None
    cur_ids = input_ids
    cur_mask = attn
    for t in range(k_max):
        out = model(input_ids=cur_ids, attention_mask=cur_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :].float() / temperature
        past = out.past_key_values
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        gen[:, t] = nxt
        log_q[:, t] = log_probs.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        cur_ids = nxt.unsqueeze(-1)
        cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=-1)

    return gen, log_q


@torch.inference_mode()
def score_with_target(
    model,
    prefix_ids: list[list[int]],
    gen_tokens: torch.Tensor,
    pad_id: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward the target once on [prefix; gen], return log p at each gen position.

    Returns:
        log_p: [B, k_max] float32
    """
    B, k_max = gen_tokens.shape
    plens = [len(p) for p in prefix_ids]
    pmax = max(plens)
    L = pmax + k_max

    input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, p in enumerate(prefix_ids):
        start = pmax - len(p)
        input_ids[i, start:pmax] = torch.tensor(p, device=device)
        input_ids[i, pmax:] = gen_tokens[i]
        attn[i, start:] = 1

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    # logit at position t predicts input_ids[t+1]; we want t in [pmax-1 .. pmax+k_max-2]
    pred_logits = out.logits[:, pmax - 1 : pmax - 1 + k_max, :].float() / temperature
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)


def _ess_regime(ess_over_n_mean: float) -> str:
    if ess_over_n_mean < 0.1:
        return "degenerate"
    if ess_over_n_mean < 0.7:
        return "mixing"
    return "converged"


def compute_metrics(
    log_w_per_step: torch.Tensor,   # [M, N, K_max]
    k_values: list[int],
) -> tuple[dict, dict]:
    """Aggregate per-step log weights into block-chi^2 / ESS / diagnostics for each K.

    Returns (summary, per_prefix). The per_prefix dict stores ess_over_n arrays
    per K (1 float per held-out prefix) for downstream prefix-difficulty analysis.
    """
    M, N, K_max = log_w_per_step.shape
    summary = {}
    per_prefix = {}
    for k in k_values:
        log_w = log_w_per_step[:, :, :k].sum(dim=-1)   # [M, N]
        flat = log_w.flatten().float()                  # [M*N]

        log_1p_chi2 = (torch.logsumexp(2.0 * flat, dim=0) - math.log(flat.numel())).item()
        chi2 = math.expm1(log_1p_chi2)

        log_ess = 2.0 * torch.logsumexp(log_w.float(), dim=1) - torch.logsumexp(2.0 * log_w.float(), dim=1)
        ess_over_n = (log_ess - math.log(N)).exp()     # [M]

        q = torch.quantile(flat, torch.tensor([0.05, 0.5, 0.95], device=flat.device))
        ess_mean = ess_over_n.mean().item()
        summary[k] = {
            "log_1p_chi2": log_1p_chi2,
            "chi2": chi2,
            "ess_over_n_mean": ess_mean,
            "ess_over_n_median": ess_over_n.median().item(),
            "ess_regime": _ess_regime(ess_mean),
            "mean_log_w": flat.mean().item(),
            "median_log_w": q[1].item(),
            "log_w_q05": q[0].item(),
            "log_w_q95": q[2].item(),
        }
        per_prefix[k] = {"ess_over_n": ess_over_n.tolist()}
    return summary, per_prefix

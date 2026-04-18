"""Top-K truncated soft cross-entropy (= forward KL up to an entropy constant).

Given target logits truncated to top-K per position, the target's renormalized
top-K distribution is the training signal. We compute the proposal's
log-softmax over the full vocab and gather at the K target indices.
"""
import torch
import torch.nn.functional as F


def top_k_soft_ce(
    proposal_logits: torch.Tensor,   # [B, T, V] any float dtype
    topk_indices: torch.Tensor,       # [B, T, K] int64
    topk_logits: torch.Tensor,        # [B, T, K] float (target's raw logits at those indices)
) -> torch.Tensor:
    p_soft = F.softmax(topk_logits.float(), dim=-1)              # [B, T, K]
    log_q = F.log_softmax(proposal_logits.float(), dim=-1)       # [B, T, V]
    log_q_k = torch.gather(log_q, dim=-1, index=topk_indices)    # [B, T, K]
    return -(p_soft * log_q_k).sum(dim=-1).mean()

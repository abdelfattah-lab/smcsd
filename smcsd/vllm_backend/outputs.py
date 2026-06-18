from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class SMCModelRunnerOutput(ModelRunnerOutput):
    # group_id -> (draft_token_ids[A, gamma+1], draft_log_probs[A, gamma],
    #              next_seed_ids[A])
    # draft_log_probs: scalar log q_draft(t_s) at each sampled draft token (steps 0..gamma-1)
    smc_draft_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    # group_id -> ancestor_indices [N_i] (CPU int64, resampled groups only)
    resampled_groups: dict[str, torch.Tensor] = field(default_factory=dict)
    # group_id -> block ids [N_i][num_blocks] after full-block remap.
    resampled_block_ids: dict[str, list[list[int]]] = field(default_factory=dict)
    # group_id -> logprob_diff[A]: sum_t log(p_target(x_t)/p_draft(x_t)) per active particle
    smc_logprob_diffs: dict[str, torch.Tensor] = field(default_factory=dict)

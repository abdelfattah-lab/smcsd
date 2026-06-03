from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class SMCModelRunnerOutput(ModelRunnerOutput):
    # group_id -> (draft_token_ids[N, gamma+1], draft_log_probs[N, gamma, V],
    #              next_seed_ids[N])
    smc_draft_results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    # group_id -> ancestor_indices [N_i] (CPU int64)
    # Present only for groups that were resampled this step.
    resampled_groups: dict[str, torch.Tensor] = field(default_factory=dict)

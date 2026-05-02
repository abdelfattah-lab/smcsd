from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


class SMCSpeculator:
    """Speculator stub for the SMC method.

    Registered via init_speculator() when speculative_config.use_smc() is True.

    Responsibilities:
      - load_model(): load the draft model weights (TODO: full implementation).
      - propose():    no-op — SMC draft is driven by SMCGPUModelRunner.smc_draft_cycle,
                      not by the base runner's propose machinery.  Returns a zero
                      tensor of the shape the base runner expects so it doesn't crash.
    """

    supports_mm_inputs: bool = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        spec = vllm_config.speculative_config
        assert spec is not None
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.draft_model: nn.Module | None = None

    def load_model(self, target_model: nn.Module) -> None:
        """Load the draft model.

        TODO: load the draft model from speculative_config.model using vllm's
        model loading infrastructure.  For now, smc_draft_cycle falls back to
        the target model when self.draft_model is None.
        """
        pass

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """No-op: returns zeros so the base runner's post-propose bookkeeping
        doesn't crash.  SMC draft tokens are produced by smc_draft_cycle."""
        return torch.zeros(
            input_batch.num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=self.device,
        )

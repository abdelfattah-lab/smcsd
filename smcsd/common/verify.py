"""Shared SMC verify primitives.

Used by the SMC scheduler / worker and by a couple of core call sites in
``cuda_graph_runner`` / ``model_runner`` that construct an idle or CUDA-
graph-captured verify ``spec_info``.  Kept in ``smc/common`` so those
core call sites don't have to import the scheduler package.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


@torch.library.custom_op("smcsd::assign_smc_cache_locs", mutates_args=())
def assign_smc_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    max_context_len: int,
    draft_token_num: int,
) -> None:
    """Vectorized assignment of cache locations for SMC draft tokens."""
    bs = req_pool_indices.shape[0]
    for i in range(bs):
        req_idx = req_pool_indices[i]
        start_pos = seq_lens[i]
        for j in range(draft_token_num):
            out_cache_loc[i * draft_token_num + j] = req_to_token[
                req_idx, start_pos + j
            ]


@assign_smc_cache_locs.register_fake
def _(
    req_pool_indices, req_to_token, seq_lens, out_cache_loc, max_context_len, draft_token_num
):
    return


@dataclasses.dataclass
class SMCVerifyInput(SpecInput):
    """Metadata for the score model verification pass."""

    draft_token_num: int = -1
    positions: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    seq_lens_sum: int = None
    seq_lens_cpu: torch.Tensor = None
    num_tokens_per_req: int = -1
    out_cache_loc: torch.Tensor = None

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (self.draft_token_num, self.draft_token_num)

    def populate_linear_verify_metadata(self, forward_batch: ForwardBatch):
        """Populate ForwardBatch fields for the score forward pass."""
        device = forward_batch.input_ids.device
        batch_size = forward_batch.batch_size // self.draft_token_num

        forward_batch.seq_lens_sum = self.seq_lens_sum
        forward_batch.extend_num_tokens = forward_batch.input_ids.shape[0]
        forward_batch.extend_start_loc = torch.arange(
            0, forward_batch.extend_num_tokens, step=self.draft_token_num,
            dtype=torch.int32, device=device,
        )
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        forward_batch.extend_prefix_lens_cpu = seq_lens_cpu
        forward_batch.extend_seq_lens_cpu = torch.full(
            (batch_size,), self.draft_token_num, dtype=torch.int32,
        )
        if self.out_cache_loc is not None:
            forward_batch.out_cache_loc = self.out_cache_loc

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "SMCVerifyInput":
        return cls(
            draft_token_num=1,
            positions=torch.empty((0,), dtype=torch.int32, device=device),
        )

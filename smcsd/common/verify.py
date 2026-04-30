"""Shared SMC verify primitives.

Used by the SMC scheduler / worker and by a couple of core call sites in
``cuda_graph_runner`` / ``model_runner`` that construct an idle or CUDA-
graph-captured verify ``spec_info``.  Kept in ``smc/common`` so those
core call sites don't have to import the scheduler package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@triton.jit
def assign_smc_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    num_tokens: tl.constexpr,
):
    """Assign cache locations for SMC decode: ``num_tokens`` slots per request."""
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    out_ptr = out_cache_loc + pid * num_tokens
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    num_loop = tl.cdiv(num_tokens, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < num_tokens
        data = tl.load(token_pool + kv_start + offset, mask=mask)
        tl.store(out_ptr + offset, data, mask=mask)


@dataclass
class SMCVerifyInput(SpecInput):
    """Spec info for SMC verify (TARGET_VERIFY mode with CUDA graph support).

    Uses linear (EXTEND-style) causal attention — no custom_mask needed.
    The triton backend recognizes use_linear_target_verify() and uses
    standard prefix_lens-based causal masking instead of custom_mask.
    """

    custom_mask: torch.Tensor = None  # Always None for SMC linear verify
    draft_token_num: int = -1
    positions: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    seq_lens_sum: int = None
    seq_lens_cpu: torch.Tensor = None
    num_tokens_per_req: int = -1

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_VERIFY)

    def get_spec_adjust_token_coefficient(self):
        return (self.draft_token_num, self.draft_token_num)

    def use_linear_target_verify(self) -> bool:
        """Signal triton backend to use EXTEND-style causal attention."""
        return True

    def populate_linear_verify_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Set EXTEND-style fields on ForwardBatch for linear causal attention."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        prefix_lens = forward_batch.seq_lens.to(dtype=torch.int32)
        extend_seq_lens = torch.full(
            (batch_size,), self.draft_token_num, dtype=torch.int32, device=device,
        )
        forward_batch.extend_prefix_lens = prefix_lens
        forward_batch.extend_seq_lens = extend_seq_lens
        forward_batch.extend_num_tokens = batch_size * self.draft_token_num
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

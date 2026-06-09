"""SMC spec info: clean separation of concerns.

- SMCDecodeContext: per-cycle state created by scheduler, consumed by worker.
  Owns prepare_for_draft / prepare_for_verify.
  Factory method from_slot_gather does vectorized KV allocation.

- SMCDraftInput: pure data carrier on batch.spec_info (no prepare methods).

- SMCVerifyInput: reused from smc_info.py (unchanged).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from smcsd.common.verify import SMCVerifyInput, assign_smc_cache_locs_kernel
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner


# ──────────────────────────────────────────────────────────────
#  SMCDecodeContext — bridge between scheduler and worker
# ──────────────────────────────────────────────────────────────


@dataclass
class SMCDecodeContext:
    """Per-decode-cycle state computed during prepare_for_decode (scheduler side),
    consumed by prepare_for_draft / prepare_for_verify (worker side).
    """

    orig_seq_lens: torch.Tensor  # (bs,) committed prefix BEFORE advance
    orig_seq_lens_cpu: torch.Tensor  # CPU copy
    orig_seq_lens_sum: int  # scalar sum
    new_seq_lens: torch.Tensor  # (bs,) AFTER advance by gamma+1
    gamma: int  # speculative steps (gamma, NOT gamma+1)

    def filter_batch(self, new_indices: torch.Tensor):
        return SMCDecodeContext(
            orig_seq_lens=self.orig_seq_lens[new_indices],
            orig_seq_lens_cpu=self.orig_seq_lens_cpu[new_indices.cpu()],
            orig_seq_lens_sum=int(self.orig_seq_lens[new_indices].sum()),
            new_seq_lens=self.new_seq_lens[new_indices],
            gamma=self.gamma,
        )

    def merge_batch(self, other: "SMCDecodeContext"):
        return SMCDecodeContext(
            orig_seq_lens=torch.cat([self.orig_seq_lens, other.orig_seq_lens]),
            orig_seq_lens_cpu=torch.cat([self.orig_seq_lens_cpu, other.orig_seq_lens_cpu]),
            orig_seq_lens_sum=self.orig_seq_lens_sum + other.orig_seq_lens_sum,
            new_seq_lens=torch.cat([self.new_seq_lens, other.new_seq_lens]),
            gamma=self.gamma,
        )

    @staticmethod
    def from_slot_gather(
        seq_lens: torch.Tensor,
        kv_allocated_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        gamma_plus_1: int,
        req_to_token_pool: ReqToTokenPool,
        tree_cache,
    ) -> Tuple["SMCDecodeContext", torch.Tensor]:
        """Vectorized KV allocation — replaces the Python loop in
        SMCDraftInput.prepare_for_decode (smc_info.py L203-217).

        Returns:
            (ctx, new_kv_allocated_lens) where new_kv_allocated_lens should be
            scattered back to the slot state.
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(seq_lens)
        seq_lens_cpu = seq_lens.cpu()
        orig_seq_lens = seq_lens.clone()
        orig_seq_lens_sum = int(seq_lens_cpu.sum().item())

        # Vectorized allocation
        alloc_start = torch.maximum(kv_allocated_lens, seq_lens)
        needed_len = seq_lens + gamma_plus_1
        new_alloc = torch.clamp(needed_len - alloc_start, min=0)
        num_needed = int(new_alloc.sum().item())

        nxt_kv_lens = alloc_start + new_alloc

        out_cache_loc = alloc_token_slots(tree_cache, num_needed)
        assign_req_to_token_pool_func(
            req_pool_indices,
            req_to_token_pool.req_to_token,
            alloc_start.to(torch.int32),
            nxt_kv_lens.to(torch.int32),
            out_cache_loc,
            bs,
        )

        new_seq_lens = seq_lens + gamma_plus_1

        ctx = SMCDecodeContext(
            orig_seq_lens=orig_seq_lens,
            orig_seq_lens_cpu=seq_lens_cpu,
            orig_seq_lens_sum=orig_seq_lens_sum,
            new_seq_lens=new_seq_lens,
            gamma=gamma_plus_1 - 1,
        )
        return ctx, nxt_kv_lens

    # ── Worker-side methods (called in SMCWorker._forward_decode) ──

    def prepare_for_draft(
        self,
        verified_id: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner,
        draft_model_runner: "ModelRunner",
    ) -> Tuple[ForwardBatch, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch and create ForwardBatch for draft AR decoding."""
        orig_seq_lens = self.orig_seq_lens
        bs = len(orig_seq_lens)
        device = orig_seq_lens.device
        gamma = self.gamma

        # Assign cache locations for gamma+1 new tokens
        out_cache_loc = torch.empty(
            bs * (gamma + 1), dtype=torch.int64, device=device
        )
        assign_smc_cache_locs_kernel[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            orig_seq_lens,
            out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            gamma + 1,
        )
        cache_locs = out_cache_loc.reshape(bs, gamma + 1)

        # Pre-compute all positions and seq_lens on GPU
        step_offsets = torch.arange(gamma + 1, device=device)
        all_positions = orig_seq_lens.unsqueeze(1) + step_offsets
        all_seq_lens = all_positions + 1

        draft_batch = copy.copy(batch)
        draft_batch.input_ids = verified_id
        draft_batch.out_cache_loc = cache_locs[:, 0].contiguous()
        draft_batch.seq_lens = all_seq_lens[:, 0].contiguous()
        draft_batch.seq_lens_sum = self.orig_seq_lens_sum + bs
        draft_batch.seq_lens_cpu = self.orig_seq_lens_cpu + 1
        draft_batch.capture_hidden_mode = CaptureHiddenMode.NULL

        draft_batch.spec_info = None
        forward_batch = ForwardBatch.init_new(draft_batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)

        return forward_batch, can_cuda_graph, cache_locs, all_positions, all_seq_lens

    def prepare_for_verify(
        self,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
        all_tokens: list,
        cache_locs: torch.Tensor,
    ) -> Tuple[ForwardBatch, bool]:
        """Prepare batch and create ForwardBatch for score model verification."""
        gamma = self.gamma
        bs = len(batch.req_pool_indices)
        device = batch.seq_lens.device
        draft_token_num = gamma + 1

        score_token_ids = torch.stack(all_tokens[: gamma + 1], dim=1)
        score_input_ids = score_token_ids.reshape(-1)

        orig_seq_lens = self.orig_seq_lens
        orig_seq_lens_cpu = self.orig_seq_lens_cpu

        step_offsets = torch.arange(draft_token_num, device=device)
        positions = (orig_seq_lens.unsqueeze(1) + step_offsets).reshape(-1)

        verify_spec_info = SMCVerifyInput(
            draft_token_num=draft_token_num,
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=self.orig_seq_lens_sum,
            seq_lens_cpu=orig_seq_lens_cpu,
            num_tokens_per_req=draft_token_num,
            out_cache_loc=cache_locs.reshape(-1),
        )

        verify_batch = copy.copy(batch)
        verify_batch.input_ids = score_input_ids
        verify_batch.out_cache_loc = cache_locs.reshape(-1)
        verify_batch.seq_lens = orig_seq_lens
        verify_batch.seq_lens_cpu = orig_seq_lens_cpu
        verify_batch.seq_lens_sum = verify_spec_info.seq_lens_sum
        verify_batch.spec_info = verify_spec_info
        verify_batch.capture_hidden_mode = CaptureHiddenMode.NULL

        is_idle = verify_batch.forward_mode.is_idle()
        verify_batch.forward_mode = (
            ForwardMode.IDLE if is_idle else ForwardMode.TARGET_VERIFY
        )

        graph_runner = target_worker.model_runner.graph_runner
        verify_forward_batch = ForwardBatch.init_new(
            verify_batch, target_worker.model_runner
        )

        can_run_cuda_graph = bool(
            graph_runner and graph_runner.can_run(verify_forward_batch)
        )

        if not is_idle:
            verify_spec_info.populate_linear_verify_metadata(verify_forward_batch)

        if can_run_cuda_graph:
            graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not is_idle:
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph


# ──────────────────────────────────────────────────────────────
#  SMCDraftInput — pure data carrier
# ──────────────────────────────────────────────────────────────


@dataclass
class SMCDraftInput(SpecInput):
    """Lightweight carrier between scheduler and worker via batch.spec_info."""

    verified_id: Optional[torch.Tensor] = None  # (bs,) last accepted token
    logprob_diff: Optional[torch.Tensor] = None  # (bs,) from last step
    num_tokens_per_req: int = -1  # gamma + 1
    decode_ctx: Optional[SMCDecodeContext] = None  # attached by prepare_for_decode

    ALLOC_LEN_PER_DECODE: ClassVar[int] = 1

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.verified_id is not None:
            self.verified_id = self.verified_id[new_indices]
        if self.logprob_diff is not None:
            self.logprob_diff = self.logprob_diff[new_indices]
        if self.decode_ctx is not None:
            self.decode_ctx = self.decode_ctx.filter_batch(new_indices)

    def merge_batch(self, other: "SMCDraftInput"):
        if self.verified_id is not None and other.verified_id is not None:
            self.verified_id = torch.cat([self.verified_id, other.verified_id])
        if self.logprob_diff is not None and other.logprob_diff is not None:
            self.logprob_diff = torch.cat([self.logprob_diff, other.logprob_diff])
        if self.decode_ctx is not None and other.decode_ctx is not None:
            self.decode_ctx = self.decode_ctx.merge_batch(other.decode_ctx)

    @property
    def new_seq_lens(self) -> Optional[torch.Tensor]:
        return self.decode_ctx.new_seq_lens if self.decode_ctx else None

    def prepare_for_decode(self, batch: ModelWorkerBatch):
        """Called by scheduler to stash per-cycle state."""
        if self.decode_ctx is not None:
            return

        bs = len(batch.seq_lens)
        orig_seq_lens = batch.seq_lens.clone()
        orig_seq_lens_cpu = batch.seq_lens_cpu.clone()
        orig_seq_lens_sum = int(orig_seq_lens.sum())
        new_seq_lens = orig_seq_lens + self.num_tokens_per_req
        
        self.decode_ctx = SMCDecodeContext(
            orig_seq_lens=orig_seq_lens,
            orig_seq_lens_cpu=orig_seq_lens_cpu,
            orig_seq_lens_sum=orig_seq_lens_sum,
            new_seq_lens=new_seq_lens,
            gamma=self.num_tokens_per_req - 1
        )

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (self.num_tokens_per_req, self.num_tokens_per_req)

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "SMCDraftInput":
        return cls(
            verified_id=torch.empty((0,), dtype=torch.int32, device=device),
            num_tokens_per_req=1,
        )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureIndices
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner


@triton.jit
def assign_smc_cache_locs_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    num_tokens: tl.constexpr,
):
    """Assign cache locations for SMC decode: num_tokens slots per request."""
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

    def populate_linear_verify_metadata(self, forward_batch: ForwardBatch) -> None:
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


@dataclass
class SMCDraftInput(SpecInput):
    """Carries state between SMC decode steps."""

    # The last accepted token id per request — starting point for next draft
    # shape: (bs,)
    verified_id: Optional[torch.Tensor] = None

    # Updated sequence lengths after accepting gamma+1 tokens
    # shape: (bs,)
    new_seq_lens: Optional[torch.Tensor] = None

    # Not needed: new_seq_lens is a schedule-stream tensor (stashed in
    # prepare_for_decode), so no cross-stream sync required.
    verify_done: None = None

    # Logprob diff per request from the last step — shape: (bs,)
    logprob_diff: Optional[torch.Tensor] = None

    # Number of tokens per request in the decode batch
    num_tokens_per_req: int = -1

    # For overlap scheduling
    future_indices: Optional[FutureIndices] = None

    # Class-level constant set during worker init
    ALLOC_LEN_PER_DECODE: ClassVar[int] = 1

    def __post_init__(self):
        super().__init__(SpecInputType.SMC_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (self.num_tokens_per_req, self.num_tokens_per_req)

    @classmethod
    def create_idle_input(cls, device: torch.device) -> SMCDraftInput:
        return cls(
            verified_id=torch.empty((0,), dtype=torch.int32, device=device),
            new_seq_lens=torch.empty((0,), dtype=torch.int32, device=device),
            num_tokens_per_req=1,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        """Called when finished requests are removed from the batch."""
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return
        # Non-overlap path: filter direct tensors
        if self.verified_id is not None and self.verified_id.numel() > 0:
            self.verified_id = self.verified_id[new_indices]
        if self.new_seq_lens is not None and self.new_seq_lens.numel() > 0:
            self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, other: SMCDraftInput):
        """Called when newly prefilled requests join the running decode batch."""
        if self.future_indices is not None:
            from sglang.srt.managers.overlap_utils import FutureIndices

            assert other.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, other.future_indices.indices]
                )
            )
            return
        # Non-overlap path: concat direct tensors
        if other.verified_id is not None:
            if self.verified_id is None or self.verified_id.numel() == 0:
                self.verified_id = other.verified_id
                self.new_seq_lens = other.new_seq_lens
            else:
                self.verified_id = torch.cat([self.verified_id, other.verified_id])
                self.new_seq_lens = torch.cat([self.new_seq_lens, other.new_seq_lens])

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate KV cache slots for the next gamma+1 tokens per request."""
        batch.maybe_evict_swa()
        # verify_done not strictly needed: filter_batch/merge_batch take the
        # future_indices early-return path and never touch forward-stream tensors.
        # batch.maybe_wait_verify_done()

        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

        # Save original seq_lens (committed prefix) for prepare_for_draft/verify.
        # Must be saved BEFORE advancing seq_lens below.
        self._orig_seq_lens = batch.seq_lens.clone()
        self._orig_seq_lens_sum = batch.seq_lens_sum  # int, no sync needed
        self._orig_seq_lens_cpu = batch.seq_lens_cpu   # already on CPU

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()
        gamma_plus_1 = self.num_tokens_per_req  # gamma + 1

        page_size = batch.token_to_kv_pool_allocator.page_size
        orig_seq_lens_cpu = batch.seq_lens_cpu  # pre-advance committed prefix
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0

        for i, r in enumerate(batch.reqs):
            # Allocate exactly gamma+1 new slots for this cycle's draft.
            # Mirror normal decode (schedule_batch.py:2050-2056): allocate,
            # then update kv_committed_len and kv_allocated_len.
            seq_len_i = int(orig_seq_lens_cpu[i].item())
            needed_len = seq_len_i + gamma_plus_1
            alloc_start = max(r.kv_allocated_len, seq_len_i)
            x = needed_len - alloc_start
            if x < 0:
                x = 0
            cur_kv_lens_cpu.append(alloc_start)
            nxt_kv_lens_cpu.append(alloc_start + x)
            num_needed_tokens += x
            r.kv_allocated_len = alloc_start + x
            r.decode_batch_idx += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        assert page_size == 1, "SMC currently only supports page_size=1"
        out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # Advance seq_lens deterministically on the schedule stream.
        # SMC always accepts gamma+1 tokens — no need to wait for the forward.
        # (Same pattern as schedule_batch.py:2059-2063 for non-spec decode)
        if batch.enable_overlap:
            batch.seq_lens = batch.seq_lens + gamma_plus_1
            batch.seq_lens_cpu = batch.seq_lens_cpu + gamma_plus_1
        else:
            batch.seq_lens.add_(gamma_plus_1)
            batch.seq_lens_cpu.add_(gamma_plus_1)
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

        # Stash the advanced seq_lens (schedule-stream tensor) so _forward_decode
        # can return it as new_seq_lens without creating a forward-stream tensor.
        # This eliminates the cross-stream dependency that required verify_done.
        self._new_seq_lens = batch.seq_lens

    def prepare_for_draft(
        self,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner,
        draft_model_runner: ModelRunner,
        gamma: int,
    ):
        """Prepare batch and create ForwardBatch for draft AR decoding.

        Returns (forward_batch, can_cuda_graph, cache_locs, all_positions, all_seq_lens).
        The caller updates forward_batch fields in-place per AR step.
        """
        # Use orig_seq_lens (committed prefix, saved in prepare_for_decode)
        # NOT batch.seq_lens which has already been advanced by gamma+1.
        orig_seq_lens = self._orig_seq_lens
        bs = len(orig_seq_lens)
        device = orig_seq_lens.device

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

        # Pre-compute all positions and seq_lens on GPU — no CPU sync
        step_offsets = torch.arange(gamma + 1, device=device)
        all_positions = orig_seq_lens.unsqueeze(1) + step_offsets  # (bs, gamma+1)
        all_seq_lens = all_positions + 1  # (bs, gamma+1)

        # Set batch fields for first step.
        # Use a shallow copy to avoid mutating the scheduler's batch.seq_lens,
        # which is needed for overlap scheduling's filter_batch/merge_batch.
        import copy
        draft_batch = copy.copy(batch)
        draft_batch.input_ids = self.verified_id
        draft_batch.out_cache_loc = cache_locs[:, 0].contiguous()
        draft_batch.seq_lens = all_seq_lens[:, 0].contiguous()
        # Use pre-advance CPU values + 1 for step 0 (no GPU→CPU sync)
        draft_batch.seq_lens_sum = self._orig_seq_lens_sum + bs
        draft_batch.seq_lens_cpu = self._orig_seq_lens_cpu + 1
        draft_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        # Set positions via spec_info so ForwardBatch.init_new picks them up
        self.positions = all_positions[:, 0].contiguous()

        # Clear spec_info for ForwardBatch creation and CUDA graph compatibility.
        # The multi-step path re-sets it before calling init_forward_metadata.
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
        gamma: int,
    ):
        """Prepare batch and create ForwardBatch for score model verification.

        Returns (forward_batch, can_run_cuda_graph).
        """
        bs = len(batch.req_pool_indices)
        device = batch.seq_lens.device
        draft_token_num = gamma + 1

        # Build score input: [x0, ..., x(gamma)]
        score_token_ids = torch.stack(all_tokens[: gamma + 1], dim=1)  # (bs, gamma+1)
        score_input_ids = score_token_ids.reshape(-1)

        # seq_lens = PREFIX ONLY (KV already in target model cache).
        # Draft tokens are handled via EXTEND-style causal self-attention.
        orig_seq_lens = self._orig_seq_lens
        orig_seq_lens_cpu = self._orig_seq_lens_cpu  # cached from prepare_for_decode

        # Positions: [seq_len, seq_len+1, ..., seq_len+gamma] per request
        step_offsets = torch.arange(draft_token_num, device=device)
        positions = (orig_seq_lens.unsqueeze(1) + step_offsets).reshape(-1)

        # Create SMCVerifyInput — no custom_mask (linear verify uses causal attention)
        verify_spec_info = SMCVerifyInput(
            draft_token_num=draft_token_num,
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=self._orig_seq_lens_sum,  # cached from prepare_for_decode
            seq_lens_cpu=orig_seq_lens_cpu,
            num_tokens_per_req=draft_token_num,
        )

        import copy

        # Create a shallow copy so we don't corrupt the scheduler's batch state.
        # The scheduler relies on batch.seq_lens reflecting the committed KV length.
        verify_batch = copy.copy(batch)
        verify_batch.input_ids = score_input_ids
        verify_batch.out_cache_loc = cache_locs.reshape(-1)
        verify_batch.seq_lens = orig_seq_lens
        verify_batch.seq_lens_cpu = orig_seq_lens_cpu
        verify_batch.seq_lens_sum = verify_spec_info.seq_lens_sum
        verify_batch.spec_info = verify_spec_info
        verify_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch = verify_batch

        # Use TARGET_VERIFY for CUDA graph path, EXTEND for non-graph path
        is_idle = batch.forward_mode.is_idle()
        batch.forward_mode = (
            ForwardMode.IDLE
            if is_idle
            else ForwardMode.TARGET_VERIFY
        )

        # Check CUDA graph eligibility before creating ForwardBatch
        graph_runner = target_worker.model_runner.graph_runner
        # Create a temporary ForwardBatch to test graph eligibility
        verify_forward_batch = ForwardBatch.init_new(
            batch, target_worker.model_runner
        )

        can_run_cuda_graph = bool(
            graph_runner
            and graph_runner.can_run(verify_forward_batch)
        )

        # Both graph and non-graph paths use TARGET_VERIFY with linear verify
        # metadata so the model returns logits for ALL tokens, not just the last.
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

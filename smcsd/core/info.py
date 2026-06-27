"""SMC spec info: clean separation of concerns.

- SMCDecodeContext: per-cycle state created by scheduler, consumed by worker.
  Owns prepare_for_draft / prepare_for_verify.
  Factory method from_slot_gather does vectorized KV allocation.

- SMCDraftInput: pure data carrier on batch.spec_info (no prepare methods).

- SMCVerifyInput: reused from smc_info.py (unchanged).
"""

from __future__ import annotations

import copy
import os
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



@dataclass
class SMCParticleOutput:
    """Side-channel message: the full particle collection for one finalized SMC
    group, sent scheduler -> engine alongside the normal token output.

    Plain-Python fields only, so it pickles cleanly over ZMQ.  ``log_Z_hat`` is
    the unbiased log normalizing-constant estimate for the group;
    ``log_w_tilde`` are the final per-particle log-weights; ``particle_output_ids``
    holds every particle's generated token ids.
    """

    rid: str
    log_Z_hat: float
    log_w_tilde: List[float]
    particle_output_ids: List[List[int]]


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
    # (bs, gamma+1) cache locations of this cycle's freshly-allocated pages,
    # carried from the fused prepare kernel (the allocation IS the per-row
    # cache-locs table under the kv_allocated == seq invariant).  None only
    # for legacy callers of from_slot_gather, which re-read the block table.
    cache_locs: Optional[torch.Tensor] = None

    @staticmethod
    def from_slot_gather(
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        kv_allocated_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        gamma_plus_1: int,
        req_to_token_pool: ReqToTokenPool,
        tree_cache,
    ) -> Tuple["SMCDecodeContext", torch.Tensor]:
        """Vectorized KV allocation, sync-free.

        Args:
            seq_lens: (bs,) int64, gathered contiguously from slot state.
            seq_lens_cpu: (bs,) int64 CPU host-shadow gather of the same
                values (see ``ScheduleBatchSMC.seq_lens_host``).  Trusted
                without a device read so this whole prepare can be enqueued
                before earlier GPU work has produced ``seq_lens``.
            kv_allocated_lens: (bs,) int64, gathered contiguously from slot state.
            req_pool_indices: (bs,) int64, gathered contiguously from slot state.
            gamma_plus_1: number of tokens per request (gamma + 1).
            req_to_token_pool: shared KV pool.
            tree_cache: for alloc_token_slots.

        Returns:
            (ctx, new_kv_allocated_lens) where new_kv_allocated_lens should be
            scattered back to the slot state.
        """
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = len(seq_lens)
        orig_seq_lens = seq_lens.clone()
        orig_seq_lens_sum = int(seq_lens_cpu.sum().item())  # CPU tensor — no sync

        # Vectorized allocation (replaces per-req Python loop).  The device
        # math handles the general alloc_start >= seq case for the block
        # table writes; the host alloc COUNT relies on the maintained
        # invariant kv_allocated_lens == seq_lens at prepare time (set at
        # allocate, both advanced to seq + gamma+1 here, both copied
        # together on resample), so every row needs exactly gamma+1 pages.
        alloc_start = torch.maximum(kv_allocated_lens, seq_lens)
        needed_len = seq_lens + gamma_plus_1
        new_alloc = torch.clamp(needed_len - alloc_start, min=0)
        num_needed = bs * gamma_plus_1

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
        """Prepare batch and create ForwardBatch for draft AR decoding.

        Returns (forward_batch, can_cuda_graph, cache_locs, all_positions, all_seq_lens).
        The caller updates forward_batch fields in-place per AR step.
        """
        orig_seq_lens = self.orig_seq_lens
        bs = len(orig_seq_lens)
        device = orig_seq_lens.device
        gamma = self.gamma

        # Cache locations for the gamma+1 new tokens: carried from the fused
        # prepare kernel when available; legacy fallback re-reads the block
        # table.
        if self.cache_locs is not None:
            cache_locs = self.cache_locs
        else:
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

        # Shallow copy to avoid mutating scheduler's batch state
        draft_batch = copy.copy(batch)
        draft_batch.input_ids = verified_id
        draft_batch.out_cache_loc = cache_locs[:, 0].contiguous()
        draft_batch.seq_lens = all_seq_lens[:, 0].contiguous()
        draft_batch.seq_lens_sum = self.orig_seq_lens_sum + bs
        draft_batch.seq_lens_cpu = self.orig_seq_lens_cpu + 1
        draft_batch.capture_hidden_mode = CaptureHiddenMode.NULL

        # Clear spec_info for ForwardBatch creation and CUDA graph compatibility.
        # Positions are derived from seq_lens via clamp_position() in init_new.
        draft_batch.spec_info = None
        forward_batch = ForwardBatch.init_new(draft_batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)

        return forward_batch, can_cuda_graph, cache_locs, all_positions, all_seq_lens

    def prepare_for_verify(
        self,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: "TpModelWorker",
        all_tokens: list,
        cache_locs: torch.Tensor,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
    ) -> Tuple[ForwardBatch, bool]:
        """Prepare batch and create ForwardBatch for score model verification.

        Returns (forward_batch, can_run_cuda_graph).
        """
        gamma = self.gamma
        bs = len(batch.req_pool_indices)
        device = batch.seq_lens.device
        draft_token_num = gamma + 1

        # Build score input: [x0, ..., x(gamma)]
        score_token_ids = torch.stack(all_tokens[: gamma + 1], dim=1)  # (bs, gamma+1)
        score_input_ids = score_token_ids.reshape(-1)

        orig_seq_lens = self.orig_seq_lens
        orig_seq_lens_cpu = self.orig_seq_lens_cpu

        # Positions: [seq_len, seq_len+1, ..., seq_len+gamma] per request
        step_offsets = torch.arange(draft_token_num, device=device)
        positions = (orig_seq_lens.unsqueeze(1) + step_offsets).reshape(-1)

        verify_spec_info = SMCVerifyInput(
            draft_token_num=draft_token_num,
            positions=positions,
            capture_hidden_mode=capture_hidden_mode,
            seq_lens_sum=self.orig_seq_lens_sum,
            seq_lens_cpu=orig_seq_lens_cpu,
            num_tokens_per_req=draft_token_num,
        )

        verify_batch = copy.copy(batch)
        verify_batch.input_ids = score_input_ids
        verify_batch.out_cache_loc = cache_locs.reshape(-1)
        verify_batch.seq_lens = orig_seq_lens
        verify_batch.seq_lens_cpu = orig_seq_lens_cpu
        verify_batch.seq_lens_sum = verify_spec_info.seq_lens_sum
        verify_batch.spec_info = verify_spec_info
        verify_batch.capture_hidden_mode = capture_hidden_mode
        batch = verify_batch

        is_idle = batch.forward_mode.is_idle()
        batch.forward_mode = (
            ForwardMode.IDLE if is_idle else ForwardMode.TARGET_VERIFY
        )

        graph_runner = target_worker.model_runner.graph_runner
        verify_forward_batch = ForwardBatch.init_new(
            batch, target_worker.model_runner
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

    def prepare_for_draft_head(
        self,
        prev_last_draft_id: torch.Tensor,
        verified_id: torch.Tensor,
        cache_locs: torch.Tensor,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        draft_model_runner: "ModelRunner",
    ) -> ForwardBatch:
        """Build the deferred-bonus 2-token draft head (eager, no CUDA graph).

        Mirrors ``prepare_for_verify`` but on the DRAFT runner with
        ``draft_token_num=2``: a linear (EXTEND-style) causal pass over
        ``[prev_last_draft_id, verified_id]`` at positions ``[S-1, S]`` with
        prefix length ``S-1``.  The draft KV is valid through ``S-2``; this head
        writes positions ``S-1`` and ``S``.  The ``S-1`` slot is the one the
        previous step deferred (its over-draft) — it may hold stale bytes (its
        own unwritten content, duplicated by the resample KV-copy), which this
        head overwrites before any read: within this forward the ``S`` token
        attends to the ``S-1`` token's freshly-computed KV, and later decodes
        run only after the head completes.  The slot is looked up live from
        ``req_to_token`` so it is correct after resampling.

        The last-position (``S`` / bonus column) logits give the proposal for
        ``d_0``.  Used on EVERY decode step: on a group's first step,
        ``prev_last_draft_id`` is the last committed prompt token (seeded at
        ``allocate_slots``), so the ``S-1`` write rewrites the prefill's
        draft KV byte-identically — which is what lets mixed-step batches
        (continuous batching) share one head with no per-row branching.
        """
        device = batch.seq_lens.device
        head_num = 2

        orig_seq_lens = self.orig_seq_lens
        # Prefix the head attends to: positions 0..S-2 (length S-1).
        prefix_lens = orig_seq_lens - 1

        # Tokens [prev, verified] per req, interleaved (req-major).
        head_token_ids = torch.stack(
            [prev_last_draft_id.to(torch.int64), verified_id.to(torch.int64)],
            dim=1,
        ).reshape(-1)

        # Positions [S-1, S] per req.
        step_offsets = torch.arange(head_num, device=device)
        positions = (prefix_lens.unsqueeze(1) + step_offsets).reshape(-1)

        # out_cache_loc [slot(S-1), slot(S)] per req.  slot(S-1) looked up live
        # from req_to_token (post-resample correct); slot(S) is this step's
        # first freshly-allocated slot (cache_locs[:, 0]).
        rp = batch.req_pool_indices
        r2t = req_to_token_pool.req_to_token
        slot_sm1 = r2t[rp, (orig_seq_lens - 1).to(torch.int64)]
        slot_s = cache_locs[:, 0]
        out_cache_loc = torch.stack(
            [slot_sm1.to(cache_locs.dtype), slot_s], dim=1
        ).reshape(-1)

        # Derived from the host-shadow CPU copy — no device read (this runs
        # inside the worker's draft launch, on the sync-free hot path).
        prefix_lens_cpu = self.orig_seq_lens_cpu - 1
        head_spec = SMCVerifyInput(
            draft_token_num=head_num,
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=self.orig_seq_lens_sum - len(prefix_lens_cpu),
            seq_lens_cpu=prefix_lens_cpu,
            num_tokens_per_req=head_num,
        )

        head_batch = copy.copy(batch)
        head_batch.input_ids = head_token_ids
        head_batch.out_cache_loc = out_cache_loc
        head_batch.seq_lens = prefix_lens
        head_batch.seq_lens_cpu = prefix_lens_cpu
        head_batch.seq_lens_sum = head_spec.seq_lens_sum
        head_batch.spec_info = head_spec
        head_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        head_batch.forward_mode = ForwardMode.TARGET_VERIFY

        forward_batch = ForwardBatch.init_new(head_batch, draft_model_runner)
        head_spec.populate_linear_verify_metadata(forward_batch)
        # Attention metadata is set up by the caller: replay_prepare (graph
        # path) or attn_backend.init_forward_metadata (eager fallback).
        return forward_batch


# ──────────────────────────────────────────────────────────────
#  SMCDraftInput — pure data carrier
# ──────────────────────────────────────────────────────────────


@dataclass
class SMCDraftInput(SpecInput):
    """Lightweight carrier between scheduler and worker via batch.spec_info.

    Has no prepare_for_decode / prepare_for_draft / prepare_for_verify methods —
    those live on SMCDecodeContext.
    """

    verified_id: Optional[torch.Tensor] = None  # (bs,) last accepted token
    # (bs,) per-row token at position S-1: the last *drafted* token
    # d_{gamma-1} from the previous step (deferred into the next step's
    # leading 2-token draft forward [prev_last_draft_id, verified_id]),
    # or — on a group's first decode step — the last committed prompt
    # token (seeded at allocate_slots), whose S-1 draft KV the head
    # rewrites idempotently.  Always populated on the decode path, so the
    # 2-token head needs no per-batch step flag and mixed-step batches
    # (continuous batching) are handled uniformly.
    prev_last_draft_id: Optional[torch.Tensor] = None
    logprob_diff: Optional[torch.Tensor] = None  # (bs, gamma) per-position, last step
    # (bs,) per-particle log-normalizer of the bonus token's power draw,
    # log Z = logsumexp(alpha*logits/T) - alpha*logsumexp(logits/T).  The bonus
    # is sampled from the locally normalized power conditional p_T^alpha / Z, so
    # under the joint-power target its incremental importance weight is Z (not 1).
    # Identically 0 at alpha=1.  Accumulated alongside logprob_diff in
    # write_back_gpu, gated by the same EOS/finish logic.
    bonus_logz: Optional[torch.Tensor] = None  # (bs,) last step
    num_tokens_per_req: int = -1  # gamma + 1
    decode_ctx: Optional[SMCDecodeContext] = None  # attached by prepare_for_decode

    # Class-level constant set during worker init
    ALLOC_LEN_PER_DECODE: ClassVar[int] = 1

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

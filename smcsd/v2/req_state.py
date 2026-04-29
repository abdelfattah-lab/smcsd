"""Slot-based persistent state for SMC particles.

Each particle occupies a fixed slot for its lifetime. Sparse slot tensors
are gathered into contiguous ModelWorkerBatch for the forward pass, and
results are scattered back after each decode cycle.

Replaces ScheduleGroupBatch's per-iteration sync_from_groups rebuild.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from smcsd.v2.info import SMCDecodeContext, SMCDraftInputV2
from smcsd.v2.stacked_state import StackedGroupState
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_config import ModelConfig

logger = logging.getLogger(__name__)

EMPTY_SLOT = -1


class ScheduleBatchSMC:
    """Slot-based SMC batch state. Each particle occupies a fixed slot.

    Sparse slot tensors ([max_slots]) persist across iterations.
    `active_slots` is the idx_mapping from contiguous batch indices to slot indices.
    Only active_slots changes between iterations — no filter_batch / merge_batch.
    """

    def __init__(
        self,
        *,
        max_num_reqs: int,
        device: torch.device,
        gamma_plus_1: int,
        vocab_size: int,
        max_output_len: int,
        max_eos_count: int = 8,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator,
        tree_cache,
        model_config: "ModelConfig",
        enable_overlap: bool = False,
        n_particles: int = 1,
    ):
        self.max_slots = max_num_reqs
        self.device = device
        self.gamma_plus_1 = gamma_plus_1
        self.vocab_size = vocab_size
        self.max_output_len = max_output_len
        self.max_eos_count = max_eos_count
        self.model_config = model_config
        self.enable_overlap = enable_overlap
        self.n_particles = n_particles

        # Pool references (shared with scheduler)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        if self.token_to_kv_pool_allocator.page_size != 1:
            raise ValueError("SMC currently only supports page_size=1")

        # ── Slot lifecycle (CPU) ──
        self.free_slots: Deque[int] = deque(range(self.max_slots))
        self.slot_to_req: Dict[int, Req] = {}
        self.slot_to_group_id: Dict[int, str] = {}

        # ── Per-slot GPU tensors [max_slots] ──
        self.req_pool_indices = torch.full(
            (self.max_slots,), EMPTY_SLOT, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros(self.max_slots, dtype=torch.int64, device=device)
        self.kv_allocated_lens = torch.zeros(
            self.max_slots, dtype=torch.int64, device=device
        )
        self.verified_ids = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.token_counts = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.group_indices = torch.full(
            (self.max_slots,), EMPTY_SLOT, dtype=torch.int32, device=device
        )
        self.particle_indices = torch.full(
            (self.max_slots,), EMPTY_SLOT, dtype=torch.int32, device=device
        )
        self.finished_mask = torch.zeros(
            self.max_slots, dtype=torch.bool, device=device
        )
        self.ignore_eos_t = torch.zeros(
            self.max_slots, dtype=torch.bool, device=device
        )
        self.max_new_tokens_t = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.eos_token_ids_t = torch.full(
            (self.max_slots, max_eos_count), -1, dtype=torch.int64, device=device
        )

        # ── Token history [max_slots, max_output_len] ──
        self.all_token_ids = torch.zeros(
            (self.max_slots, max_output_len), dtype=torch.int32, device=device
        )

        # ── EAGLE3: per-slot target hidden states [max_slots, aux_hidden_dim] ──
        # Allocated lazily on first eagle3 use to avoid memory cost in dense mode.
        # aux_hidden_dim = 3*hidden_dim (use_aux_hidden_state=True) or hidden_dim.
        self.target_hidden_states: Optional[torch.Tensor] = None
        # DFlash: variable-length full target-hidden context per slot.
        # Stored as Python tensors because sequence length grows per particle and
        # can differ across active groups. Hot tensor state remains in the dense
        # per-slot arrays above; this path is eager and correctness-first.
        self.target_hidden_contexts: Dict[int, torch.Tensor] = {}

        # ── EAGLE3: per-slot carry of the pre-sampled "first draft token" ──
        # Allocated lazily. These are the draft's own prediction of the next
        # token (in target-vocab space) sampled from the previous cycle's
        # last-position logits (prefill or rewrite). At decode step 0, this
        # is the token fed into the draft (NOT the verified_id, which was
        # already consumed by the draft during prefill / rewrite).
        self.first_draft_token_ids: Optional[torch.Tensor] = None  # (max_slots,) int64
        self.first_draft_logprobs: Optional[torch.Tensor] = None  # (max_slots,) float32

        # ── Sampling params [max_slots], static after allocation ──
        self.temperatures = torch.ones(
            self.max_slots, 1, dtype=torch.float32, device=device
        )
        self.top_ps = torch.ones(self.max_slots, dtype=torch.float32, device=device)
        self.top_ks = torch.full(
            (self.max_slots,), -1, dtype=torch.int32, device=device
        )
        self.min_ps = torch.zeros(self.max_slots, dtype=torch.float32, device=device)

        # ── Active batch index ──
        self.active_slots = torch.empty(0, dtype=torch.int64, device=device)
        self.num_active: int = 0

        # ── Group tracking ──
        self.group_slot_lists: Dict[str, List[int]] = {}
        self.group_active_indptr: List[int] = [0]
        self._sorted_group_ids: List[str] = []
        # `group_log_weights[gid]` and `group_interval_weights[gid]` are
        # VIEWS into `self.stacked.{log,interval}_weights[row, :n]` — writes
        # through to the stacked storage.  The stacked tensors are the
        # single source of truth consumed by the fused collect kernel.
        self.group_log_weights: Dict[str, torch.Tensor] = {}
        self.group_interval_weights: Dict[str, torch.Tensor] = {}
        self.group_n_particles: Dict[str, int] = {}

        # ── Stacked primary storage for per-group SMC state ──
        # max_groups bounded by #slots / #particles; conservatively
        # allow up to max_slots groups (covers any n_particles >= 1).
        self.stacked = StackedGroupState(
            max_groups=max(max_num_reqs, 1),
            n_particles=max(n_particles, 1),
            device=device,
        )

    # ────────────────────────────────────────────────────────
    #  Slot Allocation / Deallocation
    # ────────────────────────────────────────────────────────

    def allocate_slots(
        self,
        group_id: str,
        group_idx: int,
        particle_reqs: List[Req],
        shared_seq_len: int,
    ) -> List[int]:
        """Claim slots for newly materialized particles, fill from Reqs."""
        n = len(particle_reqs)
        if len(self.free_slots) < n:
            raise RuntimeError(
                f"ScheduleBatchSMC: need {n} slots, only {len(self.free_slots)} free"
            )

        slots = [self.free_slots.popleft() for _ in range(n)]

        for slot, req in zip(slots, particle_reqs):
            self.slot_to_req[slot] = req
            self.slot_to_group_id[slot] = group_id

            self.req_pool_indices[slot] = req.req_pool_idx
            self.seq_lens[slot] = shared_seq_len
            self.kv_allocated_lens[slot] = shared_seq_len
            self.verified_ids[slot] = (
                req.output_ids[-1] if req.output_ids else 0
            )
            self.token_counts[slot] = len(req.output_ids)
            self.group_indices[slot] = group_idx
            self.particle_indices[slot] = req.smc_particle_idx
            self.finished_mask[slot] = False
            self.ignore_eos_t[slot] = bool(req.sampling_params.ignore_eos)
            self.max_new_tokens_t[slot] = req.sampling_params.max_new_tokens

            # EOS token ids — collect from all sources matching v1's check_finished:
            # req.eos_token_ids, sampling_params.stop_token_ids, tokenizer EOS
            eos_ids = list(req.eos_token_ids or [])
            if req.sampling_params.stop_token_ids:
                eos_ids.extend(req.sampling_params.stop_token_ids)
            if hasattr(req, "tokenizer") and req.tokenizer is not None:
                tok = req.tokenizer
                if tok.eos_token_id is not None:
                    eos_ids.append(tok.eos_token_id)
                if getattr(tok, "additional_stop_token_ids", None):
                    eos_ids.extend(tok.additional_stop_token_ids)
            eos_ids = list(dict.fromkeys(eos_ids))  # dedup
            for j in range(self.max_eos_count):
                self.eos_token_ids_t[slot, j] = eos_ids[j] if j < len(eos_ids) else -1

            # Write initial output_ids to all_token_ids
            n_out = len(req.output_ids)
            if n_out > 0:
                self.all_token_ids[slot, :n_out] = torch.tensor(
                    req.output_ids, dtype=torch.int32, device=self.device
                )

            # Sampling params
            self.temperatures[slot, 0] = req.sampling_params.temperature
            self.top_ps[slot] = req.sampling_params.top_p
            self.top_ks[slot] = req.sampling_params.top_k
            self.min_ps[slot] = req.sampling_params.min_p

        self.group_slot_lists[group_id] = slots
        # Register this group in the stacked storage and bind the legacy
        # dict entries to views into its row.  Writes to
        # group_log_weights[gid][pidx] go through to stacked.log_weights[row, pidx].
        self.stacked.register_group(group_id, slots)
        self.group_log_weights[group_id] = self.stacked.log_weights_view(group_id, n)
        self.group_interval_weights[group_id] = self.stacked.interval_weights_view(
            group_id, n,
        )
        self.group_n_particles[group_id] = n
        self.rebuild_active_slots()
        return slots

    def set_prefill_hidden(
        self,
        slots: List[int],
        prefill_hidden: torch.Tensor,
        first_draft_token_id: Optional[torch.Tensor] = None,
        first_draft_logprob: Optional[torch.Tensor] = None,
        prefill_hidden_context: Optional[torch.Tensor] = None,
    ) -> None:
        """EAGLE3: seed particle slots for one parent.

        ``prefill_hidden`` is shared across all particles of the group (same
        target-hidden-state seed after the shared prefill). ``first_draft_token_id``
        and ``first_draft_logprob`` are PER-PARTICLE — one distinct x1 draw per
        slot — to preserve SMC diversity in the first decode cycle.

        Accepted shapes:
          prefill_hidden:        (hidden_dim,) — broadcast to all slots.
          first_draft_token_id:  (len(slots),) — scatter 1:1 with slots.
          first_draft_logprob:   (len(slots),) — scatter 1:1 with slots.
        """
        if not slots:
            return
        slot_idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        n = len(slots)

        if prefill_hidden is not None:
            new_dim = prefill_hidden.shape[-1]
            if (
                self.target_hidden_states is None
                or self.target_hidden_states.shape[-1] != new_dim
            ):
                self.target_hidden_states = torch.zeros(
                    (self.max_slots, new_dim),
                    dtype=prefill_hidden.dtype,
                    device=self.device,
                )
            self.target_hidden_states[slot_idx] = prefill_hidden.unsqueeze(0).expand(
                n, -1
            )

        if first_draft_token_id is not None:
            assert first_draft_token_id.shape[0] == n, (
                f"first_draft_token_id must have one entry per slot: "
                f"got {tuple(first_draft_token_id.shape)} for {n} slots."
            )
            if self.first_draft_token_ids is None:
                self.first_draft_token_ids = torch.zeros(
                    self.max_slots, dtype=torch.int64, device=self.device,
                )
            self.first_draft_token_ids[slot_idx] = first_draft_token_id.to(
                dtype=torch.int64
            )

        if first_draft_logprob is not None:
            assert first_draft_logprob.shape[0] == n, (
                f"first_draft_logprob must have one entry per slot: "
                f"got {tuple(first_draft_logprob.shape)} for {n} slots."
            )
            if self.first_draft_logprobs is None:
                self.first_draft_logprobs = torch.zeros(
                    self.max_slots, dtype=torch.float32, device=self.device,
                )
            self.first_draft_logprobs[slot_idx] = first_draft_logprob.to(
                dtype=torch.float32
            )

        if prefill_hidden_context is not None:
            for slot in slots:
                self.target_hidden_contexts[int(slot)] = prefill_hidden_context

    def free_group_slots(self, group_id: str) -> None:
        """Free all slots for a finalized group."""
        slots = self.group_slot_lists.pop(group_id, [])
        for slot in slots:
            pool_idx = int(self.req_pool_indices[slot].item())
            alloc_len = int(self.kv_allocated_lens[slot].item())

            if pool_idx != EMPTY_SLOT and alloc_len > 0:
                indices = self.req_to_token_pool.req_to_token[
                    pool_idx, :alloc_len
                ].to(dtype=torch.int64, copy=True)
                self.token_to_kv_pool_allocator.dec_ref_and_free(indices)
                req = self.slot_to_req.get(slot)
                if req is not None:
                    self.req_to_token_pool.free(req)

            self.req_pool_indices[slot] = EMPTY_SLOT
            self.seq_lens[slot] = 0
            self.kv_allocated_lens[slot] = 0
            self.verified_ids[slot] = 0
            self.token_counts[slot] = 0
            self.group_indices[slot] = EMPTY_SLOT
            self.particle_indices[slot] = EMPTY_SLOT
            self.finished_mask[slot] = False
            self.ignore_eos_t[slot] = False

            self.slot_to_req.pop(slot, None)
            self.slot_to_group_id.pop(slot, None)
            self.target_hidden_contexts.pop(slot, None)
            self.free_slots.append(slot)

        self.group_log_weights.pop(group_id, None)
        self.group_interval_weights.pop(group_id, None)
        self.group_n_particles.pop(group_id, None)
        self.stacked.unregister_group(group_id)
        self.rebuild_active_slots()

    def rebuild_active_slots(self) -> None:
        """Rebuild active_slots and group_active_indptr.

        active_slots is the idx_mapping: contiguous batch index → sparse slot index.
        Sorted by group so logprob_diffs can be sliced per group via indptr
        (zero-copy, no dict building). Only called when membership changes:
        allocate_slots, free_group_slots, or after a particle finishes.
        """
        self._sorted_group_ids = sorted(self.group_slot_lists.keys())
        active_list: List[int] = []
        indptr = [0]
        for group_id in self._sorted_group_ids:
            group_active = [
                s
                for s in self.group_slot_lists[group_id]
                if not self.finished_mask[s].item()
            ]
            active_list.extend(group_active)
            indptr.append(len(active_list))

        self.active_slots = torch.tensor(
            active_list, dtype=torch.int64, device=self.device
        )
        self.num_active = len(active_list)
        self.group_active_indptr = indptr

    def is_empty(self) -> bool:
        return self.num_active == 0

    # ────────────────────────────────────────────────────────
    #  Decode Preparation (sparse → vectorized KV alloc → sparse)
    # ────────────────────────────────────────────────────────

    def prepare_for_decode(self) -> SMCDraftInputV2:
        """Gather slot tensors → vectorized KV alloc → scatter back.

        Returns an SMCDraftInputV2 with decode_ctx for the worker.
        """
        if self.num_active == 0:
            return SMCDraftInputV2(
                verified_id=torch.empty(0, dtype=torch.int32, device=self.device),
                num_tokens_per_req=self.gamma_plus_1,
            )

        active = self.active_slots

        # Gather contiguous from sparse slots
        seq_lens_g = self.seq_lens[active]
        kv_alloc_g = self.kv_allocated_lens[active]
        pool_idx_g = self.req_pool_indices[active]
        verified_g = self.verified_ids[active]

        # Vectorized KV allocation via SMCDecodeContext
        ctx, new_kv_alloc = SMCDecodeContext.from_slot_gather(
            seq_lens=seq_lens_g,
            kv_allocated_lens=kv_alloc_g,
            req_pool_indices=pool_idx_g,
            gamma_plus_1=self.gamma_plus_1,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
        )

        # Scatter back to sparse slots
        self.kv_allocated_lens[active] = new_kv_alloc
        self.seq_lens[active] = ctx.new_seq_lens

        hidden_g = (
            self.target_hidden_states[active].clone()
            if self.target_hidden_states is not None
            else None
        )
        fd_ids_g = (
            self.first_draft_token_ids[active].clone()
            if self.first_draft_token_ids is not None
            else None
        )
        fd_logp_g = (
            self.first_draft_logprobs[active].clone()
            if self.first_draft_logprobs is not None
            else None
        )
        active_list = active.tolist()
        hidden_contexts_g = None
        if self.target_hidden_contexts:
            hidden_contexts_g = [
                self.target_hidden_contexts[int(slot)] for slot in active_list
            ]

        return SMCDraftInputV2(
            verified_id=verified_g,
            num_tokens_per_req=self.gamma_plus_1,
            decode_ctx=ctx,
            target_hidden_state=hidden_g,
            first_draft_token_id=fd_ids_g,
            first_draft_logprob=fd_logp_g,
            target_hidden_contexts=hidden_contexts_g,
        )

    # ────────────────────────────────────────────────────────
    #  Prepare for Extend (prefill) — not slot-based
    # ────────────────────────────────────────────────────────
    def prepare_for_extend(self):
        """Prefill uses ScheduleBatch.prepare_for_extend() — battle-tested upstream code.

        Prefill runs once per group and is not the hot loop. Keeping it on
        ScheduleBatch avoids duplicating allocation/cache-index logic that may
        change upstream. The decode loop is where the slot-based design pays off.
        """
        pass

    # ────────────────────────────────────────────────────────
    #  Build ModelWorkerBatch (sparse → contiguous gather)
    # ────────────────────────────────────────────────────────

    def build_model_worker_batch(
        self,
        draft_input: SMCDraftInputV2,
    ) -> ModelWorkerBatch:
        """Gather sparse slot state into a contiguous ModelWorkerBatch."""
        active = self.active_slots
        bs = self.num_active
        ctx = draft_input.decode_ctx

        req_pool_indices = self.req_pool_indices[active]
        seq_lens = ctx.new_seq_lens if ctx is not None else self.seq_lens[active]
        seq_lens_cpu = seq_lens.cpu()
        seq_lens_sum = int(seq_lens_cpu.sum().item())

        # Gather Req objects for ForwardBatch.init_new (needs rids)
        reqs = [self.slot_to_req[int(s.item())] for s in active]

        # Minimal SamplingBatchInfo — SMC worker does its own sampling
        sampling_info = SamplingBatchInfo(
            temperatures=self.temperatures[active],
            top_ps=self.top_ps[active],
            top_ks=self.top_ks[active],
            min_ps=self.min_ps[active],
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.vocab_size,
        )

        return ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=draft_input.verified_id,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=None,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
            return_logprob=False,
            top_logprobs_nums=[0] * bs,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=[None] * bs,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=sampling_info,
            spec_algorithm=SpeculativeAlgorithm.SMC,
            spec_info=draft_input,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            reqs=reqs,
        )

    # ────────────────────────────────────────────────────────
    #  Process Batch Result (write-back from forward pass)
    # ────────────────────────────────────────────────────────

    def process_batch_result(
        self,
        next_token_ids: torch.Tensor,
        accept_lens: torch.Tensor,
        logprob_diff: torch.Tensor,
        bonus_ids: torch.Tensor,
        *,
        next_hidden_state: Optional[torch.Tensor] = None,
        next_first_draft_token_id: Optional[torch.Tensor] = None,
        next_first_draft_logprob: Optional[torch.Tensor] = None,
        next_hidden_contexts: Optional[List[torch.Tensor]] = None,
        rebuild_active: bool = True,
    ) -> List[int]:
        """Write forward results back to slot state. Returns newly finished slots.

        Args:
            next_token_ids: (num_active * stride,) flattened accepted tokens.
            accept_lens: (num_active,) tokens accepted per particle (always gamma+1).
            logprob_diff: (num_active,) per-particle logprob diff.
            bonus_ids: (num_active,) next verified_id (bonus token).
        """
        active = self.active_slots
        bs = self.num_active
        stride = self.gamma_plus_1

        # a. Write accepted tokens to all_token_ids
        # Reshape to (bs, stride) and scatter into sparse 2D tensor
        accepted_2d = next_token_ids.reshape(bs, stride)
        offsets = self.token_counts[active].to(torch.int64)  # (bs,)
        # Vectorized scatter: build (bs, stride) index grids, write in one shot
        row_idx = active.unsqueeze(1).expand(-1, stride)  # (bs, stride)
        col_idx = offsets.unsqueeze(1) + torch.arange(
            stride, dtype=torch.int64, device=self.device,
        )  # (bs, stride)
        self.all_token_ids[row_idx, col_idx] = accepted_2d.to(self.all_token_ids.dtype)
        self.token_counts[active] += stride

        # b. Update verified_ids
        self.verified_ids[active] = bonus_ids.to(dtype=torch.int32)

        # b2. EAGLE3: scatter new target hidden states back to slot storage.
        # Dim can change across cycles (hidden_dim seed at prefill vs
        # 3*hidden_dim aux after verify) — reallocate on mismatch.
        if next_hidden_state is not None:
            new_dim = next_hidden_state.shape[-1]
            if (
                self.target_hidden_states is None
                or self.target_hidden_states.shape[-1] != new_dim
            ):
                self.target_hidden_states = torch.zeros(
                    (self.max_slots, new_dim),
                    dtype=next_hidden_state.dtype,
                    device=self.device,
                )
            self.target_hidden_states[active] = next_hidden_state

        # b3. EAGLE3: write back the next-cycle first-draft token / logprob.
        if next_first_draft_token_id is not None:
            if self.first_draft_token_ids is None:
                self.first_draft_token_ids = torch.zeros(
                    self.max_slots, dtype=torch.int64, device=self.device,
                )
            self.first_draft_token_ids[active] = next_first_draft_token_id.to(
                dtype=torch.int64
            )
        if next_first_draft_logprob is not None:
            if self.first_draft_logprobs is None:
                self.first_draft_logprobs = torch.zeros(
                    self.max_slots, dtype=torch.float32, device=self.device,
                )
            self.first_draft_logprobs[active] = next_first_draft_logprob.to(
                dtype=torch.float32
            )

        if next_hidden_contexts is not None:
            assert len(next_hidden_contexts) == bs, (
                f"next_hidden_contexts must have one tensor per active slot: "
                f"got {len(next_hidden_contexts)} for {bs} slots."
            )
            for slot, hidden_context in zip(active.tolist(), next_hidden_contexts):
                self.target_hidden_contexts[int(slot)] = hidden_context

        # c. Batched finish check
        newly_finished: List[int] = []
        updated_counts = self.token_counts[active]  # (bs,)
        max_tokens = self.max_new_tokens_t[active]  # (bs,)

        # Length check (GPU)
        length_hit = updated_counts >= max_tokens

        # EOS check: compare each accepted token against per-slot EOS ids.
        # Suppressed for slots with ignore_eos=True (matches v1's
        # _check_token_based_finish which skips all token-based checks).
        eos_ids = self.eos_token_ids_t[active]  # (bs, max_eos_count)
        eos_hit = (
            accepted_2d.unsqueeze(2).to(torch.int64)
            == eos_ids.unsqueeze(1)
        ).any(dim=2).any(dim=1)  # (bs,)
        eos_hit = eos_hit & ~self.ignore_eos_t[active]

        newly_finished_mask = (length_hit | eos_hit) & ~self.finished_mask[active]
        self.finished_mask[active] = self.finished_mask[active] | newly_finished_mask

        # d. Sync finished to Reqs (only newly finished — typically 0-2)
        # Note: we deliberately do NOT mark particle_finish on the stacked
        # state.  The legacy slow-path resample (golden truth) includes
        # finished particles in its candidate set, relying on
        # resample_copy_slot to propagate `finished_mask` from src to dst.
        # The fast path mirrors this so both produce identical results.
        if newly_finished_mask.any():
            finished_indices = newly_finished_mask.nonzero(as_tuple=True)[0]
            for idx in finished_indices.tolist():
                slot = active[idx].item()
                newly_finished.append(slot)
                req = self.slot_to_req[slot]
                count = int(self.token_counts[slot].item())
                req.kv_committed_len = int(self.seq_lens[slot].item())
                req.kv_allocated_len = int(self.kv_allocated_lens[slot].item())
                # Determine finish reason and finished_len first,
                # then set output_ids (may truncate at EOS)
                if length_hit[idx].item():
                    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
                    req.finished_reason = FINISH_LENGTH(
                        length=int(max_tokens[idx].item())
                    )
                    req.finished_len = int(max_tokens[idx].item())
                else:
                    from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
                    # Find which token matched EOS and at which position
                    eos_set = set(eos_ids[idx].tolist()) - {-1}
                    matched_tok = 0
                    eos_pos_in_stride = stride  # fallback: end of stride
                    for j, t in enumerate(accepted_2d[idx].tolist()):
                        if t in eos_set:
                            matched_tok = t
                            eos_pos_in_stride = j
                            break
                    req.finished_reason = FINISH_MATCHED_TOKEN(matched=matched_tok)
                    # finished_len = tokens before this stride + EOS position + 1
                    # (matches v1: output_ids[:finished_len] includes EOS, excludes post-EOS)
                    old_count = count - stride
                    req.finished_len = old_count + eos_pos_in_stride + 1
                # Set output_ids truncated at finished_len (excludes post-EOS tokens)
                req.output_ids = self.all_token_ids[
                    slot, : req.finished_len
                ].tolist()

        # e. Update group log_weights and interval_weights
        for g_idx, group_id in enumerate(self._sorted_group_ids):
            start = self.group_active_indptr[g_idx]
            end = self.group_active_indptr[g_idx + 1]
            if start >= end:
                continue
            diffs = logprob_diff[start:end]
            pidxs_batch = [
                self.particle_indices[active[j]].item() for j in range(start, end)
            ]
            pidx_t = torch.tensor(pidxs_batch, dtype=torch.int64, device=self.device)
            lw = self.group_log_weights[group_id]
            iw = self.group_interval_weights[group_id]
            lw[pidx_t] += diffs.to(dtype=lw.dtype, device=self.device)
            iw[pidx_t] += diffs.to(dtype=iw.dtype, device=self.device)

        # f. Rebuild active_slots if any finished (caller may defer)
        if newly_finished and rebuild_active:
            self.rebuild_active_slots()

        return newly_finished

    # ────────────────────────────────────────────────────────
    #  Resampling
    # ────────────────────────────────────────────────────────

    def resample_copy_slot(self, dst_slot: int, src_slot: int) -> None:
        """Copy all state from src_slot to dst_slot for resampling."""
        old_dst_alloc = int(self.kv_allocated_lens[dst_slot].item())
        src_seq_len = int(self.seq_lens[src_slot].item())
        # GPU tensor row copies
        self.seq_lens[dst_slot] = self.seq_lens[src_slot]
        self.kv_allocated_lens[dst_slot] = self.kv_allocated_lens[src_slot]
        self.verified_ids[dst_slot] = self.verified_ids[src_slot]
        self.finished_mask[dst_slot] = self.finished_mask[src_slot]
        if self.target_hidden_states is not None:
            self.target_hidden_states[dst_slot] = self.target_hidden_states[src_slot]
        if self.first_draft_token_ids is not None:
            self.first_draft_token_ids[dst_slot] = self.first_draft_token_ids[src_slot]
        if self.first_draft_logprobs is not None:
            self.first_draft_logprobs[dst_slot] = self.first_draft_logprobs[src_slot]
        if src_slot in self.target_hidden_contexts:
            self.target_hidden_contexts[dst_slot] = self.target_hidden_contexts[src_slot]
        else:
            self.target_hidden_contexts.pop(dst_slot, None)

        src_count = int(self.token_counts[src_slot].item())
        self.token_counts[dst_slot] = src_count
        if src_count > 0:
            self.all_token_ids[dst_slot, :src_count] = (
                self.all_token_ids[src_slot, :src_count]
            )

        # KV block table copy (through req_to_token_pool)
        src_pool = int(self.req_pool_indices[src_slot].item())
        dst_pool = int(self.req_pool_indices[dst_slot].item())

        # Dec ref on old dst KV
        if old_dst_alloc > 0:
            old_indices = self.req_to_token_pool.req_to_token[
                dst_pool, :old_dst_alloc
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.dec_ref_and_free(old_indices)

        # Copy src KV block table to dst + inc ref
        if src_seq_len > 0:
            src_indices = self.req_to_token_pool.req_to_token[
                src_pool, :src_seq_len
            ].to(dtype=torch.int64, copy=True)
            self.req_to_token_pool.write(
                (dst_pool, slice(0, src_seq_len)),
                src_indices.to(dtype=torch.int32),
            )
            self.token_to_kv_pool_allocator.inc_ref(src_indices)

        # Req-level text state (cold, for finalization/streaming)
        src_req = self.slot_to_req[src_slot]
        dst_req = self.slot_to_req[dst_slot]
        dst_req.output_ids = list(src_req.output_ids)
        dst_req.finished_reason = copy.copy(src_req.finished_reason)
        dst_req.finished_len = src_req.finished_len
        dst_req.finished_output = src_req.finished_output
        dst_req.to_finish = copy.copy(src_req.to_finish)
        dst_req.kv_committed_len = src_req.kv_committed_len
        dst_req.kv_allocated_len = src_req.kv_allocated_len
        dst_req.decoded_text = src_req.decoded_text
        dst_req.surr_offset = src_req.surr_offset
        dst_req.read_offset = src_req.read_offset
        if src_slot in self.target_hidden_contexts:
            self.target_hidden_contexts[dst_slot] = self.target_hidden_contexts[src_slot]
        else:
            self.target_hidden_contexts.pop(dst_slot, None)

    def copy_req_metadata(self, dst_slot: int, src_slot: int) -> None:
        """Copy req-side state from src to dst after fused tensor/KV copies."""
        src_req = self.slot_to_req[src_slot]
        dst_req = self.slot_to_req[dst_slot]
        dst_req.output_ids = list(src_req.output_ids)
        dst_req.finished_reason = copy.copy(src_req.finished_reason)
        dst_req.finished_len = src_req.finished_len
        dst_req.finished_output = src_req.finished_output
        dst_req.to_finish = copy.copy(src_req.to_finish)
        dst_req.kv_committed_len = src_req.kv_committed_len
        dst_req.kv_allocated_len = src_req.kv_allocated_len
        dst_req.decoded_text = src_req.decoded_text
        dst_req.surr_offset = src_req.surr_offset
        dst_req.read_offset = src_req.read_offset

    # ────────────────────────────────────────────────────────
    #  Finalization
    # ────────────────────────────────────────────────────────

    def finalize_group(self, group_id: str, parent_req: Req) -> Req:
        """Pick the best particle and copy its output to the parent req.

        Frees all group slots and returns the parent req ready for stream_output.
        """
        lw = self.group_log_weights[group_id]
        slots = self.group_slot_lists[group_id]
        best_req_by_slot = self.slot_to_req

        def visible_output_len(slot: int) -> int:
            req = best_req_by_slot[slot]
            token_count = int(self.token_counts[slot].item())
            if req.finished_len is None:
                return token_count
            return min(req.finished_len, token_count)

        # Pick best by (log_weight, output_length)
        best_slot = max(
            slots,
            key=lambda s: (
                float(lw[self.particle_indices[s]].item()),
                visible_output_len(s),
            ),
        )
        best_req = self.slot_to_req[best_slot]
        parent_req.output_ids = list(best_req.output_ids)
        if best_req.finished_reason is not None:
            import copy
            parent_req.finished_reason = copy.copy(best_req.finished_reason)
            parent_req.finished_len = best_req.finished_len
        else:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT
            parent_req.finished_reason = FINISH_ABORT(
                "SMC group finalized without a finished particle."
            )
            parent_req.finished_len = len(parent_req.output_ids)

        self.free_group_slots(group_id)
        return parent_req

    # ────────────────────────────────────────────────────────
    #  Group Queries
    # ────────────────────────────────────────────────────────

    def group_has_active(self, group_id: str) -> bool:
        slots = self.group_slot_lists.get(group_id, [])
        return any(not self.finished_mask[s].item() for s in slots)

    def sorted_group_ids(self) -> List[str]:
        return sorted(self.group_slot_lists.keys())

    def active_particle_count(self) -> int:
        return self.num_active

    def available_slot_count(self) -> int:
        return len(self.free_slots)

    def held_token_count(self) -> int:
        held: set[int] = set()
        for slot in self.slot_to_req:
            pool_idx = int(self.req_pool_indices[slot].item())
            alloc_len = int(self.kv_allocated_lens[slot].item())
            if pool_idx == EMPTY_SLOT or alloc_len <= 0:
                continue
            indices = self.req_to_token_pool.req_to_token[pool_idx, :alloc_len]
            held.update(indices.cpu().tolist())
        return len(held)

    def held_req_count(self) -> int:
        return sum(
            1
            for slot in self.slot_to_req
            if int(self.req_pool_indices[slot].item()) != EMPTY_SLOT
        )

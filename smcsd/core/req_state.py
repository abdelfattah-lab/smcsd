"""Slot-major persistent state for SMC particles.

Design at a glance
------------------

Each particle gets a fixed slot (``int in [0, max_slots)``) for its lifetime.
All per-particle state — sequence lengths, KV allocation, cumulative log
weights, output history, sampling params — lives in ``(max_slots,)``- or
``(max_slots, X)``-shaped tensors on device.  The forward pass gathers only
the LIVE subset into a contiguous ``ModelWorkerBatch`` via ``active_slots``.

Group bookkeeping is intentionally minimal.  Each active group occupies one
row in ``group_to_slots[max_groups, N]``, which the fused resample kernel
uses to look up that group's member slots.  Under the global-``N`` invariant
(every group has exactly ``server_args.smc_n_particles`` particles for its
lifetime), an in-use row is always fully populated.

Invariants
----------
* ``row_in_use[r]`` ⇒ ``group_to_slots[r, :N]`` holds N distinct slot ids,
  each assigned to exactly one particle.
* Slot sets of distinct in-use rows are disjoint.
* ``active_slots`` is the subset of allocated slots whose particle is NOT
  finished.  Finished particles remain allocated and remain in
  ``group_to_slots`` — they still participate in resampling (their
  cumulative weight is part of the SMC mixture) and may be overwritten as
  the destination of a resample copy.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from smcsd.core.info import SMCDecodeContext, SMCDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.model_config import ModelConfig

logger = logging.getLogger(__name__)

EMPTY_SLOT = -1


class ScheduleBatchSMC:
    """Slot-major SMC batch state.

    See module docstring for the layout and invariants.  Callers touch two
    coarse primitives:

    * ``allocate_slots`` / ``free_group_slots`` — rare, at group materialise
      and finalize.
    * ``prepare_for_decode`` + ``build_model_worker_batch`` +
      ``process_batch_result`` — hot path, once per decode step.
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
        self.n_particles = max(n_particles, 1)
        # Global-N invariant: every group has exactly N particles, so the
        # group lookup table has rows of fixed width N and never more than
        # ``max_slots // N`` rows in flight.
        self.max_groups = max_num_reqs // self.n_particles

        # Pool references (shared with scheduler)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tree_cache = tree_cache
        if self.token_to_kv_pool_allocator.page_size != 1:
            raise ValueError("SMC currently only supports page_size=1")

        # ── Slot lifecycle (CPU) ──
        self.free_slots: Deque[int] = deque(range(self.max_slots))
        self.slot_to_req: Dict[int, Req] = {}

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
        # Per-slot last *drafted* token d_{gamma-1} from the previous step,
        # deferred into the next step's leading 2-token draft forward.  Travels
        # with lineage exactly like verified_ids.  0 means "no predecessor"
        # (group start) → single-token head.  Carried but not yet consumed.
        self.prev_last_draft_ids = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.token_counts = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
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

        # Cumulative SMC weights per particle — slot-indexed, float64 for
        # numerical stability across long decodes.  `log_weights` is the
        # running log-weight used at finalize time.  `interval_weights` is
        # the "since last resample" accumulator consumed by the fused
        # resample kernel (zeroed per row when that row resamples).
        self.log_weights = torch.zeros(
            self.max_slots, dtype=torch.float64, device=device
        )
        self.interval_weights = torch.zeros(
            self.max_slots, dtype=torch.float64, device=device
        )

        # ── Token history [max_slots, max_output_len] ──
        self.all_token_ids = torch.zeros(
            (self.max_slots, max_output_len), dtype=torch.int32, device=device
        )

        # ── SamplingBatchInfo stubs ──
        # SMC bypasses sglang's sampler (the worker does its own draft proposal
        # and bonus sampling under engine-wide smc_* params), so per-request
        # temperature / top_p / top_k / min_p are never read on this path.  We
        # still need a SamplingBatchInfo to satisfy ModelWorkerBatch's schema,
        # so we keep one set of constant placeholder tensors sized to
        # max_slots and slice [:bs] at build time.
        self._stub_temperatures = torch.ones(
            self.max_slots, 1, dtype=torch.float32, device=device
        )
        self._stub_top_ps = torch.ones(
            self.max_slots, dtype=torch.float32, device=device
        )
        self._stub_top_ks = torch.full(
            (self.max_slots,), -1, dtype=torch.int32, device=device
        )
        self._stub_min_ps = torch.zeros(
            self.max_slots, dtype=torch.float32, device=device
        )

        # ── Active batch index ──
        # `active_slots` maps contiguous ModelWorkerBatch indices → slot ids.
        # Rebuilt on membership change (allocate / free / particle finish).
        # `_active_slots_list` mirrors `active_slots` on CPU so hot-path
        # callers (`build_model_worker_batch`) can resolve slot → Req without
        # per-element `.item()` syncs.
        self.active_slots = torch.empty(0, dtype=torch.int64, device=device)
        self._active_slots_list: List[int] = []
        self.num_active: int = 0

        # ── Group tracking ──
        # Per-group slot list (CPU authoritative view) — kept for O(1) Python
        # iteration during rebuild / finalize.  Mirrored on device as
        # `group_to_slots` for the resample kernel.
        self.group_slot_lists: Dict[str, List[int]] = {}
        self._sorted_group_ids: List[str] = []

        # ── Group → slot lookup (device, for the fused collect kernel) ──
        # group_to_slots[r, c] = slot id of particle c in group at row r,
        # or -1 if the row is free.  row_in_use[r] gates the kernel; an
        # in-use row always has all N cells populated (global-N invariant).
        self.group_to_slots = torch.full(
            (self.max_groups, self.n_particles), -1,
            dtype=torch.int32, device=device,
        )
        self.row_in_use = torch.zeros(
            self.max_groups, dtype=torch.bool, device=device,
        )
        # Per-group running unbiased log normalizing-constant estimate
        # (log Z_hat), row-aligned with group_to_slots.  Accumulated at each
        # resample boundary as logsumexp(interval_weights) - log(N) and folded
        # with the final tail at finalize_group.
        self.group_log_Z_hat = torch.zeros(
            self.max_groups, dtype=torch.float64, device=device,
        )
        self.group_id_to_row: Dict[str, int] = {}
        self.row_to_group_id: Dict[int, str] = {}
        self._free_rows: List[int] = list(range(self.max_groups))

        # Fused-collect kernel output buffers are allocated per call
        # inside `batched_collect_fused` — they are transient to one
        # kernel launch, not persistent batch state.

    # ────────────────────────────────────────────────────────
    #  Slot Allocation / Deallocation
    # ────────────────────────────────────────────────────────

    def allocate_slots(
        self,
        group_id: str,
        particle_reqs: List[Req],
        shared_seq_len: int,
    ) -> List[int]:
        """Claim N slots + one group row for a freshly materialised group.

        Writes every per-slot tensor from the particle Reqs, populates the
        device-side group lookup row, and zeroes the group's cumulative
        weights.  Triggers one ``rebuild_active_slots`` to refresh the
        forward-pass gather index.
        """
        n = len(particle_reqs)
        if n != self.n_particles:
            raise ValueError(
                f"ScheduleBatchSMC: expected {self.n_particles} particles per group, "
                f"got {n}"
            )
        if len(self.free_slots) < n:
            raise RuntimeError(
                f"ScheduleBatchSMC: need {n} slots, only {len(self.free_slots)} free"
            )
        if not self._free_rows:
            raise RuntimeError(
                f"ScheduleBatchSMC: no free group rows (max_groups={self.max_groups})"
            )

        slots = [self.free_slots.popleft() for _ in range(n)]
        row = self._free_rows.pop()
        self.group_id_to_row[group_id] = row
        self.row_to_group_id[row] = group_id

        for slot, req in zip(slots, particle_reqs):
            self.slot_to_req[slot] = req

            self.req_pool_indices[slot] = req.req_pool_idx
            self.seq_lens[slot] = shared_seq_len
            self.kv_allocated_lens[slot] = shared_seq_len
            self.verified_ids[slot] = req.output_ids[-1] if req.output_ids else 0
            # -1 sentinel = "no deferred draft token yet" (group's first decode
            # step).  The deferred-bonus path keys on this to use a 1-token head
            # for step 0 (never touching the prompt's S-1 slot) and a 2-token
            # head from step 1 on.  -1 is unambiguous since token ids are >= 0.
            self.prev_last_draft_ids[slot] = -1
            self.token_counts[slot] = len(req.output_ids)
            self.finished_mask[slot] = False
            self.ignore_eos_t[slot] = bool(req.sampling_params.ignore_eos)
            self.max_new_tokens_t[slot] = req.sampling_params.max_new_tokens

            # EOS token ids: gather from req.eos_token_ids, sampling_params
            # stop_token_ids, and the tokenizer.
            eos_ids = list(req.eos_token_ids or [])
            if req.sampling_params.stop_token_ids:
                eos_ids.extend(req.sampling_params.stop_token_ids)
            if hasattr(req, "tokenizer") and req.tokenizer is not None:
                tok = req.tokenizer
                if tok.eos_token_id is not None:
                    eos_ids.append(tok.eos_token_id)
                if getattr(tok, "additional_stop_token_ids", None):
                    eos_ids.extend(tok.additional_stop_token_ids)
            eos_ids = list(dict.fromkeys(eos_ids))
            for j in range(self.max_eos_count):
                self.eos_token_ids_t[slot, j] = eos_ids[j] if j < len(eos_ids) else -1

            # Seed the output_ids prefix into the history buffer.
            n_out = len(req.output_ids)
            if n_out > 0:
                self.all_token_ids[slot, :n_out] = torch.tensor(
                    req.output_ids, dtype=torch.int32, device=self.device
                )

        self.group_slot_lists[group_id] = slots

        # Populate the device-side group lookup row and zero this row's
        # cumulative weights in one shot.
        slots_t = torch.as_tensor(slots, dtype=torch.int32, device=self.device)
        self.group_to_slots[row, :n] = slots_t
        self.row_in_use[row] = True
        self.group_log_Z_hat[row] = 0.0
        slot_idx64 = slots_t.to(torch.int64)
        self.log_weights[slot_idx64] = 0.0
        self.interval_weights[slot_idx64] = 0.0

        self.rebuild_active_slots()
        return slots

    def free_group_slots(self, group_id: str) -> None:
        """Release every slot and the group row for a finalised group.

        Frees KV-cache refcounts for each slot's live block table, returns
        the ReqToTokenPool entry, clears per-slot tensors to sentinel
        values, and zeros the released slots' weights.  Triggers one
        ``rebuild_active_slots``.
        """
        slots = self.group_slot_lists.pop(group_id, [])
        row = self.group_id_to_row.pop(group_id, None)
        if row is not None:
            self.row_to_group_id.pop(row, None)
            self.group_to_slots[row] = -1
            self.row_in_use[row] = False
            self.group_log_Z_hat[row] = 0.0
            self._free_rows.append(row)

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
                    # For hybrid (Mamba+attention) targets with an isolated
                    # draft Mamba pool, clear the draft slot before freeing
                    # the target req — otherwise the next request to reuse
                    # this req_pool_idx would inherit stale Mamba state.
                    if (
                        hasattr(self.req_to_token_pool, "free_mamba_cache")
                        and req.mamba_pool_idx is not None
                    ):
                        saved_idx = req.mamba_pool_idx
                        self.req_to_token_pool.free_mamba_cache(req)
                        draft_pool = getattr(
                            self.req_to_token_pool,
                            "_smc_draft_hybrid_pool",
                            None,
                        )
                        from smcsd.common.utils import _clear_draft_mamba_slot
                        _clear_draft_mamba_slot(draft_pool, saved_idx)
                    self.req_to_token_pool.free(req)

            self.req_pool_indices[slot] = EMPTY_SLOT
            self.seq_lens[slot] = 0
            self.kv_allocated_lens[slot] = 0
            self.verified_ids[slot] = 0
            self.prev_last_draft_ids[slot] = 0
            self.token_counts[slot] = 0
            self.finished_mask[slot] = False
            self.ignore_eos_t[slot] = False
            self.log_weights[slot] = 0.0
            self.interval_weights[slot] = 0.0

            self.slot_to_req.pop(slot, None)
            self.free_slots.append(slot)

        self.rebuild_active_slots()

    def rebuild_active_slots(self) -> None:
        """Refresh ``active_slots``.

        ``active_slots`` is the contiguous-batch → slot gather index used to
        build a ``ModelWorkerBatch``.  Slots are grouped by group_id (sorted)
        so per-group slices of the forward-pass output tensors (e.g.
        ``logprob_diff``) are contiguous.

        Finished particles are kept in ``active_slots`` (EOS-absorbing-state
        semantics): they continue to receive forward passes so they can be
        selected as ancestors during resampling, but their weight diff is
        zeroed in ``process_batch_result`` step (e) — the absorbing state
        deterministically emits EOS, so log p(token | finished) = 0 for
        both proposal and target.  Their cumulative ``log_weights`` is
        therefore frozen at the value it had on the EOS transition step
        and stays comparable to still-running particles, so they remain
        fair candidates for resampling and finalization.  Already-finished
        slots are also guarded from being re-marked as newly finished
        (the ``& ~prev_finished_active`` term on ``newly_finished_mask``).
        """
        self._sorted_group_ids = sorted(self.group_slot_lists.keys())
        active_list: List[int] = []
        for group_id in self._sorted_group_ids:
            for s in self.group_slot_lists[group_id]:
                active_list.append(s)

        self.active_slots = torch.tensor(
            active_list, dtype=torch.int64, device=self.device
        )
        self._active_slots_list = active_list
        self.num_active = len(active_list)

    def is_empty(self) -> bool:
        return self.num_active == 0

    # ────────────────────────────────────────────────────────
    #  Decode Preparation (sparse → vectorized KV alloc → sparse)
    # ────────────────────────────────────────────────────────

    def prepare_for_decode(self) -> SMCDraftInput:
        """Gather the live slot tensors, vectorised KV allocation, scatter
        back, and return a ready-to-use ``SMCDraftInput`` for the worker.
        """
        if self.num_active == 0:
            return SMCDraftInput(
                verified_id=torch.empty(0, dtype=torch.int32, device=self.device),
                num_tokens_per_req=self.gamma_plus_1,
            )

        active = self.active_slots

        seq_lens_g = self.seq_lens[active]
        kv_alloc_g = self.kv_allocated_lens[active]
        pool_idx_g = self.req_pool_indices[active]
        verified_g = self.verified_ids[active]
        prev_last_draft_g = self.prev_last_draft_ids[active]

        ctx, new_kv_alloc = SMCDecodeContext.from_slot_gather(
            seq_lens=seq_lens_g,
            kv_allocated_lens=kv_alloc_g,
            req_pool_indices=pool_idx_g,
            gamma_plus_1=self.gamma_plus_1,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
        )

        self.kv_allocated_lens[active] = new_kv_alloc
        self.seq_lens[active] = ctx.new_seq_lens

        return SMCDraftInput(
            verified_id=verified_g,
            prev_last_draft_id=prev_last_draft_g,
            num_tokens_per_req=self.gamma_plus_1,
            decode_ctx=ctx,
        )

    def prepare_for_extend(self):
        """Prefill uses ``ScheduleBatch.prepare_for_extend`` — unchanged
        upstream code.  The slot-based design only applies to decode."""
        pass

    # ────────────────────────────────────────────────────────
    #  Build ModelWorkerBatch (slot-major → contiguous gather)
    # ────────────────────────────────────────────────────────

    def build_model_worker_batch(
        self,
        draft_input: SMCDraftInput,
    ) -> ModelWorkerBatch:
        """Assemble a contiguous ``ModelWorkerBatch`` for the worker from
        the live subset of slot-indexed tensors."""
        active = self.active_slots
        bs = self.num_active
        ctx = draft_input.decode_ctx

        req_pool_indices = self.req_pool_indices[active]
        seq_lens = ctx.new_seq_lens if ctx is not None else self.seq_lens[active]
        seq_lens_cpu = seq_lens.cpu()
        seq_lens_sum = int(seq_lens_cpu.sum().item())

        reqs = [self.slot_to_req[s] for s in self._active_slots_list]

        # Stub SamplingBatchInfo — SMC worker does its own sampling under
        # engine-wide smc_* params, so the per-row tensors are unused.  The
        # stubs are constants slot-allocated once at init.
        sampling_info = SamplingBatchInfo(
            temperatures=self._stub_temperatures[:bs],
            top_ps=self._stub_top_ps[:bs],
            top_ks=self._stub_top_ks[:bs],
            min_ps=self._stub_min_ps[:bs],
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
        prev_last_draft_ids: Optional[torch.Tensor] = None,
        rebuild_active: bool = True,
    ) -> List[int]:
        """Write forward-pass results back to slot-indexed tensors.

        Order of operations:

        a. Scatter accepted tokens into ``all_token_ids``; bump ``token_counts``.
        b. Overwrite ``verified_ids`` with next-step bonus tokens.
        c. Check finish conditions (length, EOS) batched on GPU.
        d. Sync finished particles' output to their ``Req`` objects (only
           the newly-finished ones — typically 0–2 per step).
        e. Accumulate ``logprob_diff`` into the slot-indexed
           ``log_weights`` / ``interval_weights`` — one vectorised
           index_put_ per tensor, no Python loop, no ``.item()`` syncs.
        f. Optionally rebuild the active-slot index.

        Returns the list of slot ids that just transitioned to finished.
        """
        active = self.active_slots
        bs = self.num_active
        stride = self.gamma_plus_1

        # a. Scatter accepted tokens into (bs, stride) columns starting at
        #    offsets[i] = token_counts[slot_i].
        accepted_2d = next_token_ids.reshape(bs, stride)
        offsets = self.token_counts[active].to(torch.int64)
        row_idx = active.unsqueeze(1).expand(-1, stride)
        col_idx = offsets.unsqueeze(1) + torch.arange(
            stride, dtype=torch.int64, device=self.device,
        )
        self.all_token_ids[row_idx, col_idx] = accepted_2d.to(self.all_token_ids.dtype)
        self.token_counts[active] += stride

        # b. Next step's seed token.
        self.verified_ids[active] = bonus_ids.to(dtype=torch.int32)
        # This step's last drafted token, deferred into next step's leading
        # 2-token draft forward.  Carried but not yet consumed (Step 2).
        if prev_last_draft_ids is not None:
            self.prev_last_draft_ids[active] = prev_last_draft_ids.to(
                dtype=torch.int32
            )

        # c. Batched finish check on GPU.
        updated_counts = self.token_counts[active]
        max_tokens = self.max_new_tokens_t[active]
        length_hit = updated_counts >= max_tokens

        # EOS check: any accepted token in this step matches any of this
        # slot's EOS ids.  Skipped for slots with ignore_eos=True.
        eos_ids = self.eos_token_ids_t[active]
        eos_hit = (
            accepted_2d.unsqueeze(2).to(torch.int64) == eos_ids.unsqueeze(1)
        ).any(dim=2).any(dim=1)
        eos_hit = eos_hit & ~self.ignore_eos_t[active]

        prev_finished_active = self.finished_mask[active]
        newly_finished_mask = (length_hit | eos_hit) & ~prev_finished_active
        self.finished_mask[active] = prev_finished_active | newly_finished_mask

        # Per-particle inclusive cutoff column for the weight sum below.
        # logprob_diff has `gamma` columns (the drafted positions); a particle
        # accrues weight from columns 0..cutoff inclusive.  Default keeps the
        # whole block; a particle that terminates via EOS at block column j
        # keeps only 0..j (the EOS token itself is a real sample; tokens after
        # it are post-EOS draft junk and must not contribute — EOS is an
        # absorbing state with incremental weight 1).  Already-finished
        # particles keep nothing (cutoff -1).
        n_weight_cols = logprob_diff.shape[1]
        weight_cutoff = torch.full(
            (bs,), n_weight_cols - 1, dtype=torch.int64, device=self.device
        )
        weight_cutoff[prev_finished_active] = -1

        # d. Sync the small set of newly-finished particles back to their
        #    Req objects (for finalize / streaming).  Keeps finished
        #    particles in the resample candidate set via finished_mask +
        #    ongoing participation in group_to_slots.
        newly_finished: List[int] = []
        if newly_finished_mask.any():
            finished_indices = newly_finished_mask.nonzero(as_tuple=True)[0]
            for idx in finished_indices.tolist():
                slot = int(active[idx].item())
                newly_finished.append(slot)
                req = self.slot_to_req[slot]
                count = int(self.token_counts[slot].item())
                req.kv_committed_len = int(self.seq_lens[slot].item())
                req.kv_allocated_len = int(self.kv_allocated_lens[slot].item())
                if length_hit[idx].item():
                    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
                    req.finished_reason = FINISH_LENGTH(
                        length=int(max_tokens[idx].item())
                    )
                    req.finished_len = int(max_tokens[idx].item())
                else:
                    from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
                    eos_set = set(eos_ids[idx].tolist()) - {-1}
                    matched_tok = 0
                    eos_pos_in_stride = stride
                    for j, t in enumerate(accepted_2d[idx].tolist()):
                        if t in eos_set:
                            matched_tok = t
                            eos_pos_in_stride = j
                            break
                    req.finished_reason = FINISH_MATCHED_TOKEN(matched=matched_tok)
                    old_count = count - stride
                    req.finished_len = old_count + eos_pos_in_stride + 1
                    # Keep weight columns up to and including the EOS column,
                    # drop post-EOS draft tokens.  Capped at the last weight
                    # column so an EOS in the bonus slot keeps the full block.
                    weight_cutoff[idx] = min(eos_pos_in_stride, n_weight_cols - 1)
                req.output_ids = self.all_token_ids[
                    slot, : req.finished_len
                ].tolist()

        # e. Accumulate log-weights.  `logprob_diff` is per-position
        #    (bs, gamma); each particle sums columns 0..weight_cutoff
        #    inclusive.  Already-finished slots (cutoff -1) contribute 0 — the
        #    EOS-absorbing state emits EOS deterministically under both
        #    proposal and target, so its incremental weight is 1 (log 0).  A
        #    slot that terminates this step sums only up to its EOS column, so
        #    post-EOS draft tokens — which are not part of the sequence — do
        #    not corrupt its importance weight.
        cols = torch.arange(n_weight_cols, device=self.device).unsqueeze(0)
        keep = cols <= weight_cutoff.unsqueeze(1)
        d = (logprob_diff.to(torch.float64) * keep).sum(dim=1)
        self.log_weights[active] += d
        self.interval_weights[active] += d

        # f. Optional rebuild so callers can batch it with a subsequent
        #    resample (which may also flip membership).
        if newly_finished and rebuild_active:
            self.rebuild_active_slots()

        return newly_finished

    # ────────────────────────────────────────────────────────
    #  Resampling helpers
    # ────────────────────────────────────────────────────────

    def resample_copy_slot(self, dst_slot: int, src_slot: int) -> None:
        """Python fallback for a single dst←src resample copy.

        Not on the hot path after the refactor — the fused resample kernel
        handles the KV/tensor copies — but kept as a reference and for
        any offline tooling that exercises one-at-a-time copies.  Moves
        the sequence-level tensors and the KV block table (with refcount
        adjustments), then the Req-level metadata.
        """
        old_dst_alloc = int(self.kv_allocated_lens[dst_slot].item())
        src_seq_len = int(self.seq_lens[src_slot].item())

        self.seq_lens[dst_slot] = self.seq_lens[src_slot]
        self.kv_allocated_lens[dst_slot] = self.kv_allocated_lens[src_slot]
        self.verified_ids[dst_slot] = self.verified_ids[src_slot]
        self.prev_last_draft_ids[dst_slot] = self.prev_last_draft_ids[src_slot]
        self.finished_mask[dst_slot] = self.finished_mask[src_slot]

        src_count = int(self.token_counts[src_slot].item())
        self.token_counts[dst_slot] = src_count
        if src_count > 0:
            self.all_token_ids[dst_slot, :src_count] = (
                self.all_token_ids[src_slot, :src_count]
            )

        src_pool = int(self.req_pool_indices[src_slot].item())
        dst_pool = int(self.req_pool_indices[dst_slot].item())

        if old_dst_alloc > 0:
            old_indices = self.req_to_token_pool.req_to_token[
                dst_pool, :old_dst_alloc
            ].to(dtype=torch.int64, copy=True)
            self.token_to_kv_pool_allocator.dec_ref_and_free(old_indices)

        if src_seq_len > 0:
            src_indices = self.req_to_token_pool.req_to_token[
                src_pool, :src_seq_len
            ].to(dtype=torch.int64, copy=True)
            self.req_to_token_pool.write(
                (dst_pool, slice(0, src_seq_len)),
                src_indices.to(dtype=torch.int32),
            )
            self.token_to_kv_pool_allocator.inc_ref(src_indices)

        self.copy_req_metadata(dst_slot, src_slot)

    def copy_req_metadata(self, dst_slot: int, src_slot: int) -> None:
        """Copy the Req-level text state from src to dst.

        Invoked by the fast-path dispatcher after the fused kernel has
        already copied every on-device tensor.  Mirrors the fields the
        tokenizer / stream-output pipeline reads at finalize time.
        """
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
    #  Unbiased log-Z bookkeeping
    # ────────────────────────────────────────────────────────

    def resample_logZ_increment(self) -> torch.Tensor:
        """Per-row log Z_hat increment for the current step, computed BEFORE the
        resample kernel zeroes weights.

        For each in-use row r, ``logsumexp(interval_weights[slots_r]) - log(N)``;
        free rows return 0.  The caller adds this — masked by the rows that
        actually resampled — into ``group_log_Z_hat``, which realises the
        unbiased SMC estimator's product over resample boundaries.  Free rows'
        ``group_to_slots`` cells are -1, indexing the last weight entry, but
        the caller masks those rows out via the kernel's ``resample_mask``.
        """
        iw_rows = self.interval_weights[self.group_to_slots.to(torch.int64)]
        inc = torch.logsumexp(iw_rows, dim=1) - math.log(self.n_particles)
        return torch.where(self.row_in_use, inc, torch.zeros_like(inc))

    # ────────────────────────────────────────────────────────
    #  Finalization
    # ────────────────────────────────────────────────────────

    def finalize_group(self, group_id: str, parent_req: Req) -> Req:
        """Finalize an SMC group: keep the posterior-sampled particle as the
        primary output AND attach the full particle collection + unbiased
        log Z_hat to ``parent_req``.

        ``parent_req.output_ids`` / ``finished_reason`` hold one particle drawn
        from the posterior P(slot) ∝ exp(log_weights[slot]), preserving the
        single-sequence API.  Additionally sets, for the whole group:

          * ``smc_log_Z_hat``           — unbiased log normalizing-constant est.,
            the running per-resample product closed out with the final tail
            boundary ``logsumexp(interval_weights) - log(N)``.
          * ``smc_log_w_tilde``         — final per-particle log-weights.
          * ``smc_particle_output_ids`` — every particle's output token ids.

        Frees all group slots and returns ``parent_req`` ready for
        ``stream_output``.
        """
        slots = self.group_slot_lists[group_id]
        slot_idx_t = torch.tensor(slots, dtype=torch.int64, device=self.device)

        # Collection-level unbiased log Z_hat: accumulated per-resample product
        # (group_log_Z_hat) + the final tail since the last resample.  Every
        # slot holds a finished sequence here (the group only drains once no
        # particle is active), and resample copies carry output_ids via
        # copy_req_metadata, so each slot reflects its surviving lineage.
        row = self.group_id_to_row.get(group_id)
        base = self.group_log_Z_hat[row] if row is not None else 0.0
        tail = torch.logsumexp(
            self.interval_weights[slot_idx_t], dim=0
        ) - math.log(self.n_particles)
        log_Z_hat = float((base + tail))
        log_w_tilde = [float(x) for x in self.log_weights[slot_idx_t].tolist()]
        particle_output_ids = [
            list(self.slot_to_req[s].output_ids) for s in slots
        ]

        # Posterior sample over particles for the primary output. softmax
        # handles the max-shift for numerical stability; multinomial respects
        # the global torch RNG (seeded via ServerArgs.random_seed).
        probs = torch.softmax(self.log_weights[slot_idx_t], dim=0)
        pick = int(torch.multinomial(probs, num_samples=1).item())
        best_slot = slots[pick]
        best_req = self.slot_to_req[best_slot]
        parent_req.output_ids = list(best_req.output_ids)
        if best_req.finished_reason is not None:
            parent_req.finished_reason = copy.copy(best_req.finished_reason)
            parent_req.finished_len = best_req.finished_len
        else:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT
            parent_req.finished_reason = FINISH_ABORT(
                "SMC group finalized without a finished particle."
            )
            parent_req.finished_len = len(parent_req.output_ids)

        parent_req.smc_log_Z_hat = log_Z_hat
        parent_req.smc_log_w_tilde = log_w_tilde
        parent_req.smc_particle_output_ids = particle_output_ids

        self.free_group_slots(group_id)
        return parent_req

    # ────────────────────────────────────────────────────────
    #  Group Queries
    # ────────────────────────────────────────────────────────

    def group_has_active(self, group_id: str) -> bool:
        slots = self.group_slot_lists.get(group_id, [])
        return any(not self.finished_mask[s].item() for s in slots)

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

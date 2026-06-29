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
* ``active_slots`` contains EVERY allocated slot, finished or not
  (absorbing-state semantics): finished particles keep receiving forward
  passes with their weight increment zeroed, still participate in
  resampling (their cumulative weight is part of the SMC mixture), and may
  be overwritten as the destination of a resample copy.  Membership only
  changes at ``allocate_slots`` / ``free_group_slots``.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass
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


@dataclass
class HostSnapshot:
    """Handle for one decode step's host snapshot.

    Carries the double-buffer phase the step used and the CUDA event that
    signals its pinned copies have landed.  Postprocessing calls ``wait()``
    — which completes early, during the *next* step's forward — instead of
    reading device tensors at the stream tail (which would block until all
    enqueued work, including the next forward, finished).
    """

    phase: int
    event: Optional["torch.cuda.Event"]

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()


class ScheduleBatchSMC:
    """Slot-major SMC batch state.

    See module docstring for the layout and invariants.  Callers touch two
    coarse primitives:

    * ``allocate_slots`` / ``free_group_slots`` — rare, at group materialise
      and finalize.
    * ``prepare_for_decode`` + ``build_model_worker_batch`` +
      ``write_back_gpu`` — hot path, once per decode step.
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

        # Whether host->device staging tensors can be pinned (CUDA only) so
        # the admission/teardown uploads are truly async.  See
        # ``_to_device_async``.
        self._pin_host = torch.device(device).type == "cuda"
        # Fused write-back (one triton launch) on CUDA; torch fallback
        # otherwise or via SMC_FUSED_WRITE_BACK=0.
        self._use_fused_write_back = self._pin_host and bool(
            int(os.environ.get("SMC_FUSED_WRITE_BACK", "1"))
        )

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
        # Per-slot token at position S-1: the last *drafted* token d_{gamma-1}
        # from the previous step (deferred into the next step's leading
        # 2-token draft forward), or — on a group's first decode step — the
        # last committed prompt token, whose draft KV the head rewrites
        # idempotently.  Travels with lineage exactly like verified_ids.
        self.prev_last_draft_ids = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.token_counts = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.finished_mask = torch.zeros(
            self.max_slots, dtype=torch.bool, device=device
        )
        # Finish state, tensor-resident so it travels with lineage through
        # the resample copy and is read back only at finalize time (no
        # per-step Req-object sync).  `finished_len` is the visible output
        # length; `finish_reason_code` is 0=running, 1=length, 2=EOS;
        # `matched_eos_token` holds the matched id when code==2.
        self.finished_len = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
        )
        self.finish_reason_code = torch.zeros(
            self.max_slots, dtype=torch.int8, device=device
        )
        self.matched_eos_token = torch.zeros(
            self.max_slots, dtype=torch.int32, device=device
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

        # CPU mirror of ``seq_lens``, maintained arithmetically: set at
        # ``allocate_slots``, advanced by ``gamma_plus_1`` per decode step.
        # Resampling never changes it — particles within a group are
        # step-aligned, so ``seq_lens[dst] = seq_lens[src]`` is a no-op on
        # lengths.  Lets batch construction provide ``seq_lens_cpu`` /
        # ``seq_lens_sum`` (and the KV alloc count) without a device read,
        # which is what keeps the prepare path sync-free for overlapped
        # scheduling.  Verified against the device tensor when
        self.seq_lens_host = torch.zeros(self.max_slots, dtype=torch.int64)

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
        self.active_slots_cpu = torch.empty(0, dtype=torch.int64)
        self._active_slots_list: List[int] = []
        self.num_active: int = 0
        # ModelWorkerBatch cache: everything but the per-cycle fields is
        # static between membership changes (issue #14, host-op slimming).
        self._membership_version: int = 0
        self._mwb_cache = None
        self._mwb_version: int = -1

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

        # ── Freed-page capture + host snapshot (double-buffered) ──
        # The resample kernel appends KV pages whose refcount hit zero to
        # `kv_freed_buf[phase]` through the `kv_freed_counter[phase]`
        # atomic cursor.  At the end of `_resample`, `snapshot_to_host`
        # async-copies the cursor and `finished_mask` into pinned host
        # buffers and records an event; postprocessing waits on that event
        # (NOT the stream tail), frees `kv_freed_buf[phase, :n]`, resets
        # the cursor, and drains groups from the host mask.  Two phases
        # because under overlapped scheduling step t+1's dispatch/snapshot
        # are enqueued BEFORE step t's postprocessing runs — a single
        # buffer would be clobbered before it was consumed.  Buffer rows
        # are sized to the whole KV pool — the per-step bound on frees.
        pool_cap = self.token_to_kv_pool_allocator.size + 1
        is_cuda = torch.device(device).type == "cuda"
        self._snap_phase = 0
        self.kv_freed_buf = torch.empty(
            (2, pool_cap), dtype=torch.int32, device=device
        )
        self.kv_freed_counter = torch.zeros(
            (2, 1), dtype=torch.int32, device=device
        )
        self.kv_freed_count_host = torch.zeros(
            (2, 1), dtype=torch.int32, pin_memory=is_cuda
        )
        self.finished_mask_host = torch.zeros(
            (2, self.max_slots), dtype=torch.bool, pin_memory=is_cuda
        )
        self._snap_events = [
            torch.cuda.Event() if is_cuda else None for _ in range(2)
        ]

    # ────────────────────────────────────────────────────────
    #  Slot Allocation / Deallocation
    # ────────────────────────────────────────────────────────

    def _to_device_async(self, values, dtype: torch.dtype) -> torch.Tensor:
        """Upload a host list to ``self.device`` without a blocking sync.

        Stages through pinned memory so the H2D copy is truly async on
        CUDA; the pinned caching host allocator keeps the staging block
        alive until the copy's stream event fires, so the returned tensor
        is safe to use even though the source goes out of scope here.  On
        CPU/non-pinnable devices this degrades to a plain transfer.
        """
        host = torch.tensor(values, dtype=dtype)
        if self._pin_host:
            host = host.pin_memory()
        return host.to(self.device, non_blocking=self._pin_host)

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

        Sync-free transport: per-particle Python work only appends to host
        lists; the device lands in a handful of async ops — ``index_fill_``
        for fields uniform across the group (scalar baked into the launch),
        pinned async uploads + one indexed scatter for varying / 2-D
        fields.  Replaces ~21 blocking ``tensor[slot] = scalar`` H2D copies
        per particle, which otherwise stall prefill postprocessing.
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

        # ── Pure-CPU gather: per-particle values into host lists ──
        pool_idx_list: List[int] = []
        verified_list: List[int] = []
        token_count_list: List[int] = []
        ignore_eos_list: List[bool] = []
        max_new_tokens_list: List[int] = []
        eos_rows: List[List[int]] = []
        max_n_out = 0
        for slot, req in zip(slots, particle_reqs):
            self.slot_to_req[slot] = req
            self.seq_lens_host[slot] = shared_seq_len

            pool_idx_list.append(req.req_pool_idx)
            verified_list.append(req.output_ids[-1] if req.output_ids else 0)
            token_count_list.append(len(req.output_ids))
            ignore_eos_list.append(bool(req.sampling_params.ignore_eos))
            max_new_tokens_list.append(req.sampling_params.max_new_tokens)
            max_n_out = max(max_n_out, len(req.output_ids))

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
            eos_rows.append(
                [eos_ids[j] if j < len(eos_ids) else -1
                 for j in range(self.max_eos_count)]
            )

        idx = self._to_device_async(slots, torch.int64)  # one upload, reused

        # ── Category 1: fields uniform across the group → index_fill_ ──
        self.seq_lens.index_fill_(0, idx, shared_seq_len)
        self.kv_allocated_lens.index_fill_(0, idx, shared_seq_len)
        # Seed with the LAST COMMITTED token (position S-1 of the shared
        # prefix; uniform across the group — every particle clones the same
        # parent).  Its draft KV at S-1 was already written during prefill,
        # so the deferred-bonus 2-token head's S-1 write is an idempotent
        # rewrite on a group's first decode step — the head is universally
        # valid and needs no step-0 special case.  (A -1 sentinel + batch
        # global head selection was tried before and crashes whenever a new
        # group joins a batch of already-decoding groups: the sentinel
        # reaches the embedding as a token id.)
        first_req = particle_reqs[0]
        committed = first_req.origin_input_ids + first_req.output_ids
        last_committed_token = int(committed[shared_seq_len - 1])
        self.prev_last_draft_ids.index_fill_(0, idx, last_committed_token)
        self.finished_mask.index_fill_(0, idx, 0)
        self.finished_len.index_fill_(0, idx, 0)
        self.finish_reason_code.index_fill_(0, idx, 0)
        self.matched_eos_token.index_fill_(0, idx, 0)
        self.log_weights.index_fill_(0, idx, 0.0)
        self.interval_weights.index_fill_(0, idx, 0.0)

        # ── Category 2: per-particle scalars → pinned upload + scatter ──
        self.req_pool_indices[idx] = self._to_device_async(
            pool_idx_list, torch.int64
        )
        self.verified_ids[idx] = self._to_device_async(verified_list, torch.int32)
        self.token_counts[idx] = self._to_device_async(
            token_count_list, torch.int32
        )
        self.ignore_eos_t[idx] = self._to_device_async(ignore_eos_list, torch.bool)
        self.max_new_tokens_t[idx] = self._to_device_async(
            max_new_tokens_list, torch.int32
        )

        # ── Category 3: 2-D fields → one matrix upload each ──
        self.eos_token_ids_t[idx] = self._to_device_async(
            eos_rows, torch.int64
        )
        if max_n_out > 0:
            # Right-pad ragged prefixes to a rectangle; only [:n_out] of
            # each row is meaningful (token_counts gates later reads).
            prefix_rows = [
                req.output_ids + [0] * (max_n_out - len(req.output_ids))
                for req in particle_reqs
            ]
            self.all_token_ids[idx, :max_n_out] = self._to_device_async(
                prefix_rows, torch.int32
            )

        self.group_slot_lists[group_id] = slots

        # ── Group lookup row + per-row scalars ──
        self.group_to_slots[row, :n] = idx.to(torch.int32)
        row_idx = self._to_device_async([row], torch.int64)
        self.row_in_use.index_fill_(0, row_idx, 1)
        self.group_log_Z_hat.index_fill_(0, row_idx, 0.0)

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
            # fill_ / index_fill_ keep the scalar in the kernel launch — no
            # blocking H2D (cf. the `tensor[idx] = scalar` gotcha).
            self.group_to_slots[row].fill_(-1)
            row_idx = self._to_device_async([row], torch.int64)
            self.row_in_use.index_fill_(0, row_idx, 0)
            self.group_log_Z_hat.index_fill_(0, row_idx, 0.0)
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

            self.seq_lens_host[slot] = 0
            self.slot_to_req.pop(slot, None)
            self.free_slots.append(slot)

        # Clear all released slots' device tensors to sentinels in one shot
        # per field (index_fill_, scalar in the kernel launch — no blocking
        # H2D per slot).  The per-slot .item() reads above stay; they are
        # the accepted finalize bubble.
        if slots:
            idx = self._to_device_async(slots, torch.int64)
            self.req_pool_indices.index_fill_(0, idx, EMPTY_SLOT)
            self.seq_lens.index_fill_(0, idx, 0)
            self.kv_allocated_lens.index_fill_(0, idx, 0)
            self.verified_ids.index_fill_(0, idx, 0)
            self.prev_last_draft_ids.index_fill_(0, idx, 0)
            self.token_counts.index_fill_(0, idx, 0)
            self.finished_mask.index_fill_(0, idx, 0)
            self.finished_len.index_fill_(0, idx, 0)
            self.finish_reason_code.index_fill_(0, idx, 0)
            self.matched_eos_token.index_fill_(0, idx, 0)
            self.ignore_eos_t.index_fill_(0, idx, 0)
            self.log_weights.index_fill_(0, idx, 0.0)
            self.interval_weights.index_fill_(0, idx, 0.0)

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
        zeroed in ``write_back_gpu`` step (d) — the absorbing state
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
        # CPU twin of `active_slots` for gathering host-shadow tensors
        # (seq_lens_host) without a device round trip.
        self.active_slots_cpu = torch.tensor(active_list, dtype=torch.int64)
        self._active_slots_list = active_list
        self.num_active = len(active_list)
        # Invalidate the cached ModelWorkerBatch (membership changed).
        self._membership_version += 1

    def is_empty(self) -> bool:
        return self.num_active == 0

    # ────────────────────────────────────────────────────────
    #  Decode Preparation (sparse → vectorized KV alloc → sparse)
    # ────────────────────────────────────────────────────────

    def prepare_for_decode(self) -> SMCDraftInput:
        """Allocate KV for the cycle and produce the worker's inputs via
        ONE fused kernel (issue #14, host-op slimming).

        The fused kernel does, per active row: the slot gathers
        (seq/verified/prev), the block-table write of the freshly-allocated
        pages, and the seq/kv-alloc advance — replacing the ~12 separate
        ops of the previous gather → assign → scatter sequence.  Under the
        ``kv_allocated_lens == seq_lens`` invariant every row takes exactly
        ``gamma+1`` pages, so the allocated page tensor IS the per-row
        cache-locs table (carried on the ctx; nothing re-reads the block
        table).
        """
        if self.num_active == 0:
            return SMCDraftInput(
                verified_id=torch.empty(0, dtype=torch.int32, device=self.device),
                num_tokens_per_req=self.gamma_plus_1,
            )

        from sglang.srt.mem_cache.common import alloc_token_slots
        from smcsd.core.kernels.fused_prepare import fused_prepare_decode

        active = self.active_slots
        bs = self.num_active

        # Host shadow gather — provides every CPU-side scalar the batch
        # build needs without reading the device tensors (which, under
        # overlapped scheduling, may not be computed yet).
        seq_lens_cpu_g = self.seq_lens_host[self.active_slots_cpu]

        pages = alloc_token_slots(self.tree_cache, bs * self.gamma_plus_1)
        orig_seq_lens, verified_g, prev_last_draft_g = fused_prepare_decode(
            active,
            self.seq_lens,
            self.kv_allocated_lens,
            self.req_pool_indices,
            self.verified_ids,
            self.prev_last_draft_ids,
            pages,
            self.req_to_token_pool.req_to_token,
            self.gamma_plus_1,
        )
        self.seq_lens_host[self.active_slots_cpu] = (
            seq_lens_cpu_g + self.gamma_plus_1
        )

        ctx = SMCDecodeContext(
            orig_seq_lens=orig_seq_lens,
            orig_seq_lens_cpu=seq_lens_cpu_g,
            orig_seq_lens_sum=int(seq_lens_cpu_g.sum().item()),
            new_seq_lens=orig_seq_lens + self.gamma_plus_1,
            gamma=self.gamma_plus_1 - 1,
            cache_locs=pages.view(bs, self.gamma_plus_1),
            # Per-row generated count BEFORE this block (the ramp's t). Snapshot
            # now: token_counts is advanced later by write_back_gpu. The worker
            # derives the exponent-bridging alpha(t) from this.
            gen_lens=self.token_counts[active].clone(),
        )
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
        """Assemble a contiguous ``ModelWorkerBatch`` for the worker.

        Under static membership everything except the per-cycle fields
        (input_ids / seq_lens / seq_lens_cpu / seq_lens_sum / spec_info) is
        identical between membership changes, so the batch object — incl.
        the req_pool_indices gather, the Python ``reqs`` list, and the stub
        SamplingBatchInfo — is cached and only those five fields are
        refreshed per cycle (issue #14, host-op slimming).  Safe to mutate
        in place: the previous cycle's consumers never read the batch after
        launch (the overlap queue stores it only for prefill entries).
        """
        ctx = draft_input.decode_ctx
        # CPU values from the host shadow — no device read.  The shadow was
        # already advanced by prepare_for_decode, so it equals new_seq_lens.
        seq_lens_cpu = self.seq_lens_host[self.active_slots_cpu]
        seq_lens_sum = int(seq_lens_cpu.sum().item())

        cached = self._mwb_cache
        if cached is not None and self._mwb_version == self._membership_version:
            cached.input_ids = draft_input.verified_id
            cached.seq_lens = (
                ctx.new_seq_lens if ctx is not None
                else self.seq_lens[self.active_slots]
            )
            cached.seq_lens_cpu = seq_lens_cpu
            cached.seq_lens_sum = seq_lens_sum
            cached.spec_info = draft_input
            return cached

        active = self.active_slots
        bs = self.num_active
        req_pool_indices = self.req_pool_indices[active]
        seq_lens = ctx.new_seq_lens if ctx is not None else self.seq_lens[active]
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

        self._mwb_version = self._membership_version
        self._mwb_cache = ModelWorkerBatch(
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
        return self._mwb_cache

    # ────────────────────────────────────────────────────────
    #  Process Batch Result (write-back from forward pass)
    # ────────────────────────────────────────────────────────

    def write_back_gpu(
        self,
        next_token_ids: torch.Tensor,
        logprob_diff: torch.Tensor,
        bonus_ids: torch.Tensor,
        *,
        prev_last_draft_ids: Optional[torch.Tensor] = None,
        bonus_logz: Optional[torch.Tensor] = None,
    ) -> None:
        """Write forward-pass results back to slot-indexed tensors.

        Sync-free by construction: every op here is a GPU tensor op — no
        ``.item()`` / ``.tolist()`` / Python truth-value of a CUDA tensor —
        so the whole write-back (and the resample kernels that consume
        ``interval_weights`` immediately after) can be enqueued behind the
        decode forward without a host round-trip.

        Order of operations:

        a. Scatter accepted tokens into ``all_token_ids``; bump ``token_counts``.
        b. Overwrite ``verified_ids`` with next-step bonus tokens.
        c. Check finish conditions (length, EOS) batched on GPU and record
           them tensor-resident (``finished_mask`` / ``finished_len`` /
           ``finish_reason_code`` / ``matched_eos_token``).  Req objects
           are NOT touched; ``finalize_group`` reads the tensors lazily.
        d. Accumulate ``logprob_diff`` into the slot-indexed
           ``log_weights`` / ``interval_weights``.

        On CUDA this is ONE fused triton launch (one program per row); the
        torch implementation below remains as the reference / CPU fallback
        (kill-switch: SMC_FUSED_WRITE_BACK=0).
        """
        if self._use_fused_write_back:
            from smcsd.core.kernels.fused_write_back import fused_write_back

            fused_write_back(
                self.active_slots,
                next_token_ids,
                logprob_diff,
                bonus_ids,
                prev_last_draft_ids,
                all_token_ids=self.all_token_ids,
                token_counts=self.token_counts,
                verified_ids=self.verified_ids,
                prev_ids=self.prev_last_draft_ids,
                finished_mask=self.finished_mask,
                finished_len=self.finished_len,
                finish_reason_code=self.finish_reason_code,
                matched_eos_token=self.matched_eos_token,
                ignore_eos=self.ignore_eos_t,
                max_new_tokens=self.max_new_tokens_t,
                eos_token_ids=self.eos_token_ids_t,
                log_weights=self.log_weights,
                interval_weights=self.interval_weights,
                gamma_plus_1=self.gamma_plus_1,
                bonus_logz=bonus_logz,
            )
            return

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
        # slot's EOS ids.  Skipped for slots with ignore_eos=True.  EOS-id
        # padding is -1, which no token id can match.
        eos_ids = self.eos_token_ids_t[active]
        eos_match = (
            accepted_2d.unsqueeze(2).to(torch.int64) == eos_ids.unsqueeze(1)
        ).any(dim=2)                                          # (bs, stride)
        eos_hit = eos_match.any(dim=1) & ~self.ignore_eos_t[active]

        prev_finished_active = self.finished_mask[active]
        newly_finished_mask = (length_hit | eos_hit) & ~prev_finished_active
        self.finished_mask[active] = prev_finished_active | newly_finished_mask

        # First EOS column in this step's block (== stride where no EOS):
        # min over masked positions, so "first occurrence" is guaranteed.
        # Only consumed under eos-branch masks below.
        positions = torch.arange(stride, dtype=torch.int64, device=self.device)
        first_eos = torch.where(eos_match, positions, stride).min(dim=1).values
        matched_tok = accepted_2d.gather(
            1, first_eos.clamp(max=stride - 1).unsqueeze(1)
        ).squeeze(1)

        # Finish bookkeeping, tensor-resident so it travels with lineage
        # through the resample copy and is read back lazily at finalize.
        # Length finish takes precedence over EOS (historical behaviour):
        # finished_len is max_new_tokens for length, or the position right
        # after the EOS token for EOS.
        eos_branch = newly_finished_mask & ~length_hit
        fin_len = torch.where(
            length_hit,
            max_tokens,
            (
                updated_counts.to(torch.int64) - stride + first_eos + 1
            ).to(max_tokens.dtype),
        )
        fin_code = torch.where(length_hit, 1, 2).to(self.finish_reason_code.dtype)
        self.finished_len[active] = torch.where(
            newly_finished_mask, fin_len, self.finished_len[active]
        )
        self.finish_reason_code[active] = torch.where(
            newly_finished_mask, fin_code, self.finish_reason_code[active]
        )
        self.matched_eos_token[active] = torch.where(
            eos_branch,
            matched_tok.to(torch.int32),
            self.matched_eos_token[active],
        )

        # Per-particle inclusive cutoff column for the weight sum below.
        # logprob_diff has `gamma` columns (the drafted positions); a particle
        # accrues weight from columns 0..cutoff inclusive.  Default keeps the
        # whole block; a particle that terminates via EOS at block column j
        # keeps only 0..j (the EOS token itself is a real sample; tokens after
        # it are post-EOS draft junk and must not contribute — EOS is an
        # absorbing state with incremental weight 1).  An EOS in the bonus
        # slot keeps the full block (the clamp).  Already-finished particles
        # keep nothing (cutoff -1).
        n_weight_cols = logprob_diff.shape[1]
        weight_cutoff = torch.full(
            (bs,), n_weight_cols - 1, dtype=torch.int64, device=self.device
        )
        # For the weight cutoff, EOS takes precedence even when the length
        # cap is hit in the same block (unlike the finish *reason*, where
        # length wins historically): once the sequence emitted EOS, the
        # later columns are post-EOS draft junk regardless of which finish
        # reason gets reported.
        eos_cut = newly_finished_mask & eos_hit
        weight_cutoff = torch.where(
            eos_cut, first_eos.clamp(max=n_weight_cols - 1), weight_cutoff
        )
        weight_cutoff = torch.where(
            prev_finished_active,
            torch.full_like(weight_cutoff, -1),
            weight_cutoff,
        )

        # d. Accumulate log-weights.  `logprob_diff` is per-position
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

        # Bonus-token normalizer log Z (joint-power target; 0 at alpha=1).  The
        # bonus is part of the sequence — and so weighted — unless the particle
        # was already finished or terminated via EOS within the draft columns
        # 0..gamma-1 (an EOS in the bonus column itself still emits the bonus, so
        # first_eos == n_weight_cols does NOT drop it).  Mirrors logprob_diff's
        # EOS-cutoff convention: length-only termination keeps the full block.
        if bonus_logz is not None:
            eos_in_draft = eos_cut & (first_eos < n_weight_cols)
            add_bonus = (~prev_finished_active & ~eos_in_draft).to(torch.float64)
            d = d + bonus_logz.to(torch.float64) * add_bonus

        self.log_weights[active] += d
        self.interval_weights[active] += d

    # ────────────────────────────────────────────────────────
    #  Host snapshot (postprocessing inputs, no stream-tail block)
    # ────────────────────────────────────────────────────────

    def snapshot_to_host(self) -> HostSnapshot:
        """Async-copy this step's postprocessing inputs (freed-page cursor,
        finished mask) into the current phase's pinned buffers, record the
        snapshot event, and flip the phase for the next step.

        Enqueue-only — call at the end of the GPU-side resample step.
        Postprocessing waits on the returned handle's event, reads the
        pinned buffers, frees ``kv_freed_buf[phase, :n]``, and zeroes
        ``kv_freed_counter[phase]`` (which is then stream-ordered before
        that phase's next dispatch).
        """
        p = self._snap_phase
        self.kv_freed_count_host[p].copy_(
            self.kv_freed_counter[p], non_blocking=True
        )
        self.finished_mask_host[p].copy_(self.finished_mask, non_blocking=True)
        ev = self._snap_events[p]
        if ev is not None:
            ev.record()
        self._snap_phase = 1 - p
        return HostSnapshot(phase=p, event=ev)

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

    @staticmethod
    def _finish_reason_from_code(code: int, fin_len: int, matched_tok: int):
        """Materialise a BaseFinishReason from the tensor-resident finish
        state (``finish_reason_code`` semantics: 0=running, 1=length, 2=EOS).
        """
        from sglang.srt.managers.schedule_batch import (
            FINISH_ABORT,
            FINISH_LENGTH,
            FINISH_MATCHED_TOKEN,
        )

        if code == 1:
            return FINISH_LENGTH(length=fin_len)
        if code == 2:
            return FINISH_MATCHED_TOKEN(matched=matched_tok)
        return FINISH_ABORT("SMC group finalized without a finished particle.")

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
        # particle is active), and the resample copy carries the finish-state
        # tensors (all_token_ids / finished_len / finish_reason_code), so each
        # slot reflects its surviving lineage.
        row = self.group_id_to_row.get(group_id)
        base = self.group_log_Z_hat[row] if row is not None else 0.0
        tail = torch.logsumexp(
            self.interval_weights[slot_idx_t], dim=0
        ) - math.log(self.n_particles)
        log_Z_hat = float((base + tail))
        log_w_tilde = [float(x) for x in self.log_weights[slot_idx_t].tolist()]

        # Per-particle results read lazily from the slot tensors — the first
        # (and only) time decode-time finish state crosses to the host.
        fin_lens = self.finished_len[slot_idx_t].tolist()
        fin_codes = self.finish_reason_code[slot_idx_t].tolist()
        matched_toks = self.matched_eos_token[slot_idx_t].tolist()
        particle_output_ids = [
            self.all_token_ids[s, :n].tolist() for s, n in zip(slots, fin_lens)
        ]

        # Posterior sample over particles for the primary output. softmax
        # handles the max-shift for numerical stability; multinomial respects
        # the global torch RNG (seeded via ServerArgs.random_seed).
        probs = torch.softmax(self.log_weights[slot_idx_t], dim=0)
        pick = int(torch.multinomial(probs, num_samples=1).item())
        parent_req.output_ids = list(particle_output_ids[pick])
        parent_req.finished_reason = self._finish_reason_from_code(
            fin_codes[pick], fin_lens[pick], matched_toks[pick]
        )
        parent_req.finished_len = (
            fin_lens[pick]
            if fin_codes[pick] != 0
            else len(parent_req.output_ids)
        )

        parent_req.smc_log_Z_hat = log_Z_hat
        parent_req.smc_log_w_tilde = log_w_tilde
        parent_req.smc_particle_output_ids = particle_output_ids

        self.free_group_slots(group_id)
        return parent_req

    # ────────────────────────────────────────────────────────
    #  Group Queries
    # ────────────────────────────────────────────────────────

    def group_has_active(
        self, group_id: str, finished_mask_host: torch.Tensor
    ) -> bool:
        """Whether any particle of ``group_id`` is unfinished, read from a
        pinned host snapshot (``finished_mask_host[snapshot.phase]``) —
        never from the device tensor, which would block on the stream tail
        under overlapped scheduling."""
        slots = self.group_slot_lists.get(group_id, [])
        return any(not bool(finished_mask_host[s]) for s in slots)

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

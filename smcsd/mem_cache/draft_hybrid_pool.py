"""Per-draft view of a HybridReqToTokenPool for SMC speculative decoding.

When the *target* model is hybrid (Mamba/SSM + attention) we already wire
its ``HybridReqToTokenPool`` through the SMC scheduler's slot machinery —
the per-token KV block table sits in ``req_to_token`` and the per-request
recurrent state lives in ``mamba_pool``. That works fine while the *draft*
is dense.

If the draft is *also* hybrid (e.g. Qwen3.5-9B drafting for Qwen3.6-27B),
the dense path falls apart: the draft model's Mamba layers expect their
own conv/temporal state shapes, which generally don't match the target's
(different ``hidden_size``, ``num_v_heads``, etc.). Reusing the target's
single ``MambaPool`` causes the draft to read/write tensors with the
wrong dim and the conv1d kernel asserts:

    causal_conv1d_triton.py:463
    assert (num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2])  # ← fires

``DraftHybridReqToTokenPool`` solves this by giving the draft worker its
own ``MambaPool`` (sized to the DRAFT's state shapes) and its own
``req_index_to_mamba_index_mapping``, while *aliasing* the target's
``req_to_token`` block table and ``free_slots``. So:

  * Slot indices stay coherent — target.req_to_token IS draft.req_to_token,
    SMC's ``copy_block_table`` / fused resample kernel see one truth.
  * The flat-KV allocator (``token_to_kv_pool_allocator``) is shared
    between target and draft, as before.
  * Draft mamba state is independent and correctly shaped.

Mamba slot allocation/free for the draft is *not* tied to the request
pool's ``alloc(reqs)`` call — that's owned by the target. Smcsd
explicitly calls ``alloc_mamba_for_reqs`` after a particle is materialised
and ``free_mamba_cache`` on release. ``req.draft_mamba_pool_idx`` carries
the draft-side index, parallel to upstream's ``req.mamba_pool_idx``.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool


class DraftHybridReqToTokenPool(HybridReqToTokenPool):
    """A HybridReqToTokenPool whose req-pool side mirrors a parent (target)
    pool, but whose Mamba/SSM state pool is fully independent.

    Constructed only by SMC's draft worker when both target and draft are
    hybrid models with potentially-mismatched mamba state shapes.
    """

    def __init__(
        self,
        target_pool: HybridReqToTokenPool,
        *,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: Optional[int],
        enable_overlap_schedule: bool,
        start_layer: Optional[int] = None,
    ) -> None:
        # Skip HybridReqToTokenPool.__init__ — we don't want to allocate
        # another req_to_token tensor (we'll alias the target's). Set up
        # the minimal state HybridReqToTokenPool / ReqToTokenPool consumers
        # rely on, then build our own MambaPool + index mapping.
        self._target_pool = target_pool

        # Share with target (no copy): SMC's slot machinery operates on
        # the same KV block table and the same free-slot list.
        self.req_to_token = target_pool.req_to_token
        self.size = target_pool.size
        self.free_slots = target_pool.free_slots
        self.pre_alloc_size = getattr(target_pool, "pre_alloc_size", 0)
        self.max_context_len = target_pool.max_context_len
        self.device = device
        self.enable_memory_saver = enable_memory_saver
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None

        # Draft's own MambaPool — sized for the DRAFT's state shapes.
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        self.mamba_map = {
            layer_id: i for i, layer_id in enumerate(mamba_layer_ids)
        }

        # Per-req mapping into draft's mamba pool. Parallel to the target
        # pool's req_index_to_mamba_index_mapping but independent —
        # different mamba slot for the same req_pool_idx.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=device
        )

    # ------------------------------------------------------------------
    # Allocation: the *target* pool owns req-pool slot allocation. The
    # draft only ever needs to alloc mamba slots, on demand, after the
    # target has assigned req_pool_idx.
    # ------------------------------------------------------------------

    def alloc(self, reqs):
        """The draft view never independently allocates request-pool slots.

        Smcsd allocates particle req-pool slots through the *target*'s
        ``HybridReqToTokenPool.alloc`` and then calls
        ``alloc_mamba_for_reqs`` on this draft pool. Calling ``alloc``
        here would consume from the shared ``free_slots`` twice and
        desynchronise the two views.
        """
        raise RuntimeError(
            "DraftHybridReqToTokenPool.alloc is not allowed. SMC must "
            "allocate request-pool slots via the target pool, then call "
            "alloc_mamba_for_reqs on the draft pool."
        )

    def alloc_mamba_for_reqs(self, reqs) -> None:
        """Allocate a fresh draft-mamba slot for each req that already has a
        ``req_pool_idx`` (assigned by the target's alloc).

        Stores the slot on ``req.draft_mamba_pool_idx`` (parallel to the
        target side's ``req.mamba_pool_idx``) and writes it into
        ``req_index_to_mamba_index_mapping[req_pool_idx]`` so the hybrid
        attention backend's ``get_mamba_indices`` calls return the right
        draft slot.
        """
        if not reqs:
            return
        mamba_indices: List[torch.Tensor] = []
        select_indices: List[int] = []
        for req in reqs:
            assert req.req_pool_idx is not None, (
                "alloc_mamba_for_reqs requires reqs to have req_pool_idx "
                "(target.alloc must run first)"
            )
            mid = self.mamba_pool.alloc(1)
            assert mid is not None, (
                "draft mamba pool exhausted; raise --max-mamba-cache-size "
                "or increase --mamba-full-memory-ratio for the draft worker"
            )
            mid = mid[0]
            req.draft_mamba_pool_idx = mid
            mamba_indices.append(mid)
            select_indices.append(int(req.req_pool_idx))
        select_t = torch.tensor(
            select_indices,
            dtype=torch.int64,
            device=self.req_index_to_mamba_index_mapping.device,
        )
        mamba_t = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_t] = mamba_t

    def free(self, req):  # type: ignore[override]
        # The base class would mutate the *shared* free_slots list; the
        # target pool owns that. The draft view should only release its
        # own mamba state — see ``free_mamba_cache``.
        return

    def free_mamba_cache(  # type: ignore[override]
        self, req, mamba_ping_pong_track_buffer_to_keep=None
    ):
        idx = getattr(req, "draft_mamba_pool_idx", None)
        if idx is None:
            return
        self.mamba_pool.free(idx.unsqueeze(0))
        req.draft_mamba_pool_idx = None

    # ------------------------------------------------------------------
    # Read-side: HybridLinearAttnBackend pulls these. Override to use the
    # draft's mamba_pool / mapping rather than the target's.
    # ------------------------------------------------------------------

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map, (
            f"layer_id={layer_id} not in draft mamba_map "
            f"(draft mamba layers: {sorted(self.mamba_map)})"
        )
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self):
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear(self):
        # Don't clear req_to_token / free_slots — that's the target's job.
        self.mamba_pool.clear()
        self.req_index_to_mamba_index_mapping.zero_()

    def write(self, indices, values):
        # Block-table writes are made by attention backends via the
        # target's req_to_token tensor (which we alias). Forward the call
        # so any code path holding our reference still works correctly.
        self._target_pool.write(indices, values)

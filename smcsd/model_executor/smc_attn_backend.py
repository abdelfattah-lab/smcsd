"""SMC cascade-attention backend.

Wraps SGLang's :class:`FlashInferAttnBackend` and routes the SMC verify
path (``ForwardMode.TARGET_VERIFY``) through FlashInfer's
:class:`MultiLevelCascadeAttentionWrapper` when shared-prefix metadata
is present on ``forward_batch.spec_info``.

The cascade decomposition exploits an SMC-specific invariant: every
particle in a group shares the materialise-time prompt prefix (length
``L_g``) and physically points at the same KV pages (refcounted by
``SMCRefCountedTokenAllocator``).  At verify time, the N×(γ+1) queries
within a group can be ganged against that shared prefix once and merged
with each particle's private-suffix attention via online softmax — the
"Hydragen / Cascade Inference" pattern.

All non-verify paths (PREFILL, DRAFT_EXTEND, DECODE) delegate to the
inner FlashInfer backend unchanged.  When SMC verify enters and any of
the required shared-prefix metadata is missing (``shared_prefix_lens``
is None, or ``L_g == 0`` for some group), the backend falls back to the
inner verify path so correctness is never compromised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class _CascadePlan:
    """Per-cycle cascade attention plan.

    Keeps the wrapper, the live tensors that back its plan, and the
    queries' contiguous order so :meth:`SMCFlashInferAttnBackend.forward_extend`
    can permute Q in/out to match the cascade layout.
    """

    wrapper: object  # MultiLevelCascadeAttentionWrapper
    # Permutation that maps the forward_batch query order ->
    # the (group-major, particle-major, query-major) order the
    # cascade wrapper expects.  Inverse is applied to the output.
    q_perm: torch.Tensor
    q_inv_perm: torch.Tensor
    num_groups: int
    num_particles: int
    queries_per_particle: int


class SMCFlashInferAttnBackend(FlashInferAttnBackend):
    """FlashInfer backend with an opt-in cascade verify fast-path."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ) -> None:
        super().__init__(
            model_runner,
            skip_prefill=skip_prefill,
            kv_indptr_buf=kv_indptr_buf,
            kv_last_page_len_buf=kv_last_page_len_buf,
            init_new_workspace=init_new_workspace,
        )
        # Lazily allocate the cascade workspace + wrapper; first SMC
        # verify call constructs them, subsequent cycles re-plan in place.
        self._cascade_workspace: Optional[torch.Tensor] = None
        self._cascade_wrapper = None  # MultiLevelCascadeAttentionWrapper, cached
        self._cascade_plan: Optional[_CascadePlan] = None
        # Set by ``init_forward_metadata`` if this cycle is an SMC verify
        # eligible for cascade routing; cleared otherwise.
        self._use_cascade_this_cycle: bool = False

        # Hold a handle to the request pool and model dims for plan building.
        self._req_to_token = model_runner.req_to_token_pool.req_to_token
        self._head_dim = model_runner.model_config.head_dim
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        self._tp_size = get_attention_tp_size()
        self._num_qo_heads = (
            model_runner.model_config.num_attention_heads // self._tp_size
        )
        self._num_kv_heads = model_runner.model_config.get_num_kv_heads(self._tp_size)
        self._page_size = model_runner.token_to_kv_pool_allocator.page_size
        # Cascade plan must match the runtime Q/KV dtypes.  These come from
        # the model config; bf16 and fp16 are both common.
        self._q_dtype = model_runner.dtype
        self._kv_dtype = model_runner.kv_cache_dtype

    # ──────────────────────────────────────────────────────────────
    #  Metadata
    # ──────────────────────────────────────────────────────────────

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self._use_cascade_this_cycle = False
        self._cascade_plan = None

        if forward_batch.forward_mode.is_target_verify():
            # SMC's "linear" target verify is EXTEND-shaped (extend_prefix_lens
            # + extend_seq_lens already populated by SMCVerifyInput.populate_
            # linear_verify_metadata).  SGLang's stock flashinfer verify branch
            # assumes EAGLE-style ``generate_attn_arg_prefill`` which our
            # SMCVerifyInput doesn't implement, so route through the regular
            # extend path with spec_info=None.
            self._init_smc_verify_metadata(forward_batch)

            spec = forward_batch.spec_info
            shared_prefix_lens = getattr(spec, "shared_prefix_lens", None) if spec else None
            group_row_ids = getattr(spec, "group_row_ids", None) if spec else None
            draft_token_num = getattr(spec, "draft_token_num", None) if spec else None
            if shared_prefix_lens is not None and group_row_ids is not None and draft_token_num is not None:
                plan = self._build_cascade_plan(
                    forward_batch, shared_prefix_lens, group_row_ids, draft_token_num,
                )
                if plan is not None:
                    self._cascade_plan = plan
                    self._use_cascade_this_cycle = True
            return

        # Non-verify cycles: delegate to the inner backend unchanged.
        super().init_forward_metadata(forward_batch)

    def _init_smc_verify_metadata(self, forward_batch: ForwardBatch) -> None:
        """Drive the inner FlashInfer prefill wrapper for SMC verify.

        Mirrors the regular extend branch in
        :meth:`FlashInferAttnBackend.init_forward_metadata` but uses the
        verify wrappers, with ``prefix_lens=forward_batch.extend_prefix_lens``
        and ``spec_info=None`` so the indices updater goes through the
        normal extend path (the only one compatible with SMC's linear
        verify).  The cascade plan, when built, supersedes this on the
        attention layer call.
        """
        from sglang.srt.layers.attention.flashinfer_backend import (
            PrefillMetadata, MultiItemScoringParams,
        )

        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None:
            # Defensive: shouldn't happen for SMC verify, but fall back.
            super().init_forward_metadata(forward_batch)
            return

        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.seq_lens_sum,
            prefix_lens,
            prefill_wrappers=self.prefill_wrappers_verify,
            use_ragged=False,
            encoder_lens=forward_batch.encoder_lens,
            spec_info=None,
            fixed_split_size=self.prefill_split_tile_size,
            multi_item_params=MultiItemScoringParams(),
        )
        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_verify, False, False,
        )

    # ──────────────────────────────────────────────────────────────
    #  Cascade plan construction
    # ──────────────────────────────────────────────────────────────

    def _build_cascade_plan(
        self,
        forward_batch: ForwardBatch,
        shared_prefix_lens: torch.Tensor,
        group_row_ids: torch.Tensor,
        draft_token_num: int,
    ) -> Optional[_CascadePlan]:
        """Construct the 2-level cascade plan from the verify batch.

        Plan tensors live on the device — we only sync small (bs-sized)
        scalars to the host for control flow.  Big block-table indices
        are gathered with GPU-only ops to avoid O(L) ``.cpu()`` overhead
        on long-prefix workloads.

        Returns ``None`` when prerequisites aren't met (any L_g == 0,
        groups with a single particle, group sizes mismatch) — caller
        falls back to inner backend.
        """
        from flashinfer.cascade import MultiLevelCascadeAttentionWrapper

        bs = forward_batch.batch_size
        if bs == 0:
            return None

        device = forward_batch.req_pool_indices.device

        # ── Tiny host syncs (bs ints) for control flow and grouping.
        # bs <= max_running_requests * N * (1+small overhead) — typically
        # tens of ints, never thousands; safe to copy.
        spl_cpu = shared_prefix_lens.to(torch.int64).cpu().tolist()
        grp_cpu = group_row_ids.to(torch.int64).cpu().tolist()
        seq_lens_cpu = forward_batch.seq_lens.to(torch.int64).cpu().tolist()
        # req_pool_indices stays on device — we'll index into req_to_token
        # with it directly.
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)

        groups: List[List[int]] = []
        seen: dict = {}
        for i, g in enumerate(grp_cpu):
            if g < 0:
                return None
            if g not in seen:
                seen[g] = len(groups)
                groups.append([])
            groups[seen[g]].append(i)

        n_g = len(groups)
        for batch_indices in groups:
            ls = {spl_cpu[i] for i in batch_indices}
            if len(ls) != 1:
                return None
            (L_g,) = ls
            if L_g <= 0:
                return None
            if len(batch_indices) < 2:
                return None
        n_p = sum(len(g) for g in groups)
        Ns = {len(g) for g in groups}
        if len(Ns) != 1:
            return None

        # ── Q permutation (forward order → group-major order).  Built on
        # device with arange + index_add tricks; the bs-sized cumsum is
        # the only host-side compute.
        total_q = n_p * draft_token_num
        # Flat list of forward-batch indices in group-major order.
        flat_idx_host: List[int] = []
        for batch_indices in groups:
            flat_idx_host.extend(batch_indices)
        flat_idx = torch.tensor(flat_idx_host, dtype=torch.int64, device=device)
        # Each forward-batch index i contributes draft_token_num consecutive
        # queries starting at i * draft_token_num.
        bases = flat_idx * draft_token_num
        offsets = torch.arange(draft_token_num, device=device, dtype=torch.int64)
        q_perm = (bases.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
        q_inv_perm = torch.empty_like(q_perm)
        q_inv_perm[q_perm] = torch.arange(total_q, device=device, dtype=torch.int64)

        # ── Plan tensors.  Indptr / last_page tensors are tiny and live on
        # host until the final to(device).  Indices tensors are big, so we
        # keep them on device throughout.
        qo_indptr_lvl0 = [0]
        kv_indptr_lvl0 = [0]
        kv_last_page_lvl0: List[int] = []
        kv_indices_lvl0_chunks: List[torch.Tensor] = []

        qo_indptr_lvl1 = list(range(0, total_q + 1, draft_token_num))
        kv_indptr_lvl1 = [0]
        kv_last_page_lvl1: List[int] = []
        kv_indices_lvl1_chunks: List[torch.Tensor] = []

        req_to_token = self._req_to_token  # (max_pool, max_ctx) int32 on device
        page_size = self._page_size

        for batch_indices in groups:
            L_g = spl_cpu[batch_indices[0]]
            # Level 0: shared prefix from particle 0 of this group.
            pool0 = req_pool_indices[batch_indices[0]]
            kv_indices_lvl0_chunks.append(req_to_token[pool0, :L_g].to(torch.int32))
            kv_indptr_lvl0.append(kv_indptr_lvl0[-1] + L_g)
            kv_last_page_lvl0.append(page_size if (L_g % page_size) == 0 else (L_g % page_size))
            qo_indptr_lvl0.append(qo_indptr_lvl0[-1] + len(batch_indices) * draft_token_num)

            # Level 1: per-particle private suffix.  total_kv_len = orig_seq_len + γ+1.
            for i in batch_indices:
                pool_i = req_pool_indices[i]
                total_kv = seq_lens_cpu[i] + draft_token_num
                if total_kv <= L_g:
                    return None  # nothing private; degenerate
                kv_indices_lvl1_chunks.append(
                    req_to_token[pool_i, L_g:total_kv].to(torch.int32)
                )
                suffix_len = total_kv - L_g
                kv_indptr_lvl1.append(kv_indptr_lvl1[-1] + suffix_len)
                kv_last_page_lvl1.append(
                    page_size if (suffix_len % page_size) == 0
                    else (suffix_len % page_size)
                )

        def _t_cpu(xs):  # tiny tensors built on host then moved to device
            return torch.tensor(xs, dtype=torch.int32, device=device)

        kv_indices_lvl0 = torch.cat(kv_indices_lvl0_chunks)
        kv_indices_lvl1 = torch.cat(kv_indices_lvl1_chunks)

        if self._cascade_workspace is None:
            self._cascade_workspace = torch.empty(
                256 * 1024 * 1024, dtype=torch.uint8, device=device,
            )
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                num_levels=2,
                float_workspace_buffer=self._cascade_workspace,
                kv_layout="NHD",
            )
        wrapper = self._cascade_wrapper
        wrapper.plan(
            qo_indptr_arr=[_t_cpu(qo_indptr_lvl0), _t_cpu(qo_indptr_lvl1)],
            paged_kv_indptr_arr=[_t_cpu(kv_indptr_lvl0), _t_cpu(kv_indptr_lvl1)],
            paged_kv_indices_arr=[kv_indices_lvl0, kv_indices_lvl1],
            paged_kv_last_page_len=[_t_cpu(kv_last_page_lvl0), _t_cpu(kv_last_page_lvl1)],
            num_qo_heads=self._num_qo_heads,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            page_size=self._page_size,
            causal=True,
            q_data_type=self._q_dtype,
            kv_data_type=self._kv_dtype,
        )

        return _CascadePlan(
            wrapper=wrapper,
            q_perm=q_perm,
            q_inv_perm=q_inv_perm,
            num_groups=n_g,
            num_particles=n_p,
            queries_per_particle=draft_token_num,
        )

    # ──────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if not self._use_cascade_this_cycle or self._cascade_plan is None:
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache)

        # Write the freshly computed K/V into the unified paged KV cache
        # so the cascade wrapper sees them.  Mirrors the regular
        # FlashInferAttnBackend.forward_extend non-ragged path.
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None and v is not None and save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale,
            )

        plan = self._cascade_plan
        # Permute q into (group, particle, query) order to match the plan.
        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        q_permuted = q_view.index_select(0, plan.q_perm)
        kv_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        out_permuted = plan.wrapper.run(q_permuted, kv_buffer)

        out = out_permuted.index_select(0, plan.q_inv_perm)
        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

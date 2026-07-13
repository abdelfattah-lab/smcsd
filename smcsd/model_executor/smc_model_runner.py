"""SMC variant of ModelRunner.

Overrides three extension points so core never imports from SMC:
- ``_init_pools`` swaps in ``SMCRefCountedTokenAllocator`` for refcounted
  KV slots (shared parent prefix across particles).
- ``_build_dummy_run_spec_info`` returns ``SMCVerifyInput`` during the
  autotune / warmup dummy run so attention backends see the SMC path.
- ``_get_graph_runner_class`` returns ``SMCCudaGraphRunner`` which in
  turn returns ``SMCVerifyInput`` during CUDA graph capture.

It also overrides ``_resolve_memory_pool_config`` so the target *and* draft KV
pools are co-budgeted inside ``mem_fraction_static`` instead of the draft riding
along on leftover headroom (see below).
"""

import copy
import gc
import logging
import os

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import get_available_gpu_memory
from smcsd.mem_cache.allocator import SMCRefCountedTokenAllocator

logger = logging.getLogger(__name__)


class SMCModelRunner(ModelRunner):
    def _init_pools(self):
        super()._init_pools()
        # Swap standard allocator for SMC refcount-tracking variant when the
        # standard one was constructed.  Skipped for the draft worker (which
        # is passed the target's allocator), and for SWA / paged / NPU
        # paths where a different allocator subclass was already chosen.
        if (
            type(self.token_to_kv_pool_allocator) is TokenToKVPoolAllocator
            and not self.is_draft_worker
        ):
            self.token_to_kv_pool_allocator = SMCRefCountedTokenAllocator(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            )

    def _build_dummy_run_spec_info(self, buffers, num_tokens_per_bs):
        if self.spec_algorithm.is_smc() and not self.is_draft_worker:
            from smcsd.common.verify import SMCVerifyInput

            return SMCVerifyInput(
                draft_token_num=num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_tokens_per_req=num_tokens_per_bs,
            )
        return super()._build_dummy_run_spec_info(buffers, num_tokens_per_bs)

    def _get_graph_runner_class(self):
        if self.device == "cuda":
            from smcsd.model_executor.smc_cuda_graph_runner import (
                SMCCudaGraphRunner,
            )

            return SMCCudaGraphRunner
        return super()._get_graph_runner_class()

    def _resolve_memory_pool_config(self, pre_model_load_memory):
        if self.is_draft_worker or self.spec_algorithm.is_none():
            return super()._resolve_memory_pool_config(pre_model_load_memory)

        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
        if (
            self.mambaish_config is not None
            and self.server_args.max_mamba_cache_size is None
            and self.server_args.max_running_requests is not None
        ):
            self.server_args.max_mamba_cache_size = (
                self.server_args.max_running_requests
            )

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        page_size = self.server_args.page_size
        configurator = create_memory_pool_configurator(self)

        cobudget = self._cobudget_pool_sizes(available_bytes, page_size, configurator)
        config = (
            cobudget
            if cobudget is not None
            else configurator.calculate_pool_sizes(available_bytes, page_size)
        )

        # Mirror the tail of the stock _resolve_memory_pool_config.
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(
                constrained, page_size
            )
        config.max_running_requests = self._resolve_max_num_reqs(
            config.max_total_num_tokens
        )
        config.mem_fraction_static = self.server_args.mem_fraction_static
        return config

    def _cobudget_pool_sizes(self, available_bytes, page_size, configurator):
        """Return a MemoryPoolConfig sized so target+draft KV+weights fit the
        static budget, or None if this model shape isn't supported (caller
        falls back to stock sizing)."""
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
            MemoryPoolConfig,
        )

        def _skip(reason):
            logger.warning("SMC co-budget KV: %s; using stock KV sizing.", reason)
            return None

        if not self.server_args.speculative_draft_model_path:
            return _skip("no speculative_draft_model_path")
        if not isinstance(configurator, DefaultPoolConfigurator):
            return _skip("non-default pool configurator (SWA)")
        # MLA (DeepSeek-style) KV geometry isn't handled here; Mamba/hybrid IS.
        if self.use_mla_backend or self.is_hybrid_swa:
            return _skip("MLA / hybrid-SWA target unsupported")

        try:
            draft_config = self._build_draft_model_config()
        except Exception as e:  # noqa: BLE001 - config build is best-effort
            return _skip(f"could not build draft config ({e})")

        if draft_config.attention_arch == AttentionArch.MLA or getattr(
            draft_config, "is_hybrid_swa", False
        ):
            return _skip("MLA / hybrid-SWA draft unsupported")

        tp = get_attention_tp_size()
        kv_bytes = torch._utils._element_size(self.kv_cache_dtype)
        gib = 1 << 30

        # ── Per-token full-attention KV cost for each model ──────────────────
        # For a Mamba/hybrid target, configurator._cell_size already counts only
        # the FULL-ATTENTION layers (pool_configurator uses
        # mambaish.full_attention_layer_ids), so cell_t is correct as-is.
        cell_t = configurator._cell_size
        # cell_d mirrors that: count the draft's full-attention layers only when
        # the draft is hybrid, else all of its (attention) layers.  Each carries
        # its OWN model's depth, so unequal-depth pairs are priced correctly.
        draft_mambaish = self._draft_mambaish_config(draft_config)
        if draft_mambaish is not None:
            fa_ids = getattr(draft_mambaish, "full_attention_layer_ids", None)
            if not fa_ids:
                raise RuntimeError(
                    "SMC co-budget KV: hybrid draft exposes no "
                    "full_attention_layer_ids; cannot size its KV pool."
                )
            draft_kv_layers = len(fa_ids)
        else:
            draft_kv_layers = max(
                draft_config.num_hidden_layers,
                getattr(draft_config, "num_attention_layers", 0),
            )
        cell_d = (
            draft_config.get_num_kv_heads(tp)
            * (draft_config.head_dim + draft_config.v_head_dim)
            * draft_kv_layers
            * kv_bytes
        )

        # ── Draft Mamba recurrent state (counted, not measured) ──────────────
        # The target's Mamba state (steady + verify-intermediate) was already
        # carved out of available_bytes by handle_max_mamba_cache.  The DRAFT's
        # pool (worker.py) is steady-state ONLY — it's built with
        # speculative_num_draft_tokens=None, so no verify-intermediate — sized to
        # one slot per running request (target_pool.size).
        draft_mamba_bytes = 0
        if draft_mambaish is not None:
            # Draft pool slots == target req_to_token_pool size == max_num_reqs,
            # which _resolve_max_num_reqs derives as max_running_requests //
            # self.dp_size (the token-capacity cap only binds far above SMC's
            # tiny groups*(N+1), and if it did it would shrink the real pool —
            # so this over-reserves, which is safe).
            slots = self.server_args.max_running_requests // self.dp_size
            draft_mamba_bytes = (
                draft_mambaish.mamba2_cache_params.mamba_cache_per_req * slots
            )
            # Deferred-bonus (ON by default; SMCEngine's defer_bonus=False to
            # opt out — read from server_args like the worker does) gives the
            # hybrid draft a depth-2 verify-intermediate recurrent buffer for
            # the 2-token head (speculative_num_draft_tokens=2 in
            # _maybe_isolate_dense_hybrid_draft_state).  Each intermediate
            # position is ~one per-req state (same conv-window + ssm shapes),
            # so count 1 steady + 2 intermediates.  NOTE: this is accounting
            # for buffers the pool genuinely allocates — a kernel-reuse
            # artifact of the stock TARGET_VERIFY scan (SMC accepts all
            # tokens, so only the last position's state is ever committed); a
            # direct-commit kernel variant could shrink this to 1x.
            if bool(getattr(self.server_args, "smc_defer_bonus", True)):
                draft_mamba_bytes *= 3

        # ── Exact draft weights (measured; raises on failure) ────────────────
        w_d_bytes = self._measure_draft_weight_bytes(draft_config)

        budget = available_bytes - w_d_bytes - draft_mamba_bytes
        if budget <= 0:
            raise RuntimeError(
                "SMC co-budget KV: draft weights "
                f"({w_d_bytes / gib:.2f}GB) + draft Mamba state "
                f"({draft_mamba_bytes / gib:.2f}GB) exceed the KV budget "
                f"({available_bytes / gib:.2f}GB, after target weights + target "
                "Mamba). Lower --mem-fraction-static or use a smaller draft."
            )

        max_tokens = int(budget // (cell_t + cell_d))
        max_tokens = max_tokens // page_size * page_size
        if max_tokens <= 0:
            raise RuntimeError(
                "SMC co-budget KV: co-budgeted token count is non-positive "
                f"(budget={budget / gib:.2f}GB, cell_t+cell_d={cell_t + cell_d} "
                "bytes/token). Raise --mem-fraction-static or free GPU memory."
            )

        logger.info(
            "SMC co-budget KV: cell_t=%d cell_d=%d W_t=%.2fGB W_d=%.2fGB "
            "draft_mamba=%.2fGB avail=%.2fGB -> max_total_num_tokens=%d",
            cell_t,
            cell_d,
            self.weight_load_mem_usage,
            w_d_bytes / gib,
            draft_mamba_bytes / gib,
            available_bytes / gib,
            max_tokens,
        )
        return MemoryPoolConfig(max_total_num_tokens=max_tokens)

    def _build_draft_model_config(self):
        """Build the draft ModelConfig exactly as SMCDenseDraftTpModelWorker
        will (dense AR load, no MTP rewrite; SMC_DRAFT_QUANTIZATION applied) so
        the KV cell size and the weight measurement match the model that
        actually loads."""
        from sglang.srt.configs.model_config import ModelConfig

        sa = self.server_args
        draft_quant = os.environ.get("SMC_DRAFT_QUANTIZATION")
        if draft_quant:
            sa = copy.copy(sa)
            sa.quantization = draft_quant
        return ModelConfig.from_server_args(
            sa, model_path=sa.speculative_draft_model_path, is_draft_model=False
        )

    def _draft_mambaish_config(self, draft_config):
        """Return the draft's Mamba/linear-attention config (or None if the
        draft is pure attention).

        ``mambaish_config`` and its component properties are pure functions of
        ``self.model_config`` + ``self.is_draft_worker``, so we evaluate them
        against the draft by temporarily swapping those in (single-threaded
        init; restored in ``finally``).  This reuses upstream's isinstance
        chains verbatim rather than re-deriving them here — the one nuance is
        the registry cache, which we clear and restore so the target's cached
        result isn't clobbered.
        """
        missing = object()
        orig_cfg = self.model_config
        orig_draft = self.is_draft_worker
        orig_cache = getattr(self, "_linear_attn_registry_cache", missing)
        try:
            self.model_config = draft_config
            self.is_draft_worker = True  # match the real draft runner
            if orig_cache is not missing:
                del self._linear_attn_registry_cache
            return self.mambaish_config
        finally:
            self.model_config = orig_cfg
            self.is_draft_worker = orig_draft
            if orig_cache is missing:
                if hasattr(self, "_linear_attn_registry_cache"):
                    del self._linear_attn_registry_cache
            else:
                self._linear_attn_registry_cache = orig_cache

    def _measure_draft_weight_bytes(self, draft_config):
        """Exactly measure this rank's draft weight footprint by loading the
        draft weights, reading the free-GPU-memory delta, then freeing them.

        Weights are measured (not counted) because quantization
        (SMC_DRAFT_QUANTIZATION), MoE experts, and tied/untied embeddings make an
        analytical byte count fragile — unlike the KV / Mamba caches, which are
        clean geometric tensors and ARE counted.  This is a throwaway load (the
        real draft is loaded later by SMCDenseDraftTpModelWorker), so the draft
        loads twice (~seconds for small drafts); in exchange W_d is exact,
        including quantization and TP sharding.  All TP ranks reach this method
        identically (same server_args/config), so the weight-load collectives
        stay balanced; the free-memory reads are per-rank (non-distributed)
        deltas and add no collectives of their own.

        Raises on failure — we do NOT silently fall back to an estimate, so a
        broken probe surfaces loudly rather than quietly mis-sizing the pool.
        """
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.model_loader.loader import get_model_loader

        draft_model = None
        try:
            before = get_available_gpu_memory(self.device, self.gpu_id)
            loader = get_model_loader(self.load_config, draft_config)
            draft_model = loader.load_model(
                model_config=draft_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            after = get_available_gpu_memory(self.device, self.gpu_id)
            w_d = int(max(0.0, before - after) * (1 << 30))
            if w_d <= 0:
                raise RuntimeError(
                    "measured a non-positive draft weight footprint "
                    f"(before={before:.2f}GB, after={after:.2f}GB)"
                )
            return w_d
        except Exception as e:
            raise RuntimeError(
                f"SMC co-budget KV: draft weight measurement failed ({e}). "
                "Fix the draft load rather than sizing the KV pool off a guess."
            ) from e
        finally:
            # Free the throwaway weights so the real KV pool has the room back.
            del draft_model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

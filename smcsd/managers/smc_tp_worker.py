"""SMC variants of TpModelWorker.

- ``SMCTpModelWorker`` (target): uses ``SMCModelRunner`` so the
  refcount-tracking flat-KV allocator is wired in for the target.

- ``SMCDraftTpModelWorker`` (draft, hybrid case): when both target and
  draft are hybrid (Mamba/SSM mixed with attention) and the draft's
  mamba state shapes don't match the target's, swap the inherited
  ``HybridReqToTokenPool`` for a ``DraftHybridReqToTokenPool`` whose
  ``mamba_pool`` is sized for the *draft*. The req_to_token block table
  and free-slot list are still shared with the target — that's required
  by SMC's slot machinery (copy_block_table + fused resample kernel) —
  but mamba state lives in two independent pools.

  When the draft is dense, this class is a no-op pass-through.

Upstream's ``ModelConfig._config_draft_model`` already gates the
``Qwen3-Next``/``Qwen3.5`` → MTP architecture rewrite on
``speculative_algorithm`` and skips it for SMC, so we don't need to
intercept that here.
"""

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from smcsd.model_executor.smc_model_runner import SMCModelRunner


class SMCTpModelWorker(TpModelWorker):
    def _init_model_runner(self):
        # Mirrors TpModelWorker._init_model_runner with SMCModelRunner.
        self._model_runner = SMCModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )


def _draft_text_config_with_mamba(model_config):
    """Return the draft model's text config iff it carries Mamba/GDN state
    parameters (``mamba2_cache_params``). Used to detect the
    hybrid+hybrid case where the draft also needs a Mamba pool."""
    text_cfg = model_config.hf_config.get_text_config()
    return text_cfg if hasattr(text_cfg, "mamba2_cache_params") else None


class SMCDraftTpModelWorker(TpModelWorker):
    """Draft TpModelWorker with hybrid+hybrid awareness.

    On hybrid drafts (e.g. Qwen3.5-9B drafting for Qwen3.6-27B), replaces
    the inherited ``HybridReqToTokenPool`` with a ``DraftHybridReqToTokenPool``
    that owns a draft-sized ``MambaPool`` while still aliasing the target's
    ``req_to_token`` block table.

    On dense drafts, this is a no-op subclass and behaves identically to
    plain ``TpModelWorker``.
    """

    def _init_model_runner(self):
        # By this point ``self.model_config`` is built (TpModelWorker
        # __init__ runs ``_init_model_config`` first) and
        # ``self.req_to_token_pool`` holds the *target's* pool. Detect
        # the hybrid+hybrid case and rewrap before constructing the
        # draft's ModelRunner.
        target_pool = self.req_to_token_pool
        if isinstance(target_pool, HybridReqToTokenPool):
            text_cfg = _draft_text_config_with_mamba(self.model_config)
            if text_cfg is not None:
                self.req_to_token_pool = self._build_draft_hybrid_pool(
                    target_pool, text_cfg
                )
        super()._init_model_runner()

    def _build_draft_hybrid_pool(self, target_pool, text_cfg):
        from smcsd.mem_cache.draft_hybrid_pool import DraftHybridReqToTokenPool

        cache_params = text_cfg.mamba2_cache_params
        # PP slicing of mamba layers, mirroring upstream's HybridReqToTokenPool init.
        start_layer = getattr(self._model_runner, "start_layer", 0) if hasattr(self, "_model_runner") else 0
        end_layer = getattr(self._model_runner, "end_layer", None) if hasattr(self, "_model_runner") else None
        mamba_layer_ids = list(cache_params.layers)
        if end_layer is not None:
            mamba_layer_ids = [
                i for i in mamba_layer_ids if start_layer <= i < end_layer
            ]

        return DraftHybridReqToTokenPool(
            target_pool=target_pool,
            mamba_size=self.server_args.max_mamba_cache_size,
            mamba_spec_state_size=target_pool.req_to_token.shape[0],
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=self.server_args.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
            start_layer=start_layer,
        )

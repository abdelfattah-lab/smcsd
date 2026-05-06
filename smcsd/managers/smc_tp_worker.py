"""SMC variants of TpModelWorker.

Two subclasses:

- ``SMCTpModelWorker`` (target): constructs ``SMCModelRunner`` so SMC's
  refcount-tracking allocator is used for the target model.

- ``SMCDraftTpModelWorker`` (draft): bypasses upstream's automatic
  draft-model architecture rewrite. Upstream maps the draft's HF
  architecture (e.g. ``Qwen3_5ForConditionalGeneration`` or
  ``Qwen3NextForCausalLM``) onto an MTP variant whose forward expects
  ``forward_batch.spec_info.hidden_states`` from the target — that's the
  EAGLE / NextN convention. SMC instead uses an independent draft model
  that runs its own forward without seeing target hidden states, so we
  need to keep the draft's original architecture. We do this by passing
  ``is_draft_model=False`` into ``ModelConfig.from_server_args`` and then
  re-setting ``model_config.is_draft_model = True`` so quantization and
  context-length checks that key on it still take the draft branch.
"""

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.tp_worker import TpModelWorker
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


class SMCDraftTpModelWorker(TpModelWorker):
    """Draft worker for SMC. Loads the draft as its native causal-LM
    architecture, NOT as the MTP variant upstream forces for spec decoding."""

    def _init_model_config(self):
        # Override: pass is_draft_model=False to skip the automatic
        # architecture → MTP rewrite (model_config._config_draft_model).
        # SMC doesn't want MTP semantics on the draft.
        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            (
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            (
                self.server_args.tokenizer_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_tokenizer_path
            ),
            is_draft_model=False,
        )
        # Restore the draft flag for downstream code paths (quantization
        # config skip, context-length messaging, etc.).
        self.model_config.is_draft_model = True

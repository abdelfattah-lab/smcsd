"""SMC variant of CudaGraphRunner.

Overrides ``get_spec_info`` to return ``SMCVerifyInput`` during CUDA
graph capture so Triton autotune / graph capture sees the SMC-specific
attention path.  Non-SMC paths (draft worker, other spec algos) delegate
to the base class.

For eagle3 draft mode:
- Calls ``set_eagle3_layers_to_capture()`` on the target model before
  capture so the graph captures the aux-hidden-state output path.
- Returns ``SMCVerifyInput`` with ``capture_hidden_mode=FULL`` so the
  captured graph emits hidden states on every replay.
"""

import logging

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

logger = logging.getLogger(__name__)


class SMCCudaGraphRunner(CudaGraphRunner):
    def __init__(self, model_runner):
        # For eagle3: tell the target model to emit aux hidden states.
        # Must be called BEFORE super().__init__() which triggers capture().
        if (
            model_runner.spec_algorithm.is_smc()
            and not model_runner.is_draft_worker
            and getattr(model_runner.server_args, "smc_draft_mode", "dense") == "eagle3"
            and hasattr(model_runner.model, "set_eagle3_layers_to_capture")
        ):
            layer_ids = None
            try:
                from sglang.srt.configs.model_config import ModelConfig

                draft_cfg = ModelConfig.from_server_args(
                    model_runner.server_args,
                    model_path=model_runner.server_args.speculative_draft_model_path,
                    model_revision=model_runner.server_args.speculative_draft_model_revision,
                    is_draft_model=True,
                )
                eagle_cfg = getattr(draft_cfg.hf_config, "eagle_config", None)
                if eagle_cfg is not None:
                    layer_ids = eagle_cfg.get("eagle_aux_hidden_state_layer_ids")
            except Exception as exc:
                logger.warning(
                    "SMC eagle3: failed to load draft eagle aux layer ids; "
                    "falling back to target defaults. err=%s",
                    exc,
                )
            model_runner.model.set_eagle3_layers_to_capture(layer_ids)

        super().__init__(model_runner)

    def get_spec_info(self, num_tokens: int):
        if (
            self.model_runner.spec_algorithm.is_smc()
            and not self.model_runner.is_draft_worker
        ):
            from smcsd.common.verify import SMCVerifyInput

            chm = (
                CaptureHiddenMode.FULL
                if getattr(self.model_runner.server_args, "smc_draft_mode", "dense") == "eagle3"
                else CaptureHiddenMode.NULL
            )
            return SMCVerifyInput(
                draft_token_num=self.num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=chm,
                num_tokens_per_req=self.num_tokens_per_bs,
            )
        return super().get_spec_info(num_tokens)

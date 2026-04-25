"""SMC variant of CudaGraphRunner.

Overrides ``get_spec_info`` to return ``SMCVerifyInput`` during CUDA
graph capture so Triton autotune / graph capture sees the SMC-specific
attention path.  Non-SMC paths (draft worker, other spec algos) delegate
to the base class.
"""

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


class SMCCudaGraphRunner(CudaGraphRunner):
    def __init__(self, model_runner):
        if (
            model_runner.spec_algorithm.is_smc()
            and not model_runner.is_draft_worker
            and str(getattr(model_runner.server_args, "smc_draft_mode", "dense")).startswith(
                "eagle3"
            )
            and hasattr(model_runner.model, "set_eagle3_layers_to_capture")
        ):
            model_runner.model.set_eagle3_layers_to_capture()
        super().__init__(model_runner)

    def get_spec_info(self, num_tokens: int):
        if (
            self.model_runner.spec_algorithm.is_smc()
            and not self.model_runner.is_draft_worker
        ):
            from smcsd.common.verify import SMCVerifyInput

            chm = (
                CaptureHiddenMode.FULL
                if str(
                    getattr(self.model_runner.server_args, "smc_draft_mode", "dense")
                ).startswith("eagle3")
                else CaptureHiddenMode.NULL
            )
            return SMCVerifyInput(
                draft_token_num=self.num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=chm,
                num_tokens_per_req=self.num_tokens_per_bs,
            )
        return super().get_spec_info(num_tokens)

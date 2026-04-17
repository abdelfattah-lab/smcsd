"""SMC variant of CudaGraphRunner.

Overrides ``get_spec_info`` to return ``SMCVerifyInput`` during CUDA
graph capture so Triton autotune / graph capture sees the SMC-specific
attention path.  Non-SMC paths (draft worker, other spec algos) delegate
to the base class.
"""

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode


class SMCCudaGraphRunner(CudaGraphRunner):
    def get_spec_info(self, num_tokens: int):
        if (
            self.model_runner.spec_algorithm.is_smc()
            and not self.model_runner.is_draft_worker
        ):
            from smcsd.common.verify import SMCVerifyInput

            return SMCVerifyInput(
                draft_token_num=self.num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_tokens_per_req=self.num_tokens_per_bs,
            )
        return super().get_spec_info(num_tokens)

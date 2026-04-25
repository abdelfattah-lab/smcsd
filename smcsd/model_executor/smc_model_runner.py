"""SMC variant of ModelRunner.

Overrides three extension points so core never imports from SMC:
- ``_init_pools`` swaps in ``SMCRefCountedTokenAllocator`` for refcounted
  KV slots (shared parent prefix across particles).
- ``_build_dummy_run_spec_info`` returns ``SMCVerifyInput`` during the
  autotune / warmup dummy run so attention backends see the SMC path.
- ``_get_graph_runner_class`` returns ``SMCCudaGraphRunner`` which in
  turn returns ``SMCVerifyInput`` during CUDA graph capture.
"""

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.model_runner import ModelRunner
from smcsd.mem_cache.allocator import SMCRefCountedTokenAllocator


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

            chm = (
                CaptureHiddenMode.FULL
                if str(getattr(self.server_args, "smc_draft_mode", "dense")).startswith(
                    "eagle3"
                )
                else CaptureHiddenMode.NULL
            )
            return SMCVerifyInput(
                draft_token_num=num_tokens_per_bs,
                positions=None,
                capture_hidden_mode=chm,
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

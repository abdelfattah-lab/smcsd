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
                if getattr(self.server_args, "smc_draft_mode", "dense") == "eagle3"
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


class ScoreModelRunner(SMCModelRunner):
    """SMCModelRunner for the nested-SMC SCORE model (e.g. Qwen3-32B).

    The score model must be ``is_draft_worker=False`` so SMCCudaGraphRunner
    captures TARGET_VERIFY cuda graphs (the eager score verify is the decode
    bottleneck). But it also *shares* the target's slot pool + refcounted
    allocator like a draft. The base ``_init_pools`` asserts ``is_draft_worker``
    whenever a ``req_to_token_pool`` is passed in; present as a draft only for
    that call so the assert passes. The SMC allocator swap is correctly skipped
    because the passed allocator is already an ``SMCRefCountedTokenAllocator``
    (not the plain type), so the score model keeps the *shared* allocator and
    block tables while owning its own (score-sized) KV tensor.
    """

    def _init_pools(self):
        real_is_draft = self.is_draft_worker
        self.is_draft_worker = True
        try:
            super()._init_pools()
        finally:
            self.is_draft_worker = real_is_draft

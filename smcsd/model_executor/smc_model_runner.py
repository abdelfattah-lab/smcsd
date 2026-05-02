"""SMC variant of ModelRunner.

Overrides four extension points so core never imports from SMC:
- ``_init_pools`` swaps in ``SMCRefCountedTokenAllocator`` for refcounted
  KV slots (shared parent prefix across particles).
- ``_build_dummy_run_spec_info`` returns ``SMCVerifyInput`` during the
  autotune / warmup dummy run so attention backends see the SMC path.
- ``_get_graph_runner_class`` returns ``SMCCudaGraphRunner`` which in
  turn returns ``SMCVerifyInput`` during CUDA graph capture.
- ``_get_attention_backend_from_str`` substitutes the SMC cascade-aware
  FlashInfer backend when ``--smc-shared-prefix-attn`` is set.
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

    def _get_attention_backend_from_str(self, backend_str: str, init_new_workspace: bool = False):
        """Wrap ``flashinfer`` with the SMC cascade backend when requested.

        Engaged only when:
          1. ``--smc-shared-prefix-attn`` is set on the server args.
          2. We're the target/score model (draft worker uses upstream).
          3. The user picked the ``flashinfer`` attention backend.
        """
        backend = super()._get_attention_backend_from_str(
            backend_str, init_new_workspace=init_new_workspace,
        )
        use_cascade = bool(
            getattr(self.server_args, "smc_shared_prefix_attn", False)
            and not self.is_draft_worker
            and backend_str == "flashinfer"
            and self.spec_algorithm.is_smc()
        )
        if not use_cascade:
            return backend

        # Late import to avoid hard-depending on flashinfer when the
        # flag is off.
        from smcsd.model_executor.smc_attn_backend import SMCFlashInferAttnBackend

        # Re-instantiate as the SMC cascade subclass.  We pass through
        # the same workspace flag so the inner FlashInfer wrappers behave
        # identically to the unwrapped path; the cascade plan adds its
        # own workspace lazily.
        return SMCFlashInferAttnBackend(
            self,
            skip_prefill=False,
            init_new_workspace=init_new_workspace,
        )

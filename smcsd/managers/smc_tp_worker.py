"""SMC variant of TpModelWorker.

Constructs ``SMCModelRunner`` instead of ``ModelRunner`` so SMC's
refcount-tracking allocator is used for the target model.  Draft worker
keeps using the standard ``TpModelWorker`` because the draft is passed
the target's (already-SMC) allocator via ``token_to_kv_pool_allocator``.
"""

from sglang.srt.managers.tp_worker import TpModelWorker
from smcsd.model_executor.smc_model_runner import SMCModelRunner


class SMCTpModelWorker(TpModelWorker):
    def _init_model_runner(self):
        # Mirrors TpModelWorker._init_model_runner with SMCModelRunner.
        from sglang.srt.runtime_context import get_schedule

        self._model_runner = SMCModelRunner(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

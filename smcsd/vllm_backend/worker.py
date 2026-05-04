from __future__ import annotations

import os

from vllm.v1.worker.gpu_worker import Worker

from smcsd.vllm_backend.model_runner import SMCGPUModelRunner


class SMCGPUWorker(Worker):
    """GPUWorker variant that uses SMCGPUModelRunner instead of the stock runner.

    vllm selects the worker class via parallel_config.worker_cls (a string
    qualified name).  Swapping it here is the only supported injection point
    for the runner; there is no model_runner_cls config field.
    """

    def init_device(self) -> None:
        super().init_device()
        # Replace the stock V2 runner with our SMC-aware runner.
        # super() already set self.device and initialized distributed env;
        # weights are loaded afterward in load_model(), so replacing here is safe.
        self.model_runner = SMCGPUModelRunner(self.vllm_config, self.device)

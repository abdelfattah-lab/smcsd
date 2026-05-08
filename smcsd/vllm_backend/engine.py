"""Offline SMC engine backed by vLLM.

Provides a single generate() API for batch offline inference.
Uses EngineCore directly (in-process, no ZMQ) with UniProcExecutor.

Injection points:
  - scheduler: vllm_config.scheduler_config.scheduler_cls = SMCVLLMScheduler
  - runner:    vllm_config.parallel_config.worker_cls = "smcsd.vllm_backend.worker.SMCGPUWorker"
               (SMCGPUWorker.init_device swaps in SMCGPUModelRunner)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Union

from transformers import AutoTokenizer
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor import UniProcExecutor
from vllm.v1.request import Request

from smcsd.vllm_backend.config import SMCConfig
from smcsd.vllm_backend.scheduler import SMCVLLMScheduler

_SMC_WORKER_CLS = "smcsd.vllm_backend.worker.SMCGPUWorker"


class SMCVLLMEngine:
    """Offline SMC inference engine using the vLLM backend.

    Usage::

        engine = SMCVLLMEngine(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            draft_model_path="meta-llama/Llama-3.2-1B-Instruct",
            n_particles=4,
            gamma=4,
        )
        result = engine.generate(
            "What is 2+2?",
            sampling_params={
                "draft_temperature": 1.0,
                "target_temperature": 1.0,
            },
        )
        print(result["text"])
        engine.shutdown()
    """

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        *,
        n_particles: int = 4,
        gamma: int = 4,
        tp_size: int = 1,
        max_model_len: int | None = None,
        **kwargs,
    ) -> None:
        smc_config = SMCConfig(
            draft_model_path=draft_model_path,
            n_particles=n_particles,
            gamma=gamma,
        )

        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

        engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            **kwargs,
        )
        vllm_config = engine_args.create_engine_config()

        # Attach SMC config and inject scheduler and worker.
        vllm_config.smc_config = smc_config
        vllm_config.scheduler_config.scheduler_cls = SMCVLLMScheduler
        vllm_config.parallel_config.worker_cls = _SMC_WORKER_CLS
        vllm_config.scheduler_config.async_scheduling = False

        self._engine = EngineCore(
            vllm_config=vllm_config,
            executor_class=UniProcExecutor,
            log_stats=False,
        )
        self._vllm_config = vllm_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.tokenizer,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        raw_eos = vllm_config.model_config.hf_config.eos_token_id
        if isinstance(raw_eos, int):
            self._model_stop_token_ids: list[int] = [raw_eos]
        elif isinstance(raw_eos, (list, tuple)):
            self._model_stop_token_ids = list(raw_eos)
        elif self.tokenizer.eos_token_id is not None:
            self._model_stop_token_ids = [self.tokenizer.eos_token_id]
        else:
            self._model_stop_token_ids = []

    def generate(
        self,
        prompt: Union[str, list[str], None] = None,
        sampling_params: Union[dict, list[dict], None] = None,
        input_ids: Union[list[int], list[list[int]], None] = None,
    ) -> Union[dict, list[dict]]:
        """Run offline SMC inference on one or more prompts.

        Args:
            prompt: A single string or list of strings.
            sampling_params: Sampling config dict(s) with keys accepted by
                vllm.SamplingParams plus required SMC-specific
                ``draft_temperature`` and ``target_temperature``.
                ``target_temperature`` is translated to the normal vLLM
                ``temperature`` field for the parent request.
            input_ids: Pre-tokenized input(s). Mutually exclusive with *prompt*.

        Returns:
            A dict (single prompt) or list of dicts with keys:
            ``text``, ``prompt_tokens``, ``completion_tokens``, ``output_ids``.
        """
        # Normalise inputs to lists
        is_single = isinstance(prompt, str) or (
            prompt is None
            and input_ids is not None
            and len(input_ids) > 0
            and isinstance(input_ids[0], int)
        )

        if prompt is not None:
            prompts: list[str] = [prompt] if isinstance(prompt, str) else list(prompt)
            ids_list: list[list[int]] = [
                self.tokenizer.encode(p) for p in prompts
            ]
        elif input_ids is not None:
            ids_list = [input_ids] if is_single else list(input_ids)  # type: ignore[arg-type,list-item]
            prompts = [self.tokenizer.decode(ids) for ids in ids_list]
        else:
            raise ValueError("Either prompt or input_ids must be provided.")

        if sampling_params is None:
            sp_list: list[dict] = [{}] * len(prompts)
        elif isinstance(sampling_params, dict):
            sp_list = [sampling_params] * len(prompts)
        else:
            sp_list = list(sampling_params)

        # Build and submit requests
        rids: list[str] = []
        prompt_token_counts: dict[str, int] = {}
        for ids, sp_dict in zip(ids_list, sp_list):
            rid = uuid.uuid4().hex
            rids.append(rid)
            prompt_token_counts[rid] = len(ids)
            sp_dict = dict(sp_dict)
            if "target_temperature" not in sp_dict:
                raise ValueError(
                    "sampling_params must include target_temperature"
                )
            if "draft_temperature" not in sp_dict:
                raise ValueError(
                    "sampling_params must include draft_temperature"
                )
            target_temperature = sp_dict.pop("target_temperature")
            draft_temperature = sp_dict.pop("draft_temperature")
            sp_dict["temperature"] = target_temperature
            existing_stop = list(sp_dict.pop("stop_token_ids", None) or [])
            sp_dict["stop_token_ids"] = list(set(self._model_stop_token_ids + existing_stop))
            sp = SamplingParams(**sp_dict)
            request = Request(
                request_id=rid,
                prompt_token_ids=ids,
                sampling_params=sp,
                pooling_params=None,
                arrival_time=time.time(),
            )
            request.smc_draft_temperature = float(draft_temperature)
            self._engine.add_request(request)

        # Drive the step loop until all parent requests finish.
        pending = set(rids)
        seed_tokens: dict[str, list[int]] = {rid: [] for rid in rids}

        while pending:
            engine_core_outputs, _ = self._engine.step()
            for outputs in engine_core_outputs.values():
                for out in outputs.outputs:
                    if out.request_id not in pending:
                        continue
                    # Collect the prefill seed token
                    seed_tokens[out.request_id].extend(out.new_token_ids)
                    if out.finished:
                        pending.discard(out.request_id)

        scheduler = self._engine.scheduler
        results = []
        for rid in rids:
            # Per-particle draft token sequences accumulated during draft cycles.
            draft_particles: list[list[int]] = scheduler._completed_groups.pop(rid, [])
            seed = seed_tokens[rid]
            particles = [seed + p for p in draft_particles]
            # Particle 0 as the primary output (placeholder until resampling).
            all_tokens = particles[0] if particles else seed
            results.append({
                "text": self.tokenizer.decode(all_tokens, skip_special_tokens=True),
                "output_ids": all_tokens,
                "prompt_tokens": prompt_token_counts[rid],
                "completion_tokens": len(all_tokens),
                "particles": particles,  # [N][seed + draft tokens]
            })
        return results[0] if is_single else results

    def shutdown(self) -> None:
        self._engine.shutdown()

    def __enter__(self) -> SMCVLLMEngine:
        return self

    def __exit__(self, *args: object) -> bool:
        self.shutdown()
        return False

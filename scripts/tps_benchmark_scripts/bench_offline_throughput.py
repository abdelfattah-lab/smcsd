"""Offline throughput benchmark with SMC support.

Fork of ``sglang.bench_offline_throughput`` adding a ``smc_engine`` backend
routed through :class:`smcsd.engine.SMCEngine`. Drives two sweep modes:

  * ``--backend engine`` — upstream ``sgl.Engine`` (e.g. STANDALONE spec).
  * ``--backend smc_engine`` — ``SMCEngine`` with ``--smc-*`` options.

Preserves the ``Output token throughput (tok/s):`` result line that sweep
scripts extract via ``awk '{print $NF}'``. See
``scripts/tps_benchmark_scripts/*.sh`` for invocation examples.
"""

import argparse
import asyncio
import dataclasses
import inspect
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from sglang.benchmark.datasets import DatasetRow, get_dataset
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer, set_ulimit
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs


# ---------------------------------------------------------------------------
# SMCEngine backend adapter
# ---------------------------------------------------------------------------


# Explicit keyword-only params on SMCEngine.__init__ — we pass these directly
# rather than through **kwargs so the engine gets canonical values (it also
# maps them into ServerArgs under the ``smc_*`` names).
_SMCENGINE_EXPLICIT_ARGS = {
    "model_path",
    "speculative_draft_model_path",  # → draft_model_path
    "smc_n_particles",               # → n_particles
    "smc_gamma",                     # → gamma
    "smc_draft_temperature",         # → draft_temperature
    "smc_target_temperature",        # → target_temperature
    "smc_resample_threshold",        # → resample_threshold
    "smc_resample_method",           # → resample_method
    "smc_metrics",
    "smc_metrics_log_interval",
    "smc_metrics_jsonl",
    "tp_size",
    "base_gpu_id",
}
# Args that SMCEngine's ``forced`` dict overrides unconditionally — dropping
# these from the ``**kwargs`` passthrough keeps the merged dict tidy.
_SMCENGINE_FORCED_ARGS = {
    "speculative_algorithm",
    "skip_tokenizer_init",
    "disable_overlap_schedule",
    "disable_radix_cache",
}


class _SMCEngineBackend:
    """Adapter making :class:`SMCEngine` look like ``sgl.Engine`` to the bench.

    Rewrites ``generate`` output into the upstream shape
    (``[{"text", "meta_info": {"completion_tokens"}}, ...]``) and stubs
    ``get_server_info`` (SMCEngine doesn't surface ``last_gen_throughput``).
    """

    def __init__(self, server_args: ServerArgs):
        from smcsd.engine import SMCEngine

        all_fields = dataclasses.asdict(server_args)
        extras = {
            k: v
            for k, v in all_fields.items()
            if k not in _SMCENGINE_EXPLICIT_ARGS and k not in _SMCENGINE_FORCED_ARGS
        }
        self._engine = SMCEngine(
            model_path=server_args.model_path,
            draft_model_path=server_args.speculative_draft_model_path,
            n_particles=server_args.smc_n_particles,
            gamma=server_args.smc_gamma,
            draft_temperature=server_args.smc_draft_temperature,
            target_temperature=server_args.smc_target_temperature,
            resample_threshold=server_args.smc_resample_threshold,
            resample_method=server_args.smc_resample_method,
            smc_metrics=server_args.smc_metrics,
            smc_metrics_log_interval=server_args.smc_metrics_log_interval,
            smc_metrics_jsonl=server_args.smc_metrics_jsonl,
            tp_size=server_args.tp_size,
            base_gpu_id=server_args.base_gpu_id,
            **extras,
        )

    def generate(
        self,
        *,
        prompt,
        sampling_params,
        return_logprob: bool = False,
        logprob_start_len: int = -1,
    ) -> List[Dict[str, Any]]:
        if return_logprob:
            raise ValueError(
                "--return-logprob is not supported with --backend smc_engine "
                "(SMCEngine doesn't expose per-token logprobs)."
            )
        outs = self._engine.generate(prompt=prompt, sampling_params=sampling_params)
        if not isinstance(outs, list):
            outs = [outs]
        # Reshape to the upstream contract that ``throughput_test_once``
        # expects: each item has ``meta_info.completion_tokens``.
        return [
            {
                "text": o["text"],
                "meta_info": {"completion_tokens": o["completion_tokens"]},
            }
            for o in outs
        ]

    def get_server_info(self) -> Dict[str, Any]:
        # SMCEngine doesn't plumb through ``last_gen_throughput``; return a
        # stub so the upstream print path doesn't KeyError.  The wall-clock
        # throughput numbers the bench computes itself are still accurate.
        return {"internal_states": [{"last_gen_throughput": float("nan")}]}

    def start_profile(self, *args, **kwargs):
        return self._engine.start_profile(*args, **kwargs)

    def stop_profile(self, *args, **kwargs):
        return self._engine.stop_profile(*args, **kwargs)

    def shutdown(self):
        return self._engine.shutdown()


# ---------------------------------------------------------------------------
# Bench args (mirrors upstream; trimmed to what sweep scripts actually use)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "random"
    dataset_path: str = ""
    num_prompts: int = 1000
    sharegpt_output_len: Optional[int] = None
    sharegpt_context_len: Optional[int] = None
    random_input_len: int = 1024
    random_output_len: int = 1024
    random_range_ratio: float = 0.0
    gsp_num_groups: int = 64
    gsp_prompts_per_group: int = 16
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    seed: int = 1
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    prompt_suffix: str = ""
    skip_warmup: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--backend",
            type=str,
            default=BenchArgs.backend,
            choices=["engine", "smc_engine"],
            help="engine = upstream sgl.Engine; smc_engine = smcsd SMCEngine",
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="random",
            choices=["sharegpt", "random", "generated-shared-prefix"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process.",
        )
        parser.add_argument(
            "--sharegpt-output-len",
            type=int,
            default=BenchArgs.sharegpt_output_len,
            help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
        )
        parser.add_argument(
            "--sharegpt-context-len",
            type=int,
            default=BenchArgs.sharegpt_context_len,
            help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Number of input tokens per request (random dataset).",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request (random dataset).",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix "
            "(generated-shared-prefix dataset only).",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per shared-prefix group "
            "(generated-shared-prefix dataset only).",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length "
            "(generated-shared-prefix dataset only).",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length "
            "(generated-shared-prefix dataset only).",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs "
            "(generated-shared-prefix dataset only).",
        )
        parser.add_argument("--seed", type=int, default=1, help="The random seed.")
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help="Disable ignore EOS token",
        )
        parser.add_argument(
            "--extra-request-body",
            metavar='{"key1": "value1", "key2": "value2"}',
            type=str,
            default=BenchArgs.extra_request_body,
            help="Append given JSON object to the request payload "
            "(e.g. sampling params).",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template to dataset prompts (sharegpt).",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all user prompts "
            "(followed by assistant prompt suffix).",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batch.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


# ---------------------------------------------------------------------------
# Core bench (lifted from upstream throughput_test_once, trimmed)
# ---------------------------------------------------------------------------


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[DatasetRow],
    ignore_eos: bool,
    extra_request_body: Dict,
):
    measurement_results = {
        "backend": backend_name,
        "successful_requests": len(reqs),
        "total_latency": -1,
        "total_input_tokens": sum(r.prompt_len for r in reqs),
        "total_output_tokens": -1,
        "request_throughput": -1,
        "input_throughput": -1,
        "output_throughput": -1,
        "total_throughput": -1,
    }

    prompt = [r.prompt for r in reqs]
    sampling_params = [
        {
            "temperature": 0,
            "max_new_tokens": r.output_len,
            "ignore_eos": ignore_eos,
            **extra_request_body,
        }
        for r in reqs
    ]

    st = time.perf_counter()
    gen_out = backend.generate(
        prompt=prompt,
        sampling_params=sampling_params,
    )
    latency = time.perf_counter() - st

    server_info = backend.get_server_info()

    measurement_results["total_latency"] = latency
    measurement_results["total_output_tokens"] = sum(
        o["meta_info"]["completion_tokens"] for o in gen_out
    )
    measurement_results["request_throughput"] = (
        measurement_results["successful_requests"] / latency
    )
    measurement_results["input_throughput"] = (
        measurement_results["total_input_tokens"] / latency
    )
    measurement_results["output_throughput"] = (
        measurement_results["total_output_tokens"] / latency
    )
    measurement_results["total_throughput"] = (
        measurement_results["total_input_tokens"]
        + measurement_results["total_output_tokens"]
    ) / latency

    if inspect.isawaitable(server_info):
        server_info = asyncio.run(server_info)

    measurement_results["last_gen_throughput"] = server_info["internal_states"][0][
        "last_gen_throughput"
    ]

    return measurement_results


def throughput_test(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.backend == "engine":
        backend = Engine(**dataclasses.asdict(server_args))
    elif bench_args.backend == "smc_engine":
        backend = _SMCEngineBackend(server_args)
    else:
        raise ValueError(f'Unknown backend: {bench_args.backend!r}')

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    extra_request_body: Dict = {}
    if bench_args.extra_request_body:
        extra_request_body = json.loads(bench_args.extra_request_body)

    input_requests = get_dataset(bench_args, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body=extra_request_body,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
        extra_request_body=extra_request_body,
    )
    backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print("\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print("{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"]))
    print("{:<40} {:<10.2f}".format(
        "Last generation throughput (tok/s):", result["last_gen_throughput"]))
    print("{:<40} {:<10.2f}".format(
        "Request throughput (req/s):", result["request_throughput"]))
    print("{:<40} {:<10.2f}".format(
        "Input token throughput (tok/s):", result["input_throughput"]))
    print("{:<40} {:<10.2f}".format(
        "Output token throughput (tok/s):", result["output_throughput"]))
    print("{:<40} {:<10.2f}".format(
        "Total token throughput (tok/s):", result["total_throughput"]))
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)

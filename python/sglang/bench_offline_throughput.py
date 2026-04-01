"""
Benchmark the throughput in the offline mode.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (the same as bench_serving.py).

# Usage
## Sharegpt dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10

## Random dataset with default args
python -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024
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
from typing import Dict, List, Optional

import numpy as np

from sglang.benchmark.datasets import DatasetRow, get_dataset
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer, set_ulimit
from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class BenchArgs:
    backend: str = "engine"
    result_filename: str = ""
    dataset_name: str = "sharegpt"
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
    gsm8k_num_shots: int = 5
    seed: int = 1
    disable_ignore_eos: bool = False
    extra_request_body: Optional[str] = None
    apply_chat_template: bool = False
    profile: bool = False
    profile_num_steps: Optional[int] = None
    profile_decode_only: bool = False
    skip_warmup: bool = False
    do_not_exit: bool = False
    prompt_suffix: str = ""
    return_logprob: bool = False
    logprob_start_len: int = -1

    # SSD backend options
    ssd_temperature: float = 0.0
    ssd_draft_temperature: Optional[float] = None
    ssd_num_gpus: int = 1
    ssd_speculate: bool = False
    ssd_speculate_k: int = 6
    ssd_draft_model: Optional[str] = None
    ssd_draft_async: bool = False
    ssd_async_fan_out: int = 3
    ssd_fan_out_list: Optional[List[int]] = None
    ssd_fan_out_list_miss: Optional[List[int]] = None
    ssd_sampler_x: Optional[float] = None
    ssd_backup: str = "jit"
    ssd_kvcache_block_size: int = 256
    ssd_max_num_seqs: int = 1
    ssd_max_model_len: int = 8192
    ssd_enforce_eager: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--backend",
            type=str,
            default=BenchArgs.backend,
            choices=["engine", "runtime", "ssd"],
            help="Backend used to run offline generation.",
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="sharegpt",
            choices=["sharegpt", "random", "generated-shared-prefix", "gsm8k"],
            help="Name of the dataset to benchmark on.",
        )
        parser.add_argument(
            "--dataset-path", type=str, default="", help="Path to the dataset."
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=BenchArgs.num_prompts,
            help="Number of prompts to process. Default is 1000.",
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
            help="Number of input tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-output-len",
            type=int,
            default=BenchArgs.random_output_len,
            help="Number of output tokens per request, used only for random dataset.",
        )
        parser.add_argument(
            "--random-range-ratio",
            type=float,
            default=BenchArgs.random_range_ratio,
            help="Range of sampled ratio of input/output length, "
            "used only for random dataset.",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of groups with shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-prompts-per-group",
            type=int,
            default=BenchArgs.gsp_prompts_per_group,
            help="Number of prompts per group of shared prefix, used"
            "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="System prompt length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Question length, used" "only for generate-shared-prefix",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Target length in tokens for outputs in generated-shared-prefix dataset",
        )
        parser.add_argument(
            "--gsm8k-num-shots",
            type=int,
            default=5,
            help="Number of few-shot examples for GSM8K dataset.",
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
            help="Append given JSON object to the request payload. You can use this to specify"
            "additional generate params like sampling params.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Apply chat template",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Use Torch Profiler. The endpoint must be launched with "
            "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
        )
        parser.add_argument(
            "--profile-num-steps",
            type=int,
            default=None,
            help="Number of forward steps to profile. Default profiles all steps.",
        )
        parser.add_argument(
            "--profile-decode-only",
            action="store_true",
            help="Only profile decode steps. Requires SGLANG_PROFILE_V2=1.",
        )
        parser.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Skip the warmup batches.",
        )
        parser.add_argument(
            "--do-not-exit",
            action="store_true",
            help="Do not exit the program. This is useful for nsys profile with --duration and --delay.",
        )
        parser.add_argument(
            "--prompt-suffix",
            type=str,
            default="",
            help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
        )
        parser.add_argument(
            "--return-logprob",
            action="store_true",
            help="Enable returning log probabilities.",
        )
        parser.add_argument(
            "--logprob-start-len",
            type=int,
            default=-1,
            help="Start length for logprob. -1 means only return logprobs for output tokens (default). 0 means return logprobs for all tokens including input.",
        )

        # SSD backend options
        parser.add_argument(
            "--ssd-temperature",
            type=float,
            default=BenchArgs.ssd_temperature,
            help="Sampling temperature for the SSD backend.",
        )
        parser.add_argument(
            "--ssd-draft-temperature",
            type=float,
            default=BenchArgs.ssd_draft_temperature,
            help="Optional draft temperature for the SSD backend.",
        )
        parser.add_argument(
            "--ssd-num-gpus", type=int, default=BenchArgs.ssd_num_gpus,
            help="Number of GPUs for SSD tensor parallelism.",
        )
        parser.add_argument(
            "--ssd-speculate", action="store_true",
            help="Enable speculative decoding in SSD backend.",
        )
        parser.add_argument(
            "--ssd-speculate-k", type=int, default=BenchArgs.ssd_speculate_k,
            help="Number of speculative tokens for SSD.",
        )
        parser.add_argument(
            "--ssd-draft-model", type=str, default=None,
            help="Draft model path for SSD speculative decoding.",
        )
        parser.add_argument(
            "--ssd-draft-async", action="store_true",
            help="Enable async draft in SSD.",
        )
        parser.add_argument(
            "--ssd-async-fan-out", type=int, default=BenchArgs.ssd_async_fan_out,
            help="Async fan out value for SSD.",
        )
        parser.add_argument(
            "--ssd-fan-out-list",
            type=int,
            nargs="+",
            default=BenchArgs.ssd_fan_out_list,
            help="Per-position fan out list for SSD async speculation.",
        )
        parser.add_argument(
            "--ssd-fan-out-list-miss",
            type=int,
            nargs="+",
            default=BenchArgs.ssd_fan_out_list_miss,
            help="Per-position fan out list used on cache misses for SSD async speculation.",
        )
        parser.add_argument(
            "--ssd-sampler-x",
            type=float,
            default=BenchArgs.ssd_sampler_x,
            help="SSD sampler_x rescaling factor.",
        )
        parser.add_argument(
            "--ssd-backup",
            type=str,
            choices=["jit", "fast"],
            default=BenchArgs.ssd_backup,
            help="SSD backup strategy. 'jit' matches SSD's native benchmark default.",
        )
        parser.add_argument(
            "--ssd-kvcache-block-size",
            type=int,
            default=BenchArgs.ssd_kvcache_block_size,
            help="SSD KV-cache block size.",
        )
        parser.add_argument(
            "--ssd-max-num-seqs", type=int, default=BenchArgs.ssd_max_num_seqs,
            help="Maximum batch size for SSD.",
        )
        parser.add_argument(
            "--ssd-max-model-len", type=int, default=BenchArgs.ssd_max_model_len,
            help="Maximum model length for SSD.",
        )
        parser.add_argument(
            "--ssd-enforce-eager", action="store_true",
            help="Disable CUDA graphs in SSD (eager mode).",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def throughput_test_once(
    backend_name: str,
    backend,
    reqs: List[DatasetRow],
    ignore_eos: bool,
    extra_request_body: Dict,
    profile: bool,
    profile_num_steps: Optional[int] = None,
    profile_decode_only: bool = False,
    return_logprob: bool = False,
    logprob_start_len: int = -1,
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

    if backend_name == "ssd":
        _, SSDSamplingParams = _import_ssd_backend()

        ssd_param_fields = {
            field.name for field in dataclasses.fields(SSDSamplingParams)
        }
        protected_keys = {"max_new_tokens", "ignore_eos"}
        unsupported_keys = sorted(set(extra_request_body) - ssd_param_fields)
        overridden_keys = sorted(set(extra_request_body) & protected_keys)
        if unsupported_keys:
            logging.warning(
                "Ignoring unsupported SSD sampling params: %s",
                ", ".join(unsupported_keys),
            )
        if overridden_keys:
            logging.warning(
                "Ignoring SSD sampling params managed by the benchmark: %s",
                ", ".join(overridden_keys),
            )
        ssd_extra_sampling_params = {
            key: value
            for key, value in extra_request_body.items()
            if key in ssd_param_fields and key not in protected_keys
        }

        ssd_params = [
            SSDSamplingParams(
                **{
                    "temperature": 0,
                    **ssd_extra_sampling_params,
                    "max_new_tokens": r.output_len,
                    "ignore_eos": ignore_eos,
                }
            )
            for r in reqs
        ]

        st = time.perf_counter()
        gen_out, _ssd_metrics = backend.generate(prompt, ssd_params, use_tqdm=False)
        latency = time.perf_counter() - st

        measurement_results["total_latency"] = latency
        measurement_results["total_output_tokens"] = sum(
            len(o["token_ids"]) for o in gen_out
        )
    else:
        sampling_params = [
            {
                "temperature": 0,
                "max_new_tokens": r.output_len,
                "ignore_eos": ignore_eos,
                **extra_request_body,
            }
            for r in reqs
        ]

        if profile:
            assert (
                "SGLANG_TORCH_PROFILER_DIR" in os.environ
            ), "Please set SGLANG_TORCH_PROFILER_DIR."
            os.makedirs(os.environ["SGLANG_TORCH_PROFILER_DIR"], exist_ok=True)
            profile_kwargs = {}
            if profile_num_steps is not None:
                profile_kwargs["num_steps"] = profile_num_steps
            if profile_decode_only:
                profile_kwargs["profile_by_stage"] = True
                profile_kwargs["profile_stages"] = ["decode"]
            backend.start_profile(**profile_kwargs)

        st = time.perf_counter()
        gen_out = backend.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
        )
        latency = time.perf_counter() - st

        if profile:
            dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
            known_files = set(os.listdir(dir))
            backend.stop_profile()
            monitor_trace_file(known_files, dir)

        if backend_name == "runtime":
            gen_out = json.loads(gen_out)

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

    if backend_name == "ssd":
        measurement_results["last_gen_throughput"] = (
            measurement_results["total_output_tokens"] / latency
        )
    else:
        server_info = backend.get_server_info()
        if inspect.isawaitable(server_info):
            server_info = asyncio.run(server_info)
        measurement_results["last_gen_throughput"] = server_info["internal_states"][0][
            "last_gen_throughput"
        ]

    return measurement_results


def monitor_trace_file(known_files, directory, interval=1):
    print(f"Monitoring {directory} for new trace files...")

    while True:
        flag = False
        time.sleep(interval)
        current_files = set(os.listdir(directory))

        new_files = current_files - known_files
        for new_file in new_files:
            new_file_path = os.path.join(directory, new_file)
            print(f"New file detected: {new_file}")

            previous_size = 0
            while True:
                try:
                    current_size = os.path.getsize(new_file_path)
                except FileNotFoundError:
                    print(f"File {new_file} is no longer accessible.")
                    break

                if current_size > previous_size:
                    previous_size = current_size
                else:
                    flag = True
                    break

                time.sleep(interval)
        if flag:
            break


def _import_ssd_backend():
    """Import SSD from either the installed package or the repo-local source tree."""
    import importlib

    import_attempts = []
    for module_name in ("ssd", "ssd.ssd"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            import_attempts.append(f"{module_name}: {exc!r}")
            continue

        llm_cls = getattr(module, "LLM", None)
        sampling_params_cls = getattr(module, "SamplingParams", None)
        if llm_cls is not None and sampling_params_cls is not None:
            return llm_cls, sampling_params_cls

        import_attempts.append(
            f"{module_name}: missing LLM/SamplingParams exports"
        )

    raise ImportError(
        "Failed to import SSD backend. "
        "If you are launching from the repo root, the top-level `ssd/` directory "
        "can shadow the installed package namespace. "
        + "Tried "
        + "; ".join(import_attempts)
    )


def _resolve_ssd_artifact_path(
    artifact_path: Optional[str],
    artifact_name: str,
    require_config: bool = False,
) -> Optional[str]:
    """Resolve an SSD model/tokenizer path to a local directory."""
    if not artifact_path:
        return None

    if os.path.isdir(artifact_path):
        return _resolve_ssd_snapshot_path(
            artifact_path,
            artifact_name=artifact_name,
            require_config=require_config,
        )

    if os.path.exists(artifact_path):
        raise ValueError(
            f"SSD {artifact_name} path exists but is not a directory: {artifact_path}"
        )

    if os.path.isabs(artifact_path) or artifact_path.startswith("."):
        raise ValueError(f"SSD {artifact_name} path does not exist: {artifact_path}")

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("SSD_HF_CACHE") or os.environ.get("HF_HUB_CACHE")
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true")
    offline = offline or os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in (
        "1",
        "true",
    )

    logging.info(
        "Resolving SSD %s repo '%s' to a local snapshot...",
        artifact_name,
        artifact_path,
    )
    resolved_path = snapshot_download(
        repo_id=artifact_path,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN") or None,
        local_files_only=offline,
        library_name="sglang",
    )
    logging.info(
        "Resolved SSD %s repo '%s' to '%s'",
        artifact_name,
        artifact_path,
        resolved_path,
    )
    return _resolve_ssd_snapshot_path(
        resolved_path,
        artifact_name=artifact_name,
        require_config=require_config,
    )


def _resolve_ssd_snapshot_path(
    base_path: str,
    artifact_name: str,
    require_config: bool,
) -> str:
    """Resolve a local SSD artifact path to a snapshot directory when possible."""
    if not os.path.isdir(base_path):
        return base_path

    if not require_config or os.path.exists(os.path.join(base_path, "config.json")):
        return base_path

    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        for item in sorted(os.listdir(snapshots_dir)):
            candidate = os.path.join(snapshots_dir, item)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "config.json")
            ):
                return candidate

    for item in sorted(os.listdir(base_path)):
        candidate = os.path.join(base_path, item)
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "config.json")
        ):
            return candidate

    if require_config:
        raise ValueError(
            f"SSD {artifact_name} path does not contain a snapshot with config.json: "
            f"{base_path}"
        )
    return base_path


def _infer_ssd_model_family(model_ref: str) -> str:
    model_ref_lower = model_ref.lower()
    if "llama" in model_ref_lower:
        return "llama"
    if "qwen" in model_ref_lower:
        return "qwen"
    return "unknown"


def _resolve_ssd_draft_model_ref(
    target_model_ref: str,
    requested_draft_ref: Optional[str],
) -> Optional[str]:
    """Resolve SSD draft selection using the same defaults as SSD's native benchmark."""
    target_family = _infer_ssd_model_family(target_model_ref)

    draft_aliases = {
        "llama": {
            "1": "meta-llama/Llama-3.2-1B-Instruct",
            "3": "meta-llama/Llama-3.2-3B-Instruct",
            "8": "meta-llama/Llama-3.1-8B-Instruct",
            "70": "meta-llama/Llama-3.1-70B-Instruct",
        },
        "qwen": {
            "0.6": "Qwen/Qwen3-0.6B",
            "1": "meta-llama/Llama-3.2-1B-Instruct",
        },
    }
    default_drafts = {
        "llama": "meta-llama/Llama-3.2-1B-Instruct",
        "qwen": "Qwen/Qwen3-0.6B",
    }

    if requested_draft_ref:
        if (
            requested_draft_ref in draft_aliases.get(target_family, {})
            and "/" not in requested_draft_ref
            and not os.path.exists(requested_draft_ref)
        ):
            return draft_aliases[target_family][requested_draft_ref]
        return requested_draft_ref

    if target_family not in default_drafts:
        raise ValueError(
            "SSD speculative decoding needs a draft model, but the benchmark could not "
            f"infer a default draft for target model '{target_model_ref}'. "
            "Please pass --ssd-draft-model explicitly."
        )

    default_draft_ref = default_drafts[target_family]
    logging.info(
        "Using default SSD draft model '%s' for target model '%s'",
        default_draft_ref,
        target_model_ref,
    )
    return default_draft_ref


def _validate_ssd_args(bench_args: BenchArgs):
    if bench_args.ssd_draft_async and not bench_args.ssd_speculate:
        raise ValueError("--ssd-draft-async requires --ssd-speculate")
    if bench_args.ssd_sampler_x is not None and not bench_args.ssd_draft_async:
        raise ValueError("--ssd-sampler-x requires --ssd-draft-async")
    if bench_args.ssd_fan_out_list is not None and not bench_args.ssd_draft_async:
        raise ValueError("--ssd-fan-out-list requires --ssd-draft-async")
    if bench_args.ssd_fan_out_list_miss is not None and not bench_args.ssd_draft_async:
        raise ValueError("--ssd-fan-out-list-miss requires --ssd-draft-async")
    expected_fan_out_len = bench_args.ssd_speculate_k + 1
    if (
        bench_args.ssd_fan_out_list is not None
        and len(bench_args.ssd_fan_out_list) != expected_fan_out_len
    ):
        raise ValueError(
            f"--ssd-fan-out-list must have length {expected_fan_out_len} "
            f"(ssd_speculate_k + 1), got {len(bench_args.ssd_fan_out_list)}"
        )
    if (
        bench_args.ssd_fan_out_list_miss is not None
        and len(bench_args.ssd_fan_out_list_miss) != expected_fan_out_len
    ):
        raise ValueError(
            f"--ssd-fan-out-list-miss must have length {expected_fan_out_len} "
            f"(ssd_speculate_k + 1), got {len(bench_args.ssd_fan_out_list_miss)}"
        )


def _create_ray_engine_backend(server_args: ServerArgs):
    """Create a RayEngine inside a Ray actor on a placement group.

    RayEngine requires a placement group, so we launch it inside a Ray actor
    and return a lightweight proxy that forwards calls via ray.get().
    """
    import ray
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    env_vars = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if not ray.is_initialized():
        ray.init(runtime_env=RuntimeEnv(env_vars=env_vars))

    total_gpus = server_args.tp_size * server_args.pp_size
    pg = placement_group([{"CPU": 1, "GPU": total_gpus}], strategy="STRICT_PACK")
    ray.get(pg.ready())

    @ray.remote
    class _EngineActor:
        def __init__(self, **kwargs):
            from sglang.srt.ray.engine import RayEngine

            self.engine = RayEngine(**kwargs)

        def call(self, method, **kwargs):
            return getattr(self.engine, method)(**kwargs)

    actor = _EngineActor.options(
        num_cpus=1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(**dataclasses.asdict(server_args))

    class _Proxy:
        """Forwards method calls to the remote RayEngine actor."""

        def generate(self, **kwargs):
            return ray.get(actor.call.remote("generate", **kwargs))

        def get_server_info(self, **kwargs):
            return ray.get(actor.call.remote("get_server_info", **kwargs))

        def start_profile(self, **kwargs):
            return ray.get(actor.call.remote("start_profile", **kwargs))

        def stop_profile(self, **kwargs):
            return ray.get(actor.call.remote("stop_profile", **kwargs))

        def shutdown(self):
            try:
                ray.get(actor.call.remote("shutdown"), timeout=60)
            except Exception:
                pass
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass

    return _Proxy()


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
):
    if bench_args.profile:
        os.environ.setdefault("SGLANG_PROFILE_V2", "1")

    ssd_model_path = None
    ssd_draft_model = None
    ssd_tokenizer_path = None

    if bench_args.backend == "engine":
        if server_args.use_ray:
            backend = _create_ray_engine_backend(server_args)
        else:
            backend = Engine(**dataclasses.asdict(server_args))
        if not backend:
            raise ValueError("Please provide valid engine arguments")
    elif bench_args.backend == "runtime":
        backend = Runtime(**dataclasses.asdict(server_args))
    elif bench_args.backend == "ssd":
        _validate_ssd_args(bench_args)
        SSDLLM, _ = _import_ssd_backend()
        ssd_model_path = _resolve_ssd_artifact_path(
            server_args.model_path,
            "target model",
            require_config=True,
        )
        if bench_args.ssd_speculate:
            ssd_draft_model_ref = _resolve_ssd_draft_model_ref(
                ssd_model_path or server_args.model_path,
                bench_args.ssd_draft_model,
            )
            ssd_draft_model = _resolve_ssd_artifact_path(
                ssd_draft_model_ref,
                "draft model",
                require_config=True,
            )
        ssd_tokenizer_path = _resolve_ssd_artifact_path(
            server_args.tokenizer_path, "tokenizer"
        )

        backend = SSDLLM(
            ssd_model_path,
            num_gpus=bench_args.ssd_num_gpus,
            speculate=bench_args.ssd_speculate,
            speculate_k=bench_args.ssd_speculate_k,
            draft=ssd_draft_model,
            draft_async=bench_args.ssd_draft_async,
            async_fan_out=bench_args.ssd_async_fan_out,
            fan_out_list=bench_args.ssd_fan_out_list,
            fan_out_list_miss=bench_args.ssd_fan_out_list_miss,
            sampler_x=bench_args.ssd_sampler_x,
            jit_speculate=(bench_args.ssd_backup == "jit"),
            kvcache_block_size=bench_args.ssd_kvcache_block_size,
            max_num_seqs=bench_args.ssd_max_num_seqs,
            max_model_len=bench_args.ssd_max_model_len,
            enforce_eager=bench_args.ssd_enforce_eager,
            tokenizer_path=ssd_tokenizer_path,
        )
    else:
        raise ValueError('Please set backend to "engine", "runtime", or "ssd"')

    if bench_args.backend == "ssd":
        tokenizer_id = ssd_tokenizer_path or ssd_model_path
    else:
        tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # Set global environments
    set_ulimit()
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Parse args
    extra_request_body = {}
    if bench_args.extra_request_body:
        extra_request_body = json.loads(bench_args.extra_request_body)
    if bench_args.backend == "ssd":
        extra_request_body = {
            "temperature": bench_args.ssd_temperature,
            **(
                {"draft_temperature": bench_args.ssd_draft_temperature}
                if bench_args.ssd_draft_temperature is not None
                else {}
            ),
            **extra_request_body,
        }

    # Read dataset
    input_requests = get_dataset(bench_args, tokenizer)

    warmup_requests = sample_random_requests(
        input_len=256,
        output_len=16,
        num_prompts=min(bench_args.num_prompts, 16),
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path=bench_args.dataset_path,
    )

    # Warm up
    if not bench_args.skip_warmup:
        logging.info("\nWarmup...")
        throughput_test_once(
            backend_name=bench_args.backend,
            backend=backend,
            reqs=warmup_requests,
            ignore_eos=not bench_args.disable_ignore_eos,
            extra_request_body=extra_request_body,
            profile=False,
            return_logprob=bench_args.return_logprob,
            logprob_start_len=bench_args.logprob_start_len,
        )
        time.sleep(0.5)

    logging.info("\nBenchmark...")
    result = throughput_test_once(
        backend_name=bench_args.backend,
        backend=backend,
        reqs=input_requests,
        ignore_eos=not bench_args.disable_ignore_eos,
        extra_request_body=extra_request_body,
        profile=bench_args.profile,
        profile_num_steps=bench_args.profile_num_steps,
        profile_decode_only=bench_args.profile_decode_only,
        return_logprob=bench_args.return_logprob,
        logprob_start_len=bench_args.logprob_start_len,
    )
    if bench_args.backend == "ssd":
        backend.exit(hard=False)
    else:
        backend.shutdown()

    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=50, c="=")
    )
    print("{:<40} {:<10}".format("Backend:", result["backend"]))
    print("{:<40} {:<10}".format("Successful requests:", result["successful_requests"]))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result["total_latency"]))
    print("{:<40} {:<10}".format("Total input tokens:", result["total_input_tokens"]))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", result["total_output_tokens"])
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Last generation throughput (tok/s):", result["last_gen_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result["request_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result["input_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", result["output_throughput"]
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", result["total_throughput"]
        )
    )
    print("=" * 50)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    # handling ModelScope model downloads
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() in ("true", "1"):
        if os.path.exists(args.model_path):
            print(f"Using local model path: {args.model_path}")
        else:
            try:
                from modelscope import snapshot_download

                print(f"Using ModelScope to download model: {args.model_path}")

                # download the model and replace args.model_path
                args.model_path = snapshot_download(
                    args.model_path,
                )
                print(f"Model downloaded to: {args.model_path}")
            except Exception as e:
                print(f"ModelScope download failed: {str(e)}")
                raise e

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    throughput_test(server_args, bench_args)

    while bench_args.do_not_exit:
        pass

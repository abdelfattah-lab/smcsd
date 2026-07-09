"""
Profile plain vLLM offline generation for apples-to-apples comparison with
scripts/smc_profile_vllm_engine.py.

Examples:
  source .venv/bin/activate
  python scripts/profile_vllm_baseline.py --output-dir /tmp/vllm-baseline-profile
  python scripts/profile_vllm_baseline.py --num-prompts 2 --max-tokens 128
  python scripts/profile_vllm_baseline.py --no-use-profiler
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
# DEFAULT_PROMPTS = [
#     "The capital of France is",
#     "Write one sentence about why overlap scheduling matters for inference systems.",
#     "List two prime numbers and one composite number.",
#     "In one short paragraph, explain speculative decoding.",
#     "Briefly explain why batching improves GPU utilization.",
#     "Give two benefits and one risk of speculative decoding.",
#     "What is 17 plus 28? Answer in one sentence.",
#     "Describe effective sample size in plain language.",
# ]

DEFAULT_PROMPTS  = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
    "Briefly explain why batching improves GPU utilization.",
    "Give two benefits and one risk of speculative decoding.",
    "What is 17 plus 28? Answer in one sentence.",
    "Describe effective sample size in plain language.",
    "Explain in one sentence why KV cache reuse improves inference latency.",
    "Name three programming languages commonly used in machine learning systems.",
    "If a batch contains 4 requests and each generates 16 tokens, how many total tokens are generated?",
    "Write a short explanation of why memory bandwidth can bottleneck LLM serving.",
]


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Profile plain vLLM LLM.generate() with matching knobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="/tmp/vllm-baseline-profile")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help=(
            "Number of concurrent prompts. Match this to "
            "smc_profile_vllm_engine.py --num-prompts."
        ),
    )
    parser.add_argument("--prompt", action="append", dest="prompts", default=None)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional text file with one prompt per line. Blank lines are ignored.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=["gsm8k", "default"],
        default="gsm8k",
        help=(
            "Prompt source when --prompt/--prompt-file are not provided. "
            "default uses the short built-in prompts."
        ),
    )
    parser.add_argument(
        "--gsm8k-split",
        default="test",
        help="GSM8K split to use when --prompt-source=gsm8k.",
    )
    parser.add_argument(
        "--gsm8k-start-index",
        type=int,
        default=0,
        help="Starting row in the GSM8K split.",
    )
    parser.add_argument(
        "--prompt-tokenizer-path",
        default=None,
        help=(
            "Tokenizer used to apply the chat template for GSM8K prompts. "
            "Defaults to --model-path."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use eager execution. Disable to allow CUDA graphs if supported.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-profiler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM torch profiler traces. Disable for lower-overhead "
            "throughput-only baseline runs."
        ),
    )
    parser.add_argument(
        "--with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture Python stack traces in torch profiler output. Disabled "
            "by default because it can make profiler runs much heavier."
        ),
    )
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record tensor shapes in torch profiler output.",
    )
    parser.add_argument(
        "--with-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch memory profiling.",
    )
    parser.add_argument(
        "--with-flops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch FLOP estimates where available.",
    )
    parser.add_argument(
        "--use-gzip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress Chrome trace JSON files.",
    )
    parser.add_argument(
        "--delay-iterations",
        type=int,
        default=0,
        help="Worker iterations to skip after profile start.",
    )
    parser.add_argument(
        "--max-profile-iterations",
        type=int,
        default=0,
        help="Worker iterations to record before auto-stopping; 0 records until stop.",
    )
    parser.add_argument(
        "--profiler-warmup-iterations",
        type=int,
        default=0,
        help="Torch profiler schedule warmup iterations. This is not model warmup.",
    )
    parser.add_argument(
        "--profiler-active-iterations",
        type=int,
        default=5,
        help="Torch profiler schedule active iterations when schedule is enabled.",
    )
    return parser.parse_args()


def build_run_dir(base_dir: str) -> Path:
    run_dir = Path(base_dir).expanduser().resolve() / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def read_prompt_file(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).expanduser().read_text().splitlines()
        if line.strip()
    ]


def format_gsm8k_instruction(question: str) -> str:
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k_prompts(args: argparse.Namespace) -> list[str]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if args.gsm8k_start_index < 0:
        raise SystemExit("--gsm8k-start-index cannot be negative.")
    tokenizer_path = args.prompt_tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = load_dataset("openai/gsm8k", "main", split=args.gsm8k_split)
    end = args.gsm8k_start_index + args.num_prompts
    if end > len(dataset):
        raise SystemExit(
            f"GSM8K range [{args.gsm8k_start_index}, {end}) exceeds "
            f"split length {len(dataset)}."
        )
    prompts = []
    for sample in dataset.select(range(args.gsm8k_start_index, end)):
        instruction = format_gsm8k_instruction(sample["question"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def select_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        prompts.extend(read_prompt_file(args.prompt_file))
    if args.prompts:
        prompts.extend(args.prompts)
    if not prompts:
        if args.prompt_source == "gsm8k":
            prompts = load_gsm8k_prompts(args)
        else:
            prompts = list(DEFAULT_PROMPTS)
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts must be positive.")
    while len(prompts) < args.num_prompts:
        prompts.extend(prompts[: args.num_prompts - len(prompts)])
    return prompts[: args.num_prompts]


def list_profile_artifacts(run_dir: Path) -> list[Path]:
    suffixes = (
        ".pt.trace.json",
        ".pt.trace.json.gz",
        ".trace.json",
        ".trace.json.gz",
        ".txt",
    )
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file()
        and (
            path.name.endswith(suffixes)
            or path.name.startswith("profiler_out_")
        )
    )


def wait_for_artifacts(run_dir: Path, timeout_sec: float = 30.0) -> list[Path]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        artifacts = list_profile_artifacts(run_dir)
        if artifacts:
            return artifacts
        time.sleep(0.5)
    return list_profile_artifacts(run_dir)


def synchronize_if_cuda_available() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def convert_outputs(raw_outputs: list[Any]) -> list[dict[str, Any]]:
    outputs = []
    for out in raw_outputs:
        completion = out.outputs[0]
        prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
        token_ids = list(completion.token_ids)
        outputs.append(
            {
                "text": completion.text,
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(token_ids),
                "output_ids": token_ids,
            }
        )
    return outputs


def summarize_outputs(
    outputs: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    prompt_tokens = sum(int(o.get("prompt_tokens", 0)) for o in outputs)
    completion_tokens = sum(int(o.get("completion_tokens", 0)) for o in outputs)
    total_tokens = prompt_tokens + completion_tokens
    return {
        "num_outputs": len(outputs),
        "elapsed_sec": elapsed_sec,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_throughput_per_sec": (
            len(outputs) / elapsed_sec if elapsed_sec else 0.0
        ),
        "completion_token_throughput_tok_per_sec": (
            completion_tokens / elapsed_sec if elapsed_sec else 0.0
        ),
        "total_token_throughput_tok_per_sec": (
            total_tokens / elapsed_sec if elapsed_sec else 0.0
        ),
    }


def print_throughput(summary: dict[str, Any]) -> None:
    print("THROUGHPUT")
    print(f"  elapsed_sec: {summary['elapsed_sec']:.3f}")
    print(f"  requests/s: {summary['request_throughput_per_sec']:.2f}")
    print(
        "  completion tok/s: "
        f"{summary['completion_token_throughput_tok_per_sec']:.2f} "
        f"({summary['completion_tokens']} tokens)"
    )
    print(
        "  prompt+completion tok/s: "
        f"{summary['total_token_throughput_tok_per_sec']:.2f} "
        f"({summary['total_tokens']} tokens)"
    )


def shutdown_llm(llm: Any) -> None:
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    core_shutdown = getattr(engine_core, "shutdown", None)
    if callable(core_shutdown):
        core_shutdown()


def build_llm(args: argparse.Namespace, run_dir: Path) -> Any:
    from vllm import LLM
    from vllm.config import ProfilerConfig

    profiler_config = None
    if args.use_profiler:
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(run_dir),
            torch_profiler_with_stack=args.with_stack,
            torch_profiler_with_flops=args.with_flops,
            torch_profiler_use_gzip=args.use_gzip,
            torch_profiler_record_shapes=args.record_shapes,
            torch_profiler_with_memory=args.with_memory,
            delay_iterations=args.delay_iterations,
            max_iterations=args.max_profile_iterations,
            warmup_iterations=args.profiler_warmup_iterations,
            active_iterations=args.profiler_active_iterations,
        )

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tp_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "async_scheduling": False,
    }
    if profiler_config is not None:
        llm_kwargs["profiler_config"] = profiler_config
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    return LLM(**llm_kwargs)


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive.")
    if args.profile_runs <= 0:
        raise SystemExit("--profile-runs must be positive.")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative.")

    from vllm import SamplingParams

    prompts = select_prompts(args)
    run_dir = build_run_dir(args.output_dir)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    write_json(
        run_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "output_dir": str(run_dir),
            "num_prompts": args.num_prompts,
            "warmup_runs": args.warmup_runs,
            "profile_runs": args.profile_runs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "prompt_source": args.prompt_source,
            "gsm8k_split": args.gsm8k_split,
            "gsm8k_start_index": args.gsm8k_start_index,
            "prompt_tokenizer_path": args.prompt_tokenizer_path or args.model_path,
            "profiler": {
                "enabled": args.use_profiler,
                "with_stack": args.with_stack,
                "record_shapes": args.record_shapes,
                "with_memory": args.with_memory,
                "with_flops": args.with_flops,
                "use_gzip": args.use_gzip,
                "delay_iterations": args.delay_iterations,
                "max_profile_iterations": args.max_profile_iterations,
                "profiler_warmup_iterations": args.profiler_warmup_iterations,
                "profiler_active_iterations": args.profiler_active_iterations,
            },
            "prompts": prompts,
        },
    )

    print(f"PROFILE_DIR {run_dir}")
    llm = build_llm(args, run_dir)
    try:
        engine_config = getattr(llm, "llm_engine", None)
        vllm_config = getattr(engine_config, "vllm_config", None)
        write_json(
            run_dir / "engine_config.json",
            {
                "max_num_seqs": getattr(
                    vllm_config.scheduler_config,
                    "max_num_seqs",
                    None,
                ) if vllm_config is not None else None,
                "max_num_batched_tokens": getattr(
                    vllm_config.scheduler_config,
                    "max_num_batched_tokens",
                    None,
                ) if vllm_config is not None else None,
                "max_model_len": getattr(
                    vllm_config.model_config,
                    "max_model_len",
                    None,
                ) if vllm_config is not None else None,
            },
        )

        for run_idx in range(args.warmup_runs):
            print(f"WARMUP {run_idx + 1}/{args.warmup_runs}")
            llm.generate(prompts, sampling_params, use_tqdm=False)
            synchronize_if_cuda_available()

        all_outputs: list[dict[str, Any]] = []
        if args.use_profiler:
            print("START_PROFILE")
            llm.start_profile(profile_prefix="baseline-vllm")
        try:
            tic = time.perf_counter()
            for run_idx in range(args.profile_runs):
                print(f"PROFILE_RUN {run_idx + 1}/{args.profile_runs}")
                raw_outputs = llm.generate(
                    prompts,
                    sampling_params,
                    use_tqdm=False,
                )
                all_outputs.extend(convert_outputs(raw_outputs))
            synchronize_if_cuda_available()
            elapsed_sec = time.perf_counter() - tic
        finally:
            if args.use_profiler:
                print("STOP_PROFILE")
                llm.stop_profile()

        summary = summarize_outputs(all_outputs, elapsed_sec)
        write_json(run_dir / "outputs.json", all_outputs)
        write_json(run_dir / "summary.json", summary)

        artifacts = wait_for_artifacts(run_dir) if args.use_profiler else []
        write_json(run_dir / "profile_artifacts.json", [str(path) for path in artifacts])
        if args.use_profiler and not artifacts:
            print(
                "WARNING no profiler artifacts found. Check vLLM profiler logs "
                f"and output directory: {run_dir}"
            )

        print("SUMMARY")
        print(json.dumps(summary, indent=2))
        print_throughput(summary)
        print("PROFILE_ARTIFACTS")
        for artifact in artifacts:
            print(str(artifact))
    finally:
        shutdown_llm(llm)


if __name__ == "__main__":
    main()

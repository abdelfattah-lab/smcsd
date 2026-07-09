"""Sweep GSM8K throughput and accuracy for plain vLLM vs SMC vLLM.

Runs each engine/batch-size pair in a fresh subprocess so CUDA memory is
released between measurements. Writes JSON, CSV, and a Markdown table that can
be pasted into Notion.

Example:
  source .venv/bin/activate
  python scripts/benchmark_gsm8k_vllm_smc_sweep.py \
      --num-questions 200 \
      --batch-sizes 1,4,8,16 \
      --max-tokens 512 \
      --gpu-mem 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import queue
import re
import time
import traceback
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_BATCH_SIZES = "1,4,8,16"


@dataclass
class BenchConfig:
    model: str
    draft_model: str
    tokenizer_path: str
    num_questions: int
    gsm8k_split: str
    gsm8k_start_index: int
    max_tokens: int
    temperature: float
    gpu_mem: float
    max_model_len: int
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    particles: int
    gamma: int
    resample_threshold: float
    enforce_eager: bool
    enable_prefix_caching: bool


def normalize_numeric_answer(value: str) -> Optional[str]:
    try:
        dec = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal(1)))
    return format(dec.normalize(), "f")


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return normalize_numeric_answer(match.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def format_instruction(question: str) -> str:
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k_prompts_and_labels(config: BenchConfig) -> tuple[list[str], list[str]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    dataset = load_dataset("openai/gsm8k", "main", split=config.gsm8k_split)
    start = config.gsm8k_start_index
    end = start + config.num_questions
    if start < 0:
        raise ValueError("gsm8k_start_index cannot be negative")
    if end > len(dataset):
        raise ValueError(
            f"GSM8K range [{start}, {end}) exceeds split length {len(dataset)}"
        )

    prompts: list[str] = []
    labels: list[str] = []
    for sample in dataset.select(range(start, end)):
        instruction = format_instruction(sample["question"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        label = extract_answer(sample["answer"])
        if label is None:
            raise ValueError(f"Could not parse GSM8K label: {sample['answer']!r}")
        labels.append(label)
    return prompts, labels


def maybe_add_vllm_limits(kwargs: dict[str, Any], config: BenchConfig) -> None:
    if config.max_num_seqs is not None:
        kwargs["max_num_seqs"] = config.max_num_seqs
    if config.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens


def summarize_predictions(
    *,
    engine_name: str,
    batch_size: int,
    labels: list[str],
    predictions: list[Optional[str]],
    total_completion_tokens: int,
    elapsed_sec: float,
    particle_tokens: int | None = None,
) -> dict[str, Any]:
    n = len(labels)
    correct = sum(pred == label for pred, label in zip(predictions, labels))
    invalid = sum(pred is None for pred in predictions)
    result = {
        "engine": engine_name,
        "batch_size": batch_size,
        "num_questions": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "invalid": invalid,
        "invalid_rate": invalid / n if n else 0.0,
        "completion_tokens": total_completion_tokens,
        "elapsed_sec": elapsed_sec,
        "completion_tok_s": (
            total_completion_tokens / elapsed_sec if elapsed_sec else 0.0
        ),
        "requests_s": n / elapsed_sec if elapsed_sec else 0.0,
        "status": "ok",
        "error": "",
    }
    if particle_tokens is not None:
        result["particle_tokens"] = particle_tokens
        result["particle_tok_s"] = particle_tokens / elapsed_sec if elapsed_sec else 0.0
    else:
        result["particle_tokens"] = None
        result["particle_tok_s"] = None
    return result


def run_baseline_worker(
    config_dict: dict[str, Any],
    batch_size: int,
    out_queue: mp.Queue,
) -> None:
    try:
        from vllm import LLM, SamplingParams

        config = BenchConfig(**config_dict)
        prompts, labels = load_gsm8k_prompts_and_labels(config)

        llm_kwargs: dict[str, Any] = {
            "model": config.model,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_mem,
            "enforce_eager": config.enforce_eager,
            "async_scheduling": False,
            "enable_prefix_caching": config.enable_prefix_caching,
        }
        maybe_add_vllm_limits(llm_kwargs, config)
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        predictions: list[Optional[str]] = []
        total_completion_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            raw_outputs = llm.generate(batch, sampling_params, use_tqdm=False)
            for out in raw_outputs:
                text = out.outputs[0].text
                total_completion_tokens += len(out.outputs[0].token_ids)
                predictions.append(extract_answer(text))
        elapsed_sec = time.perf_counter() - tic

        out_queue.put(
            (
                "ok",
                summarize_predictions(
                    engine_name="vllm_baseline",
                    batch_size=batch_size,
                    labels=labels,
                    predictions=predictions,
                    total_completion_tokens=total_completion_tokens,
                    elapsed_sec=elapsed_sec,
                ),
            )
        )
    except Exception as exc:
        out_queue.put(
            (
                "err",
                {
                    "engine": "vllm_baseline",
                    "batch_size": batch_size,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                },
            )
        )


def run_smc_worker(
    config_dict: dict[str, Any],
    batch_size: int,
    out_queue: mp.Queue,
) -> None:
    try:
        from smcsd.vllm_backend.engine import SMCVLLMEngine

        config = BenchConfig(**config_dict)
        prompts, labels = load_gsm8k_prompts_and_labels(config)

        engine_kwargs: dict[str, Any] = {
            "gpu_memory_utilization": config.gpu_mem,
            "enforce_eager": config.enforce_eager,
            "enable_prefix_caching": config.enable_prefix_caching,
        }
        maybe_add_vllm_limits(engine_kwargs, config)
        engine = SMCVLLMEngine(
            model_path=config.model,
            draft_model_path=config.draft_model,
            n_particles=config.particles,
            gamma=config.gamma,
            resample_threshold=config.resample_threshold,
            tp_size=1,
            max_model_len=config.max_model_len,
            **engine_kwargs,
        )
        sampling_params = {
            "draft_temperature": config.temperature,
            "target_temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        predictions: list[Optional[str]] = []
        total_completion_tokens = 0
        total_particle_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            outputs = engine.generate(prompt=batch, sampling_params=sampling_params)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                qi = start + i
                predictions.append(extract_answer(output["text"]))
                total_completion_tokens += int(output.get("completion_tokens", 0))

                particles = output.get("particles", [])
                total_particle_tokens += sum(len(particle) for particle in particles)
        elapsed_sec = time.perf_counter() - tic
        engine.shutdown()

        out_queue.put(
            (
                "ok",
                summarize_predictions(
                    engine_name="smc_vllm",
                    batch_size=batch_size,
                    labels=labels,
                    predictions=predictions,
                    total_completion_tokens=total_completion_tokens,
                    particle_tokens=total_particle_tokens,
                    elapsed_sec=elapsed_sec,
                ),
            )
        )
    except Exception as exc:
        out_queue.put(
            (
                "err",
                {
                    "engine": "smc_vllm",
                    "batch_size": batch_size,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                },
            )
        )


def run_one(
    worker_fn: Any,
    config: BenchConfig,
    batch_size: int,
    timeout_sec: float | None,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(asdict(config), batch_size, out_queue))
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "engine": (
                "smc_vllm" if worker_fn is run_smc_worker else "vllm_baseline"
            ),
            "batch_size": batch_size,
            "status": "timeout",
            "error": f"Timed out after {timeout_sec} seconds",
        }
    try:
        _status, result = out_queue.get_nowait()
        return result
    except queue.Empty:
        return {
            "engine": "smc_vllm" if worker_fn is run_smc_worker else "vllm_baseline",
            "batch_size": batch_size,
            "status": "error",
            "error": f"Subprocess exited with code {proc.exitcode} without result",
        }


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return batch_sizes


def build_run_dir(base_dir: str) -> Path:
    run_dir = Path(base_dir).expanduser().resolve() / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def result_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    engine_order = {"vllm_baseline": 0, "smc_vllm": 1}
    return (int(row.get("batch_size", 0)), engine_order.get(row.get("engine"), 99))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "engine",
        "batch_size",
        "status",
        "num_questions",
        "correct",
        "accuracy",
        "invalid",
        "invalid_rate",
        "completion_tokens",
        "elapsed_sec",
        "completion_tok_s",
        "requests_s",
        "particle_tokens",
        "particle_tok_s",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{100 * float(value):.1f}%"


def format_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def build_markdown_report(config: BenchConfig, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# GSM8K Throughput + Accuracy Benchmark",
        "",
        f"- Target model: `{config.model}`",
        f"- Draft model: `{config.draft_model}`",
        f"- Questions: `{config.num_questions}` from `{config.gsm8k_split}` "
        f"starting at `{config.gsm8k_start_index}`",
        f"- max_tokens: `{config.max_tokens}`",
        f"- temperature: `{config.temperature}`",
        f"- SMC particles/gamma: `{config.particles}` / `{config.gamma}`",
        "",
        "| Engine | Batch | Status | Accuracy | Tok/s | Req/s | Tokens | Time (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=result_sort_key):
        lines.append(
            "| {engine} | {batch} | {status} | {acc} | {tok_s} | "
            "{req_s} | {tokens} | {elapsed} |".format(
                engine=row.get("engine", "-"),
                batch=row.get("batch_size", "-"),
                status=row.get("status", "-"),
                acc=format_pct(row.get("accuracy")),
                tok_s=format_float(row.get("completion_tok_s")),
                req_s=format_float(row.get("requests_s")),
                tokens=row.get("completion_tokens", "-"),
                elapsed=format_float(row.get("elapsed_sec")),
            )
        )

    speedup_lines = [
        "",
        "## SMC Speedup vs Baseline",
        "",
        "| Batch | Baseline tok/s | SMC tok/s | Speedup |",
        "|---:|---:|---:|---:|",
    ]
    by_batch: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_batch.setdefault(int(row["batch_size"]), {})[row["engine"]] = row
    for batch_size in sorted(by_batch):
        baseline = by_batch[batch_size].get("vllm_baseline")
        smc = by_batch[batch_size].get("smc_vllm")
        if not baseline or not smc:
            continue
        baseline_tps = float(baseline["completion_tok_s"])
        smc_tps = float(smc["completion_tok_s"])
        speedup = smc_tps / baseline_tps if baseline_tps else 0.0
        speedup_lines.append(
            f"| {batch_size} | {baseline_tps:.2f} | {smc_tps:.2f} | {speedup:.2f}x |"
        )
    return "\n".join(lines + speedup_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep GSM8K throughput and accuracy for vLLM baseline and SMC vLLM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer used to build GSM8K chat prompts. Defaults to --model.",
    )
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--gsm8k-split", default="test")
    parser.add_argument("--gsm8k-start-index", type=int, default=0)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes(DEFAULT_BATCH_SIZES))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--particles", "-N", type=int, default=12)
    parser.add_argument("--gamma", "-g", type=int, default=8)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=["both", "baseline", "smc"],
        default="both",
        help="Which engine(s) to run.",
    )
    parser.add_argument("--output-dir", default="/root/smcsd/tmp/gsm8k-vllm-smc-bench")
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_questions <= 0:
        raise SystemExit("--num-questions must be positive.")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive.")

    config = BenchConfig(
        model=args.model,
        draft_model=args.draft_model,
        tokenizer_path=args.tokenizer_path or args.model,
        num_questions=args.num_questions,
        gsm8k_split=args.gsm8k_split,
        gsm8k_start_index=args.gsm8k_start_index,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        particles=args.particles,
        gamma=args.gamma,
        resample_threshold=args.resample_threshold,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    run_dir = build_run_dir(args.output_dir)
    print(f"RUN_DIR {run_dir}")
    write_json(run_dir / "run_config.json", asdict(config) | {
        "batch_sizes": args.batch_sizes,
        "mode": args.mode,
    })

    jobs: list[tuple[str, Any, int]] = []
    for batch_size in args.batch_sizes:
        if args.mode in ("both", "baseline"):
            jobs.append(("vllm_baseline", run_baseline_worker, batch_size))
        if args.mode in ("both", "smc"):
            jobs.append(("smc_vllm", run_smc_worker, batch_size))

    rows: list[dict[str, Any]] = []
    for engine_name, worker_fn, batch_size in jobs:
        print(f"RUN engine={engine_name} batch_size={batch_size}", flush=True)
        result = run_one(worker_fn, config, batch_size, args.timeout_sec)
        rows.append(result)
        if result.get("status") == "ok":
            print(
                "RESULT "
                f"engine={result['engine']} batch={result['batch_size']} "
                f"acc={100 * result['accuracy']:.1f}% "
                f"tok/s={result['completion_tok_s']:.2f} "
                f"elapsed={result['elapsed_sec']:.1f}s",
                flush=True,
            )
        else:
            print(
                f"ERROR engine={result.get('engine')} batch={batch_size}: "
                f"{result.get('error', '')}",
                flush=True,
            )

        write_json(run_dir / "results.json", sorted(rows, key=result_sort_key))
        write_csv(run_dir / "results.csv", sorted(rows, key=result_sort_key))
        (run_dir / "report.md").write_text(
            build_markdown_report(config, rows),
            encoding="utf-8",
        )

    print()
    print(build_markdown_report(config, rows))
    print(f"WROTE {run_dir / 'results.json'}")
    print(f"WROTE {run_dir / 'results.csv'}")
    print(f"WROTE {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()

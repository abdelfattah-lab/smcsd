"""GSM8K benchmark for SMC speculative decoding.

Two modes with identical preprocessing for fair comparison:
  - smc_engine: SMC via the dedicated SMCEngine (offline, no tokenizer manager)
  - baseline:   vanilla generation (no speculative decoding)

Usage:
  # SMCEngine (dedicated offline engine)
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine -N 8 -g 8

  # Baseline (no speculative decoding)
  python scripts/smc/accuracy_test_gsm8k.py --mode baseline

  # Custom models
  python scripts/smc/accuracy_test_gsm8k.py --mode smc_engine \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --draft-model meta-llama/Llama-3.2-1B-Instruct \
      -N 8 -g 8
"""

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Shared preprocessing (identical across all modes)
# ---------------------------------------------------------------------------

def normalize_numeric_answer(value: str) -> Optional[str]:
    """Normalize equivalent numeric strings, e.g. 75.00 -> 75."""
    try:
        dec = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if dec == dec.to_integral_value():
        return str(dec.quantize(Decimal(1)))
    return format(dec.normalize(), "f")

def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output or gold answer."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return normalize_numeric_answer(match.group(1))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def format_instruction(question: str) -> str:
    """Build the instruction prompt for a GSM8K question."""
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k_records(
    tokenizer,
    num_questions: int,
    *,
    disable_thinking: bool = False,
    split: str = "test",
    start_index: int = 0,
):
    """Load GSM8K into self-contained records for reproducible studies."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    end_index = start_index + num_questions
    if start_index < 0 or end_index > len(dataset):
        raise ValueError(
            f"Requested GSM8K rows [{start_index}, {end_index}), but split "
            f"{split!r} has {len(dataset)} rows."
        )

    records = []
    for dataset_index, sample in zip(
        range(start_index, end_index),
        dataset.select(range(start_index, end_index)),
    ):
        instruction = format_instruction(sample["question"])
        chat_template_kwargs = {}
        if disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        records.append(
            {
                "dataset": "openai/gsm8k",
                "dataset_config": "main",
                "split": split,
                "dataset_index": dataset_index,
                "problem_id": f"gsm8k-{split}-{dataset_index}",
                "problem": sample["question"],
                "prompt": prompt,
                "gold_answer": extract_answer(sample["answer"]),
            }
        )
    assert all(r["gold_answer"] is not None for r in records), (
        "Some gold labels could not be parsed"
    )
    return records


def load_gsm8k(
    tokenizer,
    num_questions: int,
    *,
    disable_thinking: bool = False,
    split: str = "test",
    start_index: int = 0,
):
    """Compatibility wrapper returning prompts + labels."""
    records = load_gsm8k_records(
        tokenizer,
        num_questions,
        disable_thinking=disable_thinking,
        split=split,
        start_index=start_index,
    )
    return (
        [r["prompt"] for r in records],
        [r["gold_answer"] for r in records],
    )


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def run_smc_engine_eval(args, prompts, labels):
    """Evaluation using the dedicated SMCEngine (offline, no tokenizer manager)."""
    from smcsd.engine import SMCEngine

    draft_model = args.draft_model or DEFAULT_DRAFT_MODEL
    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.resample_threshold
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    else:
        engine_kwargs["max_running_requests"] = max(args.particles + 4, 16)
    if args.max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.max_total_tokens
    if getattr(args, "dtype", None):
        engine_kwargs["dtype"] = args.dtype
    if getattr(args, "tp", 1) > 1:
        engine_kwargs["tp_size"] = args.tp
        engine_kwargs["disable_custom_all_reduce"] = True
        engine_kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
    if getattr(args, "tp_size", 1) and args.tp_size > 1:
        engine_kwargs["tp_size"] = args.tp_size
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
    }

    with SMCEngine(**engine_kwargs) as engine:
        preds = []
        _pmaj_votes = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                total_output_tokens += output["completion_tokens"]
                ptexts = output.get("smc_particle_texts")
                _pmaj_votes.append(
                    [extract_answer(t) for t in ptexts] if ptexts else None
                )
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    from collections import Counter as _Counter

    pm = pk = valid = 0
    for votes, gold in zip(_pmaj_votes, labels):
        if votes is None:
            continue
        valid += 1
        vs = [v for v in votes if v is not None]
        if vs and _Counter(vs).most_common(1)[0][0] == gold:
            pm += 1
        if gold in vs:
            pk += 1
    if valid:
        print(
            f"\n  Particle-majority:  {pm}/{valid} ({100 * pm / valid:.1f}%)"
            f"\n  Pass@particles:     {pk}/{valid} ({100 * pk / valid:.1f}%)"
        )
    return preds, total_output_tokens, latency


def run_majority_eval(args, prompts, labels, records=None):
    """Fully batched stock-engine majority@n on the extracted numeric answer.

    Returns majority-vote preds; total tokens count every sample generated
    (the honest cost of the method).
    """
    import sglang as sgl
    from collections import Counter

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
        base_gpu_id=getattr(args, "base_gpu_id", 0),
    )
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if getattr(args, "max_running_requests", None) is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if getattr(args, "tp", 1) > 1:
        engine_kwargs["tp_size"] = args.tp
        engine_kwargs["disable_custom_all_reduce"] = True
        engine_kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True
    engine = sgl.Engine(**engine_kwargs)
    try:
        n = args.n_samples
        expanded = [p for p in prompts for _ in range(n)]
        tic = time.perf_counter()
        outs = engine.generate(
            expanded,
            {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        )
        latency = time.perf_counter() - tic
    finally:
        engine.shutdown()
    total_tokens = sum(len(o["output_ids"]) for o in outs)
    dump_path = getattr(args, "dump_trajectories", None)
    if dump_path:
        if records is None:
            raise ValueError("Rich trajectory dumps require dataset records.")
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as fh:
            for i, record in enumerate(records):
                for j in range(n):
                    output = outs[i * n + j]
                    answer = extract_answer(output["text"])
                    meta = output.get("meta_info", {})
                    fh.write(
                        json.dumps(
                            {
                                "schema_version": 2,
                                **record,
                                # Legacy aliases kept for offline_policy_sim.py.
                                "qid": record["dataset_index"],
                                "sample": j,
                                "sample_id": j,
                                "generator_model": args.model,
                                "generator_seed": args.seed,
                                "temperature": args.temperature,
                                "max_new_tokens": args.max_new_tokens,
                                "generator_output_ids": output["output_ids"],
                                "full_text": output["text"],
                                "text": output["text"],
                                "completion_tokens": len(output["output_ids"]),
                                "finish_reason": meta.get("finish_reason"),
                                "extracted_answer": answer,
                                "answer": answer,
                                "gold": record["gold_answer"],
                                "correct": bool(answer == record["gold_answer"]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    preds = []
    for i in range(len(prompts)):
        votes = [extract_answer(outs[i * n + j]["text"]) for j in range(n)]
        votes = [v for v in votes if v is not None]
        preds.append(Counter(votes).most_common(1)[0][0] if votes else None)
    return preds, total_tokens, latency


def run_baseline_eval(args, prompts, labels):
    """Baseline (vanilla generation, no speculative decoding) evaluation."""
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
    )
    if getattr(args, "tp", 1) > 1:
        engine_kwargs["tp_size"] = args.tp
        engine_kwargs["disable_custom_all_reduce"] = True
        engine_kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        # v0.5.17 split the ServerArgs field into per-mode knobs.
        engine_kwargs["cuda_graph_max_bs_decode"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.max_total_tokens
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with sgl.Engine(**engine_kwargs) as engine:
        preds = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["meta_info"]["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_answer(output["text"]))
                total_output_tokens += output["meta_info"][
                    "completion_tokens"
                ]
            elapsed = time.perf_counter() - tic
            correct = sum(
                p == l for p, l in zip(preds, labels[: len(preds)])
            )
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    return preds, total_output_tokens, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    mode_label = {
        "smc_engine": "SMCEngine (dedicated offline)",
        "baseline": "Baseline (vanilla)",
        "majority": "Majority vote (batched vanilla)",
    }
    print(f"Mode: {mode_label[args.mode]} | Model: {args.model}")
    if args.mode == "smc_engine":
        draft = args.draft_model or DEFAULT_DRAFT_MODEL
        print(
            f"  particles={args.particles}, gamma={args.gamma}, "
            f"temperature={args.temperature}, draft={draft}"
        )
    print(
        f"  num_questions={args.num_questions}, max_new_tokens={args.max_new_tokens}, "
        f"ignore_eos={args.ignore_eos}"
    )
    if args.disable_thinking:
        print("  chat_template: enable_thinking=False")
    print()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load tokenizer and data (shared across all modes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_gsm8k_records(
        tokenizer,
        args.num_questions,
        disable_thinking=args.disable_thinking,
        split=args.split,
        start_index=args.start_index,
    )
    prompts = [r["prompt"] for r in records]
    labels = [r["gold_answer"] for r in records]

    # Run evaluation
    if args.mode == "smc_engine":
        preds, total_tokens, latency = run_smc_engine_eval(args, prompts, labels)
    elif args.mode == "majority":
        preds, total_tokens, latency = run_majority_eval(
            args, prompts, labels, records
        )
    else:
        preds, total_tokens, latency = run_baseline_eval(args, prompts, labels)

    # Report
    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    print(f"\n{'=' * 55}")
    print(f"  {mode_label[args.mode]}")
    if args.mode == "smc_engine":
        print(f"  N={args.particles}, γ={args.gamma}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:           {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Output throughput: {total_tokens / latency:.1f} tok/s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {latency:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument(
        "--mode",
        choices=["baseline", "smc_engine", "majority"],
        default="smc_engine",
        help="baseline = vanilla, smc_engine = dedicated SMCEngine (default: smc_engine)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"target model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=f"draft model path (default: {DEFAULT_DRAFT_MODEL})",
    )

    # SMC parameters (used by smc_engine mode)
    smc_grp = parser.add_argument_group("SMC parameters")
    smc_grp.add_argument("--particles", "-N", type=int, default=4)
    smc_grp.add_argument("--gamma", "-g", type=int, default=4)
    smc_grp.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="draft temperature (default: 0.7)",
    )
    smc_grp.add_argument(
        "--seed", type=int, default=None, help="numpy seed for reproducibility"
    )
    smc_grp.add_argument(
        "--resample-threshold", type=float, default=None,
        help="ESS resample threshold (default: 0.5, use 0 to disable resampling)",
    )
    # Benchmark
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=80)
    bench.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="GSM8K split (use train for semantic-verifier development).",
    )
    bench.add_argument("--start-index", type=int, default=0)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--n-samples", type=int, default=8, help="majority mode votes")
    bench.add_argument(
        "--dump-trajectories",
        default=None,
        help="Majority mode: write one self-contained JSONL row per sample.",
    )
    bench.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pass ignore_eos through engine sampling_params for throughput comparisons",
    )
    bench.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Pass enable_thinking=False to Qwen-style chat templates.",
    )

    # Engine overrides (smc_engine / baseline modes)
    eng = parser.add_argument_group("engine overrides (smc_engine/baseline)")
    eng.add_argument("--dtype", type=str, default=None,
                     help="model dtype override (e.g. bfloat16, float16)")
    eng.add_argument("--attention-backend", type=str, default="triton",
                      choices=["triton", "fa3"],
                      help="attention backend for smc_engine mode (default: triton)")
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=128)
    eng.add_argument("--max-running-requests", type=int, default=16)
    eng.add_argument("--tp", type=int, default=1)
    eng.add_argument("--base-gpu-id", type=int, default=0)
    eng.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Cap the KV-token memory pool size (useful for dense hybrid draft smoke tests).",
    )
    eng.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        default=False,
        help="Disable CUDA graphs (faster startup, slower decode; useful for smoke tests).",
    )
    eng.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size for the target (and draft, if shared).",
    )

    args = parser.parse_args()
    main(args)

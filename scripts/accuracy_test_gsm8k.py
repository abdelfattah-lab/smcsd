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
import re
import time
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# Shared preprocessing (identical across all modes)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from model output or gold answer."""
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return numbers[-1].replace(",", "") if numbers else None


def format_instruction(question: str) -> str:
    """Build the instruction prompt for a GSM8K question."""
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k(tokenizer, num_questions: int, enable_thinking: bool = True):
    """Load GSM8K and build chat-template prompts + gold labels.

    ``enable_thinking`` is forwarded to ``tokenizer.apply_chat_template`` for
    Qwen3-family models that gate the ``<think>`` block on this kwarg. Set
    ``False`` to skip thinking and answer directly — much shorter outputs,
    useful when the thinking budget is the bottleneck.
    """
    print(f"Loading GSM8K dataset (enable_thinking={enable_thinking}) ...")
    dataset = load_dataset("gsm8k", "main", split="test")

    prompts = []
    labels = []
    for sample in dataset.select(range(num_questions)):
        instruction = format_instruction(sample["question"])
        chat_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 / Qwen3.5 / Qwen3-Next chat templates honor enable_thinking.
        # On templates that don't, the kwarg is silently ignored.
        chat_kwargs["enable_thinking"] = enable_thinking
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                **chat_kwargs,
            )
        except TypeError:
            # Older tokenizer signatures may not accept enable_thinking.
            chat_kwargs.pop("enable_thinking")
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                **chat_kwargs,
            )
        prompts.append(prompt)
        labels.append(extract_answer(sample["answer"]))
    assert all(l is not None for l in labels), "Some gold labels could not be parsed"
    return prompts, labels


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
        tp_size=args.tp_size,
    )
    if args.context_length is not None:
        engine_kwargs["context_length"] = args.context_length
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
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
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with SMCEngine(**engine_kwargs) as engine:
        preds = []
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


def run_baseline_eval(args, prompts, labels):
    """Baseline (vanilla generation, no speculative decoding) evaluation."""
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.context_length is not None:
        engine_kwargs["context_length"] = args.context_length
    if getattr(args, "disable_cuda_graph", False):
        engine_kwargs["disable_cuda_graph"] = True
    if args.tp_size > 1:
        engine_kwargs["tp_size"] = args.tp_size

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "temperature": args.temperature,
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
    print()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load tokenizer and data (shared across all modes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, labels = load_gsm8k(
        tokenizer, args.num_questions, enable_thinking=args.enable_thinking
    )

    # Run evaluation
    if args.mode == "smc_engine":
        preds, total_tokens, latency = run_smc_engine_eval(args, prompts, labels)
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
        choices=["baseline", "smc_engine"],
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
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pass ignore_eos through engine sampling_params for throughput comparisons",
    )
    bench.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="for Qwen3 / Qwen3.5 / Qwen3-Next chat templates: include the "
             "<think> block (default True). Pass --no-enable-thinking to "
             "disable thinking — outputs are much shorter and don't blow "
             "through the max_new_tokens budget on hard questions.",
    )

    # Engine overrides (smc_engine / baseline modes)
    eng = parser.add_argument_group("engine overrides (smc_engine/baseline)")
    eng.add_argument("--attention-backend", type=str, default="triton",
                      choices=["triton", "fa3"],
                      help="attention backend for smc_engine mode (default: triton)")
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=128)
    eng.add_argument("--max-running-requests", type=int, default=16)
    eng.add_argument("--tp-size", type=int, default=1,
                      help="tensor-parallel size for the target (and draft, if shared)")
    eng.add_argument("--context-length", type=int, default=None,
                      help="cap context length to this many tokens (helps fit hybrid Qwen3.5/Next "
                           "models whose declared max_position_embeddings is much longer than "
                           "what the rope-derived effective length supports)")
    eng.add_argument("--disable-cuda-graph", action="store_true",
                      help="skip CUDA graph capture (slower decode but avoids capture-time OOMs "
                           "on tight memory budgets; useful while bringing up a new model)")

    args = parser.parse_args()
    main(args)

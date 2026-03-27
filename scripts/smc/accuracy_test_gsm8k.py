"""GSM8K benchmark for SMC speculative decoding (offline engine API).

Usage:
  # SMC mode (default) — just run it
  python scripts/smc/accuracy_test_gsm8k.py

  # Baseline (no speculative decoding)
  python scripts/smc/accuracy_test_gsm8k.py --mode baseline

  # Custom model
  python scripts/smc/accuracy_test_gsm8k.py --model meta-llama/Llama-3-8B

  # Override SMC parameters
  python scripts/smc/accuracy_test_gsm8k.py --particles 8 --gamma 6 --temperature 0.5

  # More questions for a thorough benchmark
  python scripts/smc/accuracy_test_gsm8k.py --num-questions 200
"""
import argparse
import ast
import os
import re
import time

import numpy as np
import sglang as sgl

from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def extract_first_answer(text):
    """Extract the answer from the first #### in the output."""
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    return text


def build_engine_kwargs(args):
    """Build engine kwargs from args, applying mode-based defaults."""
    kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
        log_level="info",
    )

    if args.mode == "smc":
        kwargs["speculative_algorithm"] = "SMC"
        kwargs["speculative_draft_model_path"] = args.draft_model or args.model
        kwargs["smc_n_particles"] = args.particles
        kwargs["smc_gamma"] = args.gamma
        kwargs["smc_draft_temperature"] = args.temperature
        kwargs["smc_target_temperature"] = args.temperature
        kwargs["page_size"] = 1
        kwargs["attention_backend"] = "triton"

    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        kwargs["max_running_requests"] = args.max_running_requests

    return kwargs


def main(args):
    # Print config
    print(f"Mode: {args.mode} | Model: {args.model}")
    if args.mode == "smc":
        draft = args.draft_model or args.model
        print(f"SMC: particles={args.particles}, gamma={args.gamma}, "
              f"temperature={args.temperature}, draft={draft}")
    print(f"Questions: {args.num_questions}, shots: {args.num_shots}, "
          f"max_new_tokens: {args.max_new_tokens}")
    print()

    # Read data
    data_path = args.data_path
    gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if not os.path.isfile(data_path):
        data_path = download_and_cache_file(gsm8k_url)
    lines = list(read_jsonl(data_path))

    # Construct prompts
    few_shot_examples = get_few_shot_examples(lines, args.num_shots)
    questions = []
    labels = []
    for i in range(len(lines[: args.num_questions])):
        questions.append(few_shot_examples + get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)

    engine_kwargs = build_engine_kwargs(args)
    sampling_params = {"max_new_tokens": args.max_new_tokens}

    with sgl.Engine(**engine_kwargs) as engine:
        preds = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(questions), args.batch_size):
            batch = questions[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 5:
                    print(f"--- Q{qi} ({output['meta_info']['completion_tokens']} tokens) ---")
                    print(output["text"][:500])
                    print()
                answer_str = extract_first_answer(output["text"])
                preds.append(get_answer_value(answer_str))
                total_output_tokens += output["meta_info"]["completion_tokens"]
            elapsed = time.perf_counter() - tic
            correct = sum(p == l for p, l in zip(preds, labels[:len(preds)]))
            print(
                f"\r[{len(preds)}/{len(questions)}] "
                f"acc={correct}/{len(preds)} ({correct/len(preds):.1%}) "
                f"tps={total_output_tokens/elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)
    output_throughput = total_output_tokens / latency

    print(f"\nAccuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Num questions: {args.num_questions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument("--mode", choices=["baseline", "smc"], default="smc",
                        help="baseline = vanilla, smc = speculative (default: smc)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"target model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="draft model path (default: same as --model)")

    # SMC parameters (only used when --mode smc)
    smc = parser.add_argument_group("SMC parameters (--mode smc)")
    smc.add_argument("--particles", type=int, default=4)
    smc.add_argument("--gamma", type=int, default=4)
    smc.add_argument("--temperature", type=float, default=0.7,
                     help="draft and target temperature (default: 0.7)")

    # Benchmark
    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=20)
    bench.add_argument("--num-shots", type=int, default=5)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=4)
    bench.add_argument("--data-path", type=str, default="test.jsonl")

    # Engine overrides
    eng = parser.add_argument_group("engine overrides")
    eng.add_argument("--mem-fraction-static", type=float, default=0.45)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=16)
    eng.add_argument("--max-running-requests", type=int, default=8)

    args = parser.parse_args()
    main(args)

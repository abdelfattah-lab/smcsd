"""GSM8K benchmark for SMC speculative decoding (offline engine API).

Uses sglang's offline Engine API instead of a running server.
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


def main(args):
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

    # Build engine kwargs
    engine_kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=True,
        log_level="info",
    )
    if args.speculative_algorithm:
        engine_kwargs["speculative_algorithm"] = args.speculative_algorithm
        engine_kwargs["speculative_draft_model_path"] = (
            args.speculative_draft_model_path or args.model_path
        )
        engine_kwargs["smc_n_particles"] = args.smc_n_particles
        engine_kwargs["smc_gamma"] = args.smc_gamma
        engine_kwargs["smc_draft_temperature"] = args.smc_draft_temperature
        engine_kwargs["smc_target_temperature"] = args.smc_target_temperature
        engine_kwargs["page_size"] = 1
        engine_kwargs["attention_backend"] = "triton"
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests

    sampling_params = {"max_new_tokens": args.max_new_tokens}

    batch_size = args.batch_size

    with sgl.Engine(**engine_kwargs) as engine:
        # Run inference in small batches
        preds = []
        total_output_tokens = 0
        tic = time.perf_counter()
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--speculative-algorithm", type=str, default=None,
                        help="e.g. SMC")
    parser.add_argument("--speculative-draft-model-path", type=str, default=None)
    parser.add_argument("--smc-n-particles", type=int, default=4)
    parser.add_argument("--smc-gamma", type=int, default=4)
    parser.add_argument("--smc-draft-temperature", type=float, default=0.7)
    parser.add_argument("--smc-target-temperature", type=float, default=0.7)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of questions per engine.generate() call")
    args = parser.parse_args()
    main(args)

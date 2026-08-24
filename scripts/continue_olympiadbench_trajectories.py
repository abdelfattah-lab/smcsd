"""Extend length-capped OlympiadBench trajectories from exact saved token IDs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

try:
    from scripts.accuracy_test_olympiadbench import (
        cluster_answers,
        extract_boxed_answers,
        summarize_rows,
    )
    from scripts.olympiadbench_judge import OlympiadBenchJudge
except ModuleNotFoundError:
    from accuracy_test_olympiadbench import (
        cluster_answers,
        extract_boxed_answers,
        summarize_rows,
    )
    from olympiadbench_judge import OlympiadBenchJudge


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--previous-summary", required=True)
    parser.add_argument("--additional-new-tokens", type=int, default=8192)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--dump-trajectories", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.trajectories)
    previous_summary = json.loads(Path(args.previous_summary).read_text())
    model = args.model or rows[0]["generator_model"]
    temperature = (
        args.temperature if args.temperature is not None else rows[0]["temperature"]
    )
    previous_cap = int(rows[0]["max_new_tokens"])
    total_cap = previous_cap + args.additional_new_tokens
    capped_indices = [
        index
        for index, row in enumerate(rows)
        if int(row["completion_tokens"]) >= previous_cap
    ]
    if not capped_indices:
        raise ValueError("No trajectories reached the previous token cap.")

    tokenizer = AutoTokenizer.from_pretrained(model)
    continuation_inputs = []
    for index in capped_indices:
        row = rows[index]
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        continuation_inputs.append(prompt_ids + list(row["generator_output_ids"]))

    import sglang as sgl

    initialization_started = time.perf_counter()
    engine = sgl.Engine(
        model_path=model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=args.seed,
        base_gpu_id=args.base_gpu_id,
        tp_size=args.tp,
        disable_custom_all_reduce=args.tp > 1,
        enforce_disable_flashinfer_allreduce_fusion=args.tp > 1,
    )
    initialization_time = time.perf_counter() - initialization_started
    try:
        inference_started = time.perf_counter()
        outputs = engine.generate(
            input_ids=continuation_inputs,
            sampling_params={
                "max_new_tokens": args.additional_new_tokens,
                "temperature": temperature,
            },
        )
        inference_time = time.perf_counter() - inference_started
    finally:
        engine.shutdown()

    continuation_tokens = 0
    for row_index, output in zip(capped_indices, outputs):
        row = rows[row_index]
        old_ids = list(row["generator_output_ids"])
        new_ids = list(output["output_ids"])
        combined_ids = old_ids + new_ids
        combined_text = tokenizer.decode(combined_ids, skip_special_tokens=True)
        continuation_tokens += len(new_ids)
        row.update(
            {
                "generator_output_ids": combined_ids,
                "full_text": combined_text,
                "text": combined_text,
                "completion_tokens": len(combined_ids),
                "finish_reason": output.get("meta_info", {}).get("finish_reason"),
                "raw_extracted_answer": extract_boxed_answers(combined_text),
                "max_new_tokens": total_cap,
                "continuation_seed": args.seed,
                "continuation_tokens": len(new_ids),
                "continuation_stages": 2,
            }
        )
    for index, row in enumerate(rows):
        if index not in set(capped_indices):
            row["max_new_tokens"] = total_cap
            row["continuation_tokens"] = 0
            row["continuation_stages"] = 1

    judge = OlympiadBenchJudge()
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    for problem_rows in by_problem.values():
        problem_rows.sort(key=lambda row: int(row["sample_id"]))
        raw_answers = [row["raw_extracted_answer"] for row in problem_rows]
        answers, correctness = cluster_answers(
            raw_answers,
            problem_rows[0]["gold_answer"],
            float(problem_rows[0]["grading_precision"]),
            judge,
        )
        for row, answer, correct in zip(problem_rows, answers, correctness):
            row["extracted_answer"] = answer
            row["answer"] = answer
            row["correct"] = bool(correct)

    previous_inference_time = previous_summary["cost"]["inference_wall_time_s"]
    summary = summarize_rows(rows, previous_inference_time + inference_time)
    summary["cost"].update(
        {
            "engine_initialization_time_s": initialization_time,
            "continuation_inference_wall_time_s": inference_time,
            "previous_inference_wall_time_s": previous_inference_time,
            "continuation_output_tokens": continuation_tokens,
            "trajectories_continued": len(capped_indices),
        }
    )
    summary["generation"].update(
        {
            "continuation_from_cap": previous_cap,
            "continuation_seed": args.seed,
            "sampling_is_two_stage_conditional": True,
        }
    )

    output_path = Path(args.dump_trajectories)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    quality = summary["quality"]
    cost = summary["cost"]
    print(f"continued trajectories: {len(capped_indices)}")
    print(f"continuation tokens:    {continuation_tokens}")
    print(f"sample accuracy:        {quality['sample_accuracy']:.1%}")
    print(f"self-consistency@8:     {quality['self_consistency_accuracy']:.1%}")
    print(f"oracle pass@8:          {quality['oracle_pass_at_n']:.1%}")
    print(f"mixed problems:         {quality['mixed_problems']}")
    print(f"trajectories at 16K:    {cost['trajectories_at_token_cap']}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

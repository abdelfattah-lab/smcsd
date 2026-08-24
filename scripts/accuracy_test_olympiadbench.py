"""Long-form OlympiadBench trajectory pilot for semantic test-time scaling.

The benchmark is the official text-only, English, open-ended competition-math
subset. Problems are selected from a frozen random permutation so a pilot,
development split, and holdout split can be extended without overlap.

Only stock autoregressive sampling is implemented here. The goal is to measure
whether a dataset has enough trajectory length, mixed outcomes, and oracle
headroom to justify expensive semantic-prefix scoring and online SMC work.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from scripts.olympiadbench_judge import OlympiadBenchJudge
except ModuleNotFoundError:
    from olympiadbench_judge import OlympiadBenchJudge


DATASET = "lscpku/OlympiadBench-official"
DATASET_CONFIG = "OE_TO_maths_en_COMP"
DATASET_SPLIT = "train"
INSTRUCTION = (
    "Solve the following olympiad-level mathematics problem. Develop a rigorous "
    "solution step by step. Put the final answer, and only the final answer, "
    "inside \\boxed{} at the end.\n\n"
)


def extract_boxed_answers(text: str) -> str | None:
    """Return all balanced boxed answers, joined as an unordered answer list."""
    answers = []
    start = 0
    marker = "\\boxed{"
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        index = marker_index + len(marker)
        answer_start = index
        depth = 1
        while depth and index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            return None
        answer = text[answer_start : index - 1].strip()
        if answer:
            answers.append(answer)
        start = index
    return ",".join(answers) if answers else None


def frozen_indices(dataset_size: int, selection_seed: int) -> list[int]:
    indices = list(range(dataset_size))
    random.Random(selection_seed).shuffle(indices)
    return indices


def load_olympiadbench_records(
    tokenizer,
    num_questions: int,
    *,
    start_index: int = 0,
    selection_seed: int = 0,
    disable_thinking: bool = False,
) -> list[dict]:
    dataset = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    end_index = start_index + num_questions
    if start_index < 0 or end_index > len(dataset):
        raise ValueError(
            f"Requested frozen-permutation positions [{start_index}, {end_index}), "
            f"but {DATASET_CONFIG} has {len(dataset)} rows."
        )
    selected_indices = frozen_indices(len(dataset), selection_seed)[start_index:end_index]
    chat_kwargs = {"enable_thinking": False} if disable_thinking else {}
    records = []
    for selection_position, dataset_index in enumerate(
        selected_indices, start=start_index
    ):
        row = dataset[int(dataset_index)]
        if len(row["final_answer"]) != 1:
            raise ValueError(f"Expected one answer field for dataset row {dataset_index}")
        problem = "\n\n".join(
            part for part in (row.get("context"), row["question"]) if part
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + problem}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        error = row.get("error")
        records.append(
            {
                "dataset": DATASET,
                "dataset_config": DATASET_CONFIG,
                "split": DATASET_SPLIT,
                "dataset_index": int(dataset_index),
                "dataset_id": int(row["id"]),
                "selection_seed": selection_seed,
                "selection_position": selection_position,
                "problem_id": f"olympiadbench-{DATASET_CONFIG}-{int(row['id'])}",
                "problem": problem,
                "prompt": prompt,
                "gold_answer": row["final_answer"][0],
                "answer_type": row["answer_type"],
                "is_multiple_answer": bool(row["is_multiple_answer"]),
                "unit": row.get("unit"),
                "grading_precision": float(error) if error else 1e-8,
                "subfield": row["subfield"],
            }
        )
    return records


def cluster_answers(
    raw_answers: Sequence[str | None],
    gold_answer: str,
    precision: float,
    judge: OlympiadBenchJudge,
) -> tuple[list[str | None], list[bool]]:
    """Canonicalize equivalent answers for majority voting.

    Correct answers receive the exact gold string, which keeps schema-v2's
    equality invariant. Incorrect but mutually equivalent answers share the
    first observed representative.
    """
    representatives: list[str] = []
    canonical = []
    correctness = []
    for answer in raw_answers:
        if answer is None:
            canonical.append(None)
            correctness.append(False)
            continue
        correct = judge.judge(gold_answer, answer, precision)
        correctness.append(correct)
        if correct:
            canonical.append(gold_answer)
            continue
        for representative in representatives:
            if judge.judge(representative, answer, precision) and judge.judge(
                answer, representative, precision
            ):
                canonical.append(representative)
                break
        else:
            representatives.append(answer)
            canonical.append(answer)
    return canonical, correctness


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize_rows(rows: Sequence[dict], inference_wall_time: float) -> dict:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)
    for problem_rows in by_problem.values():
        problem_rows.sort(key=lambda row: int(row["sample_id"]))

    self_consistency_correct = 0
    self_consistency_invalid = 0
    oracle_correct = 0
    pass_at_2 = 0
    majority_at_2 = 0
    first_two_agree = 0
    mixed = all_correct = all_wrong = 0
    answer_cluster_counts = []
    subfields = defaultdict(lambda: {"problems": 0, "sc_correct": 0, "oracle": 0})
    for problem_rows in by_problem.values():
        correctness = [bool(row["correct"]) for row in problem_rows]
        votes = [row["answer"] for row in problem_rows if row["answer"] is not None]
        vote = Counter(votes).most_common(1)[0][0] if votes else None
        gold = problem_rows[0]["gold_answer"]
        sc_correct = vote == gold
        self_consistency_correct += sc_correct
        self_consistency_invalid += vote is None
        oracle = any(correctness)
        oracle_correct += oracle
        if all(correctness):
            all_correct += 1
        elif oracle:
            mixed += 1
        else:
            all_wrong += 1

        first_two = problem_rows[:2]
        pass_at_2 += any(bool(row["correct"]) for row in first_two)
        two_votes = [row["answer"] for row in first_two if row["answer"] is not None]
        two_vote = Counter(two_votes).most_common(1)[0][0] if two_votes else None
        majority_at_2 += two_vote == gold
        first_two_agree += (
            len(first_two) == 2
            and first_two[0]["answer"] is not None
            and first_two[0]["answer"] == first_two[1]["answer"]
        )
        answer_cluster_counts.append(len(set(votes)))
        field = subfields[problem_rows[0]["subfield"]]
        field["problems"] += 1
        field["sc_correct"] += int(sc_correct)
        field["oracle"] += int(oracle)

    problem_count = len(by_problem)
    tokens = [int(row["completion_tokens"]) for row in rows]
    correct_trajectories = sum(bool(row["correct"]) for row in rows)
    sample_accuracy = correct_trajectories / len(rows)
    sc_accuracy = self_consistency_correct / problem_count
    oracle_accuracy = oracle_correct / problem_count
    mixed_rate = mixed / problem_count
    median_tokens = statistics.median(tokens)
    criteria = {
        "median_completion_tokens_at_least_2000": median_tokens >= 2000,
        "sample_accuracy_between_10_and_70_percent": 0.1 <= sample_accuracy <= 0.7,
        "mixed_problem_rate_at_least_30_percent": mixed_rate >= 0.3,
        "oracle_minus_self_consistency_at_least_5pp": oracle_accuracy - sc_accuracy >= 0.05,
        "oracle_minus_pass_at_2_at_least_10pp": oracle_accuracy - pass_at_2 / problem_count >= 0.1,
    }
    return {
        "schema_version": 1,
        "dataset": {
            "name": rows[0]["dataset"],
            "config": rows[0]["dataset_config"],
            "split": rows[0]["split"],
            "selection_seed": rows[0]["selection_seed"],
            "selection_positions": [
                min(row["selection_position"] for row in rows),
                max(row["selection_position"] for row in rows),
            ],
            "dataset_indices": sorted({int(row["dataset_index"]) for row in rows}),
            "n_problems": problem_count,
            "n_trajectories": len(rows),
        },
        "generation": {
            "model": rows[0]["generator_model"],
            "samples_per_problem": len(rows) // problem_count,
            "temperature": rows[0]["temperature"],
            "max_new_tokens": rows[0]["max_new_tokens"],
            "thinking": "disabled" if rows[0]["disable_thinking"] else "enabled",
            "seed": rows[0]["generator_seed"],
        },
        "quality": {
            "correct_trajectories": correct_trajectories,
            "sample_accuracy": sample_accuracy,
            "self_consistency_accuracy": sc_accuracy,
            "self_consistency_invalid": self_consistency_invalid,
            "oracle_pass_at_n": oracle_accuracy,
            "pass_at_2_fixed": pass_at_2 / problem_count,
            "majority_at_2_fixed": majority_at_2 / problem_count,
            "first_two_answer_agreement": first_two_agree / problem_count,
            "all_correct_problems": all_correct,
            "mixed_problems": mixed,
            "mixed_problem_rate": mixed_rate,
            "all_wrong_problems": all_wrong,
            "oracle_minus_self_consistency": oracle_accuracy - sc_accuracy,
            "oracle_minus_pass_at_2": oracle_accuracy - pass_at_2 / problem_count,
            "mean_distinct_answers_per_problem": statistics.mean(answer_cluster_counts),
            "by_subfield": dict(sorted(subfields.items())),
        },
        "cost": {
            "total_output_tokens": sum(tokens),
            "mean_completion_tokens": statistics.mean(tokens),
            "median_completion_tokens": median_tokens,
            "p25_completion_tokens": percentile(tokens, 0.25),
            "p75_completion_tokens": percentile(tokens, 0.75),
            "p90_completion_tokens": percentile(tokens, 0.9),
            "max_completion_tokens": max(tokens),
            "trajectories_at_token_cap": sum(
                value >= rows[0]["max_new_tokens"] for value in tokens
            ),
            "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in rows)),
            "inference_wall_time_s": inference_wall_time,
            "output_tokens_per_s": sum(tokens) / inference_wall_time,
            "allocated_generator_gpu_seconds": inference_wall_time,
        },
        "semantic_scoring_gate": {
            "criteria": criteria,
            "eligible": all(criteria.values()),
        },
    }


def generate(args, records: Sequence[dict]) -> tuple[list[dict], float, float]:
    import sglang as sgl

    initialization_started = time.perf_counter()
    engine = sgl.Engine(
        model_path=args.model,
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
        prompts = [record["prompt"] for record in records for _ in range(args.n_samples)]
        inference_started = time.perf_counter()
        outputs = engine.generate(
            prompts,
            {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        )
        inference_time = time.perf_counter() - inference_started
    finally:
        engine.shutdown()

    judge = OlympiadBenchJudge()
    rows = []
    for problem_index, record in enumerate(records):
        problem_outputs = outputs[
            problem_index * args.n_samples : (problem_index + 1) * args.n_samples
        ]
        raw_answers = [extract_boxed_answers(output["text"]) for output in problem_outputs]
        answers, correctness = cluster_answers(
            raw_answers,
            record["gold_answer"],
            record["grading_precision"],
            judge,
        )
        for sample_id, (output, raw_answer, answer, correct) in enumerate(
            zip(problem_outputs, raw_answers, answers, correctness)
        ):
            meta = output.get("meta_info", {})
            rows.append(
                {
                    "schema_version": 2,
                    **record,
                    "qid": record["dataset_index"],
                    "sample": sample_id,
                    "sample_id": sample_id,
                    "generator_model": args.model,
                    "generator_seed": args.seed,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "disable_thinking": args.disable_thinking,
                    "generator_output_ids": output["output_ids"],
                    "full_text": output["text"],
                    "text": output["text"],
                    "completion_tokens": len(output["output_ids"]),
                    "finish_reason": meta.get("finish_reason"),
                    "raw_extracted_answer": raw_answer,
                    "extracted_answer": answer,
                    "answer": answer,
                    "gold": record["gold_answer"],
                    "correct": bool(correct),
                    "grading_method": "olympiadbench_official_equivalence",
                }
            )
    return rows, initialization_time, inference_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--dump-trajectories", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_olympiadbench_records(
        tokenizer,
        args.num_questions,
        start_index=args.start_index,
        selection_seed=args.selection_seed,
        disable_thinking=args.disable_thinking,
    )
    rows, initialization_time, inference_time = generate(args, records)
    summary = summarize_rows(rows, inference_time)
    summary["cost"]["engine_initialization_time_s"] = initialization_time

    trajectory_path = Path(args.dump_trajectories)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    quality = summary["quality"]
    cost = summary["cost"]
    print("=" * 72)
    print(f"OlympiadBench pilot: {summary['dataset']['n_problems']} problems")
    print(f"sample accuracy:       {quality['sample_accuracy']:.1%}")
    samples = summary["generation"]["samples_per_problem"]
    print(f"self-consistency@{samples}:    {quality['self_consistency_accuracy']:.1%}")
    print(f"oracle pass@{samples}:         {quality['oracle_pass_at_n']:.1%}")
    print(f"mixed problems:        {quality['mixed_problems']} ({quality['mixed_problem_rate']:.1%})")
    print(f"median output tokens:  {cost['median_completion_tokens']:.0f}")
    print(f"semantic scoring gate: {summary['semantic_scoring_gate']['eligible']}")
    print(f"trajectories:          {trajectory_path}")
    print(f"summary:               {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""Run optimized likelihood-only SMC-SD on the frozen OlympiadBench slice.

The stock OlympiadBench pilot expands every prompt into independent target-model
samples.  This runner instead sends one SMC group per problem through the
repository's optimized ``SMCEngine`` and preserves the complete returned
particle collection.  It reports the engine's posterior sample as well as
particle majority, likelihood-weighted majority, maximum final weight, and the
particle-pool oracle.  All answers use the same official-equivalence judge as
the semantic study.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

try:
    from scripts.accuracy_test_olympiadbench import (
        cluster_answers,
        extract_boxed_answers,
        load_olympiadbench_records,
    )
    from scripts.olympiadbench_judge import OlympiadBenchJudge
except ModuleNotFoundError:
    from accuracy_test_olympiadbench import (
        cluster_answers,
        extract_boxed_answers,
        load_olympiadbench_records,
    )
    from olympiadbench_judge import OlympiadBenchJudge


def stable_vote(answers: Sequence[str | None]) -> str | None:
    votes = [answer for answer in answers if answer is not None]
    return Counter(votes).most_common(1)[0][0] if votes else None


def weighted_vote(
    answers: Sequence[str | None], log_weights: Sequence[float]
) -> str | None:
    if len(answers) != len(log_weights):
        raise ValueError("Answers and log weights must have equal length.")
    finite = [float(weight) for weight in log_weights if math.isfinite(float(weight))]
    if not finite:
        return stable_vote(answers)
    offset = max(finite)
    totals: dict[str, float] = {}
    first_position: dict[str, int] = {}
    for position, (answer, log_weight) in enumerate(zip(answers, log_weights)):
        if answer is None or not math.isfinite(float(log_weight)):
            continue
        totals[answer] = totals.get(answer, 0.0) + math.exp(float(log_weight) - offset)
        first_position.setdefault(answer, position)
    if not totals:
        return None
    return min(totals, key=lambda answer: (-totals[answer], first_position[answer]))


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--power-alpha", type=float, default=1.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--save-particles", required=True)
    parser.add_argument("--save-problems", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_olympiadbench_records(
        tokenizer,
        args.num_questions,
        start_index=args.start_index,
        selection_seed=args.selection_seed,
        disable_thinking=args.disable_thinking,
    )

    from smcsd.engine import SMCEngine

    initialization_started = time.perf_counter()
    engine = SMCEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        power_alpha=args.power_alpha,
        resample_threshold=args.resample_threshold,
        random_seed=args.seed,
        page_size=1,
        attention_backend=args.attention_backend,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        tp_size=args.tp,
        base_gpu_id=args.base_gpu_id,
        disable_custom_all_reduce=args.tp > 1,
        enforce_disable_flashinfer_allreduce_fusion=args.tp > 1,
    )
    initialization_time = time.perf_counter() - initialization_started
    try:
        inference_started = time.perf_counter()
        outputs = engine.generate(
            [record["prompt"] for record in records],
            {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        )
        inference_time = time.perf_counter() - inference_started
    finally:
        engine.shutdown()
    if not isinstance(outputs, list):
        outputs = [outputs]
    if len(outputs) != len(records):
        raise ValueError(f"Expected {len(records)} results, received {len(outputs)}.")

    judge = OlympiadBenchJudge()
    particle_rows = []
    problem_rows = []
    for record, output in zip(records, outputs):
        particle_ids = output.get("smc_particle_output_ids")
        particle_texts = output.get("smc_particle_texts")
        log_weights = output.get("smc_log_w_tilde")
        if not particle_ids or not particle_texts or log_weights is None:
            raise ValueError(
                f"SMC particle side-channel missing for {record['problem_id']}."
            )
        if not (
            len(particle_ids)
            == len(particle_texts)
            == len(log_weights)
            == args.particles
        ):
            raise ValueError(f"Malformed particle collection for {record['problem_id']}.")
        raw_answers = [extract_boxed_answers(text) for text in particle_texts]
        answers, correctness = cluster_answers(
            raw_answers,
            record["gold_answer"],
            record["grading_precision"],
            judge,
        )
        posterior_raw = extract_boxed_answers(output["text"])
        posterior_correct = bool(
            posterior_raw is not None
            and judge.judge(
                record["gold_answer"], posterior_raw, record["grading_precision"]
            )
        )
        majority = stable_vote(answers)
        weighted_majority = weighted_vote(answers, log_weights)
        max_weight_slot = min(
            range(args.particles),
            key=lambda slot: (-float(log_weights[slot]), slot),
        )
        problem_rows.append(
            {
                "problem_id": record["problem_id"],
                "posterior_sample_correct": posterior_correct,
                "particle_majority_correct": majority == record["gold_answer"],
                "likelihood_weighted_majority_correct": (
                    weighted_majority == record["gold_answer"]
                ),
                "max_weight_correct": bool(correctness[max_weight_slot]),
                "oracle_correct": any(correctness),
                "posterior_raw_answer": posterior_raw,
                "particle_majority_answer": majority,
                "likelihood_weighted_majority_answer": weighted_majority,
                "max_weight_slot": max_weight_slot,
                "log_Z_hat": float(output.get("smc_log_Z_hat", float("nan"))),
                "unique_final_token_sequences": len(
                    {tuple(int(token) for token in ids) for ids in particle_ids}
                ),
            }
        )
        for sample_id, (ids, text, raw_answer, answer, correct, log_weight) in enumerate(
            zip(
                particle_ids,
                particle_texts,
                raw_answers,
                answers,
                correctness,
                log_weights,
            )
        ):
            ids = [int(token) for token in ids]
            particle_rows.append(
                {
                    "schema_version": 2,
                    **record,
                    "qid": record["dataset_index"],
                    "sample": sample_id,
                    "sample_id": sample_id,
                    "generator_model": args.model,
                    "draft_model": args.draft_model,
                    "generator_seed": args.seed,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "disable_thinking": args.disable_thinking,
                    "generator_output_ids": ids,
                    "full_text": text,
                    "text": text,
                    "completion_tokens": len(ids),
                    "finish_reason": None,
                    "raw_extracted_answer": raw_answer,
                    "extracted_answer": answer,
                    "answer": answer,
                    "gold": record["gold_answer"],
                    "correct": bool(correct),
                    "grading_method": "olympiadbench_official_equivalence",
                    "smc_final_log_weight": float(log_weight),
                }
            )

    def accuracy(field: str) -> float:
        return statistics.fmean(float(row[field]) for row in problem_rows)

    total_particle_tokens = sum(row["completion_tokens"] for row in particle_rows)
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "optimized_likelihood_smc_sd",
            "model": args.model,
            "draft_model": args.draft_model,
            "uses_target_draft_likelihood_ratio": True,
            "uses_semantic_verifier": False,
            "particles": args.particles,
            "gamma": args.gamma,
            "temperature": args.temperature,
            "power_alpha": args.power_alpha,
            "resample_threshold": args.resample_threshold,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "dataset": {
            "name": records[0]["dataset"],
            "config": records[0]["dataset_config"],
            "n_problems": len(records),
            "selection_seed": args.selection_seed,
            "selection_positions": [args.start_index, args.start_index + len(records) - 1],
        },
        "quality": {
            "posterior_sample_accuracy": accuracy("posterior_sample_correct"),
            "particle_majority_accuracy": accuracy("particle_majority_correct"),
            "likelihood_weighted_majority_accuracy": accuracy(
                "likelihood_weighted_majority_correct"
            ),
            "max_weight_accuracy": accuracy("max_weight_correct"),
            "oracle_pass_at_n": accuracy("oracle_correct"),
            "mean_unique_final_token_sequences": statistics.fmean(
                row["unique_final_token_sequences"] for row in problem_rows
            ),
        },
        "cost": {
            "engine_initialization_time_s": initialization_time,
            "inference_wall_time_s": inference_time,
            "allocated_gpu_seconds": inference_time * args.tp,
            "allocated_gpu_seconds_per_problem": inference_time * args.tp / len(records),
            "resident_gpus": args.tp,
            "total_returned_particle_tokens": total_particle_tokens,
            "returned_particle_tokens_per_problem": total_particle_tokens / len(records),
            "returned_particle_tokens_per_s": total_particle_tokens / inference_time,
            "model_initialization_excluded_from_gpu_seconds": True,
        },
    }
    write_jsonl(args.save_particles, particle_rows)
    write_jsonl(args.save_problems, problem_rows)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()

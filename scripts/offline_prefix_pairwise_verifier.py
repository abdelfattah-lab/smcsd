"""Order-swapped relative verifier on equal-horizon sibling prefixes.

This is a diagnostic screen, not a deployable selector.  It deliberately
evaluates every correct-versus-incorrect pair on mixed-outcome problems to ask
whether a semantic verifier can make the within-problem distinction that SMC
particle allocation needs.  Both A/B orders are scored and symmetrized.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    from scripts import offline_pairwise_verifier as pairwise
except ModuleNotFoundError:
    import offline_pairwise_verifier as pairwise


PREFIX_PAIRWISE_TEMPLATE = """\
You are comparing two partial attempts at the same math problem. Both attempts
have been stopped after exactly {token_position} generator tokens and may be
unfinished.

Problem:
{problem}

Partial solution A:
{solution_a}

Partial solution B:
{solution_b}

Judge only the mathematical path written so far. A committed logical,
factual, or arithmetic error counts against an attempt; a missing step that
could still be supplied does not. Do not reward confidence, polish, or length.

Select the attempt that is more likely to become a fully correct solution if
both are continued with the same additional compute. If both appear viable or
both appear flawed, select the one with fewer substantive errors and the more
reliable reasoning. You must still choose one.

Return exactly A or B and no other text.
Choice:\
"""


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def write_json(path: str | Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_token_positions(value: str) -> list[int]:
    positions = sorted({int(item) for item in value.split(",") if item.strip()})
    if not positions or any(position <= 0 for position in positions):
        raise ValueError("token positions must be comma-separated positive integers.")
    return positions


def validate_trajectories(rows: Sequence[dict], token_positions: Sequence[int]) -> None:
    required = {
        "problem_id",
        "problem",
        "sample_id",
        "generator_model",
        "generator_output_ids",
        "correct",
    }
    seen = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Trajectory row {index} is missing fields: {missing}.")
        key = (str(row["problem_id"]), int(row["sample_id"]))
        if key in seen:
            raise ValueError(f"Duplicate trajectory key: {key}.")
        seen.add(key)
        if len(row["generator_output_ids"]) < max(token_positions):
            raise ValueError(
                f"Trajectory {key} has only {len(row['generator_output_ids'])} "
                f"tokens; exact checkpoint {max(token_positions)} is unavailable."
            )
        groups[key[0]].append(row)
    if not groups:
        raise ValueError("No trajectory groups found.")
    counts = {len(group) for group in groups.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError(f"Candidate counts must be uniform and >=2, got {counts}.")
    for problem_id, group in groups.items():
        if len({row["problem"] for row in group}) != 1:
            raise ValueError(f"Problem text differs within {problem_id}.")


def build_prefix_pairwise_prompt(
    tokenizer,
    *,
    problem: str,
    solution_a: str,
    solution_b: str,
    token_position: int,
) -> str:
    content = PREFIX_PAIRWISE_TEMPLATE.format(
        problem=problem,
        solution_a=solution_a,
        solution_b=solution_b,
        token_position=token_position,
    )
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **kwargs
        )


def make_jobs(
    rows: Sequence[dict],
    generator_tokenizer,
    verifier_tokenizer,
    token_positions: Sequence[int],
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)

    jobs = []
    for token_position in token_positions:
        checkpoint = f"token_{token_position}"
        for problem_id, group in groups.items():
            candidates = sorted(group, key=lambda row: int(row["sample_id"]))
            if not any(bool(row["correct"]) for row in candidates):
                continue
            if all(bool(row["correct"]) for row in candidates):
                continue
            prefixes = {
                int(row["sample_id"]): generator_tokenizer.decode(
                    row["generator_output_ids"][:token_position],
                    skip_special_tokens=True,
                )
                for row in candidates
            }
            for left, right in itertools.combinations(candidates, 2):
                if bool(left["correct"]) == bool(right["correct"]):
                    continue
                low = int(left["sample_id"])
                high = int(right["sample_id"])
                pair_id = f"{checkpoint}:{problem_id}:{low}:{high}"
                for order_index, (candidate_a, candidate_b) in enumerate(
                    ((left, right), (right, left))
                ):
                    sample_a = int(candidate_a["sample_id"])
                    sample_b = int(candidate_b["sample_id"])
                    jobs.append(
                        {
                            "job_id": f"{pair_id}:order{order_index}",
                            "pair_id": pair_id,
                            "checkpoint": checkpoint,
                            "token_position": token_position,
                            "problem_id": problem_id,
                            "sample_low": low,
                            "sample_high": high,
                            "candidate_a": sample_a,
                            "candidate_b": sample_b,
                            "correct_a": bool(candidate_a["correct"]),
                            "correct_b": bool(candidate_b["correct"]),
                            "order_index": order_index,
                            "prompt": build_prefix_pairwise_prompt(
                                verifier_tokenizer,
                                problem=left["problem"],
                                solution_a=prefixes[sample_a],
                                solution_b=prefixes[sample_b],
                                token_position=token_position,
                            ),
                        }
                    )
    return jobs


def combine_orientations(orientations: Sequence[dict]) -> list[dict]:
    metadata = {}
    for row in orientations:
        pair_id = str(row["pair_id"])
        current = (str(row["checkpoint"]), int(row["token_position"]))
        if pair_id in metadata and metadata[pair_id] != current:
            raise ValueError(f"Inconsistent checkpoint metadata for {pair_id}.")
        metadata[pair_id] = current
    combined = pairwise.combine_orientations(orientations)
    for row in combined:
        checkpoint, token_position = metadata[str(row["pair_id"])]
        row["checkpoint"] = checkpoint
        row["token_position"] = token_position
        row["correct_sample_id"] = (
            int(row["sample_low"])
            if bool(row["correct_low"])
            else int(row["sample_high"])
        )
        row["incorrect_sample_id"] = (
            int(row["sample_high"])
            if bool(row["correct_low"])
            else int(row["sample_low"])
        )
    return sorted(
        combined,
        key=lambda row: (
            int(row["token_position"]),
            str(row["problem_id"]),
            int(row["sample_low"]),
            int(row["sample_high"]),
        ),
    )


def attach_pointwise_scores(pairs: Sequence[dict], score_rows: Sequence[dict]) -> None:
    scores = {
        (
            str(row["problem_id"]),
            int(row["sample_id"]),
            str(row["checkpoint"]),
        ): float(row["score"])
        for row in score_rows
    }
    if len(scores) != len(score_rows):
        raise ValueError("Duplicate pointwise score key.")
    for row in pairs:
        problem_id = str(row["problem_id"])
        checkpoint = str(row["checkpoint"])
        correct_key = (problem_id, int(row["correct_sample_id"]), checkpoint)
        incorrect_key = (problem_id, int(row["incorrect_sample_id"]), checkpoint)
        if correct_key not in scores or incorrect_key not in scores:
            raise ValueError(
                f"Pointwise scores missing for {correct_key} or {incorrect_key}."
            )
        correct_score = scores[correct_key]
        incorrect_score = scores[incorrect_key]
        row["pointwise_score_correct"] = correct_score
        row["pointwise_score_incorrect"] = incorrect_score
        row["pointwise_margin_correct"] = correct_score - incorrect_score


def finish_reason_is_length_cap(value) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "length"
    return "length" in str(value).lower()


def attach_trajectory_outcomes(
    pairs: Sequence[dict], trajectories: Sequence[dict]
) -> None:
    by_key = {
        (str(row["problem_id"]), int(row["sample_id"])): row
        for row in trajectories
    }
    if len(by_key) != len(trajectories):
        raise ValueError("Duplicate trajectory key while attaching outcomes.")
    for row in pairs:
        key = (str(row["problem_id"]), int(row["incorrect_sample_id"]))
        trajectory = by_key.get(key)
        if trajectory is None:
            raise ValueError(f"Missing incorrect trajectory metadata for {key}.")
        row["incorrect_extracted_answer"] = trajectory.get("extracted_answer")
        row["incorrect_has_answer"] = trajectory.get("extracted_answer") is not None
        row["incorrect_finish_reason"] = trajectory.get("finish_reason")
        row["incorrect_at_length_cap"] = finish_reason_is_length_cap(
            trajectory.get("finish_reason")
        )


def hard_win(value: float, boundary: float = 0.5) -> float:
    return 1.0 if value > boundary else 0.5 if value == boundary else 0.0


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low_index = max(0, math.floor(0.025 * (len(ordered) - 1)))
    high_index = min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))
    return [float(ordered[low_index]), float(ordered[high_index])]


def bootstrap_metrics(
    rows: Sequence[dict], *, samples: int, seed: int
) -> dict[str, list[float]]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    problem_ids = sorted(by_problem)
    if samples <= 0 or len(problem_ids) < 2:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    has_pointwise = "pointwise_margin_correct" in rows[0]
    for _ in range(samples):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        groups = [by_problem[problem_id] for problem_id in drawn]
        flat = [row for group in groups for row in group]
        pairwise_wins = [hard_win(float(row["probability_correct"])) for row in flat]
        problem_pairwise = [
            mean([hard_win(float(row["probability_correct"])) for row in group])
            for group in groups
        ]
        values["pair_accuracy"].append(mean(pairwise_wins))
        values["problem_balanced_accuracy"].append(mean(problem_pairwise))
        values["mean_probability_correct"].append(
            mean([float(row["probability_correct"]) for row in flat])
        )
        if has_pointwise:
            pointwise_wins = [
                hard_win(float(row["pointwise_margin_correct"]), boundary=0.0)
                for row in flat
            ]
            problem_pointwise = [
                mean(
                    [
                        hard_win(
                            float(row["pointwise_margin_correct"]), boundary=0.0
                        )
                        for row in group
                    ]
                )
                for group in groups
            ]
            values["pointwise_pair_accuracy"].append(mean(pointwise_wins))
            values["pointwise_problem_balanced_accuracy"].append(
                mean(problem_pointwise)
            )
            values["pairwise_minus_pointwise_problem_balanced"].append(
                mean(problem_pairwise) - mean(problem_pointwise)
            )
    return {key: percentile_interval(metric) for key, metric in values.items()}


def decision_counts(values: Sequence[float], *, boundary: float) -> dict[str, int]:
    return {
        "wins": sum(value > boundary for value in values),
        "ties": sum(value == boundary for value in values),
        "losses": sum(value < boundary for value in values),
    }


def summarize_outcome_subgroup(
    rows: Sequence[dict], *, bootstrap_samples: int, seed: int
) -> dict:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    problem_accuracy = [
        mean([hard_win(float(row["probability_correct"])) for row in group])
        for group in by_problem.values()
    ]
    intervals = bootstrap_metrics(rows, samples=bootstrap_samples, seed=seed)
    return {
        "n_pairs": len(rows),
        "n_problems": len(by_problem),
        "pair_accuracy": mean(
            [hard_win(float(row["probability_correct"])) for row in rows]
        ),
        "problem_balanced_accuracy": mean(problem_accuracy),
        "problem_balanced_bootstrap_95_ci": intervals.get(
            "problem_balanced_accuracy", []
        ),
    }


def summarize_checkpoint(
    rows: Sequence[dict], orientations: Sequence[dict], *, bootstrap_samples: int, seed: int
) -> dict:
    pairwise_wins = [hard_win(float(row["probability_correct"])) for row in rows]
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    problem_accuracies = [
        mean([hard_win(float(row["probability_correct"])) for row in group])
        for group in by_problem.values()
    ]
    intervals = bootstrap_metrics(rows, samples=bootstrap_samples, seed=seed)
    checkpoint = str(rows[0]["checkpoint"])
    checkpoint_orientations = [
        row for row in orientations if str(row["checkpoint"]) == checkpoint
    ]
    result = {
        "token_position": int(rows[0]["token_position"]),
        "n_problems": len(by_problem),
        "n_correct_incorrect_pairs": len(rows),
        "pair_accuracy": mean(pairwise_wins),
        "problem_balanced_accuracy": mean(problem_accuracies),
        "mean_probability_correct": mean(
            [float(row["probability_correct"]) for row in rows]
        ),
        "pairwise_decisions": decision_counts(
            [float(row["probability_correct"]) for row in rows], boundary=0.5
        ),
        "hard_order_agreement": mean(
            [float(row["hard_order_agreement"]) for row in rows]
        ),
        "mean_absolute_order_effect": mean(
            [float(row["absolute_order_effect"]) for row in rows]
        ),
        "mean_probability_assigned_to_position_a": mean(
            [float(row["probability_a"]) for row in checkpoint_orientations]
        ),
        "hard_position_a_selection_rate": mean(
            [float(row["probability_a"] >= 0.5) for row in checkpoint_orientations]
        ),
        "bootstrap_95_ci": intervals,
    }
    if "incorrect_at_length_cap" in rows[0]:
        subgroup_specs = {
            "incorrect_at_length_cap": lambda row: bool(
                row["incorrect_at_length_cap"]
            ),
            "incorrect_stopped_before_cap": lambda row: not bool(
                row["incorrect_at_length_cap"]
            ),
            "incorrect_with_extracted_answer": lambda row: bool(
                row["incorrect_has_answer"]
            ),
            "incorrect_without_extracted_answer": lambda row: not bool(
                row["incorrect_has_answer"]
            ),
        }
        result["outcome_subgroups"] = {}
        for subgroup_index, (name, predicate) in enumerate(subgroup_specs.items()):
            subgroup = [row for row in rows if predicate(row)]
            if subgroup:
                result["outcome_subgroups"][name] = summarize_outcome_subgroup(
                    subgroup,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 100 + subgroup_index,
                )
    if "pointwise_margin_correct" in rows[0]:
        pointwise_wins = [
            hard_win(float(row["pointwise_margin_correct"]), boundary=0.0)
            for row in rows
        ]
        pointwise_problem_accuracies = [
            mean(
                [
                    hard_win(float(row["pointwise_margin_correct"]), boundary=0.0)
                    for row in group
                ]
            )
            for group in by_problem.values()
        ]
        pairwise_decisive = [
            float(row["probability_correct"]) > 0.5
            for row in rows
            if float(row["probability_correct"]) != 0.5
            and float(row["pointwise_margin_correct"]) != 0.0
        ]
        pointwise_decisive = [
            float(row["pointwise_margin_correct"]) > 0.0
            for row in rows
            if float(row["probability_correct"]) != 0.5
            and float(row["pointwise_margin_correct"]) != 0.0
        ]
        result["pointwise_comparison"] = {
            "pair_accuracy": mean(pointwise_wins),
            "problem_balanced_accuracy": mean(pointwise_problem_accuracies),
            "pairwise_minus_pointwise_problem_balanced": (
                mean(problem_accuracies) - mean(pointwise_problem_accuracies)
            ),
            "pointwise_decisions": decision_counts(
                [float(row["pointwise_margin_correct"]) for row in rows],
                boundary=0.0,
            ),
            "decisive_contingency": {
                "both_correct": sum(a and b for a, b in zip(pairwise_decisive, pointwise_decisive)),
                "pairwise_only_correct": sum(a and not b for a, b in zip(pairwise_decisive, pointwise_decisive)),
                "pointwise_only_correct": sum(not a and b for a, b in zip(pairwise_decisive, pointwise_decisive)),
                "both_wrong": sum(not a and not b for a, b in zip(pairwise_decisive, pointwise_decisive)),
                "n_pairs_both_decisive": len(pairwise_decisive),
            },
        }
    lower = intervals.get("problem_balanced_accuracy", [float("nan")])[0]
    interval = intervals.get("problem_balanced_accuracy", [])
    result["exploratory_inverted_choice"] = {
        "warning": "Post-hoc diagnostic only; direction was not predeclared.",
        "problem_balanced_accuracy": 1.0 - result["problem_balanced_accuracy"],
        "bootstrap_95_ci": (
            [1.0 - interval[1], 1.0 - interval[0]] if interval else []
        ),
    }
    result["integration_gate"] = {
        "point_estimate_at_least_60_percent": result["problem_balanced_accuracy"] >= 0.6,
        "cluster_bootstrap_lower_bound_above_chance": lower > 0.5,
        "passed": result["problem_balanced_accuracy"] >= 0.6 and lower > 0.5,
    }
    return result


def summarize(
    trajectories: Sequence[dict],
    orientations: Sequence[dict],
    pairs: Sequence[dict],
    *,
    scorer: str,
    generator_model: str,
    choice_token_ids: Sequence[int],
    token_positions: Sequence[int],
    initialization_time: float,
    inference_time: float,
    wall_time: float,
    verifier_gpus: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    checkpoints = {}
    for index, token_position in enumerate(token_positions):
        checkpoint = f"token_{token_position}"
        checkpoint_rows = [row for row in pairs if row["checkpoint"] == checkpoint]
        if checkpoint_rows:
            checkpoints[checkpoint] = summarize_checkpoint(
                checkpoint_rows,
                orientations,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
            )
    mixed_problem_ids = {str(row["problem_id"]) for row in pairs}
    return {
        "schema_version": 1,
        "experiment": {
            "method": "equal_horizon_correct_incorrect_order_swapped_pairwise_screen",
            "diagnostic_not_deployable": True,
            "scorer_model": scorer,
            "generator_model": generator_model,
            "token_positions": list(token_positions),
            "choice_labels": ["A", "B"],
            "choice_token_ids": list(choice_token_ids),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "problem",
            "seed": seed,
            "uses_generator_likelihood": False,
        },
        "dataset": {
            "name": trajectories[0].get("dataset"),
            "config": trajectories[0].get("dataset_config"),
            "split": trajectories[0].get("split"),
            "n_source_problems": len({str(row["problem_id"]) for row in trajectories}),
            "n_mixed_problems": len(mixed_problem_ids),
            "n_trajectories": len(trajectories),
        },
        "checkpoints": checkpoints,
        "cost": {
            "engine_initialization_time_s": initialization_time,
            "inference_wall_time_s": inference_time,
            "wall_time_s": wall_time,
            "verifier_gpus": verifier_gpus,
            "allocated_verifier_gpu_seconds": inference_time * verifier_gpus,
            "canonical_pairs": len(pairs),
            "order_swapped_verifier_calls": len(orientations),
            "verifier_prompt_tokens": sum(
                int(row["prompt_tokens"]) for row in orientations
            ),
            "verifier_completion_tokens": sum(
                int(row["completion_tokens"]) for row in orientations
            ),
            "mean_prompt_tokens_per_call": mean(
                [int(row["prompt_tokens"]) for row in orientations]
            ),
            "calls_per_inference_second": (
                len(orientations) / inference_time if inference_time else None
            ),
            "mean_choice_token_mass": mean(
                [float(row["choice_token_mass"]) for row in orientations]
            ),
            "selected_logprob_coverage": mean(
                [float(row["logprob_source"] == "selected") for row in orientations]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--pointwise-scores", default=None)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--generator-tokenizer", default=None)
    parser.add_argument("--token-positions", default="512,1024,2048")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-mamba-cache-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-orientations", required=True)
    parser.add_argument("--save-pairs", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--reuse-orientations", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = load_jsonl(args.trajectories)
    token_positions = parse_token_positions(args.token_positions)
    validate_trajectories(trajectories, token_positions)
    generator_model = args.generator_tokenizer or trajectories[0]["generator_model"]
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
    verifier_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    choice_token_ids = pairwise.resolve_choice_token_ids(verifier_tokenizer)
    all_jobs = make_jobs(
        trajectories,
        generator_tokenizer,
        verifier_tokenizer,
        token_positions,
    )
    jobs = all_jobs[:2] if args.probe else all_jobs
    print(
        f"source_problems={len({row['problem_id'] for row in trajectories})} "
        f"mixed_problems={len({job['problem_id'] for job in all_jobs})} "
        f"canonical_pairs={len(all_jobs) // 2} calls={len(jobs)} "
        f"scorer={args.scorer}",
        flush=True,
    )
    print(
        f"choice token IDs: A={choice_token_ids[0]} B={choice_token_ids[1]}",
        flush=True,
    )
    if args.dry_run:
        print(jobs[0]["prompt"])
        return None

    previous = (
        json.loads(Path(args.summary_output).read_text())
        if Path(args.summary_output).exists()
        else {}
    )
    initialization_time = 0.0
    inference_time = 0.0
    wall_time = 0.0
    if args.reuse_orientations:
        orientations = load_jsonl(args.save_orientations)
        expected = {job["job_id"] for job in jobs}
        observed = {row["job_id"] for row in orientations}
        if expected != observed:
            raise ValueError(
                "Saved orientations do not match requested jobs: "
                f"missing={len(expected - observed)} extra={len(observed - expected)}."
            )
        previous_cost = previous.get("cost", {})
        initialization_time = float(
            previous_cost.get("engine_initialization_time_s", 0.0)
        )
        inference_time = float(previous_cost.get("inference_wall_time_s", 0.0))
        wall_time = float(previous_cost.get("wall_time_s", 0.0))
        print(f"reusing {len(orientations)} saved calls", flush=True)
    else:
        import sglang as sgl

        engine_kwargs = dict(
            model_path=args.scorer,
            trust_remote_code=True,
            attention_backend="triton",
            mem_fraction_static=args.mem_fraction_static,
            base_gpu_id=args.base_gpu_id,
            random_seed=args.seed,
            max_running_requests=args.max_running_requests,
            max_mamba_cache_size=args.max_mamba_cache_size,
            disable_cuda_graph=args.disable_cuda_graph,
        )
        if args.dp > 1:
            engine_kwargs["dp_size"] = args.dp
            engine_kwargs["load_balance_method"] = "total_tokens"
        if args.tp > 1:
            engine_kwargs.update(
                tp_size=args.tp,
                disable_custom_all_reduce=True,
                enforce_disable_flashinfer_allreduce_fusion=True,
            )
        wall_started = time.perf_counter()
        engine = sgl.Engine(**engine_kwargs)
        initialization_time = time.perf_counter() - wall_started
        orientations = []
        inference_started = time.perf_counter()
        try:
            for start in range(0, len(jobs), args.batch_size):
                batch = jobs[start : start + args.batch_size]
                outputs = engine.generate(
                    [job["prompt"] for job in batch],
                    {"max_new_tokens": 1, "temperature": 0.0},
                    return_logprob=True,
                    top_logprobs_num=0,
                    token_ids_logprob=choice_token_ids,
                )
                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch):
                    raise ValueError(
                        f"Verifier returned {len(outputs)} outputs for {len(batch)} jobs."
                    )
                for job, output in zip(batch, outputs):
                    orientations.append(
                        {
                            "schema_version": 1,
                            **{key: value for key, value in job.items() if key != "prompt"},
                            **pairwise.choice_probability(output, choice_token_ids),
                            "scorer_model": args.scorer,
                        }
                    )
                elapsed = time.perf_counter() - inference_started
                print(
                    f"scored={len(orientations)}/{len(jobs)} "
                    f"calls/s={len(orientations) / elapsed:.2f}",
                    flush=True,
                )
                write_jsonl(args.save_orientations, orientations)
        finally:
            engine.shutdown()
        inference_time = time.perf_counter() - inference_started
        wall_time = time.perf_counter() - wall_started

    if args.probe:
        probe = {
            "schema_version": 1,
            "probe": True,
            "calls": len(orientations),
            "mean_choice_token_mass": mean(
                [float(row["choice_token_mass"]) for row in orientations]
            ),
            "selected_logprob_coverage": mean(
                [float(row["logprob_source"] == "selected") for row in orientations]
            ),
        }
        write_json(args.summary_output, probe)
        print(json.dumps(probe, indent=2), flush=True)
        return probe

    pairs = combine_orientations(orientations)
    attach_trajectory_outcomes(pairs, trajectories)
    if args.pointwise_scores:
        attach_pointwise_scores(pairs, load_jsonl(args.pointwise_scores))
    write_jsonl(args.save_pairs, pairs)
    summary = summarize(
        trajectories,
        orientations,
        pairs,
        scorer=args.scorer,
        generator_model=generator_model,
        choice_token_ids=choice_token_ids,
        token_positions=token_positions,
        initialization_time=initialization_time,
        inference_time=inference_time,
        wall_time=wall_time,
        verifier_gpus=args.dp * args.tp,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

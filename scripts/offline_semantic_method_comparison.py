"""Compare semantic TTS methods on aligned problems and accelerator cost."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)) + 1)]
    return [low, high]


def baseline_outcomes(
    trajectories: Sequence[dict], pointwise_scores: Sequence[dict]
) -> dict[str, dict[str, bool]]:
    trajectory_groups: dict[str, list[dict]] = defaultdict(list)
    score_groups: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectory_groups[str(row["problem_id"])].append(row)
    for row in pointwise_scores:
        score_groups[str(row["problem_id"])].append(row)
    outcomes = {}
    for problem_id, rows in trajectory_groups.items():
        rows.sort(key=lambda row: int(row["sample_id"]))
        votes = [row["extracted_answer"] for row in rows if row["extracted_answer"] is not None]
        majority = Counter(votes).most_common(1)[0][0] if votes else None
        scored = score_groups[problem_id]
        selected = min(
            scored,
            key=lambda row: (-float(row["score"]), int(row["sample_id"])),
        )
        outcomes[problem_id] = {
            "self_consistency": majority == rows[0]["gold_answer"],
            "terminal_pointwise": bool(selected["correct"]),
            "oracle": any(bool(row["correct"]) for row in rows),
        }
    return outcomes


def attach_outcomes(
    outcomes: dict[str, dict[str, bool]],
    rows: Sequence[dict],
    *,
    method: str,
    field: str,
) -> None:
    observed = set()
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in outcomes:
            raise ValueError(f"Unexpected problem {problem_id} in {method}.")
        outcomes[problem_id][method] = bool(row[field])
        observed.add(problem_id)
    missing = set(outcomes) - observed
    if missing:
        raise ValueError(f"{method} is missing {len(missing)} problems.")


def bootstrap(
    outcomes: dict[str, dict[str, bool]],
    methods: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> tuple[dict, dict]:
    problem_ids = sorted(outcomes)
    rng = random.Random(seed)
    accuracy_draws = {method: [] for method in methods}
    difference_draws = {
        method: {other: [] for other in methods if other != method}
        for method in methods
    }
    for _ in range(samples):
        selected = [rng.choice(problem_ids) for _ in problem_ids]
        accuracies = {
            method: statistics.fmean(
                float(outcomes[problem_id][method]) for problem_id in selected
            )
            for method in methods
        }
        for method in methods:
            accuracy_draws[method].append(accuracies[method])
            for other in methods:
                if other != method:
                    difference_draws[method][other].append(
                        accuracies[method] - accuracies[other]
                    )
    metrics = {}
    paired = {}
    for method in methods:
        point = statistics.fmean(
            float(row[method]) for row in outcomes.values()
        )
        metrics[method] = {
            "accuracy": point,
            "bootstrap_95_ci": percentile_interval(accuracy_draws[method]),
        }
        paired[method] = {}
        for other in methods:
            if method == other:
                continue
            differences = [
                float(row[method]) - float(row[other]) for row in outcomes.values()
            ]
            paired[method][other] = {
                "accuracy_difference": statistics.fmean(differences),
                "bootstrap_95_ci": percentile_interval(
                    difference_draws[method][other]
                ),
                "wins": sum(value > 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
    return metrics, paired


def cost_table(args, n_problems: int) -> dict[str, dict]:
    generation = load_json(args.generation_summary)["cost"]
    generator_gpu_seconds = float(generation["allocated_generator_gpu_seconds"])
    generator_tokens = int(generation["total_output_tokens"])
    pointwise = load_json(args.pointwise_summary)["cost"]
    knockout = load_json(args.knockout_summary)["cost"]
    smc = load_json(args.smc_summary)["cost"]
    fork = load_json(args.fork_summary)["cost"]
    costs = {
        "self_consistency": {
            "active_gpu_seconds": generator_gpu_seconds,
            "generator_tokens": generator_tokens,
            "verifier_prompt_tokens": 0,
            "verifier_calls": 0,
        },
        "terminal_pointwise": {
            "active_gpu_seconds": generator_gpu_seconds
            + args.pointwise_verifier_gpus * float(pointwise["inference_wall_time_s"]),
            "generator_tokens": generator_tokens,
            "verifier_prompt_tokens": int(pointwise["verifier_prompt_tokens"]),
            "verifier_calls": int(pointwise["prefixes_scored"]),
        },
        "terminal_knockout": {
            "active_gpu_seconds": generator_gpu_seconds
            + args.knockout_verifier_gpus * float(knockout["wall_time_s"]),
            "generator_tokens": generator_tokens,
            "verifier_prompt_tokens": int(knockout["verifier_prompt_tokens"]),
            "verifier_calls": int(knockout["order_swapped_verifier_calls"]),
        },
        "semantic_smc": {
            "active_gpu_seconds": float(smc["active_component_gpu_seconds"]),
            "static_reserved_gpu_seconds": float(smc["static_reserved_gpu_seconds"]),
            "generator_tokens": int(smc["total_generator_tokens"]),
            "verifier_prompt_tokens": int(smc["verifier_prompt_tokens"]),
            "verifier_calls": int(smc["verifier_calls"]),
        },
        "deterministic_fork": {
            "active_gpu_seconds": float(fork["active_component_gpu_seconds"]),
            "static_reserved_gpu_seconds": float(fork["static_reserved_gpu_seconds"]),
            "generator_tokens": int(fork["total_generator_tokens"]),
            "verifier_prompt_tokens": int(fork["verifier_prompt_tokens"]),
            "verifier_calls": int(fork["verifier_calls"]),
        },
    }
    for values in costs.values():
        values["active_gpu_seconds_per_problem"] = values["active_gpu_seconds"] / n_problems
        values["generator_tokens_per_problem"] = values["generator_tokens"] / n_problems
        values["verifier_prompt_tokens_per_problem"] = (
            values["verifier_prompt_tokens"] / n_problems
        )
    return costs


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--generation-summary", required=True)
    parser.add_argument("--pointwise-scores", required=True)
    parser.add_argument("--pointwise-summary", required=True)
    parser.add_argument("--knockout-rankings", required=True)
    parser.add_argument("--knockout-summary", required=True)
    parser.add_argument("--smc-problems", required=True)
    parser.add_argument("--smc-summary", required=True)
    parser.add_argument("--fork-problems", required=True)
    parser.add_argument("--fork-summary", required=True)
    parser.add_argument("--pointwise-verifier-gpus", type=int, default=4)
    parser.add_argument("--knockout-verifier-gpus", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    outcomes = baseline_outcomes(
        load_jsonl(args.trajectories), load_jsonl(args.pointwise_scores)
    )
    attach_outcomes(
        outcomes,
        load_jsonl(args.knockout_rankings),
        method="terminal_knockout",
        field="selected_correct",
    )
    attach_outcomes(
        outcomes,
        load_jsonl(args.smc_problems),
        method="semantic_smc",
        field="selected_correct",
    )
    attach_outcomes(
        outcomes,
        load_jsonl(args.fork_problems),
        method="deterministic_fork",
        field="selected_correct",
    )
    methods = [
        "self_consistency",
        "terminal_pointwise",
        "terminal_knockout",
        "semantic_smc",
        "deterministic_fork",
        "oracle",
    ]
    metrics, paired = bootstrap(
        outcomes, methods, samples=args.bootstrap_samples, seed=args.seed
    )
    costs = cost_table(args, len(outcomes))
    evaluated = [method for method in methods if method != "oracle"]
    for method in evaluated:
        dominated_by = []
        for other in evaluated:
            if method == other:
                continue
            weak_quality = metrics[other]["accuracy"] >= metrics[method]["accuracy"]
            weak_cost = (
                costs[other]["active_gpu_seconds"]
                <= costs[method]["active_gpu_seconds"]
            )
            strict = (
                metrics[other]["accuracy"] > metrics[method]["accuracy"]
                or costs[other]["active_gpu_seconds"]
                < costs[method]["active_gpu_seconds"]
            )
            if weak_quality and weak_cost and strict:
                dominated_by.append(other)
        metrics[method]["active_cost_dominated_by"] = dominated_by

    smc_quality = load_json(args.smc_summary)["quality"]
    fork_quality = load_json(args.fork_summary)["quality"]
    gate = {
        "exceeds_self_consistency": metrics["semantic_smc"]["accuracy"]
        > metrics["self_consistency"]["accuracy"],
        "exceeds_terminal_pointwise": metrics["semantic_smc"]["accuracy"]
        > metrics["terminal_pointwise"]["accuracy"],
        "exceeds_terminal_knockout": metrics["semantic_smc"]["accuracy"]
        > metrics["terminal_knockout"]["accuracy"],
        "not_active_cost_dominated": not metrics["semantic_smc"][
            "active_cost_dominated_by"
        ],
        "exceeds_fork_or_preserves_more_diversity": (
            metrics["semantic_smc"]["accuracy"]
            > metrics["deterministic_fork"]["accuracy"]
            or float(smc_quality["mean_unique_final_token_sequences"])
            > float(fork_quality["mean_unique_final_token_sequences"])
        ),
    }
    gate["passed"] = all(gate.values())
    summary = {
        "schema_version": 1,
        "evaluation_status": "exploratory_development_not_disjoint_test",
        "n_problems": len(outcomes),
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": metrics,
        "paired_differences": paired,
        "cost": costs,
        "primary_gate": gate,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

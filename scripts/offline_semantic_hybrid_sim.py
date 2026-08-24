"""Cross-validated agreement-conditioned semantic allocation simulation.

Policy:
  1. Generate two complete solutions.
  2. If their canonical answers agree, stop without a verifier call.
  3. Otherwise score both saved prefixes at one fixed token checkpoint.
  4. Expand to all eight when aggregate semantic confidence is below a
     threshold; otherwise return the higher-scored initial solution.

Checkpoint, aggregation, and threshold are selected inside each training fold
at a specified fraction of the agreement-only control's measured GPU budget.
The held-out fold is then evaluated with that frozen policy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


METRICS = (
    "accuracy",
    "generator_tokens",
    "verifier_prompt_tokens",
    "gpu_seconds",
    "expanded_fraction",
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_csv(value: str, cast=str) -> list:
    return [cast(item) for item in value.split(",") if item.strip()]


def confidence(scores: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "mean":
        return scores.mean(axis=2)
    if aggregation == "max":
        return scores.max(axis=2)
    if aggregation == "min":
        return scores.min(axis=2)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def paired_bootstrap_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        selection = rng.integers(0, len(values), size=len(values))
        means[index] = values[selection].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument(
        "--score-rubric",
        default=None,
        help="Keep only rows with this rubric from a multi-rubric score file.",
    )
    parser.add_argument("--generation-summary", required=True)
    parser.add_argument("--verifier-summary", required=True)
    parser.add_argument("--verifier-gpus", type=int, required=True)
    parser.add_argument(
        "--checkpoints", default="token_512,token_1024,token_2048"
    )
    parser.add_argument("--aggregations", default="mean,max,min")
    parser.add_argument("--budget-ratios", default="0.9,1.0")
    parser.add_argument("--threshold-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fold-seed", type=int, default=29)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args(argv)

    trajectories = load_jsonl(args.trajectories)
    score_rows = load_jsonl(args.scores)
    if args.score_rubric is not None:
        score_rows = [
            row
            for row in score_rows
            if str(row.get("rubric")) == args.score_rubric
        ]
        if not score_rows:
            raise ValueError(f"No score rows found for rubric {args.score_rubric!r}.")
    generation_summary = json.loads(Path(args.generation_summary).read_text())
    verifier_summary = json.loads(Path(args.verifier_summary).read_text())
    checkpoints = parse_csv(args.checkpoints)
    aggregations = parse_csv(args.aggregations)
    budget_ratios = parse_csv(args.budget_ratios, float)

    by_problem: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in trajectories:
        by_problem[str(row["problem_id"])][int(row["sample_id"])] = row
    scores: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in score_rows:
        scores[(str(row["problem_id"]), int(row["sample_id"]))][
            str(row["checkpoint"])
        ] = row
    problem_ids = sorted(by_problem)
    if not problem_ids or any(len(by_problem[problem_id]) != 8 for problem_id in problem_ids):
        raise ValueError("Every problem must have exactly eight trajectories.")
    expected_scores = {
        (problem_id, sample_id, checkpoint)
        for problem_id in problem_ids
        for sample_id in range(8)
        for checkpoint in checkpoints
    }
    observed_scores = {
        (problem_id, sample_id, checkpoint)
        for (problem_id, sample_id), values in scores.items()
        for checkpoint in values
        if checkpoint in checkpoints
    }
    if expected_scores != observed_scores:
        raise ValueError(
            "Semantic scores do not cover the requested jobs: "
            f"missing={len(expected_scores - observed_scores)} "
            f"extra={len(observed_scores - expected_scores)}"
        )

    generation_seconds_per_token = (
        generation_summary["cost"]["inference_wall_time_s"]
        / generation_summary["cost"]["total_output_tokens"]
    )
    verifier_prompt_tokens = verifier_summary["cost"].get(
        "verifier_prompt_tokens", verifier_summary["cost"].get("prompt_tokens")
    )
    if not verifier_prompt_tokens:
        raise ValueError("Verifier summary has no nonzero prompt-token count.")
    verifier_gpu_seconds_per_prompt_token = (
        verifier_summary["cost"]["inference_wall_time_s"]
        * args.verifier_gpus
        / verifier_prompt_tokens
    )

    def vote(problem_id: str, sample_ids: Sequence[int]):
        values = [
            by_problem[problem_id][sample_id]["answer"]
            for sample_id in sorted(sample_ids)
            if by_problem[problem_id][sample_id]["answer"] is not None
        ]
        return Counter(values).most_common(1)[0][0] if values else None

    n_problems = len(problem_ids)
    trials = args.trials
    order_rng = random.Random(args.seed)
    first_samples = np.zeros((n_problems, trials, 2), dtype=np.int16)
    agreement = np.zeros((n_problems, trials), dtype=bool)
    agreement_correct = np.zeros((n_problems, trials), dtype=bool)
    full_vote_correct = np.zeros((n_problems, trials), dtype=bool)
    initial_tokens = np.zeros((n_problems, trials), dtype=float)
    full_tokens = np.zeros((n_problems, trials), dtype=float)
    for problem_index, problem_id in enumerate(problem_ids):
        group = by_problem[problem_id]
        gold = group[0]["gold_answer"]
        full_correct = vote(problem_id, range(8)) == gold
        full_problem_tokens = sum(group[sample_id]["completion_tokens"] for sample_id in range(8))
        for trial in range(trials):
            order = order_rng.sample(range(8), 8)
            first, second = order[:2]
            first_samples[problem_index, trial] = [first, second]
            first_answer = group[first]["answer"]
            second_answer = group[second]["answer"]
            agrees = first_answer is not None and first_answer == second_answer
            agreement[problem_index, trial] = agrees
            agreement_correct[problem_index, trial] = agrees and first_answer == gold
            full_vote_correct[problem_index, trial] = full_correct
            initial_tokens[problem_index, trial] = (
                group[first]["completion_tokens"] + group[second]["completion_tokens"]
            )
            full_tokens[problem_index, trial] = full_problem_tokens

    baseline_accuracy = np.where(
        agreement, agreement_correct, full_vote_correct
    )
    baseline_generator_tokens = np.where(
        agreement, initial_tokens, full_tokens
    )
    baseline_by_problem = {
        "accuracy": baseline_accuracy.mean(axis=1),
        "generator_tokens": baseline_generator_tokens.mean(axis=1),
        "verifier_prompt_tokens": np.zeros(n_problems),
        "gpu_seconds": (
            baseline_generator_tokens.mean(axis=1) * generation_seconds_per_token
        ),
        "expanded_fraction": (~agreement).mean(axis=1),
    }

    candidates = []
    thresholds = np.linspace(0, 1, args.threshold_steps + 1)
    for checkpoint in checkpoints:
        semantic_scores = np.zeros((n_problems, trials, 2), dtype=float)
        verifier_tokens = np.zeros((n_problems, trials), dtype=float)
        selected_correct = np.zeros((n_problems, trials), dtype=bool)
        for problem_index, problem_id in enumerate(problem_ids):
            group = by_problem[problem_id]
            for trial in range(trials):
                first, second = first_samples[problem_index, trial]
                first_row = scores[(problem_id, int(first))][checkpoint]
                second_row = scores[(problem_id, int(second))][checkpoint]
                semantic_scores[problem_index, trial] = [
                    first_row["score"],
                    second_row["score"],
                ]
                verifier_tokens[problem_index, trial] = (
                    first_row["prompt_tokens"] + second_row["prompt_tokens"]
                )
                selected = int(first) if first_row["score"] >= second_row["score"] else int(second)
                selected_correct[problem_index, trial] = bool(group[selected]["correct"])
        verifier_tokens = np.where(agreement, 0, verifier_tokens)

        for aggregation in aggregations:
            aggregate_confidence = confidence(semantic_scores, aggregation)
            for threshold in thresholds:
                expand = (~agreement) & (aggregate_confidence < threshold)
                policy_accuracy = np.where(
                    agreement,
                    agreement_correct,
                    np.where(expand, full_vote_correct, selected_correct),
                )
                policy_generator_tokens = np.where(
                    expand, full_tokens, initial_tokens
                )
                policy_gpu_seconds = (
                    policy_generator_tokens * generation_seconds_per_token
                    + verifier_tokens * verifier_gpu_seconds_per_prompt_token
                )
                candidates.append(
                    {
                        "checkpoint": checkpoint,
                        "aggregation": aggregation,
                        "threshold": float(threshold),
                        "accuracy": policy_accuracy.mean(axis=1),
                        "generator_tokens": policy_generator_tokens.mean(axis=1),
                        "verifier_prompt_tokens": verifier_tokens.mean(axis=1),
                        "gpu_seconds": policy_gpu_seconds.mean(axis=1),
                        "expanded_fraction": expand.mean(axis=1),
                    }
                )

    shuffled = list(problem_ids)
    random.Random(args.fold_seed).shuffle(shuffled)
    fold = {problem_id: index % args.folds for index, problem_id in enumerate(shuffled)}
    cross_validated = {}
    for budget_index, budget_ratio in enumerate(budget_ratios):
        selected_by_problem = {
            metric: np.zeros(n_problems, dtype=float) for metric in METRICS
        }
        selected_policies = []
        for fold_index in range(args.folds):
            train = np.array(
                [index for index, problem_id in enumerate(problem_ids) if fold[problem_id] != fold_index]
            )
            test = np.array(
                [index for index, problem_id in enumerate(problem_ids) if fold[problem_id] == fold_index]
            )
            budget = (
                baseline_by_problem["gpu_seconds"][train].mean() * budget_ratio
            )
            eligible = [
                candidate
                for candidate in candidates
                if candidate["gpu_seconds"][train].mean() <= budget
            ]
            if not eligible:
                raise ValueError(f"No policy fits budget ratio {budget_ratio}")
            best = max(
                eligible,
                key=lambda candidate: (
                    candidate["accuracy"][train].mean(),
                    -candidate["gpu_seconds"][train].mean(),
                ),
            )
            for metric in METRICS:
                selected_by_problem[metric][test] = best[metric][test]
            selected_policies.append(
                {
                    "fold": fold_index,
                    "checkpoint": best["checkpoint"],
                    "aggregation": best["aggregation"],
                    "threshold": best["threshold"],
                    "training_accuracy": float(best["accuracy"][train].mean()),
                    "training_gpu_seconds": float(best["gpu_seconds"][train].mean()),
                    "test_accuracy": float(best["accuracy"][test].mean()),
                    "test_gpu_seconds": float(best["gpu_seconds"][test].mean()),
                }
            )

        metrics = {
            metric: float(selected_by_problem[metric].mean()) for metric in METRICS
        }
        accuracy_difference = (
            selected_by_problem["accuracy"] - baseline_by_problem["accuracy"]
        )
        gpu_difference = (
            selected_by_problem["gpu_seconds"] - baseline_by_problem["gpu_seconds"]
        )
        cross_validated[str(budget_ratio)] = {
            "metrics": metrics,
            "difference_vs_agreement_control": {
                "accuracy": float(accuracy_difference.mean()),
                "accuracy_bootstrap_95_ci": paired_bootstrap_ci(
                    accuracy_difference,
                    samples=args.bootstrap_samples,
                    seed=args.seed + budget_index,
                ),
                "gpu_seconds": float(gpu_difference.mean()),
                "gpu_seconds_bootstrap_95_ci": paired_bootstrap_ci(
                    gpu_difference,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 + budget_index,
                ),
            },
            "selected_policies": selected_policies,
        }

    agreement_summary = {
        metric: float(baseline_by_problem[metric].mean()) for metric in METRICS
    }
    matched_candidates = [
        candidate
        for candidate in candidates
        if candidate["gpu_seconds"].mean() <= agreement_summary["gpu_seconds"]
    ]
    in_sample = max(
        matched_candidates,
        key=lambda candidate: (
            candidate["accuracy"].mean(),
            -candidate["gpu_seconds"].mean(),
        ),
    )
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "agreement_conditioned_semantic_hybrid_cross_validation",
            "trajectories": args.trajectories,
            "scores": args.scores,
            "score_rubric": args.score_rubric,
            "checkpoints": checkpoints,
            "aggregations": aggregations,
            "threshold_steps": args.threshold_steps,
            "trials": trials,
            "folds": args.folds,
            "seed": args.seed,
            "fold_seed": args.fold_seed,
        },
        "cost_model": {
            "generator_gpu_seconds_per_output_token": generation_seconds_per_token,
            "verifier_gpu_seconds_per_prompt_token": verifier_gpu_seconds_per_prompt_token,
            "verifier_gpus_in_measurement": args.verifier_gpus,
            "model_initialization_excluded": True,
            "component_time_scaled_linearly_by_observed_tokens": True,
        },
        "agreement_adaptive_control": agreement_summary,
        "cross_validated_semantic_hybrid": cross_validated,
        "in_sample_matched_compute_upper_bound": {
            "checkpoint": in_sample["checkpoint"],
            "aggregation": in_sample["aggregation"],
            "threshold": in_sample["threshold"],
            **{metric: float(in_sample[metric].mean()) for metric in METRICS},
            "warning": "Policy selected and evaluated on the same 50 problems.",
        },
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return summary


if __name__ == "__main__":
    main()

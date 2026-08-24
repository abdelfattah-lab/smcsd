"""Distill terminal pairwise ratings into pointwise semantic prefix scores.

The student has two semantic-only features for each independently scored
prefix: the original recoverability score and the pairwise-informed error
audit. At each checkpoint a ridge-regularized logistic ranker learns the
teacher's within-problem pair preferences. Evaluation uses problem-level
out-of-fold predictions, so no problem's pairwise judgments train its own
student score.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from scripts import offline_recoverability as recoverability
except ModuleNotFoundError:
    import offline_recoverability as recoverability


def mean(values: Sequence[float]) -> float:
    return recoverability._mean(values)


def load_jsonl(path: str | Path) -> list[dict]:
    return recoverability.load_jsonl(path)


def write_json(path: str | Path, value: dict) -> None:
    recoverability.write_json(path, value)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    recoverability.write_jsonl(path, rows)


def score_key(row: dict) -> tuple[str, int, float, int]:
    return (
        str(row["problem_id"]),
        int(row["sample_id"]),
        float(row["requested_fraction"]),
        int(row["token_position"]),
    )


def make_problem_folds(
    problem_ids: Sequence[str], *, folds: int, seed: int
) -> dict[str, int]:
    unique = sorted(set(str(problem_id) for problem_id in problem_ids))
    if folds < 2 or folds > len(unique):
        raise ValueError(f"folds must be in [2, {len(unique)}].")
    random.Random(seed).shuffle(unique)
    return {problem_id: index % folds for index, problem_id in enumerate(unique)}


def align_student_rows(
    recoverability_rows: Sequence[dict],
    audit_rows: Sequence[dict],
    pairwise_rankings: Sequence[dict],
) -> list[dict]:
    recoverability_by_key = {score_key(row): row for row in recoverability_rows}
    audit_by_key = {score_key(row): row for row in audit_rows}
    if len(recoverability_by_key) != len(recoverability_rows):
        raise ValueError("Duplicate recoverability score key.")
    if len(audit_by_key) != len(audit_rows):
        raise ValueError("Duplicate audit score key.")
    if set(recoverability_by_key) != set(audit_by_key):
        raise ValueError(
            "Recoverability and audit score keys differ: "
            f"recoverability_only={len(set(recoverability_by_key) - set(audit_by_key))} "
            f"audit_only={len(set(audit_by_key) - set(recoverability_by_key))}."
        )
    pairwise_by_problem = {
        str(row["problem_id"]): row for row in pairwise_rankings
    }
    aligned = []
    for key, base in recoverability_by_key.items():
        audit = audit_by_key[key]
        problem_id, sample_id, _, _ = key
        pairwise = pairwise_by_problem.get(problem_id)
        if pairwise is None:
            raise ValueError(f"No pairwise teacher ranking for {problem_id}.")
        rating = pairwise["ratings"].get(str(sample_id))
        if rating is None:
            raise ValueError(f"No pairwise teacher rating for {problem_id}:{sample_id}.")
        if bool(base["correct"]) != bool(audit["correct"]):
            raise ValueError(f"Correctness mismatch for score key {key}.")
        aligned.append(
            {
                "schema_version": 1,
                **{
                    field: base[field]
                    for field in (
                        "problem_id",
                        "sample_id",
                        "requested_fraction",
                        "token_position",
                        "trajectory_tokens",
                        "actual_fraction",
                        "terminal",
                        "correct",
                        "extracted_answer",
                        "gold_answer",
                    )
                },
                "recoverability_score": float(base["score"]),
                "error_audit_score": float(audit["score"]),
                "pairwise_teacher_rating": float(rating),
                "prompt_tokens": int(base["prompt_tokens"])
                + int(audit["prompt_tokens"]),
                "completion_tokens": int(base["completion_tokens"])
                + int(audit["completion_tokens"]),
                "semantic_verifier_calls": 2,
                "recoverability_score_token_mass": float(base["score_token_mass"]),
                "error_audit_score_token_mass": float(audit["score_token_mass"]),
            }
        )
    return aligned


def logit(value: float, epsilon: float = 1e-5) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def feature_vector(row: dict) -> list[float]:
    return [
        logit(float(row["recoverability_score"])),
        logit(float(row["error_audit_score"])),
    ]


def make_teacher_preferences(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    differences = []
    targets = []
    for group in by_problem.values():
        for left, right in itertools.combinations(group, 2):
            left_rating = float(left["pairwise_teacher_rating"])
            right_rating = float(right["pairwise_teacher_rating"])
            if left_rating == right_rating:
                continue
            left_features = np.asarray(feature_vector(left), dtype=np.float64)
            right_features = np.asarray(feature_vector(right), dtype=np.float64)
            differences.append(left_features - right_features)
            targets.append(1.0 if left_rating > right_rating else 0.0)
    if not differences:
        raise ValueError("Pairwise teacher supplied no strict training preferences.")
    return np.asarray(differences), np.asarray(targets)


def fit_teacher_model(
    rows: Sequence[dict], *, ridge: float, max_iterations: int = 100
) -> tuple[list[float], int]:
    if not rows:
        raise ValueError("Cannot fit pairwise teacher model with no rows.")
    design, target = make_teacher_preferences(rows)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    identity = np.eye(design.shape[1], dtype=np.float64)
    for _ in range(max_iterations):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (probability - target) + ridge * coefficients
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design + ridge * identity
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    return [float(value) for value in coefficients], len(target)


def predict_teacher(row: dict, coefficients: Sequence[float]) -> float:
    linear = sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients, feature_vector(row))
    )
    return sigmoid(linear)


def cross_validated_distillation(
    rows: Sequence[dict],
    *,
    folds: int,
    ridge: float,
    seed: int,
) -> tuple[list[dict], dict]:
    fold_by_problem = make_problem_folds(
        [str(row["problem_id"]) for row in rows], folds=folds, seed=seed
    )
    by_fraction: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_fraction[float(row["requested_fraction"])].append(row)
    models = {}
    fused = []
    for fraction, checkpoint_rows in sorted(by_fraction.items()):
        checkpoint_models = {}
        for held_out_fold in range(folds):
            train = [
                row
                for row in checkpoint_rows
                if fold_by_problem[str(row["problem_id"])] != held_out_fold
            ]
            held_out = [
                row
                for row in checkpoint_rows
                if fold_by_problem[str(row["problem_id"])] == held_out_fold
            ]
            train_problem_ids = {str(row["problem_id"]) for row in train}
            held_out_problem_ids = {str(row["problem_id"]) for row in held_out}
            if train_problem_ids & held_out_problem_ids:
                raise AssertionError("Problem leakage across distillation folds.")
            coefficients, training_preferences = fit_teacher_model(
                train, ridge=ridge
            )
            checkpoint_models[str(held_out_fold)] = {
                "coefficients": {
                    "recoverability_logit": coefficients[0],
                    "error_audit_logit": coefficients[1],
                },
                "n_train_rows": len(train),
                "n_train_problems": len(train_problem_ids),
                "n_train_pairwise_preferences": training_preferences,
                "n_held_out_rows": len(held_out),
                "n_held_out_problems": len(held_out_problem_ids),
            }
            for row in held_out:
                fused.append(
                    {
                        **row,
                        "score": predict_teacher(row, coefficients),
                        "distillation_fold": held_out_fold,
                        "distillation_target": "terminal_pairwise_mean_win_probability",
                    }
                )
        deployment, deployment_preferences = fit_teacher_model(
            checkpoint_rows, ridge=ridge
        )
        checkpoint_models["all_data_deployment_fit_not_used_for_metrics"] = {
            "coefficients": {
                "recoverability_logit": deployment[0],
                "error_audit_logit": deployment[1],
            },
            "n_train_rows": len(checkpoint_rows),
            "n_train_pairwise_preferences": deployment_preferences,
        }
        models[str(fraction)] = checkpoint_models
    if len(fused) != len(rows):
        raise AssertionError(f"Expected {len(rows)} fused rows, got {len(fused)}.")
    return sorted(fused, key=score_key), models


def teacher_imitation_metrics(rows: Sequence[dict]) -> dict:
    prediction = np.asarray([float(row["score"]) for row in rows])
    teacher = np.asarray([float(row["pairwise_teacher_rating"]) for row in rows])
    correlation = float(np.corrcoef(prediction, teacher)[0, 1])
    agreements = []
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(row)
    for group in by_problem.values():
        for left, right in itertools.combinations(group, 2):
            teacher_delta = float(left["pairwise_teacher_rating"]) - float(
                right["pairwise_teacher_rating"]
            )
            if teacher_delta == 0:
                continue
            student_delta = float(left["score"]) - float(right["score"])
            agreements.append(
                1.0
                if teacher_delta * student_delta > 0
                else 0.5
                if student_delta == 0
                else 0.0
            )
    return {
        "pairwise_teacher_mse": float(np.mean((prediction - teacher) ** 2)),
        "pairwise_teacher_pearson": correlation,
        "within_problem_teacher_pair_agreement": mean(agreements),
        "n_teacher_pairs": len(agreements),
    }


def terminal_choice(rows: Sequence[dict], score_field: str) -> dict[str, bool]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["terminal"]:
            by_problem[str(row["problem_id"])].append(row)
    selected = {}
    for problem_id, group in by_problem.items():
        best = min(
            group,
            key=lambda row: (-float(row[score_field]), int(row["sample_id"])),
        )
        selected[problem_id] = bool(best["correct"])
    return selected


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return [low, high]


def terminal_comparison(
    rows: Sequence[dict],
    trajectories: Sequence[dict],
    pairwise_rankings: Sequence[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    fused = terminal_choice(rows, "score")
    base = terminal_choice(rows, "recoverability_score")
    audit = terminal_choice(rows, "error_audit_score")
    pairwise_by_problem = {
        str(row["problem_id"]): row for row in pairwise_rankings
    }
    trajectory_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectory_by_problem[str(row["problem_id"])].append(row)
    outcomes = []
    for problem_id in sorted(fused):
        trajectories_for_problem = sorted(
            trajectory_by_problem[problem_id], key=lambda row: int(row["sample_id"])
        )
        answers = [
            row["extracted_answer"]
            for row in trajectories_for_problem
            if row["extracted_answer"] is not None
        ]
        majority = Counter(answers).most_common(1)[0][0] if answers else None
        pairwise = pairwise_by_problem[problem_id]
        outcomes.append(
            {
                "problem_id": problem_id,
                "distilled": fused[problem_id],
                "recoverability": base[problem_id],
                "error_audit": audit[problem_id],
                "self_consistency": bool(
                    majority == trajectories_for_problem[0]["gold_answer"]
                ),
                "pairwise_all_pairs": bool(pairwise["selected_correct"]),
                "pairwise_knockout": bool(pairwise["knockout_selected_correct"]),
                "oracle": any(bool(row["correct"]) for row in trajectories_for_problem),
            }
        )
    references = [
        "recoverability",
        "error_audit",
        "self_consistency",
        "pairwise_all_pairs",
        "pairwise_knockout",
    ]
    result = {
        "distilled_accuracy": mean([float(row["distilled"]) for row in outcomes]),
        "oracle_pass_at_n": mean([float(row["oracle"]) for row in outcomes]),
    }
    for reference in references:
        result[f"{reference}_accuracy"] = mean(
            [float(row[reference]) for row in outcomes]
        )
        result[f"distilled_minus_{reference}"] = mean(
            [
                float(row["distilled"]) - float(row[reference])
                for row in outcomes
            ]
        )
        result[f"distilled_vs_{reference}_wins"] = sum(
            row["distilled"] and not row[reference] for row in outcomes
        )
        result[f"distilled_vs_{reference}_losses"] = sum(
            not row["distilled"] and row[reference] for row in outcomes
        )
    if bootstrap_samples > 0:
        rng = random.Random(seed)
        boot: dict[str, list[float]] = defaultdict(list)
        for _ in range(bootstrap_samples):
            drawn = [rng.choice(outcomes) for _ in outcomes]
            boot["distilled_accuracy"].append(
                mean([float(row["distilled"]) for row in drawn])
            )
            for reference in references:
                boot[f"distilled_minus_{reference}"].append(
                    mean(
                        [
                            float(row["distilled"]) - float(row[reference])
                            for row in drawn
                        ]
                    )
                )
        result["bootstrap_95_ci"] = {
            key: percentile_interval(values) for key, values in boot.items()
        }
    return result


def read_summary(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--recoverability-scores", required=True)
    parser.add_argument("--error-audit-scores", required=True)
    parser.add_argument("--pairwise-rankings", required=True)
    parser.add_argument("--recoverability-summary", default=None)
    parser.add_argument("--error-audit-summary", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--save-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    trajectories = load_jsonl(args.trajectories)
    recoverability_rows = load_jsonl(args.recoverability_scores)
    audit_rows = load_jsonl(args.error_audit_scores)
    pairwise_rankings = load_jsonl(args.pairwise_rankings)
    aligned = align_student_rows(
        recoverability_rows, audit_rows, pairwise_rankings
    )
    fused, models = cross_validated_distillation(
        aligned,
        folds=args.folds,
        ridge=args.ridge,
        seed=args.seed,
    )
    by_fraction: dict[float, list[dict]] = defaultdict(list)
    for row in fused:
        by_fraction[float(row["requested_fraction"])].append(row)
    checkpoints = {}
    for fraction, rows in sorted(by_fraction.items()):
        metrics = recoverability.summarize_checkpoint(rows)
        metrics.update(teacher_imitation_metrics(rows))
        metrics["bootstrap_95_ci"] = recoverability.bootstrap_intervals(
            rows,
            samples=args.bootstrap_samples,
            seed=args.seed + round(fraction * 1000),
        )
        checkpoints[str(fraction)] = metrics
    terminal = terminal_comparison(
        fused,
        trajectories,
        pairwise_rankings,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    base_summary = read_summary(args.recoverability_summary)
    audit_summary = read_summary(args.error_audit_summary)
    base_wall_time = float(base_summary.get("cost", {}).get("wall_time_s", 0.0))
    audit_wall_time = float(audit_summary.get("cost", {}).get("wall_time_s", 0.0))
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "problem_level_out_of_fold_pairwise_preference_distillation",
            "student_features": [
                "recoverability_score_logit",
                "error_audit_score_logit",
            ],
            "teacher": "within_problem_preferences_from_terminal_order_swapped_all_pairs_ratings",
            "folds": args.folds,
            "ridge": args.ridge,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "evaluation_status": "exploratory_cross_validation_not_frozen_test",
            "uses_generator_likelihood": False,
        },
        "dataset": {
            "name": trajectories[0].get("dataset"),
            "config": trajectories[0].get("dataset_config"),
            "split": trajectories[0].get("split"),
            "n_problems": len({str(row["problem_id"]) for row in trajectories}),
            "n_trajectories": len(trajectories),
        },
        "checkpoint_models": models,
        "checkpoints": checkpoints,
        "terminal_selection": terminal,
        "cost": {
            "semantic_verifier_calls": sum(
                int(row["semantic_verifier_calls"]) for row in fused
            ),
            "semantic_verifier_prompt_tokens": sum(
                int(row["prompt_tokens"]) for row in fused
            ),
            "semantic_verifier_completion_tokens": sum(
                int(row["completion_tokens"]) for row in fused
            ),
            "inference_wall_time_s_sum_across_source_runs": (
                base_wall_time + audit_wall_time
            ),
            "distillation_fit_requires_no_model_inference": True,
        },
    }
    write_jsonl(args.save_scores, fused)
    write_json(args.summary_output, summary)
    print("Pairwise-teacher pointwise distillation")
    print("fraction  AUROC  rank-acc  top-half  top1  teacher-r")
    for fraction, metrics in checkpoints.items():
        print(
            f"{float(fraction):7.2f}  {metrics['auroc']:5.3f}  "
            f"{metrics['within_problem_ranking_accuracy']:8.3f}  "
            f"{metrics['top_half_correct_survival']:8.3f}  "
            f"{metrics['top1_eventual_correct_rate']:4.3f}  "
            f"{metrics['pairwise_teacher_pearson']:9.3f}"
        )
    print(json.dumps(terminal, indent=2))
    print(f"wrote {args.save_scores} and {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

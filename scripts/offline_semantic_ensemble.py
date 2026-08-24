"""Problem-level out-of-fold fusion of semantic prefix scores.

Models are trained only on correct/incorrect score differences from other
problems.  The primary metric is problem-balanced sibling-pair ranking, so a
global problem-difficulty signal cannot masquerade as useful particle ranking.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


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


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_feature_spec(value: str) -> tuple[str, str, str | None]:
    if "=" not in value:
        raise ValueError("Feature must have NAME=PATH or NAME=PATH::RUBRIC format.")
    name, source = value.split("=", 1)
    if "::" in source:
        path, rubric = source.rsplit("::", 1)
    else:
        path, rubric = source, None
    if not name or not path:
        raise ValueError(f"Invalid feature specification: {value}")
    return name, path, rubric


def parse_ensemble_spec(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise ValueError("Ensemble must have NAME=FEATURE1,FEATURE2 format.")
    name, feature_csv = value.split("=", 1)
    features = [item for item in feature_csv.split(",") if item]
    if not name or len(features) < 2:
        raise ValueError(f"Invalid ensemble specification: {value}")
    return name, features


def row_key(row: dict) -> tuple[str, int, str]:
    return (
        str(row["problem_id"]),
        int(row["sample_id"]),
        str(row["checkpoint"]),
    )


def load_features(specifications: Sequence[str]) -> tuple[list[dict], dict]:
    feature_maps = {}
    metadata = {}
    common_keys = None
    base_rows = None
    for specification in specifications:
        name, path, rubric = parse_feature_spec(specification)
        if name in feature_maps:
            raise ValueError(f"Duplicate feature name: {name}.")
        rows = load_jsonl(path)
        if rubric is not None:
            rows = [row for row in rows if str(row.get("rubric")) == rubric]
        elif any("rubric" in row for row in rows):
            rubrics = sorted({str(row.get("rubric")) for row in rows})
            if len(rubrics) > 1:
                raise ValueError(
                    f"Feature {name} contains multiple rubrics {rubrics}; add ::RUBRIC."
                )
        mapping = {row_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate score key for feature {name}.")
        if common_keys is None:
            common_keys = set(mapping)
            base_rows = mapping
        elif set(mapping) != common_keys:
            raise ValueError(
                f"Feature {name} score keys differ: "
                f"missing={len(common_keys - set(mapping))} "
                f"extra={len(set(mapping) - common_keys)}."
            )
        feature_maps[name] = mapping
        metadata[name] = {
            "path": path,
            "rubric": rubric,
            "scorer_model": rows[0].get("scorer_model"),
            "calls": len(rows),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
            "mean_score_token_mass": mean(
                [float(row["score_token_mass"]) for row in rows]
            ),
            "selected_logprob_coverage": mean(
                [float(row["logprob_source"] == "selected") for row in rows]
            ),
        }
    assert common_keys is not None and base_rows is not None
    aligned = []
    for key in sorted(common_keys):
        base = base_rows[key]
        row = {
            "schema_version": 1,
            "problem_id": key[0],
            "sample_id": key[1],
            "checkpoint": key[2],
            "token_position": int(base["token_position"]),
            "correct": bool(base["correct"]),
            "extracted_answer": base.get("extracted_answer"),
            "gold_answer": base.get("gold_answer"),
            "features": {
                name: float(mapping[key]["score"])
                for name, mapping in feature_maps.items()
            },
        }
        aligned.append(row)
    return aligned, metadata


def logit(value: float, epsilon: float = 1e-5) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def mixed_problem_ids(rows: Sequence[dict]) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    return sorted(
        problem_id
        for problem_id, group in groups.items()
        if any(bool(row["correct"]) for row in group)
        and any(not bool(row["correct"]) for row in group)
    )


def make_folds(problem_ids: Sequence[str], *, folds: int, seed: int) -> dict[str, int]:
    unique = sorted(set(problem_ids))
    if folds < 2 or folds > len(unique):
        raise ValueError(f"folds must be between 2 and {len(unique)}.")
    random.Random(seed).shuffle(unique)
    return {problem_id: index % folds for index, problem_id in enumerate(unique)}


def candidate_vector(row: dict, features: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [logit(float(row["features"][feature])) for feature in features],
        dtype=np.float64,
    )


def training_design(
    rows: Sequence[dict], features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    vectors = np.asarray([candidate_vector(row, features) for row in rows])
    center = vectors.mean(axis=0)
    scale = vectors.std(axis=0)
    scale[scale < 1e-6] = 1.0
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    design = []
    targets = []
    canonical_pairs = 0
    for group in groups.values():
        correct = [row for row in group if bool(row["correct"])]
        incorrect = [row for row in group if not bool(row["correct"])]
        for positive in correct:
            for negative in incorrect:
                difference = (
                    candidate_vector(positive, features)
                    - candidate_vector(negative, features)
                ) / scale
                design.extend((difference, -difference))
                targets.extend((1.0, 0.0))
                canonical_pairs += 1
    if not design:
        raise ValueError("Training fold contains no correct/incorrect pairs.")
    return (
        np.asarray(design),
        np.asarray(targets),
        center,
        scale,
        canonical_pairs,
    )


def fit_pairwise_model(
    rows: Sequence[dict], features: Sequence[str], *, ridge: float
) -> dict:
    design, target, center, scale, canonical_pairs = training_design(rows, features)
    coefficients = np.zeros(len(features), dtype=np.float64)
    identity = np.eye(len(features), dtype=np.float64)
    for _ in range(100):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (probability - target) + ridge * coefficients
        curvature = probability * (1.0 - probability)
        hessian = (design.T * curvature) @ design + ridge * identity
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.linalg.norm(step)) < 1e-9:
            break
    return {
        "features": list(features),
        "coefficients": [float(value) for value in coefficients],
        "center": [float(value) for value in center],
        "scale": [float(value) for value in scale],
        "training_canonical_pairs": canonical_pairs,
    }


def predict(row: dict, model: dict) -> float:
    vector = candidate_vector(row, model["features"])
    center = np.asarray(model["center"])
    scale = np.asarray(model["scale"])
    coefficients = np.asarray(model["coefficients"])
    return sigmoid(float(((vector - center) / scale) @ coefficients))


def cross_validated_scores(
    rows: Sequence[dict],
    features: Sequence[str],
    *,
    folds: int,
    seed: int,
    ridge: float,
) -> tuple[dict[tuple[str, int], float], dict]:
    eligible_ids = mixed_problem_ids(rows)
    fold_by_problem = make_folds(eligible_ids, folds=folds, seed=seed)
    eligible = [row for row in rows if str(row["problem_id"]) in fold_by_problem]
    predictions = {}
    models = {}
    for held_out_fold in range(folds):
        train = [
            row
            for row in eligible
            if fold_by_problem[str(row["problem_id"])] != held_out_fold
        ]
        held_out = [
            row
            for row in eligible
            if fold_by_problem[str(row["problem_id"])] == held_out_fold
        ]
        train_ids = {str(row["problem_id"]) for row in train}
        held_out_ids = {str(row["problem_id"]) for row in held_out}
        if train_ids & held_out_ids:
            raise AssertionError("Problem leakage across ensemble folds.")
        model = fit_pairwise_model(train, features, ridge=ridge)
        model.update(
            {
                "held_out_fold": held_out_fold,
                "n_train_problems": len(train_ids),
                "n_held_out_problems": len(held_out_ids),
            }
        )
        models[str(held_out_fold)] = model
        for row in held_out:
            key = (str(row["problem_id"]), int(row["sample_id"]))
            if key in predictions:
                raise AssertionError(f"Duplicate OOF prediction: {key}.")
            predictions[key] = predict(row, model)
    if len(predictions) != len(eligible):
        raise AssertionError(
            f"Expected {len(eligible)} OOF predictions, got {len(predictions)}."
        )
    deployment = fit_pairwise_model(eligible, features, ridge=ridge)
    deployment["warning"] = "Fit on all development problems; not used for metrics."
    models["all_data_deployment_fit_not_used_for_metrics"] = deployment
    return predictions, models


def hard_win(margin: float) -> float:
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def make_pair_records(rows: Sequence[dict], score_names: Sequence[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    records = []
    for problem_id, group in groups.items():
        correct = [row for row in group if bool(row["correct"])]
        incorrect = [row for row in group if not bool(row["correct"])]
        for positive in correct:
            for negative in incorrect:
                margins = {}
                for name in score_names:
                    if name in positive["features"]:
                        positive_score = float(positive["features"][name])
                        negative_score = float(negative["features"][name])
                    else:
                        positive_score = float(positive["ensembles"][name])
                        negative_score = float(negative["ensembles"][name])
                    margins[name] = positive_score - negative_score
                records.append(
                    {
                        "problem_id": problem_id,
                        "correct_sample_id": int(positive["sample_id"]),
                        "incorrect_sample_id": int(negative["sample_id"]),
                        "margins": margins,
                    }
                )
    return records


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return [float(low), float(high)]


def summarize_checkpoint(
    rows: Sequence[dict],
    score_names: Sequence[str],
    *,
    ensemble_names: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    pairs = make_pair_records(rows, score_names)
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_problem[str(pair["problem_id"])].append(pair)
    metrics = {}
    for name in score_names:
        pair_accuracy = mean([hard_win(float(row["margins"][name])) for row in pairs])
        problem_accuracy = mean(
            [
                mean([hard_win(float(row["margins"][name])) for row in group])
                for group in by_problem.values()
            ]
        )
        candidate_groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            candidate_groups[str(row["problem_id"])].append(row)
        top1 = []
        for group in candidate_groups.values():
            if name in group[0]["features"]:
                selected = max(
                    group,
                    key=lambda row: (
                        float(row["features"][name]),
                        -int(row["sample_id"]),
                    ),
                )
            else:
                selected = max(
                    group,
                    key=lambda row: (
                        float(row["ensembles"][name]),
                        -int(row["sample_id"]),
                    ),
                )
            top1.append(float(bool(selected["correct"])))
        metrics[name] = {
            "pair_accuracy": pair_accuracy,
            "problem_balanced_accuracy": problem_accuracy,
            "top1_correct_rate_on_mixed_problems": mean(top1),
        }
    problem_ids = sorted(by_problem)
    boot: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        drawn = [rng.choice(problem_ids) for _ in problem_ids]
        problem_values = {
            name: [
                mean(
                    [
                        hard_win(float(row["margins"][name]))
                        for row in by_problem[problem_id]
                    ]
                )
                for problem_id in drawn
            ]
            for name in score_names
        }
        for name in score_names:
            boot[f"accuracy::{name}"].append(mean(problem_values[name]))
        for ensemble in ensemble_names:
            for reference in score_names:
                if reference == ensemble:
                    continue
                boot[f"difference::{ensemble}::{reference}"].append(
                    mean(problem_values[ensemble]) - mean(problem_values[reference])
                )
    for name in score_names:
        metrics[name]["problem_balanced_bootstrap_95_ci"] = percentile_interval(
            boot[f"accuracy::{name}"]
        )
    differences = {}
    for ensemble in ensemble_names:
        differences[ensemble] = {}
        for reference in score_names:
            if reference == ensemble:
                continue
            values = boot[f"difference::{ensemble}::{reference}"]
            differences[ensemble][reference] = {
                "point_difference": (
                    metrics[ensemble]["problem_balanced_accuracy"]
                    - metrics[reference]["problem_balanced_accuracy"]
                ),
                "bootstrap_95_ci": percentile_interval(values),
            }
    return {
        "n_mixed_problems": len(by_problem),
        "n_correct_incorrect_pairs": len(pairs),
        "scores": metrics,
        "paired_differences": differences,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--ensemble", action="append", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=41)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=73)
    parser.add_argument("--primary-checkpoint", default="token_1024")
    parser.add_argument("--primary-ensemble", default="all_four")
    parser.add_argument("--save-oof-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    rows, feature_metadata = load_features(args.feature)
    feature_names = list(feature_metadata)
    ensembles = dict(parse_ensemble_spec(value) for value in args.ensemble)
    for name, features in ensembles.items():
        missing = set(features) - set(feature_names)
        if missing:
            raise ValueError(f"Ensemble {name} has unknown features: {sorted(missing)}.")
    checkpoints = sorted({str(row["checkpoint"]) for row in rows})
    models = {}
    for row in rows:
        row["ensembles"] = {}
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        models[checkpoint] = {}
        for ensemble_name, features in ensembles.items():
            predictions, fold_models = cross_validated_scores(
                checkpoint_rows,
                features,
                folds=args.folds,
                seed=args.fold_seed,
                ridge=args.ridge,
            )
            models[checkpoint][ensemble_name] = fold_models
            for row in checkpoint_rows:
                key = (str(row["problem_id"]), int(row["sample_id"]))
                if key in predictions:
                    row["ensembles"][ensemble_name] = predictions[key]
    evaluation = {}
    all_score_names = feature_names + list(ensembles)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_rows = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and row["ensembles"]
        ]
        evaluation[checkpoint] = summarize_checkpoint(
            checkpoint_rows,
            all_score_names,
            ensemble_names=list(ensembles),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + checkpoint_index,
        )
    if args.primary_checkpoint not in evaluation:
        raise ValueError(f"Primary checkpoint {args.primary_checkpoint} is unavailable.")
    primary = evaluation[args.primary_checkpoint]
    if args.primary_ensemble not in primary["scores"]:
        raise ValueError(f"Primary ensemble {args.primary_ensemble} is unavailable.")
    primary_metrics = primary["scores"][args.primary_ensemble]
    single_accuracies = {
        name: primary["scores"][name]["problem_balanced_accuracy"]
        for name in feature_names
    }
    interval = primary_metrics["problem_balanced_bootstrap_95_ci"]
    gate = {
        "checkpoint": args.primary_checkpoint,
        "ensemble": args.primary_ensemble,
        "problem_balanced_accuracy_at_least_60_percent": (
            primary_metrics["problem_balanced_accuracy"] >= 0.6
        ),
        "bootstrap_lower_bound_above_chance": interval[0] > 0.5,
        "point_estimate_exceeds_every_single_feature": (
            primary_metrics["problem_balanced_accuracy"] > max(single_accuracies.values())
        ),
    }
    gate["passed"] = all(
        value for key, value in gate.items() if key not in {"checkpoint", "ensemble"}
    )
    by_checkpoint_cost = {}
    for checkpoint in checkpoints:
        feature_tokens = {}
        for specification in args.feature:
            name, path, rubric = parse_feature_spec(specification)
            source_rows = load_jsonl(path)
            if rubric is not None:
                source_rows = [row for row in source_rows if row.get("rubric") == rubric]
            source_rows = [row for row in source_rows if row["checkpoint"] == checkpoint]
            feature_tokens[name] = sum(int(row["prompt_tokens"]) for row in source_rows)
        n_prefixes = len([row for row in rows if row["checkpoint"] == checkpoint])
        by_checkpoint_cost[checkpoint] = {
            "n_prefixes": n_prefixes,
            "all_four_calls_per_prefix": len(feature_names),
            "mean_all_four_prompt_tokens_per_prefix": (
                sum(feature_tokens.values()) / n_prefixes
            ),
            "mean_all_four_prompt_tokens_for_two_particles": (
                2 * sum(feature_tokens.values()) / n_prefixes
            ),
            "feature_prompt_tokens": feature_tokens,
        }
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "problem_level_out_of_fold_pairwise_semantic_ensemble",
            "features": feature_names,
            "ensembles": ensembles,
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "ridge": args.ridge,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "problem",
            "uses_generator_likelihood": False,
            "evaluation_status": "exploratory_cross_validation_not_disjoint_test",
        },
        "dataset": {
            "n_source_problems": len({row["problem_id"] for row in rows}),
            "n_mixed_problems": len(mixed_problem_ids(rows)),
            "n_prefix_score_rows": len(rows),
        },
        "feature_metadata": feature_metadata,
        "fold_models": models,
        "checkpoints": evaluation,
        "primary_gate": gate,
        "cost": {
            "all_feature_calls": sum(meta["calls"] for meta in feature_metadata.values()),
            "all_feature_prompt_tokens": sum(
                meta["prompt_tokens"] for meta in feature_metadata.values()
            ),
            "all_feature_completion_tokens": sum(
                meta["completion_tokens"] for meta in feature_metadata.values()
            ),
            "by_checkpoint": by_checkpoint_cost,
            "fusion_requires_no_model_inference": True,
        },
    }
    write_jsonl(args.save_oof_scores, rows)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

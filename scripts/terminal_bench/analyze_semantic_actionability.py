#!/usr/bin/env python3
"""Measure whether semantic checkpoint scores can guide particle allocation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


INTERVALS = (256, 512, 2048, 8192)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    wins = 0.0
    comparisons = 0
    positive_scores = [score for score, label in zip(scores, labels) if label]
    negative_scores = [score for score, label in zip(scores, labels) if not label]
    for positive in positive_scores:
        for negative in negative_scores:
            comparisons += 1
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / comparisons


def pearson(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return float("nan")
    first_mean = mean(first)
    second_mean = mean(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    left_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    right_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if left_scale == 0 or right_scale == 0:
        return float("nan")
    return numerator / (left_scale * right_scale)


def group_metrics(
    records: Sequence[dict[str, Any]],
    *,
    include_global_auroc: bool = True,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["group_id"])].append(record)

    group_ranking: list[float] = []
    pooled_wins = 0.0
    pooled_comparisons = 0
    retention: list[float] = []
    random_retention: list[float] = []
    mixed_tasks: set[str] = set()
    for group in groups.values():
        positives = [row for row in group if row["eventual_success"]]
        negatives = [row for row in group if not row["eventual_success"]]
        if not positives or not negatives:
            continue
        mixed_tasks.add(str(group[0]["task_id"]))
        wins = 0.0
        comparisons = 0
        for positive in positives:
            for negative in negatives:
                comparisons += 1
                wins += (
                    1.0
                    if positive["score"] > negative["score"]
                    else 0.5 if positive["score"] == negative["score"] else 0.0
                )
        group_ranking.append(wins / comparisons)
        pooled_wins += wins
        pooled_comparisons += comparisons

        keep = max(1, math.ceil(len(group) / 2))
        ranked = sorted(
            group,
            key=lambda row: (-float(row["score"]), str(row["trajectory_id"])),
        )
        retained = sum(row["eventual_success"] for row in ranked[:keep])
        retention.append(retained / len(positives))
        random_retention.append(keep / len(group))

    scores = [float(record["score"]) for record in records]
    labels = [bool(record["eventual_success"]) for record in records]
    survival = mean(retention)
    baseline = mean(random_retention)
    return {
        "records": len(records),
        "groups": len(groups),
        "mixed_groups": len(group_ranking),
        "mixed_tasks": len(mixed_tasks),
        "prevalence": mean([float(label) for label in labels]),
        "global_auroc": (
            binary_auroc(scores, labels)
            if include_global_auroc
            else float("nan")
        ),
        "group_balanced_ranking_accuracy": mean(group_ranking),
        "pooled_pairwise_ranking_accuracy": (
            pooled_wins / pooled_comparisons
            if pooled_comparisons
            else float("nan")
        ),
        "pairwise_comparisons": pooled_comparisons,
        "top_half_correct_trajectory_survival": survival,
        "top_half_random_trajectory_survival": baseline,
        "top_half_survival_lift": survival - baseline,
    }


def bootstrap_tasks(
    records: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_groups[str(record["task_id"])].append(record)
    tasks = sorted(task_groups)
    if len(tasks) < 2 or samples <= 0:
        return {}
    rng = random.Random(seed)
    observations: dict[str, list[float]] = defaultdict(list)
    keys = (
        "group_balanced_ranking_accuracy",
        "pooled_pairwise_ranking_accuracy",
        "top_half_correct_trajectory_survival",
        "top_half_survival_lift",
    )
    for _ in range(samples):
        resampled: list[dict[str, Any]] = []
        for draw_index in range(len(tasks)):
            task = rng.choice(tasks)
            for record in task_groups[task]:
                clone = dict(record)
                clone["group_id"] = f"{draw_index}:{record['group_id']}"
                clone["task_id"] = f"{draw_index}:{task}"
                resampled.append(clone)
        metrics = group_metrics(resampled, include_global_auroc=False)
        for key in keys:
            value = float(metrics[key])
            if math.isfinite(value):
                observations[key].append(value)
    intervals: dict[str, list[float]] = {}
    for key, values in observations.items():
        ordered = sorted(values)
        lower = ordered[math.floor(0.025 * (len(ordered) - 1))]
        upper = ordered[math.ceil(0.975 * (len(ordered) - 1))]
        intervals[key] = [lower, upper]
    return intervals


def join_scores(
    scores: Sequence[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for score in scores:
        trajectory_id = str(score["trajectory_id"])
        if trajectory_id not in labels or trajectory_id not in trajectories:
            raise ValueError(f"unmatched trajectory ID: {trajectory_id}")
        trigger = score["trigger"]
        if trigger["kind"] == "token_interval":
            position = int(trigger["target_generated_tokens"])
            group_id = f"{score['task_id']}:token:{position}"
            preterminal = position < int(
                trajectories[trajectory_id]["counts"]["generated_tokens"]
            )
        else:
            position = int(trigger["tool_index"])
            group_id = f"{score['task_id']}:tool:{position}"
            preterminal = True
        joined.append(
            {
                **score,
                "eventual_success": bool(
                    labels[trajectory_id]["eventual_success"]
                ),
                "group_id": group_id,
                "position": position,
                "preterminal": preterminal,
            }
        )
    return joined


def view_records(
    records: Sequence[dict[str, Any]],
    view: str,
) -> list[dict[str, Any]]:
    if view == "post_tool":
        return [
            record
            for record in records
            if record["trigger"]["kind"] == "post_tool" and record["preterminal"]
        ]
    interval = int(view.removeprefix("token_"))
    return [
        record
        for record in records
        if record["trigger"]["kind"] == "token_interval"
        and record["preterminal"]
        and int(record["position"]) % interval == 0
    ]


def rank_accuracy(records: Sequence[dict[str, Any]], score_key: str) -> float:
    projected = [{**record, "score": record[score_key]} for record in records]
    return float(
        group_metrics(projected, include_global_auroc=False)[
            "group_balanced_ranking_accuracy"
        ]
    )


def out_of_task_ensemble(
    left: Sequence[dict[str, Any]],
    right_by_checkpoint: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    paired = [
        {
            **record,
            "score_left": float(record["score"]),
            "score_right": float(
                right_by_checkpoint[str(record["checkpoint_id"])]["score"]
            ),
        }
        for record in left
        if str(record["checkpoint_id"]) in right_by_checkpoint
    ]
    tasks = sorted({str(record["task_id"]) for record in paired})
    selected_weights: dict[str, float] = {}
    output: list[dict[str, Any]] = []
    candidates = [index / 20 for index in range(21)]
    for held_out in tasks:
        train = [record for record in paired if str(record["task_id"]) != held_out]
        scored_weights = []
        for weight in candidates:
            candidates_rows = [
                {
                    **record,
                    "ensemble": (
                        weight * record["score_left"]
                        + (1 - weight) * record["score_right"]
                    ),
                }
                for record in train
            ]
            accuracy = rank_accuracy(candidates_rows, "ensemble")
            scored_weights.append((accuracy, -abs(weight - 0.5), weight))
        finite = [entry for entry in scored_weights if math.isfinite(entry[0])]
        weight = max(finite)[2] if finite else 0.5
        selected_weights[held_out] = weight
        for record in paired:
            if str(record["task_id"]) != held_out:
                continue
            output.append(
                {
                    **record,
                    "score": (
                        weight * record["score_left"]
                        + (1 - weight) * record["score_right"]
                    ),
                }
            )
    return output, selected_weights


def evaluate_view(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result = group_metrics(records)
    result["bootstrap_95_ci"] = bootstrap_tasks(
        records,
        samples=bootstrap_samples,
        seed=seed,
    )
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    return value


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    def number(value: Any) -> str:
        return "-" if value is None else f"{float(value):.3f}"

    lines = [
        "# Terminal-Bench semantic actionability",
        "",
        "| scorer | view | rank accuracy | 95% CI | survival lift | mixed tasks |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    sources = {**result["models"], "out_of_task_ensemble": result["ensemble"]}
    for scorer, views in sources.items():
        for view, metrics in views.items():
            interval = (
                metrics.get("bootstrap_95_ci", {})
                .get("group_balanced_ranking_accuracy")
            )
            ci = (
                "-"
                if not interval
                else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
            )
            lines.append(
                f"| {scorer} | {view} | "
                f"{number(metrics['group_balanced_ranking_accuracy'])} | "
                f"{ci} | {number(metrics['top_half_survival_lift'])} | "
                f"{metrics['mixed_tasks']} |"
            )
    lines.extend(
        [
            "",
            f"Decision: **{result['decision']['status']}**",
            "",
            result["decision"]["reason"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--score-summary", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.scores) != 2:
        raise ValueError("exactly two heterogeneous score files are required")
    trajectory_rows = load_jsonl(args.trajectories)
    label_rows = load_jsonl(args.labels)
    manifest = json.loads(args.manifest.read_text())
    gate_config = manifest["offline_actionability_gate"]
    verifier_config = manifest["semantic_verifiers"]
    expected_models = [
        str(entry["id"]) for entry in verifier_config["models"]
    ]
    expected_criterion = str(verifier_config["criterion"])
    expected_dataset_id = str(manifest["dataset_id"])

    trajectories = {
        str(row["trajectory_id"]): row for row in trajectory_rows
    }
    labels = {str(row["trajectory_id"]): row for row in label_rows}
    if set(trajectories) != set(labels):
        raise ValueError("trajectory and label IDs do not match")
    for name, rows in (("trajectories", trajectory_rows), ("labels", label_rows)):
        if any(str(row["dataset_id"]) != expected_dataset_id for row in rows):
            raise ValueError(
                f"{name} do not match manifest dataset {expected_dataset_id}"
            )

    raw_scores = [load_jsonl(path) for path in args.scores]
    models = [str(rows[0]["scorer_model"]) for rows in raw_scores]
    if models != expected_models:
        raise ValueError(
            f"score models {models} do not match frozen models {expected_models}"
        )
    if models[0] == models[1]:
        raise ValueError("score files must use heterogeneous verifier models")
    for model, rows in zip(models, raw_scores):
        if any(str(row["scorer_model"]) != model for row in rows):
            raise ValueError(f"score file mixes verifier models: {model}")
        if any(str(row["criterion"]) != expected_criterion for row in rows):
            raise ValueError(
                f"score criterion does not match manifest: {model}"
            )
        if any(str(row["dataset_id"]) != expected_dataset_id for row in rows):
            raise ValueError(f"score dataset does not match manifest: {model}")
        checkpoint_ids = [str(row["checkpoint_id"]) for row in rows]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError(f"score file has duplicate checkpoints: {model}")
    joined = [
        join_scores(rows, labels, trajectories) for rows in raw_scores
    ]
    if {row["checkpoint_id"] for row in joined[0]} != {
        row["checkpoint_id"] for row in joined[1]
    }:
        raise ValueError("verifier score files cover different checkpoints")

    views = [f"token_{interval}" for interval in INTERVALS] + ["post_tool"]
    model_results: dict[str, dict[str, Any]] = {}
    for model_index, model in enumerate(models):
        model_results[model] = {}
        for view_index, view in enumerate(views):
            records = view_records(joined[model_index], view)
            model_results[model][view] = evaluate_view(
                records,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + model_index * 100 + view_index,
            )

    right_by_checkpoint = {
        str(row["checkpoint_id"]): row for row in joined[1]
    }
    ensemble_results: dict[str, Any] = {}
    ensemble_weights: dict[str, dict[str, float]] = {}
    for view_index, view in enumerate(views):
        left_view = view_records(joined[0], view)
        ensemble, weights = out_of_task_ensemble(left_view, right_by_checkpoint)
        ensemble_results[view] = evaluate_view(
            ensemble,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 500 + view_index,
        )
        ensemble_weights[view] = weights

    paired_errors_left: list[float] = []
    paired_errors_right: list[float] = []
    for left in joined[0]:
        right = right_by_checkpoint[str(left["checkpoint_id"])]
        label = float(left["eventual_success"])
        paired_errors_left.append(float(left["score"]) - label)
        paired_errors_right.append(float(right["score"]) - label)
    error_correlation = pearson(paired_errors_left, paired_errors_right)

    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for view, metrics in ensemble_results.items():
        if not view.startswith("token_"):
            continue
        interval = (
            metrics.get("bootstrap_95_ci", {})
            .get("group_balanced_ranking_accuracy")
        )
        lower = interval[0] if interval else float("-inf")
        lift = float(metrics["top_half_survival_lift"])
        candidates.append((lower, lift, view, metrics))
    _, _, best_view, best_metrics = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )
    single_accuracies = [
        float(model_results[model][best_view]["group_balanced_ranking_accuracy"])
        for model in models
    ]
    finite_single_accuracies = [
        value for value in single_accuracies if math.isfinite(value)
    ]
    best_single = (
        max(finite_single_accuracies)
        if finite_single_accuracies
        else float("nan")
    )
    ensemble_accuracy = float(
        ensemble_results[best_view]["group_balanced_ranking_accuracy"]
    )
    ranking_ci = (
        best_metrics.get("bootstrap_95_ci", {})
        .get("group_balanced_ranking_accuracy")
        or [float("-inf"), float("inf")]
    )
    ranking_min = float(gate_config["group_balanced_ranking_accuracy_min"])
    ranking_ci_min = float(
        gate_config["ranking_bootstrap_95_ci_lower_strictly_above"]
    )
    survival_lift_min = float(
        gate_config["top_half_correct_trajectory_survival_lift_min"]
    )
    mixed_tasks_min = int(gate_config["mixed_tasks_min"])
    comparisons_min = int(gate_config["pairwise_comparisons_min"])
    gates = {
        "ranking_accuracy_at_least_0.60": (
            float(best_metrics["group_balanced_ranking_accuracy"]) >= ranking_min
        ),
        "ranking_bootstrap_lower_above_0.50": ranking_ci[0] > ranking_ci_min,
        "survival_lift_at_least_0.10": (
            float(best_metrics["top_half_survival_lift"]) >= survival_lift_min
        ),
        "at_least_4_mixed_tasks": (
            int(best_metrics["mixed_tasks"]) >= mixed_tasks_min
        ),
        "at_least_100_pairwise_comparisons": (
            int(best_metrics["pairwise_comparisons"]) >= comparisons_min
        ),
        "ensemble_improves_best_single": (
            not bool(gate_config["ensemble_must_improve_best_single"])
            or ensemble_accuracy > best_single
        ),
    }
    passes = all(gates.values())
    failed_gates = [name for name, passed in gates.items() if not passed]
    decision = {
        "status": "promote_to_online_smc" if passes else "do_not_promote",
        "best_source": "out_of_task_ensemble",
        "best_view": best_view,
        "best_metrics": best_metrics,
        "ensemble_accuracy_at_best_view": ensemble_accuracy,
        "best_single_accuracy_at_best_view": best_single,
        "gates": gates,
        "thresholds": gate_config,
        "failed_gates": failed_gates,
        "reason": (
            "All frozen actionability gates passed."
            if passes
            else "Frozen semantic-actionability gates failed: "
            + ", ".join(failed_gates)
            + "."
        ),
    }
    label_by_task: dict[str, list[bool]] = defaultdict(list)
    for label in label_rows:
        label_by_task[str(label["task_id"])].append(
            bool(label["eventual_success"])
        )
    label_diversity = {
        task: {
            "trajectories": len(values),
            "successes": sum(values),
            "success_rate": mean([float(value) for value in values]),
            "mixed": 0 < sum(values) < len(values),
        }
        for task, values in sorted(label_by_task.items())
    }
    result = {
        "schema_version": 1,
        "dataset": {
            "trajectories": len(trajectory_rows),
            "tasks": len(label_by_task),
            "mixed_tasks": sum(row["mixed"] for row in label_diversity.values()),
            "label_diversity": label_diversity,
        },
        "models": model_results,
        "ensemble": ensemble_results,
        "ensemble_out_of_task_weights": ensemble_weights,
        "cross_verifier_error_correlation": error_correlation,
        "score_cost_summaries": (
            [
                json.loads(path.read_text())
                for path in (args.score_summary or [])
            ]
        ),
        "decision": decision,
    }
    safe_result = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe_result, indent=2, sort_keys=True) + "\n")
    write_markdown(args.markdown_output, safe_result)
    print(json.dumps(safe_result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

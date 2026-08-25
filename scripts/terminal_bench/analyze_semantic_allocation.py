#!/usr/bin/env python3
"""Consolidate matched Terminal-Bench semantic-allocation artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Sequence


SCHEMA_VERSION = 1
CELL_FIELDS = (
    "selected_reward",
    "ar1_reward",
    "random_particle_expected_reward",
    "pass_at_n",
    "population_success_rate",
    "particle_successes",
    "generator_model_calls",
    "generator_prompt_tokens",
    "generator_completion_tokens",
    "generator_tool_calls",
    "environment_replays",
    "environment_replay_wall_time_s",
    "physical_verifier_calls",
    "logical_verifier_calls",
    "verifier_prompt_tokens",
    "verifier_completion_tokens",
    "verifier_inference_wall_time_s",
    "resampling_count",
    "wall_time_s",
    "grader_wall_time_s",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)) + 1)]
    return [low, high]


def task_cluster_bootstrap(
    values: dict[str, list[float]], *, samples: int, seed: int
) -> list[float]:
    tasks = sorted(values)
    if not tasks:
        return []
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [rng.choice(tasks) for _ in tasks]
        draws.append(
            statistics.fmean(
                value for task_id in selected for value in values[task_id]
            )
        )
    return percentile_interval(draws)


def task_id(result: dict[str, Any]) -> str:
    value = (result.get("task") or {}).get("task_id")
    if not value:
        raise ValueError("result has no task.task_id")
    return str(value)


def method_id(result: dict[str, Any]) -> str:
    value = (result.get("configuration") or {}).get("method_id")
    if not value:
        raise ValueError("result has no configuration.method_id")
    return str(value)


def infer_repetition(
    report: dict[str, Any],
    result: dict[str, Any],
    counters: dict[tuple[str, str], int],
) -> int:
    if "repetition" in result:
        return int(result["repetition"])
    key = (task_id(result), method_id(result))
    selected = report.get("selected_repetitions")
    if selected is None:
        selected = list(range(int(report["repetitions"])))
    index = counters[key]
    if index >= len(selected):
        raise ValueError(
            f"cannot infer legacy repetition for {key}: "
            f"index={index}, choices={selected}"
        )
    counters[key] += 1
    return int(selected[index])


def number(mapping: dict[str, Any], key: str) -> float:
    return float(mapping.get(key, 0) or 0)


def extract_artifact(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    report = load_json(source_path)
    counters: dict[tuple[str, str], int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for result in report.get("results", []):
        task = task_id(result)
        method = method_id(result)
        repetition = infer_repetition(report, result, counters)
        record: dict[str, Any] = {
            "key": (task, method, repetition),
            "task_id": task,
            "method_id": method,
            "repetition": repetition,
            "status": str(result.get("status")),
            "experiment_id": result.get("experiment_id"),
            "source": source_path.name,
            "error": result.get("error"),
        }
        if record["status"] == "pass":
            grader = result.get("grader") or {}
            particles = grader.get("particles") or []
            rewards = [number(row, "reward") for row in particles]
            selection = result.get("selection") or {}
            baselines = result.get("selection_baselines") or {}
            generator = result.get("generator_tool_replay_cost") or {}
            verifier = result.get("semantic_verifier_cost") or {}
            record.update(
                {
                    "selected_reward": number(selection, "reward"),
                    "ar1_reward": number(
                        baselines.get("ar1_slot0") or {}, "reward"
                    ),
                    "random_particle_expected_reward": number(
                        baselines, "random_particle_expected_reward"
                    ),
                    "pass_at_n": number(baselines, "pass_at_n"),
                    "population_success_rate": number(grader, "success_rate"),
                    "particle_successes": sum(reward > 0 for reward in rewards),
                    "particle_gradings": len(rewards),
                    "generator_model_calls": number(generator, "model_calls"),
                    "generator_prompt_tokens": number(generator, "prompt_tokens"),
                    "generator_completion_tokens": number(
                        generator, "completion_tokens"
                    ),
                    "generator_tool_calls": number(generator, "tool_calls"),
                    "environment_replays": number(
                        generator, "environment_replays"
                    ),
                    "environment_replay_wall_time_s": number(
                        generator, "environment_replay_wall_time_s"
                    ),
                    "physical_verifier_calls": number(
                        verifier, "physical_verifier_calls"
                    ),
                    "logical_verifier_calls": number(
                        verifier, "logical_verifier_calls"
                    ),
                    "verifier_prompt_tokens": number(verifier, "prompt_tokens"),
                    "verifier_completion_tokens": number(
                        verifier, "completion_tokens"
                    ),
                    "verifier_inference_wall_time_s": number(
                        verifier, "inference_wall_time_s"
                    ),
                    "resampling_count": number(result, "resampling_count"),
                    "wall_time_s": number(result, "wall_time_s"),
                    "grader_wall_time_s": number(grader, "wall_time_s"),
                }
            )
        records.append(record)
    allocation = report.get("allocation_accounting") or {}
    source = {
        "file": source_path.name,
        "sha256": sha256_file(source_path),
        "bytes": source_path.stat().st_size,
        "status": report.get("status"),
        "selected_methods": list(report.get("selected_methods") or []),
        "selected_repetitions": list(report.get("selected_repetitions") or []),
        "result_records": len(records),
        "error_records": sum(row["status"] != "pass" for row in records),
        "runner_wall_time_s": number(allocation, "runner_wall_time_s"),
        "runner_allocated_accelerator_seconds": number(
            allocation, "runner_allocated_accelerator_seconds"
        ),
    }
    return records, source


def merge_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    replacements = []
    ignored_errors = []
    for row in records:
        key = row["key"]
        previous = merged.get(key)
        if previous is None:
            merged[key] = row
        elif previous["status"] != "pass" and row["status"] == "pass":
            replacements.append(
                {
                    "task_id": key[0],
                    "method_id": key[1],
                    "repetition": key[2],
                    "failed_source": previous["source"],
                    "recovery_source": row["source"],
                    "error": previous["error"],
                }
            )
            merged[key] = row
        elif previous["status"] == "pass" and row["status"] != "pass":
            ignored_errors.append(
                {
                    "task_id": key[0],
                    "method_id": key[1],
                    "repetition": key[2],
                    "source": row["source"],
                    "error": row["error"],
                }
            )
        else:
            raise ValueError(
                f"ambiguous duplicate {key}: "
                f"{previous['source']} and {row['source']}"
            )
    failures = [row["key"] for row in merged.values() if row["status"] != "pass"]
    if failures:
        raise ValueError(f"unrecovered experiment cells: {failures}")
    cells = sorted(
        merged.values(),
        key=lambda row: (row["task_id"], row["method_id"], row["repetition"]),
    )
    return cells, {
        "replacements": replacements,
        "ignored_later_errors": ignored_errors,
    }


def validate_alignment(
    cells: Sequence[dict[str, Any]], *, baseline: str
) -> tuple[list[str], list[str], list[int]]:
    methods = sorted({row["method_id"] for row in cells})
    if baseline not in methods:
        raise ValueError(f"baseline {baseline!r} is not present")
    keys_by_method = {
        method: {
            (row["task_id"], row["repetition"])
            for row in cells
            if row["method_id"] == method
        }
        for method in methods
    }
    expected = keys_by_method[baseline]
    for method, keys in keys_by_method.items():
        if keys != expected:
            raise ValueError(
                f"unaligned {method}: "
                f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
            )
    tasks = sorted({task for task, _ in expected})
    repetitions = sorted({repetition for _, repetition in expected})
    for task in tasks:
        observed = sorted(rep for name, rep in expected if name == task)
        if observed != repetitions:
            raise ValueError(
                f"task {task} repetitions {observed}, expected {repetitions}"
            )
    return methods, tasks, repetitions


def summarize_method(
    rows: Sequence[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cells": len(rows),
        "selected_successes": int(sum(row["selected_reward"] for row in rows)),
        "pass_blocks": int(sum(row["pass_at_n"] for row in rows)),
        "particle_successes": int(sum(row["particle_successes"] for row in rows)),
        "particle_gradings": int(sum(row["particle_gradings"] for row in rows)),
    }
    rate_fields = {
        "selected_reward",
        "ar1_reward",
        "random_particle_expected_reward",
        "pass_at_n",
        "population_success_rate",
    }
    for field in CELL_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[f"mean_{field}"] = statistics.fmean(values)
        if field not in rate_fields:
            summary[f"total_{field}"] = sum(values)
    summary["selection_recall_given_pass"] = (
        summary["selected_successes"] / summary["pass_blocks"]
        if summary["pass_blocks"]
        else None
    )
    values_by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values_by_task[row["task_id"]].append(float(row["selected_reward"]))
    summary["selected_reward_task_cluster_bootstrap_95_ci"] = (
        task_cluster_bootstrap(
            values_by_task, samples=bootstrap_samples, seed=seed
        )
    )
    return summary


def paired_comparison(
    baseline_rows: Sequence[dict[str, Any]],
    method_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = {
        (row["task_id"], row["repetition"]): row for row in baseline_rows
    }
    method = {
        (row["task_id"], row["repetition"]): row for row in method_rows
    }
    if baseline.keys() != method.keys():
        raise ValueError("paired comparison received unaligned cells")
    deltas_by_task: dict[str, list[float]] = defaultdict(list)
    deltas = []
    wins = []
    losses = []
    for task, repetition in sorted(baseline):
        delta = (
            float(method[(task, repetition)]["selected_reward"])
            - float(baseline[(task, repetition)]["selected_reward"])
        )
        deltas.append(delta)
        deltas_by_task[task].append(delta)
        detail = {"task_id": task, "repetition": repetition}
        if delta > 0:
            wins.append(detail)
        elif delta < 0:
            losses.append(detail)
    return {
        "selected_reward_difference": statistics.fmean(deltas),
        "task_cluster_bootstrap_95_ci": task_cluster_bootstrap(
            deltas_by_task, samples=bootstrap_samples, seed=seed
        ),
        "wins": len(wins),
        "losses": len(losses),
        "ties": len(deltas) - len(wins) - len(losses),
        "win_blocks": wins,
        "loss_blocks": losses,
    }


def analyze(
    artifact_paths: Sequence[str | Path],
    *,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    records = []
    sources = []
    for path in artifact_paths:
        extracted, source = extract_artifact(path)
        records.extend(extracted)
        sources.append(source)
    cells, recovery = merge_records(records)
    methods, tasks, repetitions = validate_alignment(cells, baseline=baseline)
    rows_by_method = {
        method: [row for row in cells if row["method_id"] == method]
        for method in methods
    }
    metrics = {
        method: summarize_method(
            rows, bootstrap_samples=bootstrap_samples, seed=seed + index
        )
        for index, (method, rows) in enumerate(rows_by_method.items())
    }
    allocated_by_method: dict[str, float] = defaultdict(float)
    for source in sources:
        selected_methods = source["selected_methods"]
        if len(selected_methods) == 1:
            allocated_by_method[str(selected_methods[0])] += float(
                source["runner_allocated_accelerator_seconds"]
            )
    for method, value in allocated_by_method.items():
        if method in metrics:
            metrics[method]["executed_allocated_accelerator_seconds"] = value
            metrics[method]["executed_allocated_accelerator_hours"] = value / 3600
            metrics[method][
                "executed_allocated_accelerator_seconds_per_valid_cell"
            ] = value / metrics[method]["cells"]
    comparisons = {}
    for index, method in enumerate(methods):
        if method != baseline:
            comparisons[f"{method}_minus_{baseline}"] = paired_comparison(
                rows_by_method[baseline],
                rows_by_method[method],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 + index,
            )
    task_outcomes: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_outcomes[task] = {}
        for method in methods:
            rows = [
                row for row in rows_by_method[method] if row["task_id"] == task
            ]
            task_outcomes[task][method] = {
                "selected_successes": int(
                    sum(row["selected_reward"] for row in rows)
                ),
                "pass_blocks": int(sum(row["pass_at_n"] for row in rows)),
                "particle_successes": int(
                    sum(row["particle_successes"] for row in rows)
                ),
                "particle_gradings": int(
                    sum(row["particle_gradings"] for row in rows)
                ),
            }
    terminal_accuracy = metrics[baseline]["mean_selected_reward"]
    strict_winners = [
        method
        for method in methods
        if method != baseline
        and metrics[method]["mean_selected_reward"] > terminal_accuracy
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "terminal-semantic-allocation-primary-analysis-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "baseline": baseline,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "bootstrap_unit": "task cluster; all repetitions retained",
            "merge_rule": (
                "a later pass replaces only an earlier error for the same "
                "task-method-repetition cell"
            ),
        },
        "coverage": {
            "tasks": tasks,
            "methods": methods,
            "repetitions": repetitions,
            "valid_cells": len(cells),
            "official_particle_gradings": int(
                sum(row["particle_gradings"] for row in cells)
            ),
            "executed_result_records": len(records),
            "executed_error_records": sum(
                row["status"] != "pass" for row in records
            ),
            "recovered_error_cells": len(recovery["replacements"]),
        },
        "source_artifacts": sources,
        "recovery_audit": recovery,
        "metrics": metrics,
        "paired_vs_terminal_bon": comparisons,
        "task_outcomes": task_outcomes,
        "decision_gate": {
            "primary_metric": "mean official selected reward across task-repeat blocks",
            "terminal_bon_selected_reward": terminal_accuracy,
            "strictly_better_intermediate_methods": strict_winners,
            "passed": bool(strict_winners),
            "scaling_sweeps_authorized": bool(strict_winners),
            "reason": (
                "At least one intermediate policy strictly beat terminal BoN."
                if strict_winners
                else "No intermediate policy strictly beat terminal BoN."
            ),
        },
        "total_executed_allocated_accelerator_seconds": sum(
            float(source["runner_allocated_accelerator_seconds"])
            for source in sources
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--baseline", default="terminal_bon")
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    report = analyze(
        args.artifact,
        baseline=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "coverage": report["coverage"],
                "metrics": report["metrics"],
                "paired_vs_terminal_bon": report["paired_vs_terminal_bon"],
                "decision_gate": report["decision_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

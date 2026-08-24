#!/usr/bin/env python3
"""Aggregate live Terminal-Bench semantic-SMC sweep reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
AXIS_FIELDS = {
    "particles": "num_particles",
    "interval": "checkpoint_interval_tokens",
    "verifier_calls": "verifier_calls_per_checkpoint",
    "beta": "semantic_beta",
    "ess": "ess_threshold_fraction",
}


def binary_auc(points: Iterable[tuple[float, float]]) -> float | None:
    """Return pairwise AUC, with tied scores receiving half credit."""
    rows = list(points)
    positives = [score for score, reward in rows if reward >= 1.0]
    negatives = [score for score, reward in rows if reward < 1.0]
    if not positives or not negatives:
        return None
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / (len(positives) * len(negatives))


def mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def result_auc(result: dict[str, Any]) -> float | None:
    rewards = {
        row["particle_id"]: float(row["reward"])
        for row in result["grader"]["particles"]
    }
    return binary_auc(
        (
            float(particle["semantic_score"]),
            rewards[particle["particle_id"]],
        )
        for particle in result["final_particles"]
    )


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    generator = [row["generator_tool_replay_cost"] for row in results]
    verifier = [row["semantic_verifier_cost"] for row in results]
    aucs = [result_auc(row) for row in results]
    selected = [
        float(row["selection"]["reward"]) >= 1.0 for row in results
    ]
    any_success = [bool(row["grader"]["any_success"]) for row in results]
    population = [
        float(row["grader"]["success_rate"]) for row in results
    ]
    return {
        "runs": len(results),
        "selected_successes": sum(selected),
        "selected_accuracy": mean(selected),
        "any_success_rate": mean(any_success),
        "mean_population_success_rate": mean(population),
        "min_population_success_rate": min(population),
        "max_population_success_rate": max(population),
        "mean_terminal_semantic_auc": mean(aucs),
        "terminal_auc_defined_runs": sum(value is not None for value in aucs),
        "mean_resampling_events": mean(
            row["resampling_count"] for row in results
        ),
        "mean_wall_time_s": mean(row["wall_time_s"] for row in results),
        "mean_generator_model_calls": mean(
            row["model_calls"] for row in generator
        ),
        "mean_generator_prompt_tokens": mean(
            row["prompt_tokens"] for row in generator
        ),
        "mean_generator_completion_tokens": mean(
            row["completion_tokens"] for row in generator
        ),
        "mean_environment_replays": mean(
            row["environment_replays"] for row in generator
        ),
        "mean_replay_wall_time_s": mean(
            row["environment_replay_wall_time_s"] for row in generator
        ),
        "mean_physical_verifier_calls": mean(
            row["physical_verifier_calls"] for row in verifier
        ),
        "mean_logical_verifier_calls": mean(
            row["logical_verifier_calls"] for row in verifier
        ),
        "mean_deduplicated_verifier_calls": mean(
            row["deduplicated_verifier_calls"] for row in verifier
        ),
        "mean_verifier_prompt_tokens": mean(
            row["prompt_tokens"] for row in verifier
        ),
        "mean_verifier_completion_tokens": mean(
            row["completion_tokens"] for row in verifier
        ),
        "mean_verifier_inference_wall_time_s": mean(
            row["inference_wall_time_s"] for row in verifier
        ),
    }


def analyze_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError(f"report is not complete/pass: {path}")
    axis = report["sweep"]["axis"]
    field = AXIS_FIELDS[axis]
    groups: dict[float | int, list[dict[str, Any]]] = defaultdict(list)
    for result in report["results"]:
        groups[result["configuration"][field]].append(result)
    return {
        "source": str(path.resolve()),
        "experiment": report["experiment"],
        "task_id": report["source"]["task_id"],
        "axis": axis,
        "axis_field": field,
        "repetitions": report["sweep"]["repetitions"],
        "generator_model": report["source"]["generator_model"],
        "verifier_model": report["verifier"]["model"],
        "external_server_startup_excluded": (
            not report["allocation_accounting"][
                "includes_external_generator_engine_startup"
            ]
            and not report["allocation_accounting"][
                "includes_external_verifier_engine_startup"
            ]
        ),
        "runner_wall_time_s": report["allocation_accounting"][
            "runner_wall_time_s"
        ],
        "runner_allocated_accelerator_seconds": report[
            "allocation_accounting"
        ]["runner_allocated_accelerator_seconds"],
        "groups": [
            {
                "axis_value": value,
                **summarize_group(groups[value]),
            }
            for value in sorted(groups)
        ],
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Terminal semantic-SMC live pilot",
        "",
        (
            "Reward labels are isolated from semantic scoring. External model "
            "server startup is excluded; runner allocated accelerator-seconds "
            "charge every generator/verifier accelerator during each sweep."
        ),
        "",
    ]
    for report in summary["reports"]:
        lines.extend(
            [
                f"## {report['axis']} sweep",
                "",
                (
                    f"Task: `{report['task_id']}`; generator: "
                    f"`{report['generator_model']}`; verifier: "
                    f"`{report['verifier_model']}`; "
                    f"allocated accelerator-seconds: "
                    f"{fmt(report['runner_allocated_accelerator_seconds'], 1)}."
                ),
                "",
                (
                    "| value | runs | selected | population | terminal AUC | "
                    "resamples | wall s | gen calls | verifier physical/logical |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in report["groups"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(group["axis_value"]),
                        str(group["runs"]),
                        fmt(group["selected_accuracy"]),
                        fmt(group["mean_population_success_rate"]),
                        fmt(group["mean_terminal_semantic_auc"]),
                        fmt(group["mean_resampling_events"]),
                        fmt(group["mean_wall_time_s"], 1),
                        fmt(group["mean_generator_model_calls"], 1),
                        (
                            f"{fmt(group['mean_physical_verifier_calls'], 1)}/"
                            f"{fmt(group['mean_logical_verifier_calls'], 1)}"
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "terminal-semantic-smc-live-pilot-analysis-v1",
        "scientific_scope": (
            "single-task engineering pilot with repeated stochastic runs; "
            "not a benchmark-level accuracy estimate"
        ),
        "reports": [analyze_report(path) for path in args.reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        markdown(summary) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

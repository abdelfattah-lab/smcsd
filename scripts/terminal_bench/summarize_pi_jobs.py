#!/usr/bin/env python3
"""Summarize agent quality and serving metrics from matrix-run Harbor jobs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([-+0-9.eE]+)$")


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def elapsed_s(interval: dict[str, Any] | None) -> float | None:
    if (
        not interval
        or not interval.get("started_at")
        or not interval.get("finished_at")
    ):
        return None
    return (
        parse_time(interval["finished_at"]) - parse_time(interval["started_at"])
    ).total_seconds()


def parse_prometheus(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = PROM_LINE.match(line)
        if match is None:
            continue
        name, raw_value = match.groups()
        value = float(raw_value)
        if math.isfinite(value):
            totals[name] = totals.get(name, 0.0) + value
    return totals


def prom_delta(job_dir: Path) -> dict[str, float]:
    before = parse_prometheus(job_dir / "prometheus_before.prom")
    after = parse_prometheus(job_dir / "prometheus_after.prom")
    return {name: after.get(name, 0.0) - value for name, value in before.items()} | {
        name: value for name, value in after.items() if name not in before
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def average_histogram(delta: dict[str, float], base: str) -> float | None:
    return ratio(delta.get(f"{base}_sum", 0.0), delta.get(f"{base}_count", 0.0))


def provider_request_count(trial_dir: Path) -> int:
    count = 0
    for capture in trial_dir.glob("agent/pi-capture/requests-*.jsonl"):
        with capture.open() as stream:
            count += sum(1 for line in stream if line.strip())
    return count


def resolved_resample_threshold(metadata: dict[str, Any]) -> float | None:
    spec = metadata["spec"]
    if spec["method"] != "smcsd":
        return None
    return spec.get("resample_threshold", 0.5)


def resolved_mem_fraction_static(metadata: dict[str, Any]) -> float:
    return (metadata.get("server_config") or {}).get("mem_fraction_static", 0.4)


def trial_rows(job_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        trial_dir = result_path.parent
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        agent_result = result.get("agent_result") or {}
        rows.append(
            {
                "job_name": metadata["job_name"],
                "run_tag": metadata.get("run_tag", "legacy"),
                "method": metadata["spec"]["method"],
                "particles": metadata["spec"].get("particles"),
                "gamma": metadata["spec"].get("gamma"),
                "resample_threshold": resolved_resample_threshold(metadata),
                "mem_fraction_static": resolved_mem_fraction_static(metadata),
                "seed": metadata["spec"]["seed"],
                "task": result.get("task_name"),
                "trial_name": result.get("trial_name"),
                "reward": reward,
                "errored": result.get("exception_info") is not None,
                "agent_wall_time_s": elapsed_s(result.get("agent_execution")),
                "trial_wall_time_s": (
                    parse_time(result["finished_at"]) - parse_time(result["started_at"])
                ).total_seconds(),
                "provider_requests": provider_request_count(trial_dir),
                "agent_input_tokens": agent_result.get("n_input_tokens", 0) or 0,
                "agent_output_tokens": agent_result.get("n_output_tokens", 0) or 0,
            }
        )
    return rows


def aggregate_job(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((job_dir / "experiment.json").read_text())
    trials = trial_rows(job_dir, metadata)
    delta = prom_delta(job_dir)
    expected_trials = len(metadata.get("tasks") or trials)
    non_error_trials = [row for row in trials if not row["errored"]]
    explicit_errors = len(trials) - len(non_error_trials)
    missing_trials = max(expected_trials - len(trials), 0)

    # Timeouts and non-zero agent exits are benchmark outcomes, not absent data.
    # Count them as zero reward and include the resources they consumed. Dropping
    # them makes both accuracy and correct-tasks/GPU-hour look artificially high.
    agent_seconds = sum(row["agent_wall_time_s"] or 0.0 for row in trials)
    reward_sum = sum(float(row["reward"] or 0.0) for row in trials)
    provider_requests = sum(row["provider_requests"] for row in trials)
    input_tokens = sum(row["agent_input_tokens"] for row in trials)
    output_tokens = sum(row["agent_output_tokens"] for row in trials)
    request_count = delta.get("sglang:num_requests_total", 0.0)
    prompt_tokens = delta.get("sglang:prompt_tokens_total", 0.0)
    generation_tokens = delta.get("sglang:generation_tokens_total", 0.0)
    model_seconds = delta.get("sglang:e2e_request_latency_seconds_sum", 0.0)
    measurement_seconds = (
        parse_time(metadata["measurement_finished_at"])
        - parse_time(metadata["measurement_started_at"])
    ).total_seconds()

    row = {
        "job_name": metadata["job_name"],
        "run_tag": metadata.get("run_tag", "legacy"),
        "method": metadata["spec"]["method"],
        "particles": metadata["spec"].get("particles"),
        "gamma": metadata["spec"].get("gamma"),
        "resample_threshold": resolved_resample_threshold(metadata),
        "mem_fraction_static": resolved_mem_fraction_static(metadata),
        "seed": metadata["spec"]["seed"],
        "n_trials": len(trials),
        "n_expected_trials": expected_trials,
        "n_completed": len(trials),
        "n_non_error": len(non_error_trials),
        "n_errors": explicit_errors + missing_trials,
        "n_missing_trials": missing_trials,
        "reward_sum": reward_sum,
        "pass_rate": ratio(reward_sum, expected_trials),
        "agent_wall_time_s": agent_seconds,
        "tasks_per_agent_gpu_hour": ratio(len(trials) * 3600.0, agent_seconds),
        "correct_tasks_per_agent_gpu_hour": ratio(reward_sum * 3600.0, agent_seconds),
        "provider_requests": provider_requests,
        "provider_requests_per_agent_s": ratio(provider_requests, agent_seconds),
        "agent_input_tokens": input_tokens,
        "agent_output_tokens": output_tokens,
        "server_measurement_wall_time_s": measurement_seconds,
        "server_requests": request_count,
        "server_prompt_tokens": prompt_tokens,
        "server_generation_tokens": generation_tokens,
        "server_time_per_query_s": average_histogram(
            delta, "sglang:e2e_request_latency_seconds"
        ),
        "server_ttft_s": average_histogram(delta, "sglang:time_to_first_token_seconds"),
        "server_inter_token_latency_s": average_histogram(
            delta, "sglang:inter_token_latency_seconds"
        ),
        "server_requests_per_s": ratio(request_count, model_seconds),
        "server_generation_tokens_per_s": ratio(generation_tokens, model_seconds),
    }
    return row, trials


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "setting",
        "seed",
        "pass",
        "agent_s",
        "correct/GPUh",
        "queries/s",
        "query_s",
        "TTFT_s",
        "gen_tok/s",
    ]
    table: list[list[str]] = []
    for row in rows:
        setting = row["method"]
        if setting == "smcsd":
            setting = f"N{row['particles']}/g{row['gamma']}"
            if row["resample_threshold"] != 0.5:
                setting += f"/r{row['resample_threshold']:g}"
        table.append(
            [
                setting,
                compact(row["seed"], 0),
                compact(row["pass_rate"]),
                compact(row["agent_wall_time_s"], 1),
                compact(row["correct_tasks_per_agent_gpu_hour"], 1),
                compact(row["server_requests_per_s"]),
                compact(row["server_time_per_query_s"]),
                compact(row["server_ttft_s"]),
                compact(row["server_generation_tokens_per_s"], 1),
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-tag")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--trials-csv-output", type=Path)
    args = parser.parse_args()

    jobs_dir = args.jobs_dir.expanduser().resolve()
    aggregate_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for experiment_path in sorted(jobs_dir.glob("*/experiment.json")):
        metadata = json.loads(experiment_path.read_text())
        if metadata.get("status") != "complete":
            continue
        if args.experiment_id and metadata.get("experiment_id") != args.experiment_id:
            continue
        if args.run_tag and metadata.get("run_tag") != args.run_tag:
            continue
        aggregate, job_trials = aggregate_job(experiment_path.parent)
        aggregate_rows.append(aggregate)
        trials.extend(job_trials)

    if not aggregate_rows:
        raise RuntimeError(f"no complete matrix jobs found under {jobs_dir}")
    print_table(aggregate_rows)
    if args.json_output:
        args.json_output.write_text(json.dumps(aggregate_rows, indent=2) + "\n")
    if args.csv_output:
        write_csv(args.csv_output, aggregate_rows)
    if args.trials_csv_output:
        write_csv(args.trials_csv_output, trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

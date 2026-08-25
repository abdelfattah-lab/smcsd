"""Analyze the frozen OlympiadBench semantic-SMC holdout experiment.

The independent candidate pool is shared by AR@1, self-consistency, terminal
semantic Best-of-N, and the terminal multi-model ensemble.  Semantic SMC is a
separate online branched run initialized from the same exact 2048-token roots.
The exact LLM-as-a-Verifier baseline is deliberately outside this analysis.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence


PRIMARY_METHODS = (
    "ar_at_1",
    "self_consistency_at_8",
    "terminal_semantic_bon_at_8",
    "terminal_multimodel_ensemble_at_8",
    "semantic_smc_at_8",
)
DIAGNOSTICS = ("independent_oracle_pass_at_8", "smc_oracle_pass_at_8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)) + 1)]
    return [low, high]


def group_trajectories(
    trajectories: Sequence[dict], *, expected_problems: int, expected_n: int
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        groups[str(row["problem_id"])].append(row)
    if len(groups) != expected_problems:
        raise ValueError(
            f"Expected {expected_problems} problems, observed {len(groups)}."
        )
    for problem_id, rows in groups.items():
        rows.sort(key=lambda row: int(row["sample_id"]))
        sample_ids = [int(row["sample_id"]) for row in rows]
        if sample_ids != list(range(expected_n)):
            raise ValueError(
                f"{problem_id} must have sample IDs 0..{expected_n - 1}; "
                f"observed {sample_ids}."
            )
        if len({str(row["gold_answer"]) for row in rows}) != 1:
            raise ValueError(f"{problem_id} has inconsistent gold answers.")
    return dict(groups)


def validate_frozen_split(
    trajectories: Sequence[dict], config: dict
) -> tuple[int, int]:
    dataset = config["dataset"]
    expected_start, expected_end = map(int, dataset["selection_positions"])
    positions = sorted({int(row["selection_position"]) for row in trajectories})
    expected_positions = list(range(expected_start, expected_end + 1))
    if positions != expected_positions:
        raise ValueError(
            "Trajectory selection positions do not match the frozen holdout: "
            f"expected {expected_start}..{expected_end}, observed "
            f"{positions[0] if positions else None}..{positions[-1] if positions else None}."
        )
    seeds = {int(row["selection_seed"]) for row in trajectories}
    if seeds != {int(dataset["selection_seed"])}:
        raise ValueError(f"Unexpected selection seeds: {sorted(seeds)}.")
    dev_start, dev_end = map(int, dataset["excluded_development_positions"])
    if set(positions) & set(range(dev_start, dev_end + 1)):
        raise ValueError("Frozen holdout overlaps excluded development positions.")
    return expected_start, expected_end


def score_map(
    scores: Sequence[dict], trajectory_groups: dict[str, list[dict]], *, label: str
) -> dict[tuple[str, int], dict]:
    expected = {
        (problem_id, int(row["sample_id"]))
        for problem_id, rows in trajectory_groups.items()
        for row in rows
    }
    mapping: dict[tuple[str, int], dict] = {}
    for row in scores:
        key = (str(row["problem_id"]), int(row["sample_id"]))
        if key in mapping:
            raise ValueError(f"Duplicate {label} terminal score for {key}.")
        if not bool(row.get("terminal", True)):
            raise ValueError(f"{label} contains a non-terminal score for {key}.")
        mapping[key] = row
    if set(mapping) != expected:
        raise ValueError(
            f"{label} score keys differ from trajectories: "
            f"missing={len(expected - set(mapping))} extra={len(set(mapping) - expected)}."
        )
    for problem_id, rows in trajectory_groups.items():
        for row in rows:
            key = (problem_id, int(row["sample_id"]))
            if bool(mapping[key]["correct"]) != bool(row["correct"]):
                raise ValueError(f"{label} correctness mismatch for {key}.")
    return mapping


def smc_map(
    rows: Sequence[dict], trajectory_groups: dict[str, list[dict]]
) -> dict[str, dict]:
    mapping = {str(row["problem_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Duplicate semantic-SMC problem rows.")
    expected = set(trajectory_groups)
    if set(mapping) != expected:
        raise ValueError(
            "Semantic-SMC problem IDs differ from the independent pool: "
            f"missing={len(expected - set(mapping))} extra={len(set(mapping) - expected)}."
        )
    return mapping


def analyze_outcomes(
    trajectory_groups: dict[str, list[dict]],
    verifier_a: dict[tuple[str, int], dict],
    verifier_b: dict[tuple[str, int], dict],
    smc: dict[str, dict],
) -> list[dict]:
    outcomes = []
    for problem_id, rows in sorted(trajectory_groups.items()):
        gold = rows[0]["gold_answer"]
        votes = [row.get("extracted_answer") for row in rows]
        valid_votes = [answer for answer in votes if answer is not None]
        majority_answer = (
            Counter(valid_votes).most_common(1)[0][0] if valid_votes else None
        )
        selected_a = min(
            rows,
            key=lambda row: (
                -float(verifier_a[(problem_id, int(row["sample_id"]))]["score"]),
                int(row["sample_id"]),
            ),
        )
        selected_ensemble = min(
            rows,
            key=lambda row: (
                -statistics.fmean(
                    (
                        float(verifier_a[(problem_id, int(row["sample_id"]))]["score"]),
                        float(verifier_b[(problem_id, int(row["sample_id"]))]["score"]),
                    )
                ),
                int(row["sample_id"]),
            ),
        )
        smc_row = smc[problem_id]
        outcomes.append(
            {
                "problem_id": problem_id,
                "dataset_index": int(rows[0]["dataset_index"]),
                "selection_position": int(rows[0]["selection_position"]),
                "subfield": rows[0]["subfield"],
                "ar_at_1": bool(rows[0]["correct"]),
                "self_consistency_at_8": majority_answer == gold,
                "terminal_semantic_bon_at_8": bool(selected_a["correct"]),
                "terminal_multimodel_ensemble_at_8": bool(
                    selected_ensemble["correct"]
                ),
                "semantic_smc_at_8": bool(smc_row["selected_correct"]),
                "independent_oracle_pass_at_8": any(
                    bool(row["correct"]) for row in rows
                ),
                "smc_oracle_pass_at_8": bool(smc_row["oracle_correct"]),
                "self_consistency_answer": majority_answer,
                "terminal_semantic_selected_sample": int(selected_a["sample_id"]),
                "terminal_ensemble_selected_sample": int(
                    selected_ensemble["sample_id"]
                ),
                "smc_selected_slot": int(smc_row["selected_slot"]),
            }
        )
    return outcomes


def bootstrap_metrics(
    outcomes: Sequence[dict], methods: Sequence[str], *, samples: int, seed: int
) -> tuple[dict, dict]:
    rng = random.Random(seed)
    accuracy_draws = {method: [] for method in methods}
    difference_draws = {
        method: {other: [] for other in methods if other != method}
        for method in methods
    }
    for _ in range(samples):
        selected = [rng.choice(outcomes) for _ in outcomes]
        accuracies = {
            method: statistics.fmean(float(row[method]) for row in selected)
            for method in methods
        }
        for method in methods:
            accuracy_draws[method].append(accuracies[method])
            for other in methods:
                if method != other:
                    difference_draws[method][other].append(
                        accuracies[method] - accuracies[other]
                    )
    metrics = {}
    paired = {}
    for method in methods:
        correct = sum(bool(row[method]) for row in outcomes)
        metrics[method] = {
            "correct": correct,
            "total": len(outcomes),
            "accuracy": correct / len(outcomes),
            "bootstrap_95_ci": percentile_interval(accuracy_draws[method]),
        }
        paired[method] = {}
        for other in methods:
            if method == other:
                continue
            differences = [
                int(bool(row[method])) - int(bool(row[other])) for row in outcomes
            ]
            paired[method][other] = {
                "accuracy_difference": statistics.fmean(differences),
                "bootstrap_95_ci": percentile_interval(
                    difference_draws[method][other]
                ),
                "wins": sum(value > 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
    return metrics, paired


def terminal_verifier_cost(summary: dict, gpus: int) -> dict:
    cost = summary["cost"]
    inference = float(cost["inference_wall_time_s"])
    return {
        "calls": int(cost["prefixes_scored"]),
        "prompt_tokens": int(cost["verifier_prompt_tokens"]),
        "completion_tokens": int(cost["verifier_completion_tokens"]),
        "inference_wall_time_s": inference,
        "active_gpu_seconds": inference * gpus,
        "gpus": gpus,
    }


def cost_table(
    trajectories: Sequence[dict],
    generation_summary: dict,
    verifier_a_summary: dict,
    verifier_b_summary: dict,
    smc_summary: dict,
    *,
    verifier_a_gpus: int,
    verifier_b_gpus: int,
) -> dict:
    generation = generation_summary["cost"]
    all_generator_tokens = int(generation["total_output_tokens"])
    all_generator_seconds = float(generation["allocated_generator_gpu_seconds"])
    ar_tokens = sum(
        int(row["completion_tokens"])
        for row in trajectories
        if int(row["sample_id"]) == 0
    )
    ar_seconds = all_generator_seconds * ar_tokens / all_generator_tokens
    verifier_a = terminal_verifier_cost(verifier_a_summary, verifier_a_gpus)
    verifier_b = terminal_verifier_cost(verifier_b_summary, verifier_b_gpus)
    common_pool = {
        "generator_tokens": all_generator_tokens,
        "generator_active_gpu_seconds": all_generator_seconds,
        "generator_inference_wall_time_s": float(generation["inference_wall_time_s"]),
    }
    costs = {
        "ar_at_1": {
            "generator_tokens": ar_tokens,
            "verifier_calls": 0,
            "verifier_prompt_tokens": 0,
            "active_component_gpu_seconds": ar_seconds,
            "active_gpu_seconds_estimated_by_generator_token_proration": True,
        },
        "self_consistency_at_8": {
            **common_pool,
            "verifier_calls": 0,
            "verifier_prompt_tokens": 0,
            "active_component_gpu_seconds": all_generator_seconds,
        },
        "terminal_semantic_bon_at_8": {
            **common_pool,
            "verifier_calls": verifier_a["calls"],
            "verifier_prompt_tokens": verifier_a["prompt_tokens"],
            "verifier_completion_tokens": verifier_a["completion_tokens"],
            "verifier_active_gpu_seconds": verifier_a["active_gpu_seconds"],
            "active_component_gpu_seconds": all_generator_seconds
            + verifier_a["active_gpu_seconds"],
            "serial_pipeline_inference_wall_time_s": float(
                generation["inference_wall_time_s"]
            )
            + verifier_a["inference_wall_time_s"],
        },
        "terminal_multimodel_ensemble_at_8": {
            **common_pool,
            "verifier_calls": verifier_a["calls"] + verifier_b["calls"],
            "verifier_prompt_tokens": verifier_a["prompt_tokens"]
            + verifier_b["prompt_tokens"],
            "verifier_completion_tokens": verifier_a["completion_tokens"]
            + verifier_b["completion_tokens"],
            "verifier_active_gpu_seconds": verifier_a["active_gpu_seconds"]
            + verifier_b["active_gpu_seconds"],
            "active_component_gpu_seconds": all_generator_seconds
            + verifier_a["active_gpu_seconds"]
            + verifier_b["active_gpu_seconds"],
            "parallel_verifier_pipeline_inference_wall_time_s": float(
                generation["inference_wall_time_s"]
            )
            + max(
                verifier_a["inference_wall_time_s"],
                verifier_b["inference_wall_time_s"],
            ),
        },
        "semantic_smc_at_8": {
            "generator_tokens": int(smc_summary["cost"]["total_generator_tokens"]),
            "verifier_calls": int(smc_summary["cost"]["verifier_calls"]),
            "verifier_prompt_tokens": int(
                smc_summary["cost"]["verifier_prompt_tokens"]
            ),
            "verifier_completion_tokens": int(
                smc_summary["cost"]["verifier_completion_tokens"]
            ),
            "active_component_gpu_seconds": float(
                smc_summary["cost"]["active_component_gpu_seconds"]
            ),
            "static_reserved_gpu_seconds": float(
                smc_summary["cost"]["static_reserved_gpu_seconds"]
            ),
            "algorithm_wall_time_s": float(
                smc_summary["cost"]["algorithm_wall_time_s"]
            ),
        },
    }
    n_problems = int(generation_summary["dataset"]["n_problems"])
    for method_cost in costs.values():
        method_cost["generator_tokens_per_problem"] = (
            method_cost["generator_tokens"] / n_problems
        )
        method_cost["verifier_calls_per_problem"] = (
            method_cost["verifier_calls"] / n_problems
        )
        method_cost["verifier_prompt_tokens_per_problem"] = (
            method_cost["verifier_prompt_tokens"] / n_problems
        )
        method_cost["active_component_gpu_seconds_per_problem"] = (
            method_cost["active_component_gpu_seconds"] / n_problems
        )
    return costs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--generation-summary", required=True)
    parser.add_argument("--verifier-a-scores", required=True)
    parser.add_argument("--verifier-a-summary", required=True)
    parser.add_argument("--verifier-a-gpus", type=int, required=True)
    parser.add_argument("--verifier-b-scores", required=True)
    parser.add_argument("--verifier-b-summary", required=True)
    parser.add_argument("--verifier-b-gpus", type=int, required=True)
    parser.add_argument("--smc-problems", required=True)
    parser.add_argument("--smc-summary", required=True)
    parser.add_argument("--save-outcomes", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_json(args.config)
    trajectories = load_jsonl(args.trajectories)
    expected_problems = int(config["dataset"]["problems"])
    expected_n = int(config["generator"]["samples_or_particles_per_problem"])
    if expected_n != 8:
        raise ValueError("This frozen v1 analysis requires N=8.")
    validate_frozen_split(trajectories, config)
    trajectory_groups = group_trajectories(
        trajectories,
        expected_problems=expected_problems,
        expected_n=expected_n,
    )
    verifier_a_rows = load_jsonl(args.verifier_a_scores)
    verifier_b_rows = load_jsonl(args.verifier_b_scores)
    verifier_a = score_map(
        verifier_a_rows, trajectory_groups, label="verifier A"
    )
    verifier_b = score_map(
        verifier_b_rows, trajectory_groups, label="verifier B"
    )
    scorer_a = {str(row["scorer_model"]) for row in verifier_a_rows}
    scorer_b = {str(row["scorer_model"]) for row in verifier_b_rows}
    if len(scorer_a) != 1 or len(scorer_b) != 1 or scorer_a == scorer_b:
        raise ValueError(
            f"Expected two distinct single-model score files; got {scorer_a} and {scorer_b}."
        )
    smc = smc_map(load_jsonl(args.smc_problems), trajectory_groups)
    outcomes = analyze_outcomes(trajectory_groups, verifier_a, verifier_b, smc)

    inference = config["inference"]
    methods = (*PRIMARY_METHODS, *DIAGNOSTICS)
    metrics, paired = bootstrap_metrics(
        outcomes,
        methods,
        samples=int(inference["bootstrap_samples"]),
        seed=int(inference["bootstrap_seed"]),
    )
    generation_summary = load_json(args.generation_summary)
    smc_summary = load_json(args.smc_summary)
    if int(generation_summary["dataset"]["n_problems"]) != expected_problems:
        raise ValueError("Generation summary problem count does not match config.")
    if int(smc_summary["dataset"]["n_problems"]) != expected_problems:
        raise ValueError("SMC summary problem count does not match config.")
    costs = cost_table(
        trajectories,
        generation_summary,
        load_json(args.verifier_a_summary),
        load_json(args.verifier_b_summary),
        smc_summary,
        verifier_a_gpus=args.verifier_a_gpus,
        verifier_b_gpus=args.verifier_b_gpus,
    )
    smc_vs_terminal = paired["semantic_smc_at_8"]["terminal_semantic_bon_at_8"]
    smc_vs_ensemble = paired["semantic_smc_at_8"][
        "terminal_multimodel_ensemble_at_8"
    ]
    summary = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "evaluation_status": config["evaluation_status"],
        "research_question": config["research_question"],
        "dataset": {
            **config["dataset"],
            "dataset_indices": sorted(
                {int(row["dataset_index"]) for row in trajectories}
            ),
        },
        "protocol": {
            "generator": config["generator"],
            "methods": config["methods"],
            "verifier_a": next(iter(scorer_a)),
            "verifier_b": next(iter(scorer_b)),
            "ensemble_rule": "unweighted arithmetic mean of the two expected validity scores",
            "uses_generator_likelihood": False,
            "llm_as_a_verifier_included": False,
        },
        "metrics": metrics,
        "paired_differences": paired,
        "cost": costs,
        "smc_allocation": smc_summary["allocation"],
        "primary_answer": {
            "smc_minus_terminal_semantic_bon": smc_vs_terminal,
            "smc_minus_terminal_multimodel_ensemble": smc_vs_ensemble,
            "smc_improves_over_both_terminal_selectors": (
                smc_vs_terminal["accuracy_difference"] > 0
                and smc_vs_ensemble["accuracy_difference"] > 0
            ),
            "uncertainty_note": "Percentile intervals use problem-level paired bootstrap resampling.",
        },
    }
    write_jsonl(args.save_outcomes, outcomes)
    write_json(args.output, summary)
    print("\nFrozen OlympiadBench semantic-SMC holdout")
    print("method                                      correct   accuracy       95% CI")
    for method in methods:
        row = metrics[method]
        low, high = row["bootstrap_95_ci"]
        print(
            f"{method:42s} {row['correct']:3d}/{row['total']:<3d} "
            f"{row['accuracy']:8.1%}  [{low:6.1%}, {high:6.1%}]"
        )
    print(f"\nwrote {args.output}")
    return summary


if __name__ == "__main__":
    main()

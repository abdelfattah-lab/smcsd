"""Compare likelihood-SMC selection with terminal semantic combinations.

This consumes a completed optimized SMC particle pool and pointwise terminal
semantic scores.  It does not replay or regenerate trajectories.  For each
predeclared beta it evaluates both the maximum combined particle weight and an
answer-cluster vote using ``exp(log_w + beta * semantic_score)``.  Bootstrap
intervals resample whole problems.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_vote(answers: Sequence[str | None]) -> str | None:
    valid = [answer for answer in answers if answer is not None]
    return Counter(valid).most_common(1)[0][0] if valid else None


def weighted_vote(
    answers: Sequence[str | None], log_weights: Sequence[float]
) -> str | None:
    finite = [float(weight) for weight in log_weights if math.isfinite(float(weight))]
    if not finite:
        return stable_vote(answers)
    offset = max(finite)
    totals: dict[str, float] = {}
    first_position: dict[str, int] = {}
    for position, (answer, weight) in enumerate(zip(answers, log_weights)):
        if answer is None or not math.isfinite(float(weight)):
            continue
        totals[answer] = totals.get(answer, 0.0) + math.exp(float(weight) - offset)
        first_position.setdefault(answer, position)
    if not totals:
        return None
    return min(totals, key=lambda answer: (-totals[answer], first_position[answer]))


def bootstrap_accuracy(
    values: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict:
    delta = left - right
    return {
        "difference": float(delta.mean()),
        "bootstrap_95_ci": bootstrap_accuracy(delta, samples=samples, seed=seed),
        "wins": int(np.sum(delta > 0)),
        "losses": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--likelihood-particles", required=True)
    parser.add_argument("--likelihood-problems", required=True)
    parser.add_argument("--likelihood-summary", required=True)
    parser.add_argument("--semantic-scores", required=True)
    parser.add_argument("--semantic-summary", required=True)
    parser.add_argument("--semantic-gpus", type=int, required=True)
    parser.add_argument("--self-consistency-trajectories", required=True)
    parser.add_argument("--self-consistency-summary", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--summary-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_json(args.config)
    particles = load_jsonl(args.likelihood_particles)
    likelihood_problem_rows = {
        str(row["problem_id"]): row for row in load_jsonl(args.likelihood_problems)
    }
    likelihood_summary = load_json(args.likelihood_summary)
    semantic_summary = load_json(args.semantic_summary)
    sc_summary = load_json(args.self_consistency_summary)
    sc_rows = load_jsonl(args.self_consistency_trajectories)

    semantic_rows = load_jsonl(args.semantic_scores)
    semantic = {}
    for row in semantic_rows:
        key = (str(row["problem_id"]), int(row["sample_id"]))
        if key in semantic:
            raise ValueError(f"Duplicate semantic terminal score for {key}.")
        semantic[key] = float(row["score"])

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in particles:
        by_problem[str(row["problem_id"])].append(row)
    for rows in by_problem.values():
        rows.sort(key=lambda row: int(row["sample_id"]))
    problem_ids = sorted(by_problem)
    expected_count = int(config["likelihood_smc"]["particles"])
    if not problem_ids or any(len(by_problem[key]) != expected_count for key in problem_ids):
        raise ValueError("Every problem must contain the configured particle count.")
    expected_scores = {
        (problem_id, sample_id)
        for problem_id in problem_ids
        for sample_id in range(expected_count)
    }
    if set(semantic) != expected_scores:
        raise ValueError(
            "Terminal semantic scores do not align with likelihood particles: "
            f"missing={len(expected_scores - set(semantic))} "
            f"extra={len(set(semantic) - expected_scores)}"
        )
    if set(likelihood_problem_rows) != set(problem_ids):
        raise ValueError("Likelihood problem rows do not align with particles.")

    sc_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in sc_rows:
        sc_by_problem[str(row["problem_id"])].append(row)
    if set(sc_by_problem) != set(problem_ids):
        raise ValueError("Self-consistency problems do not align with likelihood SMC.")

    outcomes: dict[str, list[float]] = defaultdict(list)
    selected_slots: dict[str, list[int]] = defaultdict(list)
    betas = [float(value) for value in config["semantic_terminal_factor"]["semantic_betas"]]
    for problem_id in problem_ids:
        rows = by_problem[problem_id]
        gold = rows[0]["gold_answer"]
        answers = [row["extracted_answer"] for row in rows]
        likelihood = [float(row["smc_final_log_weight"]) for row in rows]
        scores = [semantic[(problem_id, sample_id)] for sample_id in range(expected_count)]

        sc_answers = [row["extracted_answer"] for row in sc_by_problem[problem_id]]
        outcomes["self_consistency_at_8"].append(float(stable_vote(sc_answers) == gold))
        p_row = likelihood_problem_rows[problem_id]
        for name, field in (
            ("likelihood_posterior_sample", "posterior_sample_correct"),
            ("likelihood_particle_majority", "particle_majority_correct"),
            (
                "likelihood_weighted_answer_majority",
                "likelihood_weighted_majority_correct",
            ),
            ("likelihood_max_weight", "max_weight_correct"),
            ("likelihood_pool_oracle", "oracle_correct"),
        ):
            outcomes[name].append(float(bool(p_row[field])))

        semantic_slot = min(range(expected_count), key=lambda i: (-scores[i], i))
        selected_slots["terminal_semantic_argmax"].append(semantic_slot)
        outcomes["terminal_semantic_argmax"].append(float(bool(rows[semantic_slot]["correct"])))

        for beta in betas:
            combined = [lw + beta * score for lw, score in zip(likelihood, scores)]
            suffix = f"beta_{beta:g}"
            particle_slot = min(
                range(expected_count), key=lambda i: (-combined[i], i)
            )
            selected_slots[f"combined_particle_argmax_{suffix}"].append(particle_slot)
            outcomes[f"combined_particle_argmax_{suffix}"].append(
                float(bool(rows[particle_slot]["correct"]))
            )
            answer = weighted_vote(answers, combined)
            outcomes[f"combined_weighted_answer_{suffix}"].append(float(answer == gold))

    arrays = {name: np.asarray(values, dtype=float) for name, values in outcomes.items()}
    correct_particle_counts = [
        sum(bool(row["correct"]) for row in by_problem[problem_id])
        for problem_id in problem_ids
    ]
    final_ess = []
    final_logweight_ranges = []
    for problem_id in problem_ids:
        log_weights = [
            float(row["smc_final_log_weight"]) for row in by_problem[problem_id]
        ]
        maximum = max(log_weights)
        weights = [math.exp(value - maximum) for value in log_weights]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        final_ess.append(1.0 / sum(value * value for value in probabilities))
        final_logweight_ranges.append(max(log_weights) - min(log_weights))
    likelihood_cost = float(likelihood_summary["cost"]["allocated_gpu_seconds_per_problem"])
    semantic_prompt_tokens = float(
        semantic_summary["cost"].get(
            "verifier_prompt_tokens", semantic_summary["cost"].get("prompt_tokens", 0)
        )
    )
    if args.semantic_gpus <= 0:
        raise ValueError("--semantic-gpus must be positive.")
    semantic_gpus = args.semantic_gpus
    semantic_total_gpu_seconds = (
        float(semantic_summary["cost"]["inference_wall_time_s"]) * semantic_gpus
    )
    semantic_cost = semantic_total_gpu_seconds / len(problem_ids)
    sc_cost = float(sc_summary["cost"]["allocated_generator_gpu_seconds"]) / len(problem_ids)

    method_results = {}
    for index, (name, values) in enumerate(arrays.items()):
        uses_semantic = name.startswith("terminal_semantic") or name.startswith("combined_")
        if name == "self_consistency_at_8":
            gpu_seconds = sc_cost
        else:
            gpu_seconds = likelihood_cost + (semantic_cost if uses_semantic else 0.0)
        method_results[name] = {
            "accuracy": float(values.mean()),
            "bootstrap_95_ci": bootstrap_accuracy(
                values, samples=args.bootstrap_samples, seed=args.seed + index
            ),
            "active_gpu_seconds_per_problem": gpu_seconds,
            "uses_semantic_verifier": uses_semantic,
        }

    candidate_names = [
        name
        for name in arrays
        if name.startswith("combined_") and not name.endswith("beta_0")
    ]
    best_hybrid = min(
        candidate_names,
        key=lambda name: (-method_results[name]["accuracy"], name),
    )
    strongest_likelihood = min(
        (
            "likelihood_posterior_sample",
            "likelihood_particle_majority",
            "likelihood_weighted_answer_majority",
            "likelihood_max_weight",
        ),
        key=lambda name: (-method_results[name]["accuracy"], name),
    )
    comparisons = {
        f"{best_hybrid}_minus_self_consistency": paired_difference(
            arrays[best_hybrid],
            arrays["self_consistency_at_8"],
            samples=args.bootstrap_samples,
            seed=args.seed + 1000,
        ),
        f"{best_hybrid}_minus_{strongest_likelihood}": paired_difference(
            arrays[best_hybrid],
            arrays[strongest_likelihood],
            samples=args.bootstrap_samples,
            seed=args.seed + 1001,
        ),
        "terminal_semantic_minus_strongest_likelihood": paired_difference(
            arrays["terminal_semantic_argmax"],
            arrays[strongest_likelihood],
            samples=args.bootstrap_samples,
            seed=args.seed + 1002,
        ),
    }
    empirical_best = min(
        (
            name
            for name in method_results
            if name != "likelihood_pool_oracle"
        ),
        key=lambda name: (
            -method_results[name]["accuracy"],
            method_results[name]["active_gpu_seconds_per_problem"],
            name,
        ),
    )
    result = {
        "schema_version": 1,
        "evaluation_status": config["evaluation_status"],
        "n_problems": len(problem_ids),
        "semantic_betas": betas,
        "methods": method_results,
        "best_development_hybrid": best_hybrid,
        "strongest_likelihood_selection": strongest_likelihood,
        "empirical_best_accuracy_then_cost": empirical_best,
        "paired_comparisons": comparisons,
        "diagnostics": {
            "likelihood_particle_accuracy": float(
                statistics.fmean(bool(row["correct"]) for row in particles)
            ),
            "independent_target_particle_accuracy": float(
                statistics.fmean(bool(row["correct"]) for row in sc_rows)
            ),
            "correct_particle_count_histogram": {
                str(count): frequency
                for count, frequency in sorted(Counter(correct_particle_counts).items())
            },
            "mean_final_ess": float(statistics.fmean(final_ess)),
            "median_final_ess": float(statistics.median(final_ess)),
            "mean_final_logweight_range": float(
                statistics.fmean(final_logweight_ranges)
            ),
            "mean_unique_final_token_sequences": float(
                likelihood_summary["quality"]["mean_unique_final_token_sequences"]
            ),
            "particles_at_token_cap": sum(
                int(row["completion_tokens"]) >= int(config["likelihood_smc"]["max_new_tokens"])
                for row in particles
            ),
        },
        "cost": {
            "self_consistency_gpu_seconds_per_problem": sc_cost,
            "likelihood_smc_gpu_seconds_per_problem": likelihood_cost,
            "terminal_semantic_gpu_seconds_per_problem": semantic_cost,
            "terminal_semantic_prompt_tokens": semantic_prompt_tokens,
            "semantic_resident_gpus": semantic_gpus,
        },
        "caveats": [
            "The semantic beta sweep and method choice use the same development problems.",
            "This is an exact terminal L1+S combination, not periodic intermediate semantic resampling.",
            "Model initialization is excluded consistently from active GPU seconds.",
        ],
    }
    write_json(args.summary_output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()

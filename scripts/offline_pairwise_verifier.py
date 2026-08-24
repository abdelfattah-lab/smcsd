"""Order-swapped terminal pairwise LLM-as-a-Verifier baseline.

For every pair of completed candidate solutions to the same problem, the
verifier judges both A/B orders. The two probabilities are symmetrized to
remove first-position bias, and each candidate's mean pairwise win probability
defines its final ranking. This is a terminal reranking baseline, not online
ParticleScale and not a prefix potential for semantic SMC.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PAIRWISE_TEMPLATE = """\
You are judging two proposed solutions to the same math problem.

Problem:
{problem}

Solution A:
{solution_a}

Solution B:
{solution_b}

Evaluate the mathematical reasoning and final answer in each solution. Select
the solution that is more likely to be fully correct. If both appear correct
or both appear incorrect, select the one with fewer substantive errors and the
more reliable reasoning. You must still choose one.

Return exactly A or B and no other text.
Choice:\
"""


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def write_json(path: str | Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_trajectories(rows: Sequence[dict]) -> None:
    required = {
        "problem_id",
        "problem",
        "sample_id",
        "generator_model",
        "full_text",
        "extracted_answer",
        "gold_answer",
        "correct",
    }
    seen = set()
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Trajectory row {index} is missing fields: {missing}.")
        key = (str(row["problem_id"]), int(row["sample_id"]))
        if key in seen:
            raise ValueError(f"Duplicate trajectory key: {key}.")
        seen.add(key)
        if bool(row["correct"]) != (
            row["extracted_answer"] == row["gold_answer"]
        ):
            raise ValueError(f"Trajectory row {index} has inconsistent correctness.")
        by_problem[key[0]].append(row)
    sample_counts = {len(group) for group in by_problem.values()}
    if len(sample_counts) != 1 or next(iter(sample_counts)) < 2:
        raise ValueError(
            "Every problem must have the same number of at least two candidates; "
            f"got counts {sorted(sample_counts)}."
        )
    for problem_id, group in by_problem.items():
        if len({row["problem"] for row in group}) != 1:
            raise ValueError(f"Problem text differs within {problem_id}.")
        if len({row["gold_answer"] for row in group}) != 1:
            raise ValueError(f"Gold answer differs within {problem_id}.")


def select_problem_rows(
    rows: Sequence[dict], max_problems: int | None
) -> list[dict]:
    if max_problems is None:
        return list(rows)
    if max_problems <= 0:
        raise ValueError("max_problems must be positive.")
    selected_ids = []
    seen = set()
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in seen:
            seen.add(problem_id)
            selected_ids.append(problem_id)
        if len(selected_ids) == max_problems:
            break
    selected = set(selected_ids)
    return [row for row in rows if str(row["problem_id"]) in selected]


def build_pairwise_prompt(tokenizer, problem: str, solution_a: str, solution_b: str) -> str:
    content = PAIRWISE_TEMPLATE.format(
        problem=problem,
        solution_a=solution_a,
        solution_b=solution_b,
    )
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **kwargs
        )


def resolve_choice_token_ids(tokenizer) -> list[int]:
    token_ids = []
    for label in ("A", "B"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Pairwise label {label!r} is not one token: {encoded}.")
        token_ids.append(int(encoded[0]))
    if token_ids[0] == token_ids[1]:
        raise ValueError("Pairwise A/B labels map to the same token ID.")
    return token_ids


def make_jobs(rows: Sequence[dict], tokenizer) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)

    jobs = []
    for problem_id, group in groups.items():
        candidates = sorted(group, key=lambda row: int(row["sample_id"]))
        for left, right in itertools.combinations(candidates, 2):
            low = int(left["sample_id"])
            high = int(right["sample_id"])
            pair_id = f"{problem_id}:{low}:{high}"
            for order_index, (candidate_a, candidate_b) in enumerate(
                ((left, right), (right, left))
            ):
                jobs.append(
                    {
                        "job_id": f"{pair_id}:order{order_index}",
                        "pair_id": pair_id,
                        "problem_id": problem_id,
                        "sample_low": low,
                        "sample_high": high,
                        "candidate_a": int(candidate_a["sample_id"]),
                        "candidate_b": int(candidate_b["sample_id"]),
                        "correct_a": bool(candidate_a["correct"]),
                        "correct_b": bool(candidate_b["correct"]),
                        "order_index": order_index,
                        "prompt": build_pairwise_prompt(
                            tokenizer,
                            problem=left["problem"],
                            solution_a=candidate_a["full_text"],
                            solution_b=candidate_b["full_text"],
                        ),
                    }
                )
    return jobs


def entry_logprob_and_id(entry) -> tuple[float, int]:
    if isinstance(entry, dict):
        return float(entry["logprob"]), int(entry["token_id"])
    return float(entry[0]), int(entry[1])


def choice_probability(output: dict, choice_token_ids: Sequence[int]) -> dict:
    meta = output.get("meta_info", {})
    positions = meta.get("output_token_ids_logprobs")
    source = "selected"
    if not positions:
        positions = meta.get("output_top_logprobs")
        source = "top"
    if not positions or not positions[0]:
        raise ValueError("Pairwise verifier returned no choice-token logprobs.")
    by_id = {
        token_id: logprob
        for logprob, token_id in (
            entry_logprob_and_id(entry) for entry in positions[0]
        )
    }
    missing = [token_id for token_id in choice_token_ids if token_id not in by_id]
    if missing:
        raise ValueError(f"Pairwise output is missing choice token IDs: {missing}.")
    log_a, log_b = (by_id[token_id] for token_id in choice_token_ids)
    maximum = max(log_a, log_b)
    weight_a = math.exp(log_a - maximum)
    weight_b = math.exp(log_b - maximum)
    probability_a = weight_a / (weight_a + weight_b)
    log_mass = maximum + math.log(weight_a + weight_b)
    return {
        "probability_a": probability_a,
        "probability_b": 1.0 - probability_a,
        "choice_token_mass": math.exp(min(log_mass, 0.0)),
        "logprob_source": source,
        "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
        "generated_token_id": (
            int(output["output_ids"][0]) if output.get("output_ids") else None
        ),
    }


def combine_orientations(orientations: Sequence[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in orientations:
        groups[str(row["pair_id"])].append(row)
    pairs = []
    for pair_id, group in groups.items():
        if len(group) != 2:
            raise ValueError(f"Pair {pair_id} has {len(group)} orientations, expected 2.")
        low = int(group[0]["sample_low"])
        high = int(group[0]["sample_high"])
        low_first = next((row for row in group if int(row["candidate_a"]) == low), None)
        low_second = next((row for row in group if int(row["candidate_b"]) == low), None)
        if low_first is None or low_second is None:
            raise ValueError(f"Pair {pair_id} does not contain both A/B orders.")
        probability_low_first = float(low_first["probability_a"])
        probability_low_second = float(low_second["probability_b"])
        probability_low = (probability_low_first + probability_low_second) / 2
        correct_low = bool(low_first["correct_a"])
        correct_high = bool(low_first["correct_b"])
        if correct_low != correct_high:
            probability_correct = (
                probability_low if correct_low else 1.0 - probability_low
            )
        else:
            probability_correct = None
        pairs.append(
            {
                "pair_id": pair_id,
                "problem_id": low_first["problem_id"],
                "sample_low": low,
                "sample_high": high,
                "correct_low": correct_low,
                "correct_high": correct_high,
                "probability_low_first": probability_low_first,
                "probability_low_second": probability_low_second,
                "probability_low": probability_low,
                "probability_high": 1.0 - probability_low,
                "probability_correct": probability_correct,
                "hard_order_agreement": (
                    (probability_low_first >= 0.5)
                    == (probability_low_second >= 0.5)
                ),
                "absolute_order_effect": abs(
                    probability_low_first - probability_low_second
                ),
            }
        )
    return pairs


def majority_prediction(group: Sequence[dict], sample_order: Sequence[int]) -> str | None:
    by_sample = {int(row["sample_id"]): row for row in group}
    answers = [
        by_sample[sample_id]["extracted_answer"]
        for sample_id in sample_order
        if by_sample[sample_id]["extracted_answer"] is not None
    ]
    return Counter(answers).most_common(1)[0][0] if answers else None


def aggregate_rankings(
    trajectories: Sequence[dict], pairs: Sequence[dict]
) -> list[dict]:
    trajectories_by_problem: dict[str, list[dict]] = defaultdict(list)
    pairs_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectories_by_problem[str(row["problem_id"])].append(row)
    for pair in pairs:
        pairs_by_problem[str(pair["problem_id"])].append(pair)

    results = []
    for problem_id, group in trajectories_by_problem.items():
        candidates = sorted(group, key=lambda row: int(row["sample_id"]))
        expected_pairs = math.comb(len(candidates), 2)
        problem_pairs = pairs_by_problem.get(problem_id, [])
        if len(problem_pairs) != expected_pairs:
            raise ValueError(
                f"Problem {problem_id} has {len(problem_pairs)} pairs, "
                f"expected {expected_pairs}."
            )
        rating_sum = defaultdict(float)
        comparison_count = Counter()
        for pair in problem_pairs:
            low = int(pair["sample_low"])
            high = int(pair["sample_high"])
            rating_sum[low] += float(pair["probability_low"])
            rating_sum[high] += float(pair["probability_high"])
            comparison_count[low] += 1
            comparison_count[high] += 1
        ratings = {
            int(row["sample_id"]): (
                rating_sum[int(row["sample_id"])]
                / comparison_count[int(row["sample_id"])]
            )
            for row in candidates
        }
        ranked = sorted(ratings, key=lambda sample_id: (-ratings[sample_id], sample_id))
        by_sample = {int(row["sample_id"]): row for row in candidates}
        gold = candidates[0]["gold_answer"]
        top_k_correct = {}
        for k in range(1, len(candidates) + 1):
            prediction = majority_prediction(candidates, ranked[:k])
            top_k_correct[str(k)] = bool(prediction == gold)
        sample_order = [int(row["sample_id"]) for row in candidates]
        self_consistency = majority_prediction(candidates, sample_order)
        results.append(
            {
                "problem_id": problem_id,
                "ratings": {str(key): value for key, value in ratings.items()},
                "ranking": ranked,
                "selected_sample_id": ranked[0],
                "selected_correct": bool(by_sample[ranked[0]]["correct"]),
                "selected_answer": by_sample[ranked[0]]["extracted_answer"],
                "top_k_majority_correct": top_k_correct,
                "self_consistency_correct": bool(self_consistency == gold),
                "oracle_correct": any(bool(row["correct"]) for row in candidates),
            }
        )
    return results


def attach_knockout_outcomes(
    problems: Sequence[dict],
    trajectories: Sequence[dict],
    pairs: Sequence[dict],
    orientations: Sequence[dict],
) -> None:
    """Simulate a fixed order-swapped knockout using only seven pairs for n=8."""
    trajectories_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectories_by_problem[str(row["problem_id"])].append(row)
    pair_by_key = {
        (
            str(row["problem_id"]),
            int(row["sample_low"]),
            int(row["sample_high"]),
        ): row
        for row in pairs
    }
    calls_by_pair = Counter(str(row["pair_id"]) for row in orientations)
    prompt_tokens_by_pair: dict[str, int] = defaultdict(int)
    completion_tokens_by_pair: dict[str, int] = defaultdict(int)
    for row in orientations:
        pair_id = str(row["pair_id"])
        prompt_tokens_by_pair[pair_id] += int(row["prompt_tokens"])
        completion_tokens_by_pair[pair_id] += int(row["completion_tokens"])

    for problem in problems:
        problem_id = str(problem["problem_id"])
        candidates = sorted(
            trajectories_by_problem[problem_id],
            key=lambda row: int(row["sample_id"]),
        )
        candidate_count = len(candidates)
        if candidate_count < 2 or candidate_count & (candidate_count - 1):
            raise ValueError(
                "Knockout requires a power-of-two candidate count; "
                f"got {candidate_count} for {problem_id}."
            )
        by_sample = {int(row["sample_id"]): row for row in candidates}
        alive = sorted(by_sample)
        match_rows = []
        round_index = 0
        while len(alive) > 1:
            winners = []
            for match_index, (first, second) in enumerate(
                zip(alive[::2], alive[1::2])
            ):
                low, high = sorted((first, second))
                pair = pair_by_key.get((problem_id, low, high))
                if pair is None:
                    raise ValueError(
                        f"Missing knockout pair for {problem_id}: {low} vs {high}."
                    )
                probability_first = float(
                    pair["probability_low"]
                    if first == low
                    else pair["probability_high"]
                )
                winner = first if probability_first >= 0.5 else second
                winners.append(winner)
                pair_id = str(pair["pair_id"])
                match_rows.append(
                    {
                        "round": round_index,
                        "match": match_index,
                        "first": first,
                        "second": second,
                        "probability_first": probability_first,
                        "winner": winner,
                        "pair_id": pair_id,
                    }
                )
            alive = winners
            round_index += 1
        winner = alive[0]
        selected_pair_ids = [row["pair_id"] for row in match_rows]
        problem.update(
            {
                "knockout_selected_sample_id": winner,
                "knockout_selected_correct": bool(by_sample[winner]["correct"]),
                "knockout_matches": match_rows,
                "knockout_verifier_calls": sum(
                    calls_by_pair[pair_id] for pair_id in selected_pair_ids
                ),
                "knockout_verifier_prompt_tokens": sum(
                    prompt_tokens_by_pair[pair_id] for pair_id in selected_pair_ids
                ),
                "knockout_verifier_completion_tokens": sum(
                    completion_tokens_by_pair[pair_id]
                    for pair_id in selected_pair_ids
                ),
            }
        )


def attach_pointwise_outcomes(problems: Sequence[dict], score_rows: Sequence[dict]) -> None:
    terminal_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in score_rows:
        if row.get("terminal"):
            terminal_by_problem[str(row["problem_id"])].append(row)
    for problem in problems:
        problem_id = str(problem["problem_id"])
        candidates = terminal_by_problem.get(problem_id, [])
        if not candidates:
            raise ValueError(f"No terminal pointwise scores for {problem_id}.")
        selected = min(
            candidates,
            key=lambda row: (-float(row["score"]), int(row["sample_id"])),
        )
        problem["pointwise_terminal_selected_correct"] = bool(selected["correct"])


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return [low, high]


def bootstrap_metrics(
    problems: Sequence[dict],
    pairs: Sequence[dict],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0 or len(problems) < 2:
        return {}
    pairs_by_problem: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["probability_correct"] is not None:
            pairs_by_problem[str(pair["problem_id"])].append(pair)
    rng = random.Random(seed)
    boot = defaultdict(list)
    candidate_count = len(problems[0]["top_k_majority_correct"])
    for _ in range(samples):
        drawn = [rng.choice(problems) for _ in problems]
        boot["pairwise_selection_accuracy"].append(
            mean([float(row["selected_correct"]) for row in drawn])
        )
        boot["pairwise_minus_self_consistency"].append(
            mean(
                [
                    float(row["selected_correct"])
                    - float(row["self_consistency_correct"])
                    for row in drawn
                ]
            )
        )
        if "knockout_selected_correct" in drawn[0]:
            boot["knockout_selection_accuracy"].append(
                mean([float(row["knockout_selected_correct"]) for row in drawn])
            )
            boot["knockout_minus_self_consistency"].append(
                mean(
                    [
                        float(row["knockout_selected_correct"])
                        - float(row["self_consistency_correct"])
                        for row in drawn
                    ]
                )
            )
        if "pointwise_terminal_selected_correct" in drawn[0]:
            boot["pairwise_minus_pointwise_terminal_bon"].append(
                mean(
                    [
                        float(row["selected_correct"])
                        - float(row["pointwise_terminal_selected_correct"])
                        for row in drawn
                    ]
                )
            )
            if "knockout_selected_correct" in drawn[0]:
                boot["knockout_minus_pointwise_terminal_bon"].append(
                    mean(
                        [
                            float(row["knockout_selected_correct"])
                            - float(row["pointwise_terminal_selected_correct"])
                            for row in drawn
                        ]
                    )
                )
        for k in range(1, candidate_count + 1):
            boot[f"top_{k}_majority_accuracy"].append(
                mean(
                    [
                        float(row["top_k_majority_correct"][str(k)])
                        for row in drawn
                    ]
                )
            )
        eligible = [
            pair
            for row in drawn
            for pair in pairs_by_problem.get(str(row["problem_id"]), [])
        ]
        if eligible:
            boot["correct_vs_incorrect_pair_accuracy"].append(
                mean(
                    [
                        1.0
                        if float(pair["probability_correct"]) > 0.5
                        else 0.5
                        if float(pair["probability_correct"]) == 0.5
                        else 0.0
                        for pair in eligible
                    ]
                )
            )
    return {key: percentile_interval(values) for key, values in boot.items()}


def summarize(
    trajectories: Sequence[dict],
    orientations: Sequence[dict],
    pairs: Sequence[dict],
    problems: Sequence[dict],
    *,
    scorer: str,
    choice_token_ids: Sequence[int],
    wall_time: float,
    bootstrap_samples: int,
    seed: int,
    pointwise_summary: dict | None,
) -> dict:
    eligible_pairs = [pair for pair in pairs if pair["probability_correct"] is not None]
    pair_wins = [
        1.0
        if float(pair["probability_correct"]) > 0.5
        else 0.5
        if float(pair["probability_correct"]) == 0.5
        else 0.0
        for pair in eligible_pairs
    ]
    n_candidates = len(trajectories) // len(problems)
    selection_by_k = {
        str(k): mean(
            [float(row["top_k_majority_correct"][str(k)]) for row in problems]
        )
        for k in range(1, n_candidates + 1)
    }
    baselines = {
        "self_consistency_accuracy": mean(
            [float(row["self_consistency_correct"]) for row in problems]
        ),
        "oracle_pass_at_n": mean([float(row["oracle_correct"]) for row in problems]),
    }
    paired = {
        "pairwise_minus_self_consistency": mean(
            [
                float(row["selected_correct"])
                - float(row["self_consistency_correct"])
                for row in problems
            ]
        ),
        "pairwise_vs_self_consistency_wins": sum(
            row["selected_correct"] and not row["self_consistency_correct"]
            for row in problems
        ),
        "pairwise_vs_self_consistency_losses": sum(
            not row["selected_correct"] and row["self_consistency_correct"]
            for row in problems
        ),
    }
    if "knockout_selected_correct" in problems[0]:
        paired.update(
            {
                "knockout_selection_accuracy": mean(
                    [float(row["knockout_selected_correct"]) for row in problems]
                ),
                "knockout_minus_self_consistency": mean(
                    [
                        float(row["knockout_selected_correct"])
                        - float(row["self_consistency_correct"])
                        for row in problems
                    ]
                ),
                "knockout_vs_self_consistency_wins": sum(
                    row["knockout_selected_correct"]
                    and not row["self_consistency_correct"]
                    for row in problems
                ),
                "knockout_vs_self_consistency_losses": sum(
                    not row["knockout_selected_correct"]
                    and row["self_consistency_correct"]
                    for row in problems
                ),
            }
        )
    if "pointwise_terminal_selected_correct" in problems[0]:
        baselines["pointwise_terminal_bon_accuracy"] = mean(
            [
                float(row["pointwise_terminal_selected_correct"])
                for row in problems
            ]
        )
        paired.update(
            {
                "pairwise_minus_pointwise_terminal_bon": mean(
                    [
                        float(row["selected_correct"])
                        - float(row["pointwise_terminal_selected_correct"])
                        for row in problems
                    ]
                ),
                "pairwise_vs_pointwise_wins": sum(
                    row["selected_correct"]
                    and not row["pointwise_terminal_selected_correct"]
                    for row in problems
                ),
                "pairwise_vs_pointwise_losses": sum(
                    not row["selected_correct"]
                    and row["pointwise_terminal_selected_correct"]
                    for row in problems
                ),
            }
        )
        if "knockout_selected_correct" in problems[0]:
            paired.update(
                {
                    "knockout_minus_pointwise_terminal_bon": mean(
                        [
                            float(row["knockout_selected_correct"])
                            - float(row["pointwise_terminal_selected_correct"])
                            for row in problems
                        ]
                    ),
                    "knockout_vs_pointwise_wins": sum(
                        row["knockout_selected_correct"]
                        and not row["pointwise_terminal_selected_correct"]
                        for row in problems
                    ),
                    "knockout_vs_pointwise_losses": sum(
                        not row["knockout_selected_correct"]
                        and row["pointwise_terminal_selected_correct"]
                        for row in problems
                    ),
                }
            )
    if pointwise_summary is not None:
        baselines["pointwise_terminal_bon_accuracy"] = pointwise_summary.get(
            "selection_baselines", {}
        ).get("terminal_pointwise_bon_accuracy")
    return {
        "schema_version": 1,
        "experiment": {
            "method": "order_swapped_all_pairs_mean_win_probability",
            "scorer_model": scorer,
            "choice_labels": ["A", "B"],
            "choice_token_ids": list(choice_token_ids),
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
        },
        "dataset": {
            "name": trajectories[0].get("dataset"),
            "config": trajectories[0].get("dataset_config"),
            "split": trajectories[0].get("split"),
            "n_problems": len(problems),
            "n_candidates_per_problem": n_candidates,
            "n_trajectories": len(trajectories),
        },
        "metrics": {
            "pairwise_selection_accuracy": selection_by_k["1"],
            "selection_by_top_k_majority": selection_by_k,
            "correct_vs_incorrect_pair_accuracy": mean(pair_wins),
            "mean_probability_assigned_to_correct": mean(
                [float(pair["probability_correct"]) for pair in eligible_pairs]
            ),
            "n_correct_vs_incorrect_pairs": len(eligible_pairs),
            "hard_order_agreement": mean(
                [float(pair["hard_order_agreement"]) for pair in pairs]
            ),
            "mean_absolute_order_effect": mean(
                [float(pair["absolute_order_effect"]) for pair in pairs]
            ),
            **paired,
            "bootstrap_95_ci": bootstrap_metrics(
                problems,
                pairs,
                samples=bootstrap_samples,
                seed=seed,
            ),
        },
        "baselines": baselines,
        "cost": {
            "wall_time_s": wall_time,
            "canonical_pairs": len(pairs),
            "order_swapped_verifier_calls": len(orientations),
            "verifier_prompt_tokens": sum(row["prompt_tokens"] for row in orientations),
            "verifier_completion_tokens": sum(
                row["completion_tokens"] for row in orientations
            ),
            "mean_prompt_tokens_per_call": mean(
                [row["prompt_tokens"] for row in orientations]
            ),
            "calls_per_s": len(orientations) / wall_time,
            "mean_choice_token_mass": mean(
                [row["choice_token_mass"] for row in orientations]
            ),
            "selected_logprob_coverage": mean(
                [float(row["logprob_source"] == "selected") for row in orientations]
            ),
            "knockout_tournament": (
                {
                    "canonical_pairs": sum(
                        len(row["knockout_matches"]) for row in problems
                    ),
                    "order_swapped_verifier_calls": sum(
                        row["knockout_verifier_calls"] for row in problems
                    ),
                    "verifier_prompt_tokens": sum(
                        row["knockout_verifier_prompt_tokens"] for row in problems
                    ),
                    "verifier_completion_tokens": sum(
                        row["knockout_verifier_completion_tokens"]
                        for row in problems
                    ),
                }
                if "knockout_matches" in problems[0]
                else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--pointwise-summary", default=None)
    parser.add_argument("--pointwise-scores", default=None)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-gpu-id", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-mamba-cache-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable decode graphs; pairwise evaluation generates only one token.",
    )
    parser.add_argument("--save-orientations", required=True)
    parser.add_argument("--save-pairs", required=True)
    parser.add_argument("--save-rankings", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-orientations",
        action="store_true",
        help="Skip inference and re-aggregate the existing --save-orientations file.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Score one canonical pair in both orders and skip aggregation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = select_problem_rows(
        load_jsonl(args.trajectories), args.max_problems
    )
    validate_trajectories(trajectories)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    choice_token_ids = resolve_choice_token_ids(tokenizer)
    jobs = make_jobs(trajectories, tokenizer)
    if args.probe:
        jobs = jobs[:2]
    print(
        f"problems={len({row['problem_id'] for row in trajectories})} "
        f"trajectories={len(trajectories)} calls={len(jobs)} scorer={args.scorer}",
        flush=True,
    )
    print(f"choice token IDs: A={choice_token_ids[0]} B={choice_token_ids[1]}", flush=True)
    if args.dry_run:
        print(jobs[0]["prompt"])
        return None

    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.scorer,
        trust_remote_code=True,
        attention_backend="triton",
        mem_fraction_static=args.mem_fraction_static,
        base_gpu_id=args.base_gpu_id,
        random_seed=args.seed,
        max_running_requests=args.max_running_requests,
        max_mamba_cache_size=args.max_mamba_cache_size,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    if args.dp > 1:
        engine_kwargs["dp_size"] = args.dp
        engine_kwargs["load_balance_method"] = "total_tokens"
    if args.tp > 1:
        engine_kwargs.update(
            tp_size=args.tp,
            disable_custom_all_reduce=True,
            enforce_disable_flashinfer_allreduce_fusion=True,
        )

    if args.reuse_orientations:
        orientations = load_jsonl(args.save_orientations)
        expected_jobs = {job["job_id"] for job in jobs}
        observed_jobs = {row["job_id"] for row in orientations}
        if expected_jobs != observed_jobs:
            raise ValueError(
                "Saved orientations do not match the requested trajectories: "
                f"missing={len(expected_jobs - observed_jobs)} "
                f"extra={len(observed_jobs - expected_jobs)}."
            )
        previous_summary = (
            json.loads(Path(args.summary_output).read_text())
            if Path(args.summary_output).exists()
            else {}
        )
        wall_time = float(previous_summary.get("cost", {}).get("wall_time_s", 0.0))
        print(f"reusing {len(orientations)} saved orientation calls", flush=True)
    else:
        orientations = []
        started = time.perf_counter()
        engine = sgl.Engine(**engine_kwargs)
        try:
            for start in range(0, len(jobs), args.batch_size):
                batch = jobs[start : start + args.batch_size]
                outputs = engine.generate(
                    [job["prompt"] for job in batch],
                    {"max_new_tokens": 1, "temperature": 0.0},
                    return_logprob=True,
                    top_logprobs_num=0,
                    token_ids_logprob=choice_token_ids,
                )
                if not isinstance(outputs, list):
                    outputs = [outputs]
                if len(outputs) != len(batch):
                    raise ValueError(
                        f"Verifier returned {len(outputs)} outputs for {len(batch)} jobs."
                    )
                for job, output in zip(batch, outputs):
                    orientations.append(
                        {
                            "schema_version": 1,
                            **{
                                key: value
                                for key, value in job.items()
                                if key != "prompt"
                            },
                            **choice_probability(output, choice_token_ids),
                            "scorer_model": args.scorer,
                        }
                    )
                elapsed = time.perf_counter() - started
                print(
                    f"scored={len(orientations)}/{len(jobs)} "
                    f"calls/s={len(orientations) / elapsed:.2f}",
                    flush=True,
                )
                if args.probe:
                    print(json.dumps(outputs[0].get("meta_info", {}), indent=2)[:3000])
        finally:
            engine.shutdown()
        wall_time = time.perf_counter() - started
        write_jsonl(args.save_orientations, orientations)

    if args.probe:
        probe = {
            "schema_version": 1,
            "probe": True,
            "choice_token_ids": choice_token_ids,
            "orientations": orientations,
            "wall_time_s": wall_time,
        }
        write_json(args.summary_output, probe)
        print(f"wrote probe outputs to {args.save_orientations}")
        return probe

    pairs = combine_orientations(orientations)
    rankings = aggregate_rankings(trajectories, pairs)
    attach_knockout_outcomes(rankings, trajectories, pairs, orientations)
    if args.pointwise_scores:
        attach_pointwise_outcomes(rankings, load_jsonl(args.pointwise_scores))
    pointwise_summary = (
        json.loads(Path(args.pointwise_summary).read_text())
        if args.pointwise_summary
        else None
    )
    summary = summarize(
        trajectories,
        orientations,
        pairs,
        rankings,
        scorer=args.scorer,
        choice_token_ids=choice_token_ids,
        wall_time=wall_time,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        pointwise_summary=pointwise_summary,
    )
    write_jsonl(args.save_pairs, pairs)
    write_jsonl(args.save_rankings, rankings)
    write_json(args.summary_output, summary)

    print("\nTerminal pairwise LLM-as-a-Verifier")
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["baselines"], indent=2))
    print(json.dumps(summary["cost"], indent=2))
    print(f"wrote {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

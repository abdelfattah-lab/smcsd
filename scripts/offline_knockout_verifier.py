"""Stage-wise 14-call order-swapped knockout semantic verifier.

For eight completed candidates, each of seven bracket matches is judged in
both A/B orders. The two probabilities are symmetrized before the winner
advances. Unlike the all-pairs ablation, later match prompts are generated only
after the preceding round, so this executes exactly 14 verifier calls/problem.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

try:
    from scripts import offline_pairwise_verifier as pairwise
except ModuleNotFoundError:
    import offline_pairwise_verifier as pairwise


def group_trajectories(rows: Sequence[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    for problem_id, group in groups.items():
        group.sort(key=lambda row: int(row["sample_id"]))
        count = len(group)
        if count < 2 or count & (count - 1):
            raise ValueError(
                "Knockout requires a power-of-two candidate count; "
                f"got {count} for {problem_id}."
            )
    return groups


def make_round_jobs(
    alive_by_problem: dict[str, list[int]],
    trajectories_by_problem: dict[str, list[dict]],
    tokenizer,
    *,
    round_index: int,
) -> list[dict]:
    jobs = []
    for problem_id, alive in alive_by_problem.items():
        by_sample = {
            int(row["sample_id"]): row
            for row in trajectories_by_problem[problem_id]
        }
        for match_index, (first, second) in enumerate(
            zip(alive[::2], alive[1::2])
        ):
            low, high = sorted((first, second))
            pair_id = f"{problem_id}:{low}:{high}"
            left = by_sample[low]
            right = by_sample[high]
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
                        "round": round_index,
                        "match": match_index,
                        "prompt": pairwise.build_pairwise_prompt(
                            tokenizer,
                            problem=left["problem"],
                            solution_a=candidate_a["full_text"],
                            solution_b=candidate_b["full_text"],
                        ),
                    }
                )
    return jobs


def advance_round(
    alive_by_problem: dict[str, list[int]],
    round_pairs: Sequence[dict],
    *,
    round_index: int,
) -> tuple[dict[str, list[int]], list[dict]]:
    pair_by_key = {
        (
            str(row["problem_id"]),
            int(row["sample_low"]),
            int(row["sample_high"]),
        ): row
        for row in round_pairs
    }
    next_alive = {}
    matches = []
    for problem_id, alive in alive_by_problem.items():
        winners = []
        for match_index, (first, second) in enumerate(
            zip(alive[::2], alive[1::2])
        ):
            low, high = sorted((first, second))
            pair = pair_by_key.get((problem_id, low, high))
            if pair is None:
                raise ValueError(
                    f"Missing round {round_index} pair {problem_id}:{low}:{high}."
                )
            probability_first = float(
                pair["probability_low"]
                if first == low
                else pair["probability_high"]
            )
            winner = first if probability_first >= 0.5 else second
            winners.append(winner)
            matches.append(
                {
                    "problem_id": problem_id,
                    "pair_id": pair["pair_id"],
                    "round": round_index,
                    "match": match_index,
                    "first": first,
                    "second": second,
                    "probability_first": probability_first,
                    "winner": winner,
                }
            )
        next_alive[problem_id] = winners
    return next_alive, matches


def pointwise_terminal_choices(score_rows: Sequence[dict]) -> dict[str, bool]:
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in score_rows:
        if row.get("terminal"):
            by_problem[str(row["problem_id"])].append(row)
    selected = {}
    for problem_id, group in by_problem.items():
        best = min(
            group,
            key=lambda row: (-float(row["score"]), int(row["sample_id"])),
        )
        selected[problem_id] = bool(best["correct"])
    return selected


def build_problem_results(
    trajectories_by_problem: dict[str, list[dict]],
    alive_by_problem: dict[str, list[int]],
    matches: Sequence[dict],
    pointwise_scores: Sequence[dict] | None,
) -> list[dict]:
    matches_by_problem: dict[str, list[dict]] = defaultdict(list)
    for match in matches:
        matches_by_problem[str(match["problem_id"])].append(match)
    pointwise_outcomes = (
        pointwise_terminal_choices(pointwise_scores)
        if pointwise_scores is not None
        else {}
    )
    problems = []
    for problem_id, group in trajectories_by_problem.items():
        if len(alive_by_problem[problem_id]) != 1:
            raise ValueError(f"Tournament for {problem_id} did not produce one winner.")
        winner = alive_by_problem[problem_id][0]
        by_sample = {int(row["sample_id"]): row for row in group}
        sample_order = sorted(by_sample)
        majority = pairwise.majority_prediction(group, sample_order)
        result = {
            "problem_id": problem_id,
            "selected_sample_id": winner,
            "selected_correct": bool(by_sample[winner]["correct"]),
            "selected_answer": by_sample[winner]["extracted_answer"],
            "self_consistency_correct": bool(majority == group[0]["gold_answer"]),
            "oracle_correct": any(bool(row["correct"]) for row in group),
            "matches": sorted(
                matches_by_problem[problem_id],
                key=lambda row: (int(row["round"]), int(row["match"])),
            ),
        }
        if pointwise_scores is not None:
            if problem_id not in pointwise_outcomes:
                raise ValueError(f"No terminal pointwise score for {problem_id}.")
            result["pointwise_terminal_selected_correct"] = pointwise_outcomes[
                problem_id
            ]
        problems.append(result)
    return problems


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, int(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)) + 1)]
    return [low, high]


def paired_metrics(problems: Sequence[dict], reference: str, name: str) -> dict:
    return {
        f"knockout_minus_{name}": pairwise.mean(
            [
                float(row["selected_correct"]) - float(row[reference])
                for row in problems
            ]
        ),
        f"knockout_vs_{name}_wins": sum(
            row["selected_correct"] and not row[reference] for row in problems
        ),
        f"knockout_vs_{name}_losses": sum(
            not row["selected_correct"] and row[reference] for row in problems
        ),
    }


def bootstrap_metrics(
    problems: Sequence[dict], *, samples: int, seed: int
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    references = {
        "self_consistency": "self_consistency_correct",
        "pointwise_terminal_bon": "pointwise_terminal_selected_correct",
    }
    boot: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = [rng.choice(problems) for _ in problems]
        boot["knockout_selection_accuracy"].append(
            pairwise.mean([float(row["selected_correct"]) for row in drawn])
        )
        for name, field in references.items():
            if field in drawn[0]:
                boot[f"knockout_minus_{name}"].append(
                    pairwise.mean(
                        [
                            float(row["selected_correct"]) - float(row[field])
                            for row in drawn
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
) -> dict:
    eligible_pairs = [row for row in pairs if row["probability_correct"] is not None]
    metrics = {
        "knockout_selection_accuracy": pairwise.mean(
            [float(row["selected_correct"]) for row in problems]
        ),
        **paired_metrics(problems, "self_consistency_correct", "self_consistency"),
    }
    baselines = {
        "self_consistency_accuracy": pairwise.mean(
            [float(row["self_consistency_correct"]) for row in problems]
        ),
        "oracle_pass_at_n": pairwise.mean(
            [float(row["oracle_correct"]) for row in problems]
        ),
    }
    if "pointwise_terminal_selected_correct" in problems[0]:
        baselines["pointwise_terminal_bon_accuracy"] = pairwise.mean(
            [float(row["pointwise_terminal_selected_correct"]) for row in problems]
        )
        metrics.update(
            paired_metrics(
                problems,
                "pointwise_terminal_selected_correct",
                "pointwise_terminal_bon",
            )
        )
    if eligible_pairs:
        metrics.update(
            {
                "correct_vs_incorrect_match_accuracy": pairwise.mean(
                    [
                        1.0
                        if float(row["probability_correct"]) > 0.5
                        else 0.5
                        if float(row["probability_correct"]) == 0.5
                        else 0.0
                        for row in eligible_pairs
                    ]
                ),
                "mean_probability_assigned_to_correct": pairwise.mean(
                    [float(row["probability_correct"]) for row in eligible_pairs]
                ),
                "n_correct_vs_incorrect_matches": len(eligible_pairs),
            }
        )
    metrics.update(
        {
            "hard_order_agreement": pairwise.mean(
                [float(row["hard_order_agreement"]) for row in pairs]
            ),
            "mean_absolute_order_effect": pairwise.mean(
                [float(row["absolute_order_effect"]) for row in pairs]
            ),
            "bootstrap_95_ci": bootstrap_metrics(
                problems, samples=bootstrap_samples, seed=seed
            ),
        }
    )
    return {
        "schema_version": 1,
        "experiment": {
            "method": "stagewise_fixed_bracket_order_swapped_knockout",
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
            "start_index": min(int(row["dataset_index"]) for row in trajectories),
            "n_problems": len(problems),
            "n_candidates_per_problem": len(trajectories) // len(problems),
            "n_trajectories": len(trajectories),
        },
        "metrics": metrics,
        "baselines": baselines,
        "cost": {
            "wall_time_s": wall_time,
            "canonical_matches": len(pairs),
            "order_swapped_verifier_calls": len(orientations),
            "verifier_prompt_tokens": sum(row["prompt_tokens"] for row in orientations),
            "verifier_completion_tokens": sum(
                row["completion_tokens"] for row in orientations
            ),
            "mean_prompt_tokens_per_call": pairwise.mean(
                [row["prompt_tokens"] for row in orientations]
            ),
            "calls_per_s": len(orientations) / wall_time if wall_time else None,
            "mean_choice_token_mass": pairwise.mean(
                [row["choice_token_mass"] for row in orientations]
            ),
            "selected_logprob_coverage": pairwise.mean(
                [float(row["logprob_source"] == "selected") for row in orientations]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--pointwise-scores", default=None)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
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
    )
    parser.add_argument("--save-orientations", required=True)
    parser.add_argument("--save-matches", required=True)
    parser.add_argument("--save-rankings", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--reuse-orientations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    trajectories = pairwise.select_problem_rows(
        pairwise.load_jsonl(args.trajectories), args.max_problems
    )
    pairwise.validate_trajectories(trajectories)
    trajectories_by_problem = group_trajectories(trajectories)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    choice_token_ids = pairwise.resolve_choice_token_ids(tokenizer)
    alive = {
        problem_id: [int(row["sample_id"]) for row in group]
        for problem_id, group in trajectories_by_problem.items()
    }
    first_jobs = make_round_jobs(
        alive, trajectories_by_problem, tokenizer, round_index=0
    )
    print(
        f"problems={len(alive)} trajectories={len(trajectories)} "
        f"expected_calls={2 * (len(trajectories) - len(alive))} scorer={args.scorer}",
        flush=True,
    )
    if args.dry_run:
        print(first_jobs[0]["prompt"])
        return None

    if args.reuse_orientations:
        orientations = pairwise.load_jsonl(args.save_orientations)
        try:
            previous = json.loads(Path(args.summary_output).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        wall_time = float(previous.get("cost", {}).get("wall_time_s", 0.0))
        engine = None
        started = None
    else:
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
        engine = sgl.Engine(**engine_kwargs)
        orientations = []
        started = time.perf_counter()

    all_pairs = []
    all_matches = []
    round_index = 0
    try:
        while len(next(iter(alive.values()))) > 1:
            jobs = make_round_jobs(
                alive,
                trajectories_by_problem,
                tokenizer,
                round_index=round_index,
            )
            expected_job_ids = {job["job_id"] for job in jobs}
            if args.reuse_orientations:
                round_orientations = [
                    row for row in orientations if int(row["round"]) == round_index
                ]
                observed_job_ids = {row["job_id"] for row in round_orientations}
                if expected_job_ids != observed_job_ids:
                    raise ValueError(
                        f"Saved round {round_index} jobs differ: "
                        f"missing={len(expected_job_ids - observed_job_ids)} "
                        f"extra={len(observed_job_ids - expected_job_ids)}."
                    )
            else:
                round_orientations = []
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
                            f"Verifier returned {len(outputs)} outputs for "
                            f"{len(batch)} jobs."
                        )
                    for job, output in zip(batch, outputs):
                        row = {
                            "schema_version": 1,
                            **{key: value for key, value in job.items() if key != "prompt"},
                            **pairwise.choice_probability(output, choice_token_ids),
                            "scorer_model": args.scorer,
                        }
                        orientations.append(row)
                        round_orientations.append(row)
                pairwise.write_jsonl(args.save_orientations, orientations)
            round_pairs = pairwise.combine_orientations(round_orientations)
            alive, matches = advance_round(
                alive, round_pairs, round_index=round_index
            )
            all_pairs.extend(round_pairs)
            all_matches.extend(matches)
            print(
                f"round={round_index} calls={len(round_orientations)} "
                f"survivors/problem={len(next(iter(alive.values())))}",
                flush=True,
            )
            round_index += 1
    finally:
        if engine is not None:
            engine.shutdown()
    if started is not None:
        wall_time = time.perf_counter() - started

    pointwise_scores = (
        pairwise.load_jsonl(args.pointwise_scores) if args.pointwise_scores else None
    )
    problems = build_problem_results(
        trajectories_by_problem,
        alive,
        all_matches,
        pointwise_scores,
    )
    summary = summarize(
        trajectories,
        orientations,
        all_pairs,
        problems,
        scorer=args.scorer,
        choice_token_ids=choice_token_ids,
        wall_time=wall_time,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    pairwise.write_jsonl(args.save_matches, all_matches)
    pairwise.write_jsonl(args.save_rankings, problems)
    pairwise.write_json(args.summary_output, summary)
    print("\nStage-wise order-swapped knockout")
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["baselines"], indent=2))
    print(json.dumps(summary["cost"], indent=2))
    print(f"wrote {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

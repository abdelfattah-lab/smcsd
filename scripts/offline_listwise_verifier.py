"""One-call terminal listwise LLM-as-a-Verifier baseline.

Each problem's completed candidates are placed in a deterministic random order
and labeled A, B, ....  The verifier sees all candidates in one prompt.  Exact
label-token log probabilities define a distribution and ranking, so this uses
one verifier call per problem rather than all-pairs comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


LISTWISE_HEADER = """\
You are judging several proposed solutions to the same math problem.

Problem:
{problem}

"""

LISTWISE_FOOTER = """\
Evaluate the mathematical reasoning and final answer in every solution. Select
the one solution that is most likely to be fully correct. If multiple solutions
appear equally strong, select the one with the most reliable reasoning and
fewest substantive errors. You must still choose one.

Return exactly one label from {labels} and no other text.
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
    counts = {len(group) for group in by_problem.values()}
    if len(counts) != 1 or not 2 <= next(iter(counts)) <= 26:
        raise ValueError(
            "Every problem must have the same number of 2--26 candidates; "
            f"got counts {sorted(counts)}."
        )


def select_problem_rows(
    rows: Sequence[dict], max_problems: int | None
) -> list[dict]:
    if max_problems is None:
        return list(rows)
    if max_problems <= 0:
        raise ValueError("max_problems must be positive.")
    selected_ids = []
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id not in selected_ids:
            selected_ids.append(problem_id)
        if len(selected_ids) == max_problems:
            break
    selected = set(selected_ids)
    return [row for row in rows if str(row["problem_id"]) in selected]


def labels_for_count(candidate_count: int) -> list[str]:
    if not 2 <= candidate_count <= 26:
        raise ValueError("Listwise labels support 2--26 candidates.")
    return [chr(ord("A") + index) for index in range(candidate_count)]


def resolve_choice_token_ids(tokenizer, labels: Sequence[str]) -> list[int]:
    token_ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Listwise label {label!r} is not one token: {encoded}.")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Listwise labels do not map to unique token IDs.")
    return token_ids


def build_listwise_prompt(
    tokenizer,
    problem: str,
    labeled_solutions: Sequence[tuple[str, str]],
) -> str:
    sections = [LISTWISE_HEADER.format(problem=problem)]
    for label, solution in labeled_solutions:
        sections.append(f"Solution {label}:\n{solution}\n\n")
    sections.append(
        LISTWISE_FOOTER.format(
            labels=", ".join(label for label, _ in labeled_solutions)
        )
    )
    content = "".join(sections)
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


def make_jobs(rows: Sequence[dict], tokenizer, *, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["problem_id"])].append(row)
    candidate_count = len(next(iter(groups.values())))
    labels = labels_for_count(candidate_count)
    jobs = []
    for problem_id, group in groups.items():
        candidates = sorted(group, key=lambda row: int(row["sample_id"]))
        random.Random(f"{seed}:{problem_id}").shuffle(candidates)
        label_to_sample = {
            label: int(candidate["sample_id"])
            for label, candidate in zip(labels, candidates)
        }
        label_to_correct = {
            label: bool(candidate["correct"])
            for label, candidate in zip(labels, candidates)
        }
        jobs.append(
            {
                "job_id": problem_id,
                "problem_id": problem_id,
                "label_to_sample": label_to_sample,
                "label_to_correct": label_to_correct,
                "prompt": build_listwise_prompt(
                    tokenizer,
                    candidates[0]["problem"],
                    [
                        (label, candidate["full_text"])
                        for label, candidate in zip(labels, candidates)
                    ],
                ),
            }
        )
    return jobs


def entry_logprob_and_id(entry) -> tuple[float, int]:
    if isinstance(entry, dict):
        return float(entry["logprob"]), int(entry["token_id"])
    return float(entry[0]), int(entry[1])


def choice_distribution(
    output: dict,
    labels: Sequence[str],
    choice_token_ids: Sequence[int],
) -> dict:
    meta = output.get("meta_info", {})
    positions = meta.get("output_token_ids_logprobs")
    source = "selected"
    if not positions:
        positions = meta.get("output_top_logprobs")
        source = "top"
    if not positions or not positions[0]:
        raise ValueError("Listwise verifier returned no choice-token logprobs.")
    by_id = {
        token_id: logprob
        for logprob, token_id in (
            entry_logprob_and_id(entry) for entry in positions[0]
        )
    }
    missing = [token_id for token_id in choice_token_ids if token_id not in by_id]
    if missing:
        raise ValueError(f"Listwise output is missing choice token IDs: {missing}.")
    logprobs = [by_id[token_id] for token_id in choice_token_ids]
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    denominator = sum(weights)
    probabilities = {
        label: weight / denominator for label, weight in zip(labels, weights)
    }
    log_mass = maximum + math.log(denominator)
    generated_token_id = (
        int(output["output_ids"][0]) if output.get("output_ids") else None
    )
    token_to_label = dict(zip(choice_token_ids, labels))
    return {
        "probabilities": probabilities,
        "choice_token_mass": math.exp(min(log_mass, 0.0)),
        "logprob_source": source,
        "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
        "generated_token_id": generated_token_id,
        "generated_label": token_to_label.get(generated_token_id),
    }


def majority_prediction(group: Sequence[dict], sample_order: Sequence[int]) -> str | None:
    by_sample = {int(row["sample_id"]): row for row in group}
    answers = [
        by_sample[sample_id]["extracted_answer"]
        for sample_id in sample_order
        if by_sample[sample_id]["extracted_answer"] is not None
    ]
    return Counter(answers).most_common(1)[0][0] if answers else None


def aggregate_rankings(
    trajectories: Sequence[dict], jobs: Sequence[dict], score_rows: Sequence[dict]
) -> list[dict]:
    trajectories_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectories_by_problem[str(row["problem_id"])].append(row)
    job_by_id = {str(row["job_id"]): row for row in jobs}
    score_by_id = {str(row["job_id"]): row for row in score_rows}
    if set(job_by_id) != set(score_by_id):
        raise ValueError("Listwise score rows do not match jobs.")

    problems = []
    for problem_id, group in trajectories_by_problem.items():
        job = job_by_id[problem_id]
        score = score_by_id[problem_id]
        probabilities = {
            str(label): float(value)
            for label, value in score["probabilities"].items()
        }
        ranked_labels = sorted(probabilities, key=lambda x: (-probabilities[x], x))
        ranked_samples = [int(job["label_to_sample"][label]) for label in ranked_labels]
        by_sample = {int(row["sample_id"]): row for row in group}
        selected_sample = ranked_samples[0]
        gold = group[0]["gold_answer"]
        top_k_correct = {}
        for k in range(1, len(group) + 1):
            prediction = majority_prediction(group, ranked_samples[:k])
            top_k_correct[str(k)] = bool(prediction == gold)
        sample_order = sorted(by_sample)
        self_consistency = majority_prediction(group, sample_order)
        problems.append(
            {
                "problem_id": problem_id,
                "label_to_sample": job["label_to_sample"],
                "probabilities": probabilities,
                "ranking_labels": ranked_labels,
                "ranking": ranked_samples,
                "selected_label": ranked_labels[0],
                "selected_sample_id": selected_sample,
                "selected_correct": bool(by_sample[selected_sample]["correct"]),
                "top_k_majority_correct": top_k_correct,
                "self_consistency_correct": bool(self_consistency == gold),
                "oracle_correct": any(bool(row["correct"]) for row in group),
                "generated_label": score.get("generated_label"),
            }
        )
    return problems


def attach_pointwise_outcomes(problems: Sequence[dict], score_rows: Sequence[dict]) -> None:
    terminal_by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in score_rows:
        if row.get("terminal"):
            terminal_by_problem[str(row["problem_id"])].append(row)
    for problem in problems:
        candidates = terminal_by_problem.get(str(problem["problem_id"]), [])
        if not candidates:
            raise ValueError(f"No terminal pointwise scores for {problem['problem_id']}.")
        selected = min(
            candidates,
            key=lambda row: (-float(row["score"]), int(row["sample_id"])),
        )
        problem["pointwise_terminal_selected_correct"] = bool(selected["correct"])


def attach_pairwise_outcomes(problems: Sequence[dict], rankings: Sequence[dict]) -> None:
    by_problem = {str(row["problem_id"]): row for row in rankings}
    for problem in problems:
        reference = by_problem.get(str(problem["problem_id"]))
        if reference is None:
            raise ValueError(f"No pairwise ranking for {problem['problem_id']}.")
        problem["pairwise_selected_correct"] = bool(reference["selected_correct"])
        if "knockout_selected_correct" in reference:
            problem["knockout_selected_correct"] = bool(
                reference["knockout_selected_correct"]
            )


def percentile_interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return [low, high]


def bootstrap_metrics(
    problems: Sequence[dict], *, samples: int, seed: int
) -> dict[str, list[float]]:
    if samples <= 0 or len(problems) < 2:
        return {}
    rng = random.Random(seed)
    boot: dict[str, list[float]] = defaultdict(list)
    comparisons = {
        "self_consistency": "self_consistency_correct",
        "pointwise_terminal_bon": "pointwise_terminal_selected_correct",
        "pairwise_all_pairs": "pairwise_selected_correct",
        "pairwise_knockout": "knockout_selected_correct",
    }
    for _ in range(samples):
        drawn = [rng.choice(problems) for _ in problems]
        boot["listwise_selection_accuracy"].append(
            mean([float(row["selected_correct"]) for row in drawn])
        )
        for name, field in comparisons.items():
            if field in drawn[0]:
                boot[f"listwise_minus_{name}"].append(
                    mean(
                        [
                            float(row["selected_correct"]) - float(row[field])
                            for row in drawn
                        ]
                    )
                )
    return {key: percentile_interval(values) for key, values in boot.items()}


def comparison_metrics(
    problems: Sequence[dict], name: str, reference_field: str
) -> dict:
    return {
        f"listwise_minus_{name}": mean(
            [
                float(row["selected_correct"]) - float(row[reference_field])
                for row in problems
            ]
        ),
        f"listwise_vs_{name}_wins": sum(
            row["selected_correct"] and not row[reference_field]
            for row in problems
        ),
        f"listwise_vs_{name}_losses": sum(
            not row["selected_correct"] and row[reference_field]
            for row in problems
        ),
    }


def summarize(
    trajectories: Sequence[dict],
    score_rows: Sequence[dict],
    problems: Sequence[dict],
    *,
    scorer: str,
    labels: Sequence[str],
    choice_token_ids: Sequence[int],
    wall_time: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    n_candidates = len(labels)
    baselines = {
        "self_consistency_accuracy": mean(
            [float(row["self_consistency_correct"]) for row in problems]
        ),
        "oracle_pass_at_n": mean([float(row["oracle_correct"]) for row in problems]),
    }
    metrics = {
        "listwise_selection_accuracy": mean(
            [float(row["selected_correct"]) for row in problems]
        ),
        "selection_by_top_k_majority": {
            str(k): mean(
                [float(row["top_k_majority_correct"][str(k)]) for row in problems]
            )
            for k in range(1, n_candidates + 1)
        },
        "selected_label_counts": dict(Counter(row["selected_label"] for row in problems)),
        "generated_label_counts": dict(
            Counter(row["generated_label"] or "<other>" for row in problems)
        ),
    }
    references = {
        "self_consistency": "self_consistency_correct",
        "pointwise_terminal_bon": "pointwise_terminal_selected_correct",
        "pairwise_all_pairs": "pairwise_selected_correct",
        "pairwise_knockout": "knockout_selected_correct",
    }
    baseline_names = {
        "pointwise_terminal_bon": "pointwise_terminal_bon_accuracy",
        "pairwise_all_pairs": "pairwise_all_pairs_accuracy",
        "pairwise_knockout": "pairwise_knockout_accuracy",
    }
    for name, field in references.items():
        if field in problems[0]:
            metrics.update(comparison_metrics(problems, name, field))
            if name in baseline_names:
                baselines[baseline_names[name]] = mean(
                    [float(row[field]) for row in problems]
                )
    metrics["bootstrap_95_ci"] = bootstrap_metrics(
        problems, samples=bootstrap_samples, seed=seed
    )
    return {
        "schema_version": 1,
        "experiment": {
            "method": "single_call_randomized_listwise_exact_label_distribution",
            "scorer_model": scorer,
            "choice_labels": list(labels),
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
        "metrics": metrics,
        "baselines": baselines,
        "cost": {
            "wall_time_s": wall_time,
            "verifier_calls": len(score_rows),
            "verifier_prompt_tokens": sum(row["prompt_tokens"] for row in score_rows),
            "verifier_completion_tokens": sum(
                row["completion_tokens"] for row in score_rows
            ),
            "mean_prompt_tokens_per_call": mean(
                [row["prompt_tokens"] for row in score_rows]
            ),
            "calls_per_s": len(score_rows) / wall_time if wall_time else None,
            "mean_choice_token_mass": mean(
                [row["choice_token_mass"] for row in score_rows]
            ),
            "selected_logprob_coverage": mean(
                [float(row["logprob_source"] == "selected") for row in score_rows]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--pointwise-scores", default=None)
    parser.add_argument("--pairwise-rankings", default=None)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-gpu-id", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-mamba-cache-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-scores", required=True)
    parser.add_argument("--save-rankings", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--probe", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = select_problem_rows(load_jsonl(args.trajectories), args.max_problems)
    validate_trajectories(trajectories)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    candidate_count = len(trajectories) // len(
        {str(row["problem_id"]) for row in trajectories}
    )
    labels = labels_for_count(candidate_count)
    choice_token_ids = resolve_choice_token_ids(tokenizer, labels)
    jobs = make_jobs(trajectories, tokenizer, seed=args.seed)
    if args.probe:
        jobs = jobs[:1]
    print(
        f"problems={len(jobs)} trajectories={len(trajectories)} "
        f"calls={len(jobs)} scorer={args.scorer}",
        flush=True,
    )
    print(
        "choice token IDs: "
        + " ".join(f"{label}={token_id}" for label, token_id in zip(labels, choice_token_ids)),
        flush=True,
    )
    if args.dry_run:
        print(jobs[0]["prompt"])
        return None

    if args.reuse_scores:
        score_rows = load_jsonl(args.save_scores)
        expected_jobs = {job["job_id"] for job in jobs}
        observed_jobs = {row["job_id"] for row in score_rows}
        if expected_jobs != observed_jobs:
            raise ValueError(
                "Saved listwise scores do not match requested jobs: "
                f"missing={len(expected_jobs - observed_jobs)} "
                f"extra={len(observed_jobs - expected_jobs)}."
            )
        try:
            summary_text = Path(args.summary_output).read_text()
        except FileNotFoundError:
            summary_text = ""
        try:
            previous_summary = json.loads(summary_text) if summary_text else {}
        except json.JSONDecodeError:
            wall_time_match = re.search(
                r'"wall_time_s"\s*:\s*([0-9.eE+-]+)', summary_text
            )
            previous_summary = {
                "cost": {
                    "wall_time_s": (
                        float(wall_time_match.group(1)) if wall_time_match else 0.0
                    )
                }
            }
        wall_time = float(previous_summary.get("cost", {}).get("wall_time_s", 0.0))
        print(f"reusing {len(score_rows)} saved listwise calls", flush=True)
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
        score_rows = []
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
                    score_rows.append(
                        {
                            "schema_version": 1,
                            **{key: value for key, value in job.items() if key != "prompt"},
                            **choice_distribution(output, labels, choice_token_ids),
                            "scorer_model": args.scorer,
                        }
                    )
                elapsed = time.perf_counter() - started
                print(
                    f"scored={len(score_rows)}/{len(jobs)} "
                    f"calls/s={len(score_rows) / elapsed:.2f}",
                    flush=True,
                )
                if args.probe:
                    print(json.dumps(outputs[0].get("meta_info", {}), indent=2)[:5000])
        finally:
            engine.shutdown()
        wall_time = time.perf_counter() - started
        write_jsonl(args.save_scores, score_rows)

    if args.probe:
        probe = {
            "schema_version": 1,
            "probe": True,
            "choice_labels": labels,
            "choice_token_ids": choice_token_ids,
            "scores": score_rows,
            "wall_time_s": wall_time,
        }
        write_json(args.summary_output, probe)
        return probe

    problems = aggregate_rankings(trajectories, jobs, score_rows)
    if args.pointwise_scores:
        attach_pointwise_outcomes(problems, load_jsonl(args.pointwise_scores))
    if args.pairwise_rankings:
        attach_pairwise_outcomes(problems, load_jsonl(args.pairwise_rankings))
    summary = summarize(
        trajectories,
        score_rows,
        problems,
        scorer=args.scorer,
        labels=labels,
        choice_token_ids=choice_token_ids,
        wall_time=wall_time,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_jsonl(args.save_rankings, problems)
    write_json(args.summary_output, summary)
    print("\nTerminal one-call listwise LLM-as-a-Verifier")
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["baselines"], indent=2))
    print(json.dumps(summary["cost"], indent=2))
    print(f"wrote {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

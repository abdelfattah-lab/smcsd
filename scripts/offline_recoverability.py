"""Semantic verifier validation on exact generator-token prefixes.

This is the Phase-2 gate before implementing online semantic SMC. It consumes
self-contained trajectory JSONL emitted by accuracy_test_gsm8k.py or
accuracy_test_math500.py and:

* cuts prefixes at exact generator-token positions;
* obtains every configured score-token logprob from an LLM verifier;
* computes a continuous expected recoverability score in [0, 1];
* reports eventual-correctness AUROC/AUPRC, calibration, within-problem
  ranking, top-half correct-path survival, terminal Best-of-N, and cost.

The main path uses one clean pointwise prompt. Paper-style pairwise ranking is
a separate terminal baseline; online ParticleScale requires pointwise prefix
potentials whose score differences telescope through time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SCORE_LABELS = tuple("ABCDEFGHIJKLMNOPQRST")

RECOVERABILITY_TEMPLATE = """\
You are verifying an unfinished solution to a math problem.

Problem:
{problem}

Partial solution:
{prefix}

Criterion:
Estimate the probability that this partial solution can still be completed
into a fully correct solution. Account for irreversible logical, factual, or
arithmetic errors already present. Do not penalize the solution merely because
it is incomplete.

Score scale:
{scale}

Return exactly one score label and no other text.
Score:\
"""


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def parse_fractions(value: str) -> list[float]:
    fractions = [float(item) for item in value.split(",") if item.strip()]
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise argparse.ArgumentTypeError(
            "fractions must be a comma-separated list in (0, 1]"
        )
    if len(set(fractions)) != len(fractions):
        raise argparse.ArgumentTypeError("fractions must be unique")
    return sorted(fractions)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}.")
    return rows


def validate_trajectory_rows(rows: Sequence[dict]) -> None:
    required = {
        "problem_id",
        "problem",
        "sample_id",
        "generator_model",
        "generator_output_ids",
        "full_text",
        "extracted_answer",
        "gold_answer",
        "correct",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(
                f"Trajectory row {index} is missing required fields: {missing}. "
                "Regenerate it with the schema-v2 benchmark dump."
            )
        if not row["generator_output_ids"]:
            raise ValueError(f"Trajectory row {index} has no generator output IDs.")
        if bool(row["correct"]) != (
            row["extracted_answer"] == row["gold_answer"]
        ):
            raise ValueError(f"Trajectory row {index} has inconsistent correctness.")
    models = {row["generator_model"] for row in rows}
    if len(models) != 1:
        raise ValueError(
            "One semantic study must use exactly one generator model; got "
            f"{sorted(models)}."
        )


def score_scale(labels: Sequence[str]) -> tuple[list[float], str]:
    if len(labels) < 2:
        raise ValueError("At least two ordered score labels are required.")
    values = [index / (len(labels) - 1) for index in range(len(labels))]
    descriptions = ", ".join(
        f"{label}={round(value * 100):d}%" for label, value in zip(labels, values)
    )
    return values, descriptions


def resolve_score_token_ids(tokenizer, labels: Sequence[str]) -> list[int]:
    token_ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Score label {label!r} is not one token for the verifier: "
                f"{encoded}. Choose a single-token ordered label set."
            )
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Score labels do not map to unique verifier token IDs.")
    return token_ids


def make_prefix_jobs(
    rows: Sequence[dict],
    generator_tokenizer,
    fractions: Sequence[float],
    *,
    min_prefix_tokens: int = 8,
) -> list[dict]:
    """Create one job per trajectory/fraction using generator token IDs."""
    jobs = []
    for trajectory_index, row in enumerate(rows):
        output_ids = list(row["generator_output_ids"])
        n_tokens = len(output_ids)
        seen_positions: set[int] = set()
        for fraction in fractions:
            position = min(
                n_tokens,
                max(min_prefix_tokens, math.ceil(n_tokens * fraction)),
            )
            if position in seen_positions:
                continue
            seen_positions.add(position)
            prefix = generator_tokenizer.decode(
                output_ids[:position], skip_special_tokens=True
            )
            jobs.append(
                {
                    "trajectory_index": trajectory_index,
                    "problem_id": row["problem_id"],
                    "sample_id": row["sample_id"],
                    "problem": row["problem"],
                    "prefix": prefix,
                    "requested_fraction": float(fraction),
                    "token_position": position,
                    "trajectory_tokens": n_tokens,
                    "actual_fraction": position / n_tokens,
                    "terminal": position == n_tokens,
                    "correct": bool(row["correct"]),
                    "extracted_answer": row["extracted_answer"],
                    "gold_answer": row["gold_answer"],
                }
            )
    return jobs


def build_verifier_prompt(
    tokenizer,
    *,
    problem: str,
    prefix: str,
    labels: Sequence[str],
) -> str:
    _, scale = score_scale(labels)
    content = RECOVERABILITY_TEMPLATE.format(
        problem=problem,
        prefix=prefix,
        scale=scale,
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
            [{"role": "user", "content": content}],
            **kwargs,
        )


def _entry_logprob_and_id(entry) -> tuple[float, int]:
    if isinstance(entry, dict):
        return float(entry["logprob"]), int(entry["token_id"])
    return float(entry[0]), int(entry[1])


def expected_score_from_output(
    output: dict,
    score_token_ids: Sequence[int],
    score_values: Sequence[float],
) -> dict:
    """Extract the exact configured-token distribution from an SGLang result."""
    meta = output.get("meta_info", {})
    positions = meta.get("output_token_ids_logprobs")
    source = "selected"
    if not positions:
        positions = meta.get("output_top_logprobs")
        source = "top"
    if not positions or not positions[0]:
        raise ValueError("Verifier output contains no score-token logprobs.")

    by_id = {
        token_id: logprob
        for logprob, token_id in (
            _entry_logprob_and_id(entry) for entry in positions[0]
        )
    }
    missing = [
        token_id for token_id in score_token_ids if token_id not in by_id
    ]
    if missing:
        raise ValueError(
            f"Verifier output is missing {len(missing)} configured score tokens "
            f"from {source} logprobs: {missing}."
        )

    logprobs = [by_id[token_id] for token_id in score_token_ids]
    max_logprob = max(logprobs)
    unnormalized = [math.exp(logprob - max_logprob) for logprob in logprobs]
    normalizer = sum(unnormalized)
    probabilities = [value / normalizer for value in unnormalized]
    expected = sum(
        probability * value
        for probability, value in zip(probabilities, score_values)
    )
    log_mass = max_logprob + math.log(normalizer)
    token_mass = math.exp(min(log_mass, 0.0))
    return {
        "score": expected,
        "score_probabilities": probabilities,
        "score_token_mass": token_mass,
        "logprob_source": source,
        "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
    }


def binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC with half credit for ties."""
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = (start + end + 1) / 2.0
        rank_sum += average_rank * sum(
            bool(pairs[index][1]) for index in range(start, end)
        )
        start = end
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Threshold-grouped average precision, with prevalence as chance level."""
    positives = sum(bool(label) for label in labels)
    if positives == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    previous_true_positives = 0
    area = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        true_positives += sum(bool(label) for _, label in pairs[start:end])
        recall_gain = (true_positives - previous_true_positives) / positives
        area += recall_gain * (true_positives / end)
        previous_true_positives = true_positives
        start = end
    return area


def within_problem_ranking_accuracy(records: Sequence[dict]) -> tuple[float, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["problem_id"])].append(record)
    wins = 0.0
    comparisons = 0
    for group in groups.values():
        correct = [record["score"] for record in group if record["correct"]]
        incorrect = [
            record["score"] for record in group if not record["correct"]
        ]
        for positive in correct:
            for negative in incorrect:
                comparisons += 1
                wins += (
                    1.0
                    if positive > negative
                    else 0.5 if positive == negative else 0.0
                )
    return (
        wins / comparisons if comparisons else float("nan"),
        comparisons,
    )


def top_fraction_survival(
    records: Sequence[dict], fraction: float = 0.5
) -> tuple[float, float, int]:
    """Observed and random-baseline chance of retaining a correct path."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["problem_id"])].append(record)
    observed = []
    random_baselines = []
    for group in groups.values():
        n = len(group)
        correct = sum(bool(record["correct"]) for record in group)
        if correct == 0:
            continue
        keep = max(1, math.ceil(n * fraction))
        ranked = sorted(
            group,
            key=lambda record: (-record["score"], int(record["sample_id"])),
        )
        observed.append(any(record["correct"] for record in ranked[:keep]))
        if keep > n - correct:
            random_baselines.append(1.0)
        else:
            random_baselines.append(
                1.0 - math.comb(n - correct, keep) / math.comb(n, keep)
            )
    return (
        _mean([float(value) for value in observed]),
        _mean(random_baselines),
        len(observed),
    )


def calibration_curve(
    scores: Sequence[float], labels: Sequence[bool], n_bins: int = 10
) -> tuple[list[dict], float]:
    bins = []
    expected_calibration_error = 0.0
    total = len(scores)
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = [
            (score, bool(label))
            for score, label in zip(scores, labels)
            if lower <= score < upper or (index == n_bins - 1 and score == 1)
        ]
        if not members:
            continue
        mean_score = _mean([score for score, _ in members])
        accuracy = _mean([float(label) for _, label in members])
        expected_calibration_error += (
            len(members) / total * abs(mean_score - accuracy)
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_score": mean_score,
                "accuracy": accuracy,
            }
        )
    return bins, expected_calibration_error


def summarize_checkpoint(records: Sequence[dict]) -> dict:
    scores = [float(record["score"]) for record in records]
    labels = [bool(record["correct"]) for record in records]
    positives = sum(labels)
    ranking_accuracy, comparisons = within_problem_ranking_accuracy(records)
    survival, random_survival, eligible = top_fraction_survival(records)
    calibration, ece = calibration_curve(scores, labels)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["problem_id"])].append(record)
    top1 = []
    for group in groups.values():
        pick = min(
            group,
            key=lambda record: (-record["score"], int(record["sample_id"])),
        )
        top1.append(float(bool(pick["correct"])))
    return {
        "n_prefixes": len(records),
        "n_problems": len(groups),
        "prevalence": positives / len(records),
        "auroc": binary_auroc(scores, labels),
        "auprc": average_precision(scores, labels),
        "brier": _mean(
            [(score - float(label)) ** 2 for score, label in zip(scores, labels)]
        ),
        "ece_10": ece,
        "mean_score": _mean(scores),
        "mean_score_correct": _mean(
            [score for score, label in zip(scores, labels) if label]
        ),
        "mean_score_incorrect": _mean(
            [score for score, label in zip(scores, labels) if not label]
        ),
        "within_problem_ranking_accuracy": ranking_accuracy,
        "within_problem_comparisons": comparisons,
        "top_half_correct_survival": survival,
        "top_half_random_baseline": random_survival,
        "top_half_survival_lift": survival - random_survival,
        "top_half_eligible_problems": eligible,
        "top1_eventual_correct_rate": _mean(top1),
        "calibration": calibration,
    }


def bootstrap_intervals(
    records: Sequence[dict],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["problem_id"])].append(record)
    problem_ids = sorted(groups)
    if len(problem_ids) < 2:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        resampled = []
        for draw_index in range(len(problem_ids)):
            problem_id = rng.choice(problem_ids)
            for record in groups[problem_id]:
                clone = dict(record)
                clone["problem_id"] = f"{draw_index}:{problem_id}"
                resampled.append(clone)
        summary = summarize_checkpoint(resampled)
        for key in (
            "auroc",
            "auprc",
            "within_problem_ranking_accuracy",
            "top_half_correct_survival",
            "top_half_survival_lift",
            "top1_eventual_correct_rate",
        ):
            value = float(summary[key])
            if math.isfinite(value):
                values[key].append(value)

    intervals = {}
    for key, observed in values.items():
        ordered = sorted(observed)
        if not ordered:
            continue
        lo = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
        hi = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
        intervals[key] = [lo, hi]
    return intervals


def selection_baselines(trajectories: Sequence[dict], terminal: Sequence[dict]) -> dict:
    trajectory_groups: dict[str, list[dict]] = defaultdict(list)
    for row in trajectories:
        trajectory_groups[str(row["problem_id"])].append(row)
    terminal_groups: dict[str, list[dict]] = defaultdict(list)
    for row in terminal:
        terminal_groups[str(row["problem_id"])].append(row)

    majority_correct = 0
    terminal_bon_correct = 0
    oracle_correct = 0
    invalid_majority = 0
    for problem_id, group in trajectory_groups.items():
        answers = [
            row["extracted_answer"]
            for row in sorted(group, key=lambda row: int(row["sample_id"]))
            if row["extracted_answer"] is not None
        ]
        votes = Counter(answers)
        if votes:
            prediction = votes.most_common(1)[0][0]
            majority_correct += prediction == group[0]["gold_answer"]
        else:
            invalid_majority += 1
        oracle_correct += any(bool(row["correct"]) for row in group)

        scored = terminal_groups.get(problem_id, [])
        if scored:
            pick = min(
                scored,
                key=lambda row: (-row["score"], int(row["sample_id"])),
            )
            terminal_bon_correct += bool(pick["correct"])

    n = len(trajectory_groups)
    return {
        "n_problems": n,
        "self_consistency_accuracy": majority_correct / n,
        "self_consistency_invalid": invalid_majority,
        "terminal_pointwise_bon_accuracy": terminal_bon_correct / n,
        "oracle_pass_at_n": oracle_correct / n,
    }


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--generator-tokenizer", default=None)
    parser.add_argument(
        "--fractions",
        type=parse_fractions,
        default=parse_fractions("0.25,0.5,0.75,1.0"),
    )
    parser.add_argument("--score-labels", default=",".join(DEFAULT_SCORE_LABELS))
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--min-prefix-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-gpu-id", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-mamba-cache-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--save-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Score only two prefixes and print result metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = load_jsonl(args.trajectories)
    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]
    validate_trajectory_rows(trajectories)
    generator_model = args.generator_tokenizer or trajectories[0]["generator_model"]
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
    verifier_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    labels = [label.strip() for label in args.score_labels.split(",") if label.strip()]
    score_values, scale_description = score_scale(labels)
    score_token_ids = resolve_score_token_ids(verifier_tokenizer, labels)

    jobs = make_prefix_jobs(
        trajectories,
        generator_tokenizer,
        args.fractions,
        min_prefix_tokens=args.min_prefix_tokens,
    )
    for job in jobs:
        job["verifier_prompt"] = build_verifier_prompt(
            verifier_tokenizer,
            problem=job["problem"],
            prefix=job["prefix"],
            labels=labels,
        )
    if args.probe:
        jobs = jobs[:2]

    print(
        f"trajectories={len(trajectories)} prefixes={len(jobs)} "
        f"generator={generator_model} verifier={args.scorer}",
        flush=True,
    )
    print(f"score labels: {dict(zip(labels, score_token_ids))}", flush=True)
    if args.dry_run:
        print(jobs[0]["verifier_prompt"])
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
    )
    if args.tp > 1:
        engine_kwargs.update(
            tp_size=args.tp,
            disable_custom_all_reduce=True,
            enforce_disable_flashinfer_allreduce_fusion=True,
        )

    scored = []
    started = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    try:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start : start + args.batch_size]
            outputs = engine.generate(
                [job["verifier_prompt"] for job in batch],
                {"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=score_token_ids,
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            for job, output in zip(batch, outputs):
                extracted = expected_score_from_output(
                    output, score_token_ids, score_values
                )
                record = {
                    "schema_version": 1,
                    **{key: value for key, value in job.items() if key != "verifier_prompt"},
                    **extracted,
                    "scorer_model": args.scorer,
                    "score_labels": labels,
                    "score_token_ids": score_token_ids,
                }
                scored.append(record)
            elapsed = time.perf_counter() - started
            print(
                f"scored={len(scored)}/{len(jobs)} "
                f"prefixes/s={len(scored) / elapsed:.2f}",
                flush=True,
            )
            if args.probe:
                print(json.dumps(outputs[0].get("meta_info", {}), indent=2)[:3000])
    finally:
        engine.shutdown()
    wall_time = time.perf_counter() - started

    write_jsonl(args.save_scores, scored)
    by_fraction: dict[float, list[dict]] = defaultdict(list)
    for record in scored:
        by_fraction[float(record["requested_fraction"])].append(record)
    checkpoint_summaries = {}
    for fraction, records in sorted(by_fraction.items()):
        checkpoint = summarize_checkpoint(records)
        checkpoint["bootstrap_95_ci"] = bootstrap_intervals(
            records,
            samples=args.bootstrap_samples,
            seed=args.seed + round(fraction * 1000),
        )
        checkpoint_summaries[str(fraction)] = checkpoint

    terminal_records = [record for record in scored if record["terminal"]]
    summary = {
        "schema_version": 1,
        "experiment": {
            "trajectories": str(args.trajectories),
            "generator_model": generator_model,
            "scorer_model": args.scorer,
            "fractions": args.fractions,
            "score_labels": labels,
            "score_token_ids": score_token_ids,
            "score_scale": scale_description,
            "criterion": "recoverability",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "dataset": {
            "name": trajectories[0].get("dataset"),
            "config": trajectories[0].get("dataset_config"),
            "split": trajectories[0].get("split"),
            "n_trajectories": len(trajectories),
            "n_problems": len({row["problem_id"] for row in trajectories}),
        },
        "checkpoints": checkpoint_summaries,
        "selection_baselines": selection_baselines(
            trajectories, terminal_records
        ) if terminal_records else None,
        "cost": {
            "wall_time_s": wall_time,
            "prefixes_scored": len(scored),
            "prefixes_per_s": len(scored) / wall_time,
            "verifier_prompt_tokens": sum(
                record["prompt_tokens"] for record in scored
            ),
            "verifier_completion_tokens": sum(
                record["completion_tokens"] for record in scored
            ),
            "mean_prompt_tokens_per_prefix": _mean(
                [record["prompt_tokens"] for record in scored]
            ),
            "mean_score_token_mass": _mean(
                [record["score_token_mass"] for record in scored]
            ),
            "selected_logprob_coverage": _mean(
                [
                    float(record["logprob_source"] == "selected")
                    for record in scored
                ]
            ),
        },
    }
    write_json(args.summary_output, summary)

    print("\nSemantic verifier validation")
    print("fraction  AUROC  AUPRC  rank-acc  top-half  top1  Brier  ECE")
    for fraction, checkpoint in checkpoint_summaries.items():
        print(
            f"{float(fraction):7.2f}  "
            f"{checkpoint['auroc']:5.3f}  "
            f"{checkpoint['auprc']:5.3f}  "
            f"{checkpoint['within_problem_ranking_accuracy']:8.3f}  "
            f"{checkpoint['top_half_correct_survival']:8.3f}  "
            f"{checkpoint['top1_eventual_correct_rate']:4.3f}  "
            f"{checkpoint['brier']:5.3f}  "
            f"{checkpoint['ece_10']:5.3f}"
        )
    print(json.dumps(summary["selection_baselines"], indent=2))
    print(f"wrote {args.save_scores} and {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

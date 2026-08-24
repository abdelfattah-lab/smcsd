"""Pairwise-informed semantic error audit on exact generator-token prefixes.

This remains a pointwise verifier: every prefix is scored independently and
no generator likelihood or sibling candidate is visible.  The rubric distills
the behavior that made order-swapped pairwise judging effective: explicitly
look for substantive mathematical errors before rewarding polish or length.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

try:
    from scripts import offline_recoverability as recoverability
except ModuleNotFoundError:
    import offline_recoverability as recoverability

_mean = recoverability._mean
bootstrap_intervals = recoverability.bootstrap_intervals
expected_score_from_output = recoverability.expected_score_from_output
load_jsonl = recoverability.load_jsonl
make_prefix_jobs = recoverability.make_prefix_jobs
resolve_score_token_ids = recoverability.resolve_score_token_ids
score_scale = recoverability.score_scale
selection_baselines = recoverability.selection_baselines
summarize_checkpoint = recoverability.summarize_checkpoint
validate_trajectory_rows = recoverability.validate_trajectory_rows
write_json = recoverability.write_json
write_jsonl = recoverability.write_jsonl


ERROR_AUDIT_LABELS = tuple("ABCDE")

ERROR_AUDIT_TEMPLATE = """\
You are independently auditing one partial solution to a math problem.

Problem:
{problem}

Partial solution:
{prefix}

Audit criterion:
Check the mathematical claims, equations, deductions, and any stated final
answer. Focus on substantive errors that make the current reasoning path
unlikely to yield a fully correct solution. Do not reward confidence, polish,
length, or merely matching a familiar-looking answer.

If the solution is unfinished, do not penalize missing later work. Judge only
whether the work already written remains viable: an omitted step can be added,
but a committed logical, factual, or arithmetic error counts against it. If
the solution is complete, require both a correct final answer and reasoning
without a substantive error.

Score the probability that the current reasoning path is still free of a
substantive error and can yield a fully correct solution:
{scale}

You must return exactly one of these labels: {labels}. Do not explain your
choice and do not output any other text.
Score:\
"""


def build_error_audit_prompt(
    tokenizer,
    *,
    problem: str,
    prefix: str,
    labels: Sequence[str] = ERROR_AUDIT_LABELS,
) -> str:
    _, scale = score_scale(labels)
    content = ERROR_AUDIT_TEMPLATE.format(
        problem=problem,
        prefix=prefix,
        scale=scale,
        labels=", ".join(labels),
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


def score_key(row: dict) -> tuple[str, int, str, int]:
    return (
        str(row["problem_id"]),
        int(row["sample_id"]),
        str(row.get("checkpoint", row.get("requested_fraction"))),
        int(row["token_position"]),
    )


def make_fixed_token_jobs(
    rows: Sequence[dict],
    generator_tokenizer,
    token_positions: Sequence[int],
    *,
    include_terminal: bool,
    min_prefix_tokens: int = 8,
) -> list[dict]:
    """Create online-style jobs at the same absolute token horizon."""
    jobs = []
    for trajectory_index, row in enumerate(rows):
        output_ids = list(row["generator_output_ids"])
        n_tokens = len(output_ids)
        for requested_position in token_positions:
            position = min(n_tokens, max(min_prefix_tokens, requested_position))
            jobs.append(
                {
                    "trajectory_index": trajectory_index,
                    "problem_id": row["problem_id"],
                    "sample_id": row["sample_id"],
                    "problem": row["problem"],
                    "prefix": generator_tokenizer.decode(
                        output_ids[:position], skip_special_tokens=True
                    ),
                    "checkpoint": f"token_{requested_position}",
                    "requested_token_position": requested_position,
                    "token_position": position,
                    "trajectory_tokens": n_tokens,
                    "actual_fraction": position / n_tokens,
                    "terminal": position == n_tokens,
                    "correct": bool(row["correct"]),
                    "extracted_answer": row["extracted_answer"],
                    "gold_answer": row["gold_answer"],
                }
            )
        if include_terminal:
            jobs.append(
                {
                    "trajectory_index": trajectory_index,
                    "problem_id": row["problem_id"],
                    "sample_id": row["sample_id"],
                    "problem": row["problem"],
                    "prefix": generator_tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ),
                    "checkpoint": "terminal",
                    "requested_token_position": None,
                    "token_position": n_tokens,
                    "trajectory_tokens": n_tokens,
                    "actual_fraction": 1.0,
                    "terminal": True,
                    "correct": bool(row["correct"]),
                    "extracted_answer": row["extracted_answer"],
                    "gold_answer": row["gold_answer"],
                }
            )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--generator-tokenizer", default=None)
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--token-positions",
        default=None,
        help="Use fixed absolute generator-token checkpoints, e.g. 512,1024,2048.",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="With --token-positions, also score every completed trajectory.",
    )
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--sample-ids", type=parse_sample_ids, default=None)
    parser.add_argument("--min-prefix-tokens", type=int, default=8)
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
    parser.add_argument("--save-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    return parser


def parse_fractions(value: str) -> list[float]:
    fractions = sorted(float(item) for item in value.split(",") if item.strip())
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("fractions must be a comma-separated list in (0, 1].")
    return fractions


def parse_token_positions(value: str) -> list[int]:
    positions = sorted({int(item) for item in value.split(",") if item.strip()})
    if not positions or any(position <= 0 for position in positions):
        raise ValueError("token positions must be comma-separated positive integers.")
    return positions


def parse_sample_ids(value: str) -> list[int]:
    sample_ids = sorted({int(item) for item in value.split(",") if item.strip()})
    if not sample_ids or any(sample_id < 0 for sample_id in sample_ids):
        raise ValueError("sample IDs must be comma-separated non-negative integers.")
    return sample_ids


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = load_jsonl(args.trajectories)
    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]
    if args.sample_ids is not None:
        allowed_sample_ids = set(args.sample_ids)
        trajectories = [
            row
            for row in trajectories
            if int(row["sample_id"]) in allowed_sample_ids
        ]
    validate_trajectory_rows(trajectories)
    generator_model = args.generator_tokenizer or trajectories[0]["generator_model"]
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
    verifier_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    labels = list(ERROR_AUDIT_LABELS)
    score_values, scale_description = score_scale(labels)
    score_token_ids = resolve_score_token_ids(verifier_tokenizer, labels)
    if args.token_positions:
        token_positions = parse_token_positions(args.token_positions)
        fractions = None
        jobs = make_fixed_token_jobs(
            trajectories,
            generator_tokenizer,
            token_positions,
            include_terminal=args.include_terminal,
            min_prefix_tokens=args.min_prefix_tokens,
        )
        checkpoint_mode = "fixed_generator_tokens"
    else:
        fractions = parse_fractions(args.fractions)
        token_positions = None
        jobs = make_prefix_jobs(
            trajectories,
            generator_tokenizer,
            fractions,
            min_prefix_tokens=args.min_prefix_tokens,
        )
        checkpoint_mode = "trajectory_fraction"
    for job in jobs:
        job["verifier_prompt"] = build_error_audit_prompt(
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
    print(f"error-audit labels: {dict(zip(labels, score_token_ids))}", flush=True)
    if args.dry_run:
        print(jobs[0]["verifier_prompt"])
        return None

    if args.reuse_scores:
        scored = load_jsonl(args.save_scores)
        expected = {score_key(job) for job in jobs}
        observed = {score_key(row) for row in scored}
        if expected != observed:
            raise ValueError(
                "Saved audit scores do not match jobs: "
                f"missing={len(expected - observed)} extra={len(observed - expected)}."
            )
        try:
            previous = json.loads(Path(args.summary_output).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        wall_time = float(previous.get("cost", {}).get("wall_time_s", 0.0))
        initialization_time = float(
            previous.get("cost", {}).get("engine_initialization_time_s", 0.0)
        )
        inference_wall_time = float(
            previous.get("cost", {}).get("inference_wall_time_s", wall_time)
        )
        print(f"reusing {len(scored)} saved audit scores", flush=True)
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

        scored = []
        total_started = time.perf_counter()
        engine = sgl.Engine(**engine_kwargs)
        initialization_time = time.perf_counter() - total_started
        inference_started = time.perf_counter()
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
                if len(outputs) != len(batch):
                    raise ValueError(
                        f"Verifier returned {len(outputs)} outputs for {len(batch)} jobs."
                    )
                for job, output in zip(batch, outputs):
                    scored.append(
                        {
                            "schema_version": 1,
                            **{
                                key: value
                                for key, value in job.items()
                                if key != "verifier_prompt"
                            },
                            **expected_score_from_output(
                                output, score_token_ids, score_values
                            ),
                            "scorer_model": args.scorer,
                            "score_labels": labels,
                            "score_token_ids": score_token_ids,
                            "criterion": "pairwise_informed_error_audit",
                        }
                    )
                elapsed = time.perf_counter() - inference_started
                print(
                    f"scored={len(scored)}/{len(jobs)} "
                    f"prefixes/s={len(scored) / elapsed:.2f}",
                    flush=True,
                )
                if args.probe:
                    print(json.dumps(outputs[0].get("meta_info", {}), indent=2)[:4000])
        finally:
            inference_wall_time = time.perf_counter() - inference_started
            engine.shutdown()
        wall_time = time.perf_counter() - total_started
        write_jsonl(args.save_scores, scored)

    if args.probe:
        probe = {
            "schema_version": 1,
            "probe": True,
            "labels": labels,
            "score_token_ids": score_token_ids,
            "scores": scored,
            "wall_time_s": wall_time,
            "engine_initialization_time_s": initialization_time,
            "inference_wall_time_s": inference_wall_time,
        }
        write_json(args.summary_output, probe)
        return probe

    by_checkpoint: dict[str, list[dict]] = defaultdict(list)
    for record in scored:
        checkpoint = str(
            record.get("checkpoint", record.get("requested_fraction"))
        )
        by_checkpoint[checkpoint].append(record)
    checkpoint_summaries = {}
    checkpoint_order = (
        [f"token_{position}" for position in token_positions] + ["terminal"]
        if token_positions
        else [str(fraction) for fraction in fractions]
    )
    for checkpoint_index, checkpoint_name in enumerate(checkpoint_order):
        records = by_checkpoint.get(checkpoint_name, [])
        if not records:
            continue
        checkpoint = summarize_checkpoint(records)
        checkpoint["bootstrap_95_ci"] = bootstrap_intervals(
            records,
            samples=args.bootstrap_samples,
            seed=args.seed + checkpoint_index + 1,
        )
        checkpoint_summaries[checkpoint_name] = checkpoint
    terminal_records = [
        record
        for record in scored
        if record["terminal"]
        and (not token_positions or record.get("checkpoint") == "terminal")
    ]
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "pairwise_informed_pointwise_error_audit",
            "trajectories": str(args.trajectories),
            "generator_model": generator_model,
            "scorer_model": args.scorer,
            "fractions": fractions,
            "token_positions": token_positions,
            "checkpoint_mode": checkpoint_mode,
            "score_labels": labels,
            "score_token_ids": score_token_ids,
            "score_scale": scale_description,
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
        "selection_baselines": (
            selection_baselines(trajectories, terminal_records)
            if terminal_records
            else None
        ),
        "cost": {
            "wall_time_s": wall_time,
            "engine_initialization_time_s": initialization_time,
            "inference_wall_time_s": inference_wall_time,
            "prefixes_scored": len(scored),
            "prefixes_per_s": (
                len(scored) / inference_wall_time if inference_wall_time else None
            ),
            "verifier_prompt_tokens": sum(row["prompt_tokens"] for row in scored),
            "verifier_completion_tokens": sum(
                row["completion_tokens"] for row in scored
            ),
            "mean_prompt_tokens_per_prefix": _mean(
                [row["prompt_tokens"] for row in scored]
            ),
            "mean_score_token_mass": _mean(
                [row["score_token_mass"] for row in scored]
            ),
            "selected_logprob_coverage": _mean(
                [float(row["logprob_source"] == "selected") for row in scored]
            ),
        },
    }
    write_json(args.summary_output, summary)
    print("\nPairwise-informed pointwise error audit")
    print("checkpoint     AUROC  rank-acc  top-half  top1")
    for checkpoint_name, checkpoint in checkpoint_summaries.items():
        print(
            f"{checkpoint_name:12s}  {checkpoint['auroc']:5.3f}  "
            f"{checkpoint['within_problem_ranking_accuracy']:8.3f}  "
            f"{checkpoint['top_half_correct_survival']:8.3f}  "
            f"{checkpoint['top1_eventual_correct_rate']:4.3f}"
        )
    print(json.dumps(summary["selection_baselines"], indent=2))
    print(json.dumps(summary["cost"], indent=2))
    print(f"wrote {args.summary_output}")
    return summary


if __name__ == "__main__":
    main()

"""Score fixed prefixes for mathematical validity and budgeted progress.

The two semantic rubrics are deliberately separate.  Validity asks whether the
reasoning remains mathematically viable.  Progress asks whether the path is
likely to finish within the remaining generator-token budget while ignoring
whether its mathematical conclusion is correct.  Neither rubric uses
generator likelihoods or sibling candidates.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

try:
    from scripts import offline_pointwise_error_audit as error_audit
except ModuleNotFoundError:
    import offline_pointwise_error_audit as error_audit


LABELS = tuple("ABCDE")
VALID_RUBRICS = ("validity", "progress")

PROGRESS_TEMPLATE = """\
You are independently auditing the progress of one partial solution to a math
problem. The solution was stopped after exactly {token_position} generator
tokens. Its total output budget is {total_budget} tokens, leaving at most
{remaining_budget} additional tokens.

Problem:
{problem}

Partial solution:
{prefix}

Judge progress toward producing a complete final solution before the remaining
budget expires. Look for concrete advancement: a usable approach, resolved key
obstacles, completed cases or calculations, and proximity to a final boxed
answer. Penalize looping, repeated restarts, excessive restatement, unresolved
core obstacles, or a path whose remaining work is unlikely to fit the budget.

This is a progress audit, not a correctness audit. A concise wrong derivation
can be high-progress, while a mathematically promising but stalled derivation
can be low-progress. Do not reward polish, verbosity, or confidence by itself.

Score the probability that continuation by the same generator will reach a
complete boxed final answer within the remaining token budget:
{scale}

You must return exactly one of these labels: {labels}. Do not explain your
choice and do not output any other text.
Score:\
"""


def parse_rubrics(value: str) -> list[str]:
    rubrics = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not rubrics or any(rubric not in VALID_RUBRICS for rubric in rubrics):
        raise ValueError(f"rubrics must be drawn from {VALID_RUBRICS}.")
    return rubrics


def build_progress_prompt(
    tokenizer,
    *,
    problem: str,
    prefix: str,
    token_position: int,
    total_budget: int,
    labels: Sequence[str] = LABELS,
) -> str:
    if token_position >= total_budget:
        raise ValueError("Progress checkpoints must precede the total budget.")
    _, scale = error_audit.score_scale(labels)
    content = PROGRESS_TEMPLATE.format(
        problem=problem,
        prefix=prefix,
        token_position=token_position,
        total_budget=total_budget,
        remaining_budget=total_budget - token_position,
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


def score_key(row: dict) -> tuple[str, int, str, str]:
    return (
        str(row["problem_id"]),
        int(row["sample_id"]),
        str(row["checkpoint"]),
        str(row["rubric"]),
    )


def make_jobs(
    trajectories: Sequence[dict],
    generator_tokenizer,
    verifier_tokenizer,
    token_positions: Sequence[int],
    rubrics: Sequence[str],
    *,
    total_budget: int,
) -> list[dict]:
    base = error_audit.make_fixed_token_jobs(
        trajectories,
        generator_tokenizer,
        token_positions,
        include_terminal=False,
    )
    jobs = []
    for row in base:
        for rubric in rubrics:
            job = dict(row)
            job["rubric"] = rubric
            if rubric == "validity":
                prompt = error_audit.build_error_audit_prompt(
                    verifier_tokenizer,
                    problem=row["problem"],
                    prefix=row["prefix"],
                    labels=LABELS,
                )
            else:
                prompt = build_progress_prompt(
                    verifier_tokenizer,
                    problem=row["problem"],
                    prefix=row["prefix"],
                    token_position=int(row["token_position"]),
                    total_budget=total_budget,
                    labels=LABELS,
                )
            job["verifier_prompt"] = prompt
            jobs.append(job)
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--generator-tokenizer", default=None)
    parser.add_argument("--token-positions", default="512,1024,2048")
    parser.add_argument("--total-budget", type=int, default=16384)
    parser.add_argument("--rubrics", default="validity,progress")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-gpu-id", type=int, default=0)
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


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)

    from transformers import AutoTokenizer

    trajectories = error_audit.load_jsonl(args.trajectories)
    error_audit.validate_trajectory_rows(trajectories)
    token_positions = error_audit.parse_token_positions(args.token_positions)
    if max(token_positions) >= args.total_budget:
        raise ValueError("All progress checkpoints must precede --total-budget.")
    rubrics = parse_rubrics(args.rubrics)
    generator_model = args.generator_tokenizer or trajectories[0]["generator_model"]
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
    verifier_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    labels = list(LABELS)
    score_values, scale = error_audit.score_scale(labels)
    score_token_ids = error_audit.resolve_score_token_ids(verifier_tokenizer, labels)
    all_jobs = make_jobs(
        trajectories,
        generator_tokenizer,
        verifier_tokenizer,
        token_positions,
        rubrics,
        total_budget=args.total_budget,
    )
    jobs = all_jobs[:2] if args.probe else all_jobs
    print(
        f"trajectories={len(trajectories)} rubrics={rubrics} "
        f"calls={len(jobs)} scorer={args.scorer}",
        flush=True,
    )
    print(f"score tokens: {dict(zip(labels, score_token_ids))}", flush=True)
    if args.dry_run:
        for rubric in rubrics:
            print(f"\n--- {rubric} ---")
            print(next(job["verifier_prompt"] for job in jobs if job["rubric"] == rubric))
        return None

    previous = (
        json.loads(Path(args.summary_output).read_text())
        if Path(args.summary_output).exists()
        else {}
    )
    if args.reuse_scores:
        scored = error_audit.load_jsonl(args.save_scores)
        expected = {score_key(job) for job in jobs}
        observed = {score_key(row) for row in scored}
        if expected != observed:
            raise ValueError(
                "Saved scores do not match jobs: "
                f"missing={len(expected - observed)} extra={len(observed - expected)}."
            )
        cost = previous.get("cost", {})
        initialization_time = float(cost.get("engine_initialization_time_s", 0.0))
        inference_time = float(cost.get("inference_wall_time_s", 0.0))
        wall_time = float(cost.get("wall_time_s", 0.0))
        print(f"reusing {len(scored)} saved scores", flush=True)
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
        wall_started = time.perf_counter()
        engine = sgl.Engine(**engine_kwargs)
        initialization_time = time.perf_counter() - wall_started
        inference_started = time.perf_counter()
        scored = []
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
                            **error_audit.expected_score_from_output(
                                output, score_token_ids, score_values
                            ),
                            "scorer_model": args.scorer,
                            "score_labels": labels,
                            "score_token_ids": score_token_ids,
                            "criterion": (
                                "pairwise_informed_error_audit"
                                if job["rubric"] == "validity"
                                else "budgeted_progress_audit"
                            ),
                        }
                    )
                elapsed = time.perf_counter() - inference_started
                print(
                    f"scored={len(scored)}/{len(jobs)} "
                    f"calls/s={len(scored) / elapsed:.2f}",
                    flush=True,
                )
                error_audit.write_jsonl(args.save_scores, scored)
        finally:
            inference_time = time.perf_counter() - inference_started
            engine.shutdown()
        wall_time = time.perf_counter() - wall_started

    if args.probe:
        probe = {
            "schema_version": 1,
            "probe": True,
            "scorer_model": args.scorer,
            "scores": scored,
        }
        error_audit.write_json(args.summary_output, probe)
        return probe

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in scored:
        grouped[str(row["rubric"])][str(row["checkpoint"])].append(row)
    rubric_summaries = {}
    for rubric_index, rubric in enumerate(rubrics):
        rubric_summaries[rubric] = {}
        for checkpoint_index, token_position in enumerate(token_positions):
            checkpoint = f"token_{token_position}"
            rows = grouped[rubric].get(checkpoint, [])
            if not rows:
                continue
            metrics = error_audit.summarize_checkpoint(rows)
            metrics["bootstrap_95_ci"] = error_audit.bootstrap_intervals(
                rows,
                samples=args.bootstrap_samples,
                seed=args.seed + 100 * rubric_index + checkpoint_index + 1,
            )
            rubric_summaries[rubric][checkpoint] = metrics
    by_rubric_cost = {}
    for rubric in rubrics:
        rows = [row for row in scored if row["rubric"] == rubric]
        by_rubric_cost[rubric] = {
            "calls": len(rows),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
        }
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": "separate_validity_and_budgeted_progress_semantic_audits",
            "trajectories": args.trajectories,
            "generator_model": generator_model,
            "scorer_model": args.scorer,
            "rubrics": rubrics,
            "token_positions": token_positions,
            "total_generator_token_budget": args.total_budget,
            "score_labels": labels,
            "score_token_ids": score_token_ids,
            "score_scale": scale,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "uses_generator_likelihood": False,
        },
        "dataset": {
            "name": trajectories[0].get("dataset"),
            "config": trajectories[0].get("dataset_config"),
            "split": trajectories[0].get("split"),
            "n_problems": len({str(row["problem_id"]) for row in trajectories}),
            "n_trajectories": len(trajectories),
        },
        "rubrics": rubric_summaries,
        "cost": {
            "wall_time_s": wall_time,
            "engine_initialization_time_s": initialization_time,
            "inference_wall_time_s": inference_time,
            "verifier_gpus": args.dp * args.tp,
            "allocated_verifier_gpu_seconds": inference_time * args.dp * args.tp,
            "calls": len(scored),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in scored),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in scored),
            "mean_score_token_mass": error_audit._mean(
                [float(row["score_token_mass"]) for row in scored]
            ),
            "selected_logprob_coverage": error_audit._mean(
                [float(row["logprob_source"] == "selected") for row in scored]
            ),
            "by_rubric": by_rubric_cost,
        },
    }
    error_audit.write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

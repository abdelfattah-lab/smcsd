#!/usr/bin/env python3
"""Score leakage-safe Terminal-Bench checkpoints with one semantic verifier.

This process reads checkpoints.jsonl only. It never opens labels.jsonl.
Scoring is append-only and resumable by stable verifier-call ID.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_SCORE_LABELS = tuple("ABCDEFGHIJKLMNOPQRST")
CRITERION_VERSION = "terminal_recoverability_v1"

VERIFIER_TEMPLATE = """\
You are evaluating the current state of a terminal-based software task.

Task:
{task}

Agent-visible transcript at the checkpoint:
{transcript}

Criterion:
Estimate the probability that continuing from exactly this state will
eventually produce an environment that passes the task's hidden evaluator.
Use only the visible task and transcript. Consider progress, discovered facts,
tool results, destructive or irreversible mistakes, remaining recoverability,
and whether the current partial assistant generation is heading toward a useful
next action. Do not reward verbosity or confidence. Do not assume access to
hidden tests, future tool output, or the final reward.

Score scale:
{scale}

Return exactly one score label and no other text.
Score:"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(*parts: object, length: int = 24) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:length]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                pieces.append(str(block))
            elif block.get("type") == "text":
                pieces.append(block.get("text") or "")
            else:
                pieces.append(canonical_json(block))
        return "".join(pieces)
    return str(content)


def compact_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n[...middle truncated by frozen semantic-input policy...]\n"
    available = max(limit - len(marker), 0)
    head = min(512, available // 3)
    tail = max(available - head, 0)
    return text[:head] + marker + text[-tail:], True


def render_message(
    message: dict[str, Any],
    *,
    tool_output_max_chars: int,
) -> tuple[str, bool]:
    role = message.get("role") or "unknown"
    truncated = False
    if role == "assistant":
        pieces = [flatten_content(message.get("content"))]
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = canonical_json(arguments or {})
            pieces.append(
                "\n[TOOL CALL "
                + str(function.get("name") or "unknown")
                + "] "
                + arguments
            )
        return "[ASSISTANT]\n" + "".join(pieces).strip(), False
    if role == "tool":
        content, truncated = compact_text(
            flatten_content(message.get("content")),
            tool_output_max_chars,
        )
        label = message.get("name") or message.get("tool_call_id") or "tool"
        return f"[TOOL RESULT {label}]\n{content.strip()}", truncated
    if role == "user":
        return "[USER]\n" + flatten_content(message.get("content")).strip(), False
    rendered = (
        f"[{str(role).upper()}]\n"
        f"{flatten_content(message.get('content')).strip()}"
    )
    return "".join(rendered), False


def checkpoint_transcript(
    checkpoint: dict[str, Any],
    *,
    transcript_max_chars: int,
    tool_output_max_chars: int,
) -> tuple[str, str, dict[str, int]]:
    semantic_input = checkpoint["semantic_input"]
    messages = semantic_input["messages"]
    user_messages = [
        flatten_content(message.get("content")).strip()
        for message in messages
        if message.get("role") == "user"
    ]
    if not user_messages:
        raise ValueError(
            f"checkpoint {checkpoint['checkpoint_id']} has no task message"
        )
    task = user_messages[0]

    blocks: list[str] = []
    tool_truncations = 0
    skipped_initial_user = False
    for message in messages:
        if message.get("role") == "system":
            continue
        if message.get("role") == "user" and not skipped_initial_user:
            skipped_initial_user = True
            continue
        rendered, truncated = render_message(
            message,
            tool_output_max_chars=tool_output_max_chars,
        )
        tool_truncations += int(truncated)
        if rendered.strip():
            blocks.append(rendered)
    partial = semantic_input.get("partial_assistant_rendered")
    if partial:
        blocks.append("[CURRENT PARTIAL ASSISTANT GENERATION]\n" + str(partial))

    selected: list[str] = []
    remaining = transcript_max_chars
    omitted_blocks = 0
    for block in reversed(blocks):
        separator_cost = 2 if selected else 0
        if len(block) + separator_cost <= remaining:
            selected.append(block)
            remaining -= len(block) + separator_cost
            continue
        omitted_blocks += 1
        if not selected and remaining > 128:
            marker = "[...older content truncated...]\n"
            selected.append(marker + block[-max(remaining - len(marker), 0) :])
            remaining = 0
        omitted_blocks += len(blocks) - len(selected) - omitted_blocks
        break
    selected.reverse()
    if omitted_blocks:
        selected.insert(
            0,
            f"[{omitted_blocks} older transcript blocks omitted by frozen policy]",
        )
    transcript = "\n\n".join(selected) or "[No assistant or tool activity yet]"
    return task, transcript, {
        "source_messages": len(messages),
        "included_blocks": len(selected),
        "omitted_blocks": omitted_blocks,
        "tool_outputs_compacted": tool_truncations,
        "transcript_chars": len(transcript),
    }


def score_scale(labels: Sequence[str]) -> tuple[list[float], str]:
    if len(labels) < 2:
        raise ValueError("at least two score labels are required")
    values = [index / (len(labels) - 1) for index in range(len(labels))]
    scale = ", ".join(
        f"{label}={round(value * 100):d}%" for label, value in zip(labels, values)
    )
    return values, scale


def resolve_score_token_ids(tokenizer: Any, labels: Sequence[str]) -> list[int]:
    token_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"score label {label!r} is not one token: {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("score labels do not map to unique tokens")
    return token_ids


def build_verifier_prompt(
    tokenizer: Any,
    *,
    task: str,
    transcript: str,
    labels: Sequence[str],
) -> str:
    _, scale = score_scale(labels)
    content = VERIFIER_TEMPLATE.format(task=task, transcript=transcript, scale=scale)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
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


def _entry_logprob_and_id(entry: Any) -> tuple[float, int]:
    if isinstance(entry, dict):
        return float(entry["logprob"]), int(entry["token_id"])
    return float(entry[0]), int(entry[1])


def expected_score_from_output(
    output: dict[str, Any],
    score_token_ids: Sequence[int],
    score_values: Sequence[float],
) -> dict[str, Any]:
    meta = output.get("meta_info") or {}
    positions = meta.get("output_token_ids_logprobs")
    source = "selected"
    if not positions:
        positions = meta.get("output_top_logprobs")
        source = "top"
    if not positions or not positions[0]:
        raise ValueError("verifier output has no score-token logprobs")
    by_id = {
        token_id: logprob
        for logprob, token_id in (
            _entry_logprob_and_id(entry) for entry in positions[0]
        )
    }
    missing = [token_id for token_id in score_token_ids if token_id not in by_id]
    if missing:
        raise ValueError(f"verifier output is missing score tokens: {missing}")
    logprobs = [by_id[token_id] for token_id in score_token_ids]
    maximum = max(logprobs)
    weights = [math.exp(logprob - maximum) for logprob in logprobs]
    normalizer = sum(weights)
    probabilities = [weight / normalizer for weight in weights]
    log_mass = maximum + math.log(normalizer)
    return {
        "score": sum(
            probability * value
            for probability, value in zip(probabilities, score_values)
        ),
        "score_probabilities": probabilities,
        "score_token_mass": math.exp(min(log_mass, 0.0)),
        "logprob_source": source,
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
    }


def load_existing(
    path: Path, scorer_model: str
) -> tuple[set[str], list[dict[str, Any]]]:
    if not path.exists():
        return set(), []
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    wrong = {
        row.get("scorer_model")
        for row in rows
        if row.get("scorer_model") != scorer_model
    }
    if wrong:
        raise ValueError(f"existing output contains different scorer models: {wrong}")
    return {str(row["verifier_call_id"]) for row in rows}, rows


def iter_checkpoints(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {
                "final_reward",
                "eventual_success",
                "reward",
                "verifier_result",
            } & set(row)
            if forbidden:
                raise ValueError(
                    f"label leakage at {path}:{line_number}: {sorted(forbidden)}"
                )
            yield row


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        stream.flush()


def summarize_scores(
    scores_path: Path,
    summary_path: Path,
    run_costs_path: Path,
    *,
    scorer_model: str,
    criterion: str,
    tp_size: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in scores_path.read_text().splitlines()
        if line.strip()
    ]
    run_costs = [
        json.loads(line)
        for line in run_costs_path.read_text().splitlines()
        if line.strip()
    ]
    batch_cost: dict[tuple[str, int], float] = {}
    for row in rows:
        batch_cost[(row["run_id"], int(row["batch_index"]))] = float(
            row["batch_wall_time_s"]
        )
    inference_wall_time = sum(batch_cost.values())
    allocation_wall_time = sum(
        float(row["allocation_wall_time_s"]) for row in run_costs
    )
    summary = {
        "schema_version": 1,
        "scorer_model": scorer_model,
        "criterion": criterion,
        "created_at": utc_now(),
        "counts": {
            "verifier_calls": len(rows),
            "unique_checkpoints": len({row["checkpoint_id"] for row in rows}),
            "scorer_process_runs": len(run_costs),
        },
        "cost": {
            "wall_time_s": allocation_wall_time,
            "allocation_wall_time_s": allocation_wall_time,
            "inference_wall_time_s": inference_wall_time,
            "engine_startup_wall_time_s": sum(
                float(row["engine_startup_wall_time_s"]) for row in run_costs
            ),
            "tensor_parallel_size": tp_size,
            "allocated_accelerator_seconds": allocation_wall_time * tp_size,
            "inference_active_accelerator_seconds": (
                inference_wall_time * tp_size
            ),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
            "mean_prompt_tokens": statistics.fmean(
                int(row["prompt_tokens"]) for row in rows
            )
            if rows
            else None,
            "mean_score_token_mass": statistics.fmean(
                float(row["score_token_mass"]) for row in rows
            )
            if rows
            else None,
        },
        "run_costs_file": {
            "path": run_costs_path.name,
            "sha256": hashlib.sha256(run_costs_path.read_bytes()).hexdigest(),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--run-cost-output", type=Path)
    parser.add_argument("--score-labels", default=",".join(DEFAULT_SCORE_LABELS))
    parser.add_argument("--criterion", default=CRITERION_VERSION)
    parser.add_argument("--transcript-max-chars", type=int, default=30000)
    parser.add_argument("--tool-output-max-chars", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-gpu-id", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-mamba-cache-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completed-score", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    labels = [label.strip() for label in args.score_labels.split(",") if label.strip()]
    run_cost_output = args.run_cost_output or args.summary_output.with_suffix(
        ".runs.jsonl"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.scorer,
        trust_remote_code=True,
        local_files_only=True,
    )
    score_values, _ = score_scale(labels)
    score_token_ids = resolve_score_token_ids(tokenizer, labels)
    completed, _ = load_existing(args.scores_output, args.scorer)
    for completed_path in args.completed_score:
        external_completed, _ = load_existing(completed_path, args.scorer)
        completed.update(external_completed)
    source_total = sum(1 for _ in iter_checkpoints(args.checkpoints))
    if args.max_checkpoints is not None:
        source_total = min(source_total, args.max_checkpoints)
    total = sum(
        checkpoint_index % args.num_shards == args.shard_index
        for checkpoint_index in range(source_total)
    )
    print(
        f"checkpoints={total} already_scored={len(completed)} "
        f"scorer={args.scorer} shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )

    if args.dry_run:
        checkpoint = next(iter(iter_checkpoints(args.checkpoints)))
        task, transcript, compaction = checkpoint_transcript(
            checkpoint,
            transcript_max_chars=args.transcript_max_chars,
            tool_output_max_chars=args.tool_output_max_chars,
        )
        prompt = build_verifier_prompt(
            tokenizer,
            task=task,
            transcript=transcript,
            labels=labels,
        )
        print(json.dumps(compaction, indent=2, sort_keys=True))
        print(prompt)
        return 0

    import sglang as sgl

    engine_kwargs: dict[str, Any] = {
        "model_path": args.scorer,
        "trust_remote_code": True,
        "attention_backend": "triton",
        "mem_fraction_static": args.mem_fraction_static,
        "base_gpu_id": args.base_gpu_id,
        "random_seed": args.seed,
        "max_running_requests": args.max_running_requests,
        "max_mamba_cache_size": args.max_mamba_cache_size,
    }
    if args.tp > 1:
        engine_kwargs.update(
            tp_size=args.tp,
            disable_custom_all_reduce=True,
            enforce_disable_flashinfer_allreduce_fusion=True,
        )

    run_id = stable_id(
        args.scorer, args.criterion, args.seed, args.shard_index, utc_now()
    )
    allocation_started = time.perf_counter()
    engine = sgl.Engine(**engine_kwargs)
    engine_startup_wall_time = time.perf_counter() - allocation_started
    scored_this_run = 0
    batch_index = 0
    run_error: str | None = None
    try:
        batch: list[dict[str, Any]] = []
        for checkpoint_index, checkpoint in enumerate(
            iter_checkpoints(args.checkpoints)
        ):
            if (
                args.max_checkpoints is not None
                and checkpoint_index >= args.max_checkpoints
            ):
                break
            if checkpoint_index % args.num_shards != args.shard_index:
                continue
            call_id = stable_id(
                checkpoint["dataset_id"],
                checkpoint["checkpoint_id"],
                args.scorer,
                args.criterion,
                0,
            )
            if call_id in completed:
                continue
            task, transcript, compaction = checkpoint_transcript(
                checkpoint,
                transcript_max_chars=args.transcript_max_chars,
                tool_output_max_chars=args.tool_output_max_chars,
            )
            prompt = build_verifier_prompt(
                tokenizer,
                task=task,
                transcript=transcript,
                labels=labels,
            )
            batch.append(
                {
                    "checkpoint": checkpoint,
                    "verifier_call_id": call_id,
                    "prompt": prompt,
                    "compaction": compaction,
                }
            )
            if len(batch) < args.batch_size:
                continue

            started = time.perf_counter()
            outputs = engine.generate(
                [entry["prompt"] for entry in batch],
                {"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=score_token_ids,
            )
            batch_wall = time.perf_counter() - started
            if not isinstance(outputs, list):
                outputs = [outputs]
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"verifier returned {len(outputs)} outputs for {len(batch)} prompts"
                )
            records = []
            for entry, output in zip(batch, outputs):
                checkpoint = entry["checkpoint"]
                records.append(
                    {
                        "schema_version": 1,
                        "dataset_id": checkpoint["dataset_id"],
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trajectory_id": checkpoint["trajectory_id"],
                        "task_id": checkpoint["task_id"],
                        "trigger": checkpoint["trigger"],
                        "verifier_call_id": entry["verifier_call_id"],
                        "verifier_call_index": 0,
                        "scorer_model": args.scorer,
                        "criterion": args.criterion,
                        "score_labels": labels,
                        "score_token_ids": score_token_ids,
                        "semantic_input_sha256": hashlib.sha256(
                            canonical_json(checkpoint["semantic_input"]).encode()
                        ).hexdigest(),
                        "compaction": entry["compaction"],
                        **expected_score_from_output(
                            output,
                            score_token_ids,
                            score_values,
                        ),
                        "run_id": run_id,
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "batch_wall_time_s": batch_wall,
                    }
                )
            append_jsonl(args.scores_output, records)
            completed.update(record["verifier_call_id"] for record in records)
            scored_this_run += len(records)
            batch_index += 1
            print(
                f"scored={len(completed)}/{total} "
                f"new={scored_this_run} batch_s={batch_wall:.2f}",
                flush=True,
            )
            batch = []

        if batch:
            started = time.perf_counter()
            outputs = engine.generate(
                [entry["prompt"] for entry in batch],
                {"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=score_token_ids,
            )
            batch_wall = time.perf_counter() - started
            if not isinstance(outputs, list):
                outputs = [outputs]
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"verifier returned {len(outputs)} outputs for {len(batch)} prompts"
                )
            records = []
            for entry, output in zip(batch, outputs):
                checkpoint = entry["checkpoint"]
                records.append(
                    {
                        "schema_version": 1,
                        "dataset_id": checkpoint["dataset_id"],
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trajectory_id": checkpoint["trajectory_id"],
                        "task_id": checkpoint["task_id"],
                        "trigger": checkpoint["trigger"],
                        "verifier_call_id": entry["verifier_call_id"],
                        "verifier_call_index": 0,
                        "scorer_model": args.scorer,
                        "criterion": args.criterion,
                        "score_labels": labels,
                        "score_token_ids": score_token_ids,
                        "semantic_input_sha256": hashlib.sha256(
                            canonical_json(checkpoint["semantic_input"]).encode()
                        ).hexdigest(),
                        "compaction": entry["compaction"],
                        **expected_score_from_output(
                            output,
                            score_token_ids,
                            score_values,
                        ),
                        "run_id": run_id,
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "batch_wall_time_s": batch_wall,
                    }
                )
            append_jsonl(args.scores_output, records)
            completed.update(record["verifier_call_id"] for record in records)
            scored_this_run += len(records)
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        engine.shutdown()
        allocation_wall_time = time.perf_counter() - allocation_started
        append_jsonl(
            run_cost_output,
            [
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "scorer_model": args.scorer,
                    "criterion": args.criterion,
                    "created_at": utc_now(),
                    "successful": run_error is None,
                    "error": run_error,
                    "verifier_calls": scored_this_run,
                    "tensor_parallel_size": args.tp,
                    "engine_startup_wall_time_s": engine_startup_wall_time,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "allocation_wall_time_s": allocation_wall_time,
                }
            ],
        )

    summary = summarize_scores(
        args.scores_output,
        args.summary_output,
        run_cost_output,
        scorer_model=args.scorer,
        criterion=args.criterion,
        tp_size=args.tp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export leakage-safe Terminal-Bench checkpoints for semantic scoring.

Pi stores the complete request before each model turn and the finalized
assistant message in its session JSONL.  Together with the model tokenizer,
those artifacts recover the exact generated token IDs, including prefixes
inside a model turn.  Post-tool checkpoints are copied from the next provider
request, where the tool result is already part of the agent-visible history.

The exporter deliberately writes final rewards to a separate labels file.
Semantic scorers should receive only checkpoints.jsonl.  Existing Harbor
artifacts do not include restorable container snapshots, so the output marks
terminal-environment cloning as unsupported rather than implying that this
offline dataset is already an online trajectory-level SMC runtime.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "configs/terminal_bench/semantic_actionability_smoke_v1.json"
)
LABEL_ONLY_KEYS = {
    "correct",
    "eventual_success",
    "final_reward",
    "gold_answer",
    "grader",
    "hidden_tests",
    "reward",
    "verifier_result",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(*parts: object, length: int = 24) -> str:
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def task_id_from_result(result: dict[str, Any]) -> str:
    task_name = result.get("task_name") or ""
    if not task_name:
        raise ValueError("trial result is missing task_name")
    return task_name.rsplit("/", 1)[-1]


def pi_assistant_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.get("content") or []:
        block_type = block.get("type")
        if block_type == "text":
            text.append(block.get("text") or "")
        elif block_type in {"thinking", "reasoning"}:
            reasoning.append(block.get("thinking") or block.get("text") or "")
        elif block_type == "toolCall":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": copy.deepcopy(block.get("arguments") or {}),
                    },
                }
            )
        else:
            raise ValueError(f"unsupported Pi assistant content block: {block_type}")

    converted: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text) or None,
    }
    if reasoning:
        converted["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        converted["tool_calls"] = tool_calls
    return converted


def normalize_messages_for_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON tool arguments in call {call.get('id')}"
                    ) from exc
    return normalized


def chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        normalize_messages_for_template(messages),
        tools=copy.deepcopy(tools),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **(template_kwargs or {}),
    )
    if hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("expected one tokenized chat, received a batch")
        rendered = rendered[0]
    return [int(token_id) for token_id in rendered]

def structural_template_suffix(tokenizer: Any, suffix: str) -> bool:
    residual = suffix
    for special_token in getattr(tokenizer, "all_special_tokens", []):
        if special_token:
            residual = residual.replace(str(special_token), "")
    closing_tags = (
        "</IMPORTANT>",
        "</function>",
        "</parameter>",
        "</think>",
        "</tool_call>",
        "</tool_response>",
        "</tools>",
    )
    contained_template_close = any(tag in residual for tag in closing_tags)
    for tag in closing_tags:
        residual = residual.replace(tag, "")
    stripped = residual.strip()
    return not stripped or (contained_template_close and stripped == ">")



def generation_token_ids(
    tokenizer: Any,
    request_payload: dict[str, Any],
    assistant_message: dict[str, Any],
    expected_tokens: int,
) -> tuple[list[int], list[int], bool]:
    messages = request_payload["messages"]
    tools = request_payload.get("tools")
    template_kwargs = request_payload.get("chat_template_kwargs") or {}
    prompt_ids = chat_token_ids(
        tokenizer,
        messages,
        tools,
        add_generation_prompt=True,
        template_kwargs=template_kwargs,
    )
    complete_ids = chat_token_ids(
        tokenizer,
        messages + [pi_assistant_to_openai(assistant_message)],
        tools,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    content = assistant_message.get("content") or []
    has_tool_call = any(
        isinstance(block, dict) and block.get("type") == "toolCall"
        for block in content
    )
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        if assistant_message.get("stopReason") == "toolUse" and has_tool_call:
            # Structured-output parsing and reasoning-tag recovery can rewrite
            # the completed chat prefix. Preserve the exact provider prompt,
            # but do not fabricate completion token IDs for this turn.
            return prompt_ids, [], False
        raise ValueError(
            "assistant chat rendering does not preserve the request prefix"
        )

    output_ids = complete_ids[len(prompt_ids) :]
    if len(output_ids) > expected_tokens:
        suffix = tokenizer.decode(
            output_ids[expected_tokens:], skip_special_tokens=False
        )
        if not structural_template_suffix(tokenizer, suffix):
            if assistant_message.get("stopReason") == "toolUse" and has_tool_call:
                return prompt_ids, [], False
            raise ValueError(
                "rendered generation has a non-whitespace suffix beyond Pi usage: "
                f"expected={expected_tokens}, rendered={len(output_ids)}, "
                f"suffix={suffix!r}"
            )
        output_ids = output_ids[:expected_tokens]
    if len(output_ids) != expected_tokens:
        if (
            expected_tokens > len(output_ids)
            and assistant_message.get("stopReason") == "toolUse"
            and has_tool_call
        ):
            return prompt_ids, output_ids, False
        raise ValueError(
            "rendered generation does not match Pi completion-token usage: "
            f"expected={expected_tokens}, rendered={len(output_ids)}"
        )
    return prompt_ids, output_ids, True


def load_requests(trial_dir: Path) -> list[dict[str, Any]]:
    captures = load_jsonl(trial_dir.glob("agent/pi-capture/requests-*.jsonl"))
    requests = [row for row in captures if isinstance(row.get("payload"), dict)]
    requests.sort(key=lambda row: (row.get("t", 0), row.get("pid", 0)))
    if not requests:
        raise ValueError(f"no captured provider requests under {trial_dir}")
    return requests


def load_session_messages(trial_dir: Path) -> list[dict[str, Any]]:
    session_paths = sorted(trial_dir.glob("agent/pi/sessions/*.jsonl"))
    if len(session_paths) != 1:
        raise ValueError(
            f"expected one non-subagent Pi session under {trial_dir}, "
            f"found {len(session_paths)}"
        )
    rows = load_jsonl(session_paths)
    return [
        row["message"]
        for row in rows
        if row.get("type") == "message" and isinstance(row.get("message"), dict)
    ]


def label_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key.lower() in LABEL_ONLY_KEYS:
                found.append(path)
            found.extend(label_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(label_keys(child, f"{prefix}[{index}]"))
    return found


def count_tools(messages: list[dict[str, Any]]) -> int:
    return sum(message.get("role") == "tool" for message in messages)


def generator_cost_so_far(
    turns: list[dict[str, Any]],
    completed_turns: int,
    generated_tokens: int,
    *,
    current_turn_started: bool,
) -> dict[str, Any]:
    included = completed_turns + int(current_turn_started)
    return {
        "calls": included,
        "input_tokens": sum(turn["input_tokens"] for turn in turns[:included]),
        "output_tokens": generated_tokens,
        "active_accelerator_seconds": None,
        "active_accelerator_seconds_status": "not_available_per_request",
    }


def base_checkpoint(
    *,
    dataset_id: str,
    trajectory_id: str,
    task_id: str,
    order: int,
    checkpoint_id: str,
    trigger: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    state: dict[str, Any],
    costs_so_far: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "checkpoint_id": checkpoint_id,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "order": order,
        "trigger": trigger,
        "semantic_input": {
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "partial_assistant_rendered": state.get("partial_generation_text"),
        },
        "resume_state": state,
        "costs_so_far": costs_so_far,
    }
    leaked = label_keys(row)
    if leaked:
        raise ValueError(f"checkpoint contains label-only keys: {leaked}")
    return row


def export_trial(
    *,
    dataset_id: str,
    task_id: str,
    trial_dir: Path,
    job_metadata: dict[str, Any],
    attempt_index: int,
    tokenizer: Any,
    base_interval: int,
    derived_intervals: list[int],
    accelerators_allocated: int,
    success_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result = json.loads((trial_dir / "result.json").read_text())
    errored = result.get("exception_info") is not None
    if task_id_from_result(result) != task_id:
        raise ValueError(f"task mismatch for {trial_dir}")

    job_name = job_metadata["job_name"]
    trajectory_id = stable_id(
        dataset_id,
        job_name,
        task_id,
        job_metadata["spec"]["seed"],
        attempt_index,
    )
    requests = load_requests(trial_dir)
    session_messages = load_session_messages(trial_dir)
    assistants = [
        message for message in session_messages if message.get("role") == "assistant"
    ]
    tool_results = [
        message for message in session_messages if message.get("role") == "toolResult"
    ]
    incomplete_requests_dropped = 0
    if len(requests) == len(assistants) + 1 and errored:
        incomplete_requests_dropped = 1
        requests = requests[: len(assistants)]
    if len(requests) != len(assistants):
        raise ValueError(
            f"request/assistant count mismatch for {trial_dir}: "
            f"{len(requests)} requests, {len(assistants)} assistants"
        )

    turns: list[dict[str, Any]] = []
    for turn_index, (request, assistant) in enumerate(zip(requests, assistants)):
        usage = assistant.get("usage") or {}
        expected_output = int(usage.get("output") or 0)
        try:
            prompt_ids, output_ids, alignment_exact = generation_token_ids(
                tokenizer,
                request["payload"],
                assistant,
                expected_output,
            )
        except ValueError as exc:
            raise ValueError(
                f"token reconstruction failed for {trial_dir} turn {turn_index}: {exc}"
            ) from exc
        turns.append(
            {
                "turn_index": turn_index,
                "request_t_ms": int(request.get("t") or 0),
                "request_payload": request["payload"],
                "assistant": assistant,
                "prompt_token_ids": prompt_ids,
                "output_token_ids": output_ids,
                "input_tokens": int(usage.get("input") or 0),
                "output_tokens": expected_output,
                "output_token_alignment_exact": alignment_exact,
                "output_token_alignment_delta": (
                    expected_output - len(output_ids)
                ),
            }
        )

    agent_result = result.get("agent_result") or {}
    session_input_tokens = sum(turn["input_tokens"] for turn in turns)
    session_output_tokens = sum(turn["output_tokens"] for turn in turns)
    agent_result_input_tokens = int(agent_result.get("n_input_tokens") or 0)
    agent_result_output_tokens = int(agent_result.get("n_output_tokens") or 0)
    input_token_delta = session_input_tokens - agent_result_input_tokens
    output_token_delta = session_output_tokens - agent_result_output_tokens
    if input_token_delta and not errored:
        raise ValueError(
            f"input token mismatch for {trial_dir}: session={session_input_tokens}, "
            f"result={agent_result.get('n_input_tokens')}"
        )
    if output_token_delta and not errored:
        raise ValueError(
            f"output token mismatch for {trial_dir}: session={session_output_tokens}, "
            f"result={agent_result.get('n_output_tokens')}"
        )
    token_accounting_status = (
        "session_and_agent_result_match"
        if not input_token_delta and not output_token_delta
        else "session_sum_preserved_agent_result_differs_on_errored_trajectory"
    )

    checkpoints: list[dict[str, Any]] = []
    order = 0
    next_token_target = base_interval
    cumulative_output = 0
    global_tool_index = 0
    agent_started = (result.get("agent_execution") or {}).get("started_at")
    agent_started_ms = (
        int(parse_time(agent_started).timestamp() * 1000) if agent_started else None
    )

    prefix_alignment_exact = True
    skipped_token_checkpoints = 0
    for turn_index, turn in enumerate(turns):
        payload = turn["request_payload"]
        messages = payload["messages"]
        tools = payload.get("tools")

        if turn_index:
            previous_messages = turns[turn_index - 1]["request_payload"]["messages"]
            if messages[: len(previous_messages)] != previous_messages:
                raise ValueError(
                    f"request history was compacted or rewritten at turn {turn_index} "
                    f"for {trial_dir}; exact incremental tool checkpoints are unsafe"
                )
            appended = messages[len(previous_messages) :]
            for appended_index, message in enumerate(appended):
                if message.get("role") != "tool":
                    continue
                message_index = len(previous_messages) + appended_index
                visible_messages = messages[: message_index + 1]
                global_tool_index += 1
                checkpoint_id = stable_id(
                    dataset_id,
                    trajectory_id,
                    "post_tool",
                    global_tool_index,
                )
                elapsed_agent = None
                if agent_started_ms is not None and turn["request_t_ms"]:
                    elapsed_agent = max(
                        (turn["request_t_ms"] - agent_started_ms) / 1000.0,
                        0.0,
                    )
                state = {
                    "kind": "provider_request_boundary",
                    "model": payload.get("model"),
                    "request": {
                        **copy.deepcopy(payload),
                        "messages": copy.deepcopy(visible_messages),
                    },
                    "partial_generation_token_ids": [],
                    "partial_generation_text": None,
                    "cumulative_generated_tokens": cumulative_output,
                    "model_state_serializable": True,
                    "model_state_resumable": prefix_alignment_exact,
                    "model_state_status": (
                        "exact" if prefix_alignment_exact else "prior_turn_nonalignable"
                    ),
                    "terminal_environment_cloneable": False,
                    "terminal_environment_status": (
                        "not_captured_by_existing_harbor_artifacts"
                    ),
                }
                checkpoints.append(
                    base_checkpoint(
                        dataset_id=dataset_id,
                        trajectory_id=trajectory_id,
                        task_id=task_id,
                        order=order,
                        checkpoint_id=checkpoint_id,
                        trigger={
                            "kind": "post_tool",
                            "tool_index": global_tool_index,
                            "tool_call_id": message.get("tool_call_id"),
                            "generated_tokens": cumulative_output,
                        },
                        messages=visible_messages,
                        tools=tools,
                        state=state,
                        costs_so_far={
                            "generator": generator_cost_so_far(
                                turns,
                                turn_index,
                                cumulative_output,
                                current_turn_started=False,
                            ),
                            "tools": {
                                "calls": count_tools(visible_messages),
                                "wall_time_s": None,
                                "wall_time_status": "not_available_per_tool",
                            },
                            "semantic_verifier": {
                                "calls": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "wall_time_s": 0.0,
                                "active_accelerator_seconds": 0.0,
                            },
                            "agent_elapsed_wall_time_s": elapsed_agent,
                        },
                    )
                )
                order += 1

        turn_end = cumulative_output + turn["output_tokens"]
        while next_token_target <= turn_end:
            offset = next_token_target - cumulative_output
            if not turn["output_token_alignment_exact"]:
                skipped_token_checkpoints += 1
                next_token_target += base_interval
                continue

            partial_ids = turn["output_token_ids"][:offset]
            partial_text = tokenizer.decode(partial_ids, skip_special_tokens=False)
            checkpoint_id = stable_id(
                dataset_id,
                trajectory_id,
                "token_interval",
                next_token_target,
            )
            state = {
                "kind": "partial_model_generation",
                "model": payload.get("model"),
                "request": copy.deepcopy(payload),
                "prompt_token_count": len(turn["prompt_token_ids"]),
                "prompt_token_sha256": hashlib.sha256(
                    canonical_json(turn["prompt_token_ids"]).encode()
                ).hexdigest(),
                "partial_generation_token_ids": partial_ids,
                "partial_generation_text": partial_text,
                "turn_index": turn_index,
                "turn_generated_tokens": offset,
                "cumulative_generated_tokens": next_token_target,
                "model_state_serializable": True,
                "model_state_resumable": prefix_alignment_exact,
                "model_state_status": (
                    "exact" if prefix_alignment_exact else "prior_turn_nonalignable"
                ),
                "terminal_environment_cloneable": False,
                "terminal_environment_status": (
                    "not_captured_by_existing_harbor_artifacts"
                ),
            }
            checkpoints.append(
                base_checkpoint(
                    dataset_id=dataset_id,
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    order=order,
                    checkpoint_id=checkpoint_id,
                    trigger={
                        "kind": "token_interval",
                        "base_interval_tokens": base_interval,
                        "target_generated_tokens": next_token_target,
                        "actual_generated_tokens": next_token_target,
                        "derived_interval_views": [
                            interval
                            for interval in derived_intervals
                            if next_token_target % interval == 0
                        ],
                    },
                    messages=messages,
                    tools=tools,
                    state=state,
                    costs_so_far={
                        "generator": generator_cost_so_far(
                            turns,
                            turn_index,
                            next_token_target,
                            current_turn_started=True,
                        ),
                        "tools": {
                            "calls": count_tools(messages),
                            "wall_time_s": None,
                            "wall_time_status": "not_available_per_tool",
                        },
                        "semantic_verifier": {
                            "calls": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "wall_time_s": 0.0,
                            "active_accelerator_seconds": 0.0,
                        },
                        "agent_elapsed_wall_time_s": None,
                    },
                )
            )
            order += 1
            next_token_target += base_interval
        cumulative_output = turn_end

        prefix_alignment_exact &= bool(turn["output_token_alignment_exact"])
    reward = (
        ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    )
    agent_wall = elapsed_s(result.get("agent_execution"))
    trial_wall = (
        parse_time(result["finished_at"]) - parse_time(result["started_at"])
    ).total_seconds()
    trajectory = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "source": {
            "job_name": job_name,
            "trial_name": result.get("trial_name"),
            "method": job_metadata["spec"]["method"],
            "seed": job_metadata["spec"]["seed"],
            "attempt_index": attempt_index,
            "model": job_metadata["models"]["target"],
            "revisions": job_metadata.get("revisions"),
        },
        "counts": {
            "model_turns": len(turns),
            "tool_calls": len(tool_results),
            "checkpoints": len(checkpoints),
            "generated_tokens": session_output_tokens,
            "incomplete_model_requests_dropped": incomplete_requests_dropped,
            "nonalignable_model_turns": sum(
                not bool(turn["output_token_alignment_exact"])
                for turn in turns
            ),
            "token_checkpoints_skipped_for_alignment": skipped_token_checkpoints,
            "agent_result_token_accounting_mismatch": bool(
                input_token_delta or output_token_delta
            ),
        },
        "costs": {
            "generator": {
                "calls": len(turns),
                "input_tokens": session_input_tokens,
                "output_tokens": session_output_tokens,
                "agent_result_input_tokens": agent_result_input_tokens,
                "agent_result_output_tokens": agent_result_output_tokens,
                "session_minus_agent_result_input_tokens": input_token_delta,
                "session_minus_agent_result_output_tokens": output_token_delta,
                "token_accounting_status": token_accounting_status,
                "active_accelerator_seconds": None,
                "active_accelerator_seconds_status": (
                    "available_only_as_job_aggregate_in_prometheus"
                ),
            },
            "tools": {
                "calls": len(tool_results),
                "errored_calls": sum(
                    bool(message.get("isError")) for message in tool_results
                ),
                "wall_time_s": None,
                "wall_time_status": "not_available_per_tool",
            },
            "semantic_verifier": {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_s": 0.0,
                "active_accelerator_seconds": 0.0,
            },
            "total": {
                "agent_wall_time_s": agent_wall,
                "trial_wall_time_s": trial_wall,
                "accelerators_allocated": accelerators_allocated,
                "allocated_accelerator_seconds": None,
                "allocated_accelerator_seconds_status": (
                    "unavailable_per_trajectory_under_concurrent_source_serving"
                ),
            },
        },
        "resume_capability": {
            "serialized_model_prefix": prefix_alignment_exact,
            "model_prefix_resumable_by_engine": prefix_alignment_exact,
            "model_prefix_status": (
                "exact" if prefix_alignment_exact
                else "at_least_one_tool_use_control_token_not_reconstructable"
            ),
            "terminal_environment_cloneable": False,
            "terminal_environment_status": (
                "requires_a_future_snapshot_or_replay_backend"
            ),
        },
    }
    label = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "final_reward": reward,
        "eventual_success": (
            False if reward is None else float(reward) >= success_threshold
        ),
        "errored": errored,
    }
    return trajectory, checkpoints, label


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("only semantic actionability manifest schema_version=1 works")
    checkpoint = manifest["checkpoint_policy"]
    base = int(checkpoint["base_interval_tokens"])
    derived = [int(value) for value in checkpoint["derived_interval_tokens"]]
    if base <= 0 or any(value < base or value % base for value in derived):
        raise ValueError(
            "derived intervals must be positive multiples of base interval"
        )
    axes = manifest["online_scaling_axes"]
    if 256 not in axes["checkpoint_interval_tokens"]:
        raise ValueError("online scaling axes must retain the 256-token interval")
    if 64 not in axes["particles"]:
        raise ValueError("online scaling axes must retain N=64")
    if 8 not in axes["verifier_calls_per_checkpoint"]:
        raise ValueError("online scaling axes must retain eight verifier calls")


def load_tokenizer(manifest: dict[str, Any], tokenizer_path: Path | None) -> Any:
    from transformers import AutoTokenizer

    spec = manifest["tokenizer"]
    source = str(tokenizer_path) if tokenizer_path else spec["model"]
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(spec.get("trust_remote_code", True)),
        "local_files_only": bool(spec.get("local_files_only", True)),
    }
    if not tokenizer_path and spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    return AutoTokenizer.from_pretrained(source, **kwargs)


def export_dataset(
    manifest: dict[str, Any],
    jobs_dir: Path,
    output_dir: Path,
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    validate_manifest(manifest)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    dataset_id = manifest["dataset_id"]
    task_ids = [entry["id"] for entry in manifest["benchmark"]["tasks"]]
    base_interval = int(manifest["checkpoint_policy"]["base_interval_tokens"])
    derived_intervals = [
        int(value)
        for value in manifest["checkpoint_policy"]["derived_interval_tokens"]
    ]
    accelerators = int(manifest["cost_accounting"]["accelerators_allocated"])
    success_threshold = float(manifest["labels"]["success_threshold"])

    trajectories: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_job_costs: list[dict[str, Any]] = []
    for source in manifest["source_jobs"]:
        job_dir = jobs_dir / source["job_name"]
        metadata_path = job_dir / "experiment.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing experiment metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") != "complete":
            raise ValueError(f"source job is not complete: {job_dir}")
        started_at = metadata.get("started_at")
        finished_at = metadata.get("finished_at")
        if not started_at or not finished_at:
            raise ValueError(f"source job lacks allocation timestamps: {job_dir}")
        allocation_wall_time_s = (
            parse_time(finished_at) - parse_time(started_at)
        ).total_seconds()
        all_result_paths = sorted(job_dir.glob("*__*/result.json"))
        exported_for_job = 0
        for task_id in task_ids:
            result_paths = sorted(job_dir.glob(f"{task_id}__*/result.json"))
            if not result_paths:
                raise FileNotFoundError(f"no {task_id} result in {job_dir}")
            for attempt_index, result_path in enumerate(result_paths):
                trajectory, trial_checkpoints, label = export_trial(
                    dataset_id=dataset_id,
                    task_id=task_id,
                    trial_dir=result_path.parent,
                    job_metadata=metadata,
                    attempt_index=attempt_index,
                    tokenizer=tokenizer,
                    base_interval=base_interval,
                    derived_intervals=derived_intervals,
                    accelerators_allocated=accelerators,
                    success_threshold=success_threshold,
                )
                trajectories.append(trajectory)
                checkpoints.extend(trial_checkpoints)
                labels.append(label)
                exported_for_job += 1
        source_job_costs.append(
            {
                "job_name": metadata["job_name"],
                "allocation_started_at": started_at,
                "allocation_finished_at": finished_at,
                "allocation_wall_time_s": allocation_wall_time_s,
                "accelerators_allocated": accelerators,
                "allocated_accelerator_seconds": (
                    allocation_wall_time_s * accelerators
                ),
                "exported_trajectories": exported_for_job,
                "job_result_trajectories": len(all_result_paths),
                "covers_exported_dataset_exactly": (
                    set(metadata.get("tasks") or []) == set(task_ids)
                    and exported_for_job == len(all_result_paths)
                ),
            }
        )

    collection = manifest.get("collection") or manifest.get("smoke")
    if not collection:
        raise ValueError("manifest must define collection or smoke expectations")
    expected = int(collection["expected_trajectories"])
    if len(trajectories) != expected:
        raise ValueError(
            f"expected {expected} smoke trajectories, exported {len(trajectories)}"
        )
    checkpoint_ids = [row["checkpoint_id"] for row in checkpoints]
    trajectory_ids = [row["trajectory_id"] for row in trajectories]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("checkpoint IDs are not unique")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("trajectory IDs are not unique")
    for checkpoint in checkpoints:
        leaked = label_keys(checkpoint)
        if leaked:
            raise ValueError(
                f"label leakage in {checkpoint['checkpoint_id']}: {leaked}"
            )

    trajectories_path = output_dir / "trajectories.jsonl"
    checkpoints_path = output_dir / "checkpoints.jsonl"
    labels_path = output_dir / "labels.jsonl"
    write_jsonl(trajectories_path, trajectories)
    write_jsonl(checkpoints_path, checkpoints)
    write_jsonl(labels_path, labels)
    trigger_counts = Counter(row["trigger"]["kind"] for row in checkpoints)
    exact_alignment = all(
        row["trigger"]["target_generated_tokens"]
        == row["trigger"]["actual_generated_tokens"]
        for row in checkpoints
        if row["trigger"]["kind"] == "token_interval"
    )
    summary = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "manifest_sha256": hashlib.sha256(
            canonical_json(manifest).encode()
        ).hexdigest(),
        "checkpoint_policy": manifest["checkpoint_policy"],
        "online_scaling_axes": manifest["online_scaling_axes"],
        "counts": {
            "tasks": len(task_ids),
            "trajectories": len(trajectories),
            "checkpoints": len(checkpoints),
            "token_interval_checkpoints": trigger_counts["token_interval"],
            "post_tool_checkpoints": trigger_counts["post_tool"],
            "errored_trajectories": sum(
                bool(label["errored"]) for label in labels
            ),
            "incomplete_model_requests_dropped": sum(
                row["counts"]["incomplete_model_requests_dropped"]
                for row in trajectories
            ),
            "trajectories_with_nonalignable_turns": sum(
                row["counts"]["nonalignable_model_turns"] > 0
                for row in trajectories
            ),
            "nonalignable_model_turns": sum(
                row["counts"]["nonalignable_model_turns"] for row in trajectories
            ),
            "token_checkpoints_skipped_for_alignment": sum(
                row["counts"]["token_checkpoints_skipped_for_alignment"]
                for row in trajectories
            ),
            "errored_trajectories_with_token_accounting_mismatch": sum(
                row["counts"]["agent_result_token_accounting_mismatch"]
                for row in trajectories
            ),
            "session_minus_agent_result_input_tokens": sum(
                row["costs"]["generator"][
                    "session_minus_agent_result_input_tokens"
                ]
                for row in trajectories
            ),
            "session_minus_agent_result_output_tokens": sum(
                row["costs"]["generator"][
                    "session_minus_agent_result_output_tokens"
                ]
                for row in trajectories
            ),
        },
        "source_generation_cost": {
            "allocation_scope": "one generator server per source job",
            "source_jobs": source_job_costs,
            "allocation_wall_time_s": sum(
                row["allocation_wall_time_s"] for row in source_job_costs
            ),
            "allocated_accelerator_seconds": sum(
                row["allocated_accelerator_seconds"] for row in source_job_costs
            ),
            "covers_exported_dataset_exactly": all(
                row["covers_exported_dataset_exactly"] for row in source_job_costs
            ),
            "per_trajectory_attribution": "not_identifiable_under_concurrency",
        },
        "files": {
            "trajectories": {
                "path": trajectories_path.name,
                "sha256": sha256_file(trajectories_path),
            },
            "checkpoints": {
                "path": checkpoints_path.name,
                "sha256": sha256_file(checkpoints_path),
                "safe_for_semantic_verifier": True,
            },
            "labels": {
                "path": labels_path.name,
                "sha256": sha256_file(labels_path),
                "must_not_be_given_to_semantic_verifier": True,
            },
        },
        "validation": {
            "offline_export_passed": True,
            "exact_token_checkpoint_alignment": exact_alignment,
            "stable_unique_ids": True,
            "checkpoint_records_have_no_label_fields": True,
            "all_serialized_model_prefixes_exact": all(
                row["resume_capability"]["serialized_model_prefix"]
                for row in trajectories
            ),
            "all_model_prefixes_resumable_by_engine": all(
                row["resume_capability"]["model_prefix_resumable_by_engine"]
                for row in trajectories
            ),
            "terminal_environment_cloneable": False,
            "terminal_environment_status": (
                "blocked_until_snapshot_or_deterministic_replay_backend_exists"
            ),
            "cost_fields": {
                "token_counts": (
                    "exact_per_turn_session_usage; agent-result disagreements "
                    "retained_on_errored_trajectories"
                ),
                "agent_and_trial_wall_time": "exact",
                "allocated_accelerator_time": (
                    "exact_per_complete_source_job_not_per_trajectory"
                ),
                "generator_active_time": "prometheus_job_aggregate_only",
                "per_tool_wall_time": "unavailable",
                "semantic_verifier_cost": "exact_zero_before_scoring",
            },
        },
    }
    write_json(output_dir / "manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("~/agentbench/jobs"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help=(
            "Optional local tokenizer snapshot; otherwise use manifest "
            "model/revision."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    tokenizer = load_tokenizer(
        manifest,
        args.tokenizer_path.expanduser().resolve() if args.tokenizer_path else None,
    )
    summary = export_dataset(
        manifest,
        args.jobs_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        tokenizer=tokenizer,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

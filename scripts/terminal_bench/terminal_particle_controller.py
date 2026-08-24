#!/usr/bin/env python3
"""Drive online Terminal-Bench particles with coupled model and Docker state.

Each particle owns one Docker environment, OpenAI-compatible message prefix,
replay manifest, lineage, and cost record. The controller calls a shared model
server, routes tool calls to the matching container, seals the resulting state,
and can resample a survivor by validated deterministic replay.

The live-smoke command starts two siblings from a captured Pi checkpoint,
advances both with independent sampling seeds, clones one survivor, proves that
environment and model-prefix state were copied together, and advances the
resampled pair again.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import posixpath
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import smcsd_pi_contract
import terminal_particle_backend as backend


CONTROLLER_SCHEMA_VERSION = 1
MAX_TOOL_OUTPUT_CHARS = 50 * 1024
MAX_TOOL_OUTPUT_LINES = 2000


class ControllerError(RuntimeError):
    """A coupled particle transition cannot be completed safely."""


class SafeToolError(ControllerError):
    """A tool failed validation before any environment mutation began."""


@dataclass
class ToolExecution:
    content: str
    is_error: bool
    wall_time_s: float
    return_code: int | None = None


@dataclass
class Particle:
    slot: int
    container_name: str
    container_id: str
    manifest: dict[str, Any]
    messages: list[dict[str, Any]]
    lineage: dict[str, Any]
    state: dict[str, Any]
    replay_cost: dict[str, Any]
    rounds: int = 0
    finished: bool = False
    resampled_from: str | None = None
    cost: dict[str, float | int] = field(
        default_factory=lambda: {
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_wall_time_s": 0.0,
            "tool_calls": 0,
            "tool_wall_time_s": 0.0,
        }
    )

    @property
    def particle_id(self) -> str:
        return str(self.lineage["particle_id"])

    @property
    def model_prefix_sha256(self) -> str:
        return backend.canonical_json_sha256(self.messages)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def epoch_milliseconds() -> int:
    return int(time.time() * 1000)


def load_captured_checkpoint(
    capture_path: Path,
    *,
    after_tool_call_id: str,
) -> dict[str, Any]:
    """Return the first provider request immediately after a selected tool."""
    matches: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        capture_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ControllerError(
                f"capture row {line_number} has no provider payload"
            )
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        last = messages[-1]
        if (
            isinstance(last, dict)
            and last.get("role") == "tool"
            and last.get("tool_call_id") == after_tool_call_id
        ):
            matches.append(copy.deepcopy(payload))
    if len(matches) != 1:
        raise ControllerError(
            f"expected one provider request after {after_tool_call_id}, "
            f"found {len(matches)}"
        )
    payload = matches[0]
    if not isinstance(payload.get("model"), str):
        raise ControllerError("captured request has no model")
    if not isinstance(payload.get("tools"), list):
        raise ControllerError("captured request has no tool schemas")
    return payload


def validate_checkpoint_binding(
    manifest: dict[str, Any],
    request_payload: dict[str, Any],
    *,
    after_tool_call_id: str,
) -> None:
    events = manifest.get("tool_events") or []
    if not events or events[-1].get("call_id") != after_tool_call_id:
        raise ControllerError(
            "replay manifest and provider checkpoint end at different tool calls"
        )
    messages = request_payload.get("messages") or []
    if (
        not messages
        or messages[-1].get("role") != "tool"
        or messages[-1].get("tool_call_id") != after_tool_call_id
    ):
        raise ControllerError("provider checkpoint does not end at selected tool")


def truncate_tool_output(value: str) -> str:
    lines = value.splitlines(keepends=True)
    if len(lines) > MAX_TOOL_OUTPUT_LINES:
        value = "".join(lines[-MAX_TOOL_OUTPUT_LINES:])
    encoded = value.encode("utf-8", "replace")
    if len(encoded) > MAX_TOOL_OUTPUT_CHARS:
        encoded = encoded[-MAX_TOOL_OUTPUT_CHARS:]
        value = encoded.decode("utf-8", "replace")
    return value


def format_bash_result(
    stdout: str,
    stderr: str,
    return_code: int,
) -> str:
    content = stdout + stderr
    if return_code != 0:
        content = content.rstrip()
        if content:
            content += "\n\n"
        content += f"Command exited with code {return_code}"
    elif not content:
        content = "(no output)"
    return truncate_tool_output(content)


def resolve_tool_path(path: Any, workdir: str) -> str:
    if not isinstance(path, str) or not path:
        raise SafeToolError("tool path must be a non-empty string")
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(workdir, path))


def read_text_tool(
    docker: backend.DockerCLI,
    container: str,
    path: str,
    *,
    offset: Any = None,
    limit: Any = None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="smcsd-live-read-") as raw:
        local = Path(raw) / "value"
        docker.copy_from(container, path, local)
        try:
            value = local.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise SafeToolError(
                f"read only supports UTF-8 text: {path}"
            ) from error
    lines = value.splitlines(keepends=True)
    start = 1 if offset is None else int(offset)
    if start < 1:
        raise SafeToolError("read offset must be at least 1")
    selected = lines[start - 1 :]
    if limit is not None:
        count = int(limit)
        if count < 0:
            raise SafeToolError("read limit must be non-negative")
        selected = selected[:count]
    return truncate_tool_output("".join(selected))


def write_text_tool(
    docker: backend.DockerCLI,
    container: str,
    path: str,
    content: Any,
) -> None:
    if not isinstance(content, str):
        raise SafeToolError("write content must be a string")
    parent = posixpath.dirname(path) or "/"
    docker.run(["exec", container, "mkdir", "-p", parent])
    docker.write_text(container, path, content)


def edit_text_tool(
    docker: backend.DockerCLI,
    container: str,
    path: str,
    raw_edits: Any,
) -> None:
    edits = backend.normalized_edits(raw_edits)
    with tempfile.TemporaryDirectory(prefix="smcsd-live-edit-") as raw:
        local = Path(raw) / "value"
        try:
            docker.copy_from(container, path, local)
            original = local.read_text(encoding="utf-8")
        except (OSError, UnicodeError, backend.ReplayError) as error:
            raise SafeToolError(f"cannot read edit target: {path}") from error
        replacements: list[tuple[int, int, str]] = []
        for index, edit in enumerate(edits):
            old = edit.get("oldText")
            new = edit.get("newText")
            if not isinstance(old, str) or not isinstance(new, str):
                raise SafeToolError(
                    f"edit {index} is missing string oldText/newText"
                )
            occurrences = original.count(old)
            if occurrences != 1:
                raise SafeToolError(
                    f"edit {index} expected one oldText occurrence, "
                    f"found {occurrences}: {path}"
                )
            start = original.index(old)
            replacements.append((start, start + len(old), new))
        ordered = sorted(replacements)
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] > current[0]:
                raise SafeToolError("edit replacements overlap")
        value = original
        for start, end, replacement in reversed(ordered):
            value = value[:start] + replacement + value[end:]
        local.write_text(value, encoding="utf-8")
        docker.copy_to(local, container, path)


def execute_live_tool(
    docker: backend.DockerCLI,
    container: str,
    tool_call: dict[str, Any],
    *,
    workdir: str,
) -> tuple[ToolExecution, dict[str, Any]]:
    function = tool_call.get("function") or {}
    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )
    except json.JSONDecodeError as error:
        raise ControllerError(f"invalid arguments for {name}") from error
    if not isinstance(arguments, dict):
        raise ControllerError(f"arguments for {name} are not an object")

    started = time.perf_counter()
    safe_no_mutation = False
    try:
        if name == "bash":
            command = arguments.get("command")
            if not isinstance(command, str):
                raise SafeToolError("bash command must be a string")
            process = docker.exec_bash(container, command, workdir=workdir)
            result = ToolExecution(
                content=format_bash_result(
                    process.stdout,
                    process.stderr,
                    process.returncode,
                ),
                is_error=process.returncode != 0,
                return_code=process.returncode,
                wall_time_s=time.perf_counter() - started,
            )
        elif name == "read":
            path = resolve_tool_path(arguments.get("path"), workdir)
            result = ToolExecution(
                content=read_text_tool(
                    docker,
                    container,
                    path,
                    offset=arguments.get("offset"),
                    limit=arguments.get("limit"),
                ),
                is_error=False,
                wall_time_s=time.perf_counter() - started,
            )
        elif name == "write":
            path = resolve_tool_path(arguments.get("path"), workdir)
            write_text_tool(
                docker,
                container,
                path,
                arguments.get("content"),
            )
            arguments["path"] = path
            result = ToolExecution(
                content="(no output)",
                is_error=False,
                wall_time_s=time.perf_counter() - started,
            )
        elif name == "edit":
            path = resolve_tool_path(arguments.get("path"), workdir)
            edit_text_tool(
                docker,
                container,
                path,
                arguments.get("edits"),
            )
            arguments["path"] = path
            result = ToolExecution(
                content="(no output)",
                is_error=False,
                wall_time_s=time.perf_counter() - started,
            )
        else:
            raise ControllerError(f"unsupported live tool: {name}")
    except SafeToolError as error:
        safe_no_mutation = True
        result = ToolExecution(
            content=str(error),
            is_error=True,
            wall_time_s=time.perf_counter() - started,
        )
    except ControllerError:
        raise
    except Exception as error:
        if name in {"write", "edit"}:
            raise ControllerError(
                f"{name} failed with ambiguous mutation state"
            ) from error
        result = ToolExecution(
            content=str(error),
            is_error=True,
            wall_time_s=time.perf_counter() - started,
        )

    event = {
        "call_id": str(tool_call.get("id") or uuid.uuid4().hex),
        "name": name,
        "arguments": arguments,
        "assistant_timestamp_ms": epoch_milliseconds(),
        "result_timestamp_ms": epoch_milliseconds(),
        "expected_error": result.is_error,
        "result_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
    }
    if safe_no_mutation and name in {"write", "edit"}:
        event["mutation_status"] = "none"
    return result, event


ModelCaller = Callable[[dict[str, Any], float], dict[str, Any]]


class TerminalParticleController:
    """Own coupled environment/model state and account every actual operation."""

    def __init__(
        self,
        docker: backend.DockerCLI,
        *,
        request_template: dict[str, Any],
        base_url: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_s: float,
        model_caller: ModelCaller | None = None,
    ) -> None:
        self.docker = docker
        self.request_template = copy.deepcopy(request_template)
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.model_caller = model_caller or self._post_stream
        self._lock = threading.Lock()
        self.cost: dict[str, float | int] = {
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_request_wall_time_s": 0.0,
            "tool_calls": 0,
            "tool_wall_time_s": 0.0,
            "environment_replays": 0,
            "environment_replay_wall_time_s": 0.0,
        }

    def _post_stream(
        self,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        return smcsd_pi_contract.post_stream(
            self.base_url,
            payload,
            timeout_s,
        )

    def _record_cost(self, values: dict[str, float | int]) -> None:
        with self._lock:
            for key, value in values.items():
                self.cost[key] = self.cost.get(key, 0) + value

    def _model_payload(self, particle: Particle, seed: int) -> dict[str, Any]:
        payload = {
            "model": self.request_template["model"],
            "messages": copy.deepcopy(particle.messages),
            "tools": copy.deepcopy(self.request_template["tools"]),
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": seed,
            "max_tokens": self.max_tokens,
        }
        if "chat_template_kwargs" in self.request_template:
            payload["chat_template_kwargs"] = copy.deepcopy(
                self.request_template["chat_template_kwargs"]
            )
        return payload

    def spawn(
        self,
        manifest: dict[str, Any],
        messages: Sequence[dict[str, Any]],
        *,
        slot: int,
        name: str,
        resampled_from: str | None = None,
    ) -> Particle:
        result = backend.fork_manifest(self.docker, manifest, name=name)
        child_manifest = copy.deepcopy(manifest)
        child_manifest["lineage"] = copy.deepcopy(result["lineage"])
        child_manifest["expected_state"] = copy.deepcopy(result["state"])
        child_messages = copy.deepcopy(list(messages))
        child_manifest["model_prefix"] = {
            "checkpoint_id": manifest.get("model_prefix", {}).get(
                "checkpoint_id"
            ),
            "sha256": backend.canonical_json_sha256(child_messages),
            "representation": "openai_messages_v1",
            "kv_cache_handle": None,
        }
        replay_cost = copy.deepcopy(result["cost"])
        self._record_cost(
            {
                "environment_replays": 1,
                "environment_replay_wall_time_s": float(
                    replay_cost["replay_wall_time_s"]
                ),
            }
        )
        return Particle(
            slot=slot,
            container_name=name,
            container_id=result["container_id"],
            manifest=child_manifest,
            messages=child_messages,
            lineage=copy.deepcopy(result["lineage"]),
            state=copy.deepcopy(result["state"]),
            replay_cost=replay_cost,
            resampled_from=resampled_from,
        )

    def seal(self, particle: Particle) -> None:
        environment = particle.manifest["base_environment"]
        particle.state = {
            "filesystem_sha256": self.docker.filesystem_digest(
                particle.container_name,
                environment["state_roots"],
            ),
            "process_fingerprint": self.docker.process_fingerprint(
                particle.container_name
            ),
        }
        particle.manifest["expected_state"] = copy.deepcopy(particle.state)
        particle.manifest["lineage"] = copy.deepcopy(particle.lineage)
        particle.manifest["model_prefix"] = {
            "checkpoint_id": (
                f"{particle.particle_id}:assistant-round-{particle.rounds}"
            ),
            "sha256": particle.model_prefix_sha256,
            "representation": "openai_messages_v1",
            "kv_cache_handle": None,
        }
        transcript = particle.manifest.setdefault("transcript", {})
        events = particle.manifest["tool_events"]
        transcript["completed_tool_calls"] = len(events)
        transcript["events_sha256"] = backend.canonical_json_sha256(events)
        transcript["controller_messages_sha256"] = particle.model_prefix_sha256
        particle.manifest["sealed_from"] = {
            "container_id": particle.container_id,
            "container_name": particle.container_name,
        }

    def advance(self, particle: Particle, *, seed: int) -> dict[str, Any]:
        if particle.finished:
            return {
                "particle_id": particle.particle_id,
                "slot": particle.slot,
                "status": "already_finished",
            }
        request_prefix = particle.model_prefix_sha256
        result = self.model_caller(
            self._model_payload(particle, seed),
            self.timeout_s,
        )
        tool_calls = copy.deepcopy(result.get("tool_calls") or [])
        for call in tool_calls:
            function = call.get("function")
            if function is None:
                function = {
                    "name": call.get("name") or "",
                    "arguments": call.get("arguments") or "{}",
                }
                call["function"] = function
                call.pop("name", None)
                call.pop("arguments", None)
            try:
                json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as error:
                raise ControllerError(
                    f"model returned invalid JSON arguments: {call}"
                ) from error
            call.setdefault("id", f"call_{uuid.uuid4().hex}")
            call.setdefault("type", "function")

        assistant = {
            "role": "assistant",
            "content": result.get("content") or None,
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        particle.messages.append(assistant)

        workdir = str(particle.manifest["base_environment"]["workdir"])
        tool_rows: list[dict[str, Any]] = []
        for call in tool_calls:
            tool_result, event = execute_live_tool(
                self.docker,
                particle.container_name,
                call,
                workdir=workdir,
            )
            particle.manifest["tool_events"].append(event)
            particle.messages.append(
                {
                    "role": "tool",
                    "content": tool_result.content,
                    "tool_call_id": event["call_id"],
                }
            )
            particle.cost["tool_calls"] += 1
            particle.cost["tool_wall_time_s"] += tool_result.wall_time_s
            self._record_cost(
                {
                    "tool_calls": 1,
                    "tool_wall_time_s": tool_result.wall_time_s,
                }
            )
            tool_rows.append(
                {
                    "call_id": event["call_id"],
                    "name": event["name"],
                    "arguments": copy.deepcopy(event["arguments"]),
                    "is_error": event["expected_error"],
                    "result_sha256": event["result_sha256"],
                    "wall_time_s": tool_result.wall_time_s,
                }
            )

        usage = result.get("usage") or {}
        prompt_tokens = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        completion_tokens = int(
            usage.get(
                "completion_tokens",
                usage.get("output_tokens", 0),
            )
            or 0
        )
        model_wall_time = float(result.get("latency_s") or 0.0)
        particle.cost["model_calls"] += 1
        particle.cost["prompt_tokens"] += prompt_tokens
        particle.cost["completion_tokens"] += completion_tokens
        particle.cost["model_wall_time_s"] += model_wall_time
        self._record_cost(
            {
                "model_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_request_wall_time_s": model_wall_time,
            }
        )
        particle.rounds += 1
        particle.finished = not tool_calls
        self.seal(particle)
        return {
            "particle_id": particle.particle_id,
            "slot": particle.slot,
            "seed": seed,
            "request_prefix_sha256": request_prefix,
            "response_prefix_sha256": particle.model_prefix_sha256,
            "finish_reason": result.get("finish_reason"),
            "finished": particle.finished,
            "content_sha256": hashlib.sha256(
                (result.get("content") or "").encode()
            ).hexdigest(),
            "tool_calls": tool_rows,
            "usage": usage,
            "model_wall_time_s": model_wall_time,
        }

    def advance_many(
        self,
        particles: Sequence[Particle],
        *,
        seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        if len(particles) != len(seeds):
            raise ValueError("particles and seeds must have equal length")
        with ThreadPoolExecutor(max_workers=len(particles)) as executor:
            futures = [
                executor.submit(self.advance, particle, seed=seed)
                for particle, seed in zip(particles, seeds)
            ]
            return [future.result() for future in futures]

    def resample(
        self,
        *,
        survivor: Particle,
        replaced: Particle,
        name: str,
    ) -> Particle:
        self.seal(survivor)
        child = self.spawn(
            survivor.manifest,
            survivor.messages,
            slot=replaced.slot,
            name=name,
            resampled_from=survivor.particle_id,
        )
        if child.model_prefix_sha256 != survivor.model_prefix_sha256:
            self.docker.remove(child.container_name)
            raise ControllerError("resampling copied the wrong model prefix")
        if child.state != survivor.state:
            self.docker.remove(child.container_name)
            raise ControllerError("resampling copied the wrong environment state")
        self.docker.remove(replaced.container_name)
        return child

    def remove_particles(self, particles: Sequence[Particle]) -> None:
        for name in sorted({particle.container_name for particle in particles}):
            self.docker.remove(name)


def particle_summary(particle: Particle) -> dict[str, Any]:
    return {
        "slot": particle.slot,
        "particle_id": particle.particle_id,
        "parent_particle_id": particle.lineage["parent_particle_id"],
        "generation": particle.lineage["generation"],
        "container_name": particle.container_name,
        "model_prefix_sha256": particle.model_prefix_sha256,
        "state": copy.deepcopy(particle.state),
        "rounds": particle.rounds,
        "finished": particle.finished,
        "resampled_from": particle.resampled_from,
        "cost": copy.deepcopy(particle.cost),
        "replay_cost": copy.deepcopy(particle.replay_cost),
        "tool_event_count": len(particle.manifest["tool_events"]),
    }


def particle_checkpoint(particle: Particle) -> dict[str, Any]:
    """Return a replay-restorable particle artifact with its exact transcript."""
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "controller": "openai_terminal_particle_controller_v1",
        "saved_at": utc_now(),
        "restorable_by_replay": True,
        "particle": particle_summary(particle),
        "manifest": copy.deepcopy(particle.manifest),
        "messages": copy.deepcopy(particle.messages),
        "message_prefix_sha256": particle.model_prefix_sha256,
    }


def write_particle_checkpoints(
    report_path: Path,
    particles: Sequence[Particle],
) -> list[str]:
    directory = report_path.parent / f"{report_path.stem}.particles"
    paths: list[str] = []
    for particle in sorted(particles, key=lambda item: item.slot):
        path = directory / f"slot-{particle.slot}.json"
        backend.atomic_write_json(path, particle_checkpoint(particle))
        paths.append(str(path.resolve()))
    return paths


def load_particle_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != CONTROLLER_SCHEMA_VERSION:
        raise ControllerError("unsupported controller checkpoint schema")
    if checkpoint.get("controller") != "openai_terminal_particle_controller_v1":
        raise ControllerError("unsupported controller checkpoint")
    if checkpoint.get("restorable_by_replay") is not True:
        raise ControllerError("particle checkpoint is not replay-restorable")
    messages = checkpoint.get("messages")
    manifest = checkpoint.get("manifest")
    if not isinstance(messages, list) or not isinstance(manifest, dict):
        raise ControllerError("particle checkpoint is missing messages/manifest")
    prefix = backend.canonical_json_sha256(messages)
    if checkpoint.get("message_prefix_sha256") != prefix:
        raise ControllerError("particle checkpoint message hash mismatch")
    if manifest.get("model_prefix", {}).get("sha256") != prefix:
        raise ControllerError("manifest and controller model prefixes differ")
    events = manifest.get("tool_events") or []
    if (
        manifest.get("transcript", {}).get("events_sha256")
        != backend.canonical_json_sha256(events)
    ):
        raise ControllerError("particle checkpoint event hash mismatch")
    if manifest.get("expected_state") != checkpoint.get("particle", {}).get(
        "state"
    ):
        raise ControllerError("particle checkpoint state binding mismatch")
    return checkpoint


def restore_particle_checkpoint(
    docker: backend.DockerCLI,
    *,
    checkpoint_path: Path,
    name: str,
) -> dict[str, Any]:
    checkpoint = load_particle_checkpoint(checkpoint_path)
    result = backend.fork_manifest(
        docker,
        checkpoint["manifest"],
        name=name,
    )
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "status": "pass",
        "checkpoint": str(checkpoint_path.resolve()),
        "container_name": name,
        "message_prefix_sha256": checkpoint["message_prefix_sha256"],
        "message_count": len(checkpoint["messages"]),
        "fork": result,
    }


def run_live_smoke(
    docker: backend.DockerCLI,
    *,
    manifest_path: Path,
    capture_path: Path,
    after_tool_call_id: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    pre_resample_turns: int,
    post_resample_turns: int,
    seed_base: int,
    name_prefix: str,
    output: Path | None,
    keep_containers: bool,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = load_captured_checkpoint(
        capture_path,
        after_tool_call_id=after_tool_call_id,
    )
    validate_checkpoint_binding(
        manifest,
        checkpoint,
        after_tool_call_id=after_tool_call_id,
    )
    controller = TerminalParticleController(
        docker,
        request_template=checkpoint,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    suffix = uuid.uuid4().hex[:8]
    particles: list[Particle] = []
    all_particles: list[Particle] = []
    started = time.perf_counter()
    try:
        for slot in range(2):
            particle = controller.spawn(
                manifest,
                checkpoint["messages"],
                slot=slot,
                name=f"{name_prefix}-{suffix}-p{slot}",
            )
            particles.append(particle)
            all_particles.append(particle)

        initial = [particle_summary(particle) for particle in particles]
        common_environment = particles[0].state == particles[1].state
        common_model_prefix = (
            particles[0].model_prefix_sha256
            == particles[1].model_prefix_sha256
        )
        if not common_environment or not common_model_prefix:
            raise ControllerError("initial siblings do not share one checkpoint")

        pre_rounds: list[list[dict[str, Any]]] = []
        for round_index in range(pre_resample_turns):
            rows = controller.advance_many(
                particles,
                seeds=[
                    seed_base + round_index * 100 + particle.slot
                    for particle in particles
                ],
            )
            pre_rounds.append(rows)
            if all(particle.finished for particle in particles):
                break

        pre_prefixes = {
            particle.model_prefix_sha256 for particle in particles
        }
        pre_states = {
            particle.state["filesystem_sha256"] for particle in particles
        }
        continuation_diverged = len(pre_prefixes) > 1 or len(pre_states) > 1
        if not continuation_diverged:
            raise ControllerError(
                "independent sibling continuations did not diverge; "
                "increase pre-resample turns or sampling temperature"
            )

        survivor = particles[0]
        replaced = particles[1]
        survivor_before = particle_summary(survivor)
        replaced_before = particle_summary(replaced)
        clone = controller.resample(
            survivor=survivor,
            replaced=replaced,
            name=f"{name_prefix}-{suffix}-p1-resampled",
        )
        all_particles.append(clone)
        particles = [survivor, clone]
        resample_equivalence = {
            "model_prefix": (
                survivor.model_prefix_sha256 == clone.model_prefix_sha256
            ),
            "filesystem": (
                survivor.state["filesystem_sha256"]
                == clone.state["filesystem_sha256"]
            ),
            "processes": (
                survivor.state["process_fingerprint"]
                == clone.state["process_fingerprint"]
            ),
            "lineage_parent": (
                clone.lineage["parent_particle_id"] == survivor.particle_id
            ),
        }
        if not all(resample_equivalence.values()):
            raise ControllerError("resampled particle coupling check failed")

        post_rounds: list[list[dict[str, Any]]] = []
        for round_index in range(post_resample_turns):
            rows = controller.advance_many(
                particles,
                seeds=[
                    seed_base + 10_000 + round_index * 100 + particle.slot
                    for particle in particles
                ],
            )
            post_rounds.append(rows)
            if all(particle.finished for particle in particles):
                break

        report = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "status": "pass",
            "generated_at": utc_now(),
            "controller": "openai_terminal_particle_controller_v1",
            "checkpoint": {
                "manifest": str(manifest_path.resolve()),
                "capture": str(capture_path.resolve()),
                "after_tool_call_id": after_tool_call_id,
                "source_events": len(manifest["tool_events"]),
                "model": checkpoint["model"],
                "model_prefix_sha256": backend.canonical_json_sha256(
                    checkpoint["messages"]
                ),
            },
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed_base": seed_base,
            },
            "initial_siblings": initial,
            "initial_coupling": {
                "environment": common_environment,
                "model_prefix": common_model_prefix,
            },
            "pre_resample_rounds": pre_rounds,
            "pre_resample_divergence": {
                "continuation_diverged": continuation_diverged,
                "unique_model_prefixes": len(pre_prefixes),
                "unique_filesystems": len(pre_states),
            },
            "resampling": {
                "survivor_before": survivor_before,
                "replaced_before": replaced_before,
                "clone": particle_summary(clone),
                "equivalence": resample_equivalence,
            },
            "post_resample_rounds": post_rounds,
            "final_particles": [
                particle_summary(particle) for particle in particles
            ],
            "global_cost": copy.deepcopy(controller.cost),
            "wall_time_s": time.perf_counter() - started,
            "limitations": {
                "kv_cache_handle": None,
                "model_continuation": (
                    "exact serialized OpenAI messages; server prefix cache may "
                    "reuse tokens but no persistent KV handle is exported"
                ),
                "selection": (
                    "slot 0 is selected deterministically in this coupling "
                    "test; semantic resampling policy is the next experiment"
                ),
            },
        }
        if output is not None:
            report["particle_checkpoints"] = write_particle_checkpoints(
                output,
                particles,
            )
            backend.atomic_write_json(output, report)
        return report
    finally:
        if not keep_containers:
            controller.remove_particles(all_particles)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-binary", default="docker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live-smoke")
    live.add_argument("--manifest", type=Path, required=True)
    live.add_argument("--capture", type=Path, required=True)
    live.add_argument("--after-tool-call-id", required=True)
    live.add_argument("--base-url", default="http://127.0.0.1:30000")
    live.add_argument("--temperature", type=float, default=0.8)
    live.add_argument("--top-p", type=float, default=0.95)
    live.add_argument("--max-tokens", type=int, default=512)
    live.add_argument("--timeout", type=float, default=300.0)
    live.add_argument("--pre-resample-turns", type=int, default=2)
    live.add_argument("--post-resample-turns", type=int, default=1)
    live.add_argument("--seed-base", type=int, default=1701)
    live.add_argument("--name-prefix", default="smcsd-live-particle")
    live.add_argument("--output", type=Path)
    live.add_argument("--keep-containers", action="store_true")
    live.set_defaults(action="live-smoke")

    restore = subparsers.add_parser("restore")
    restore.add_argument("--checkpoint", type=Path, required=True)
    restore.add_argument("--name", required=True)
    restore.add_argument("--ledger-output", type=Path)
    restore.add_argument("--cleanup", action="store_true")
    restore.set_defaults(action="restore")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    docker = backend.DockerCLI(args.docker_binary)
    if args.action == "restore":
        report = restore_particle_checkpoint(
            docker,
            checkpoint_path=args.checkpoint,
            name=args.name,
        )
        if args.ledger_output is not None:
            backend.atomic_write_json(args.ledger_output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checkpoint": report["checkpoint"],
                    "container_name": report["container_name"],
                    "message_count": report["message_count"],
                    "message_prefix_sha256": report[
                        "message_prefix_sha256"
                    ],
                    "state": report["fork"]["state"],
                    "cost": report["fork"]["cost"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.cleanup:
            docker.remove(args.name)
        return 0
    if args.action == "live-smoke":
        report = run_live_smoke(
            docker,
            manifest_path=args.manifest,
            capture_path=args.capture,
            after_tool_call_id=args.after_tool_call_id,
            base_url=args.base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            pre_resample_turns=args.pre_resample_turns,
            post_resample_turns=args.post_resample_turns,
            seed_base=args.seed_base,
            name_prefix=args.name_prefix,
            output=args.output,
            keep_containers=args.keep_containers,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "initial_coupling": report["initial_coupling"],
                    "pre_resample_divergence": report[
                        "pre_resample_divergence"
                    ],
                    "resample_equivalence": report["resampling"][
                        "equivalence"
                    ],
                    "global_cost": report["global_cost"],
                    "wall_time_s": report["wall_time_s"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unknown action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Clone Terminal-Bench particle environments by validated tool replay.

Docker CRIU checkpoints are preferred when the daemon supports them. The
current development host does not, so this module implements the explicitly
auditable fallback: start a pinned task image, replay every completed Pi tool
call, and verify filesystem plus surviving-process fingerprints.

The manifest also binds the terminal state to transcript and model-prefix
identities. Unsupported or ambiguous mutating tools fail closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
READ_ONLY_TOOLS = frozenset({"read"})
MUTATING_TOOLS = frozenset({"bash", "write", "edit"})
CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ReplayError(RuntimeError):
    """The recorded environment cannot be reproduced safely."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def result_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def extract_pi_tool_events(
    session_path: Path,
    *,
    max_tool_calls: int | None = None,
) -> list[dict[str, Any]]:
    """Extract completed tool calls and their results in transcript order."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    completed: set[str] = set()

    for line_number, line in enumerate(
        session_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "message":
            continue
        message = record.get("message") or {}
        if message.get("role") == "assistant":
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                call_id = str(block.get("id") or "")
                if not call_id:
                    raise ReplayError(
                        f"tool call without an ID at {session_path}:{line_number}"
                    )
                if call_id in calls:
                    raise ReplayError(f"duplicate tool call ID: {call_id}")
                arguments = block.get("arguments")
                if not isinstance(arguments, dict):
                    raise ReplayError(
                        f"tool call arguments are not an object: {call_id}"
                    )
                calls[call_id] = {
                    "call_id": call_id,
                    "name": str(block.get("name") or ""),
                    "arguments": arguments,
                    "assistant_message_id": str(record.get("id") or ""),
                    "assistant_timestamp_ms": record.get("timestamp"),
                }
                order.append(call_id)
        elif message.get("role") == "toolResult":
            call_id = str(message.get("toolCallId") or "")
            if call_id not in calls:
                raise ReplayError(f"tool result has no preceding call: {call_id}")
            if call_id in completed:
                raise ReplayError(f"duplicate tool result: {call_id}")
            text = result_text(message)
            calls[call_id].update(
                {
                    "result_message_id": str(record.get("id") or ""),
                    "result_timestamp_ms": record.get("timestamp"),
                    "expected_error": bool(message.get("isError")),
                    "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            completed.add(call_id)

    events = [calls[call_id] for call_id in order if call_id in completed]
    if max_tool_calls is not None:
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        events = events[:max_tool_calls]
    return events


class DockerCLI:
    """Small Docker CLI adapter with deterministic, testable command assembly."""

    def __init__(self, binary: str = "docker") -> None:
        self.binary = binary

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout: float = 300,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            [self.binary, *arguments],
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and process.returncode != 0:
            command = " ".join([self.binary, *arguments[:4]])
            raise ReplayError(
                f"{command} failed ({process.returncode}): "
                f"{process.stderr.strip() or process.stdout.strip()}"
            )
        return process

    def inspect(self, reference: str) -> dict[str, Any]:
        process = self.run(["inspect", reference])
        rows = json.loads(process.stdout)
        if len(rows) != 1:
            raise ReplayError(f"docker inspect returned {len(rows)} rows")
        return rows[0]

    def image_identity(self, image: str) -> dict[str, str]:
        row = self.inspect(image)
        image_id = str(row.get("Id") or "")
        if not image_id.startswith("sha256:"):
            raise ReplayError(f"image is not content-addressed: {image}")
        return {"reference": image, "image_id": image_id}

    def create(
        self,
        *,
        name: str,
        image: str,
        workdir: str,
        network: str,
        command: Sequence[str],
        labels: dict[str, str],
    ) -> str:
        if not CONTAINER_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid Docker container name: {name!r}")
        arguments = [
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            network,
            "--workdir",
            workdir,
        ]
        for key, value in sorted(labels.items()):
            arguments.extend(["--label", f"{key}={value}"])
        arguments.extend([image, *command])
        return self.run(arguments).stdout.strip()

    def remove(self, container: str) -> None:
        self.run(["rm", "--force", container], check=False)

    def exec_bash(
        self,
        container: str,
        command: str,
        *,
        workdir: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        arguments = ["exec", "--workdir", workdir]
        for key, value in sorted((environment or {}).items()):
            arguments.extend(["--env", f"{key}={value}"])
        arguments.extend(
            [
                container,
                "/bin/bash",
                "-lc",
                command,
            ]
        )
        return self.run(arguments, check=False)

    def copy_from(self, container: str, source: str, destination: Path) -> None:
        self.run(["cp", f"{container}:{source}", str(destination)])

    def copy_to(self, source: Path, container: str, destination: str) -> None:
        self.run(["cp", str(source), f"{container}:{destination}"])

    def write_text(self, container: str, path: str, content: str) -> None:
        parent = os.path.dirname(path) or "/"
        self.run(["exec", container, "mkdir", "-p", parent])
        with tempfile.TemporaryDirectory(prefix="smcsd-replay-write-") as raw:
            local = Path(raw) / "value"
            local.write_text(content, encoding="utf-8")
            self.copy_to(local, container, path)

    def edit_text(
        self,
        container: str,
        path: str,
        edits: Sequence[dict[str, str]],
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="smcsd-replay-edit-") as raw:
            local = Path(raw) / "value"
            self.copy_from(container, path, local)
            content = local.read_text(encoding="utf-8")
            for index, edit in enumerate(edits):
                old = edit.get("oldText")
                new = edit.get("newText")
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ReplayError(f"edit {index} is missing oldText/newText")
                occurrences = content.count(old)
                if occurrences != 1:
                    raise ReplayError(
                        f"edit {index} expected one oldText occurrence, "
                        f"found {occurrences}: {path}"
                    )
                content = content.replace(old, new, 1)
            local.write_text(content, encoding="utf-8")
            self.copy_to(local, container, path)

    def _filesystem_digest_with_tar(
        self,
        container: str,
        roots: Sequence[str],
        ignore_runtime_caches: bool = False,
    ) -> str:
        script = r"""
set -euo pipefail
ignore_runtime_caches="$1"
shift
exclude_args=()
if [[ "$ignore_runtime_caches" == "1" ]]; then
    exclude_args+=(--exclude='*/__pycache__' --exclude='*/__pycache__/*')
    exclude_args+=(--exclude='*/.pytest_cache' --exclude='*/.pytest_cache/*')
    exclude_args+=(--exclude='*.pyc' --exclude='*.pyo')
fi
emit_git_command() {
    repository="$1"
    label="$2"
    shift 2
    temporary="$(mktemp -d /tmp/smcsd-git-digest.XXXXXX)"
    if git -C "$repository" "$@" >"$temporary/stdout" 2>"$temporary/stderr"; then
        status=0
    else
        status=$?
    fi
    printf 'COMMAND\000%s\000STATUS\000%s\000STDOUT\000' "$label" "$status"
    cat "$temporary/stdout"
    printf '\000STDERR\000'
    cat "$temporary/stderr"
    printf '\000'
    rm -- "$temporary/stdout" "$temporary/stderr"
    rmdir -- "$temporary"
}
emit_git_semantic_state() {
    repository="$1"
    printf 'GIT_SEMANTIC\000%s\000' "$repository"
    emit_git_command "$repository" 'ls-files --stage -z' \
        ls-files --stage -z
    emit_git_command "$repository" \
        'status --porcelain=v1 -z --untracked-files=all' \
        status --porcelain=v1 -z --untracked-files=all
    emit_git_command "$repository" \
        'for-each-ref --format=%(refname)%00%(objectname)' \
        for-each-ref '--format=%(refname)%00%(objectname)'
    emit_git_command "$repository" 'rev-parse --verify HEAD' \
        rev-parse --verify HEAD
    emit_git_command "$repository" 'rev-parse --abbrev-ref HEAD' \
        rev-parse --abbrev-ref HEAD
}
{
    for root in "$@"; do
        printf 'ROOT\000%s\000' "$root"
        if [[ ! -e "$root" && ! -L "$root" ]]; then
            printf 'MISSING\000'
            continue
        fi
        relative="${root#/}"
        if [[ -z "$relative" ]]; then
            relative="."
        fi
        tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
            --format=gnu --exclude='*/.git/index' --exclude='*/.git/logs' \
            "${exclude_args[@]}" \
            -cf - -C / "$relative"
        while IFS= read -r -d '' git_dir; do
            emit_git_semantic_state "${git_dir%/.git}"
        done < <(find "$root" -type d -name .git -print0 2>/dev/null | sort -z)
    done
} | sha256sum
"""
        process = self.run(
            [
                "exec",
                container,
                "/bin/bash",
                "-c",
                script,
                "smcsd-filesystem-digest",
                "1" if ignore_runtime_caches else "0",
                *sorted(roots),
            ],
            check=False,
        )
        value = process.stdout.strip().split(" ", 1)[0]
        if process.returncode != 0 or not re.fullmatch(r"[0-9a-f]{64}", value):
            detail = process.stderr.strip() or process.stdout.strip()
            raise ReplayError(f"portable filesystem digest failed: {detail}")
        return value

    def filesystem_digest(
        self,
        container: str,
        roots: Sequence[str],
        ignore_runtime_caches: bool = False,
    ) -> str:
        python = self.run(
            ["exec", container, "/bin/bash", "-lc", "command -v python3"],
            check=False,
        )
        if python.returncode != 0:
            return self._filesystem_digest_with_tar(
                container, roots, ignore_runtime_caches
            )
        script = r"""
import hashlib, json, os, stat, subprocess, sys
roots = json.loads(sys.argv[1])
ignore_runtime_caches = sys.argv[2] == "1"
digest = hashlib.sha256()
git_repositories = set()
def add(value):
    if isinstance(value, str):
        value = value.encode("utf-8", "surrogateescape")
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
def volatile_git_path(relative):
    parts = relative.split(os.sep)
    if ".git" not in parts:
        return False
    tail = parts[parts.index(".git") + 1:]
    return tail == ["index"] or (tail and tail[0] == "logs")
def volatile_path(relative):
    if volatile_git_path(relative):
        return True
    parts = relative.split(os.sep)
    return ignore_runtime_caches and (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or relative.endswith((".pyc", ".pyo"))
    )
for root in sorted(roots):
    root = os.path.abspath(root)
    add("ROOT")
    add(root)
    if not os.path.lexists(root):
        add("MISSING")
        continue
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        if ".git" in directories:
            git_repositories.add(current)
        directories[:] = [
            name
            for name in directories
            if not volatile_path(os.path.relpath(os.path.join(current, name), root))
        ]
        files = [
            name
            for name in files
            if not volatile_path(os.path.relpath(os.path.join(current, name), root))
        ]
        entries = [(name, "D") for name in directories]
        entries.extend((name, "F") for name in files)
        for name, hint in entries:
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root)
            info = os.lstat(path)
            add(relative)
            add(oct(stat.S_IMODE(info.st_mode) & 0o111))
            if stat.S_ISLNK(info.st_mode):
                add("L")
                add(os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                add("D")
            elif stat.S_ISREG(info.st_mode):
                add("F")
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        add(chunk)
            else:
                add(hint)
                add(str(info.st_mode))
for repository in sorted(git_repositories):
    add("GIT_SEMANTIC")
    add(repository)
    commands = (
        ("ls-files", "--stage", "-z"),
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        ("for-each-ref", "--format=%(refname)%00%(objectname)"),
        ("rev-parse", "--verify", "HEAD"),
        ("rev-parse", "--abbrev-ref", "HEAD"),
    )
    for command in commands:
        result = subprocess.run(
            ["git", "-C", repository, *command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        add(" ".join(command))
        add(str(result.returncode))
        add(result.stdout)
        add(result.stderr)
print(digest.hexdigest())
"""
        process = self.run(
            [
                "exec",
                container,
                "python3",
                "-c",
                script,
                json.dumps(list(roots)),
                "1" if ignore_runtime_caches else "0",
            ]
        )
        value = process.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ReplayError(f"invalid filesystem digest: {value!r}")
        return value

    def process_fingerprint(self, container: str) -> list[str]:
        process = self.run(
            ["top", container, "-eo", "pid,user,stat,comm,args"],
        )
        rows: list[str] = []
        for index, raw in enumerate(process.stdout.splitlines()):
            line = " ".join(raw.split())
            if not line or index == 0:
                continue
            fields = line.split(" ", 4)
            if len(fields) < 5:
                raise ReplayError(f"unexpected docker top row: {raw!r}")
            _pid, user, stat_value, command, arguments = fields
            normalized_stat = re.sub(r"[^A-Z]", "", stat_value.upper())
            rows.append(
                "\t".join([user, normalized_stat, command, arguments])
            )
        return sorted(rows)

    def capabilities(self) -> dict[str, Any]:
        version = json.loads(
            self.run(["version", "--format", "{{json .}}"]).stdout
        )
        server = version.get("Server") or {}
        experimental = str(server.get("Experimental", "false")).lower() == "true"
        return {
            "docker_server_version": server.get("Version"),
            "docker_api_version": server.get("ApiVersion"),
            "docker_experimental": experimental,
            "criu_checkpoint_restore": experimental and shutil.which("criu") is not None,
            "deterministic_tool_replay": True,
        }


def normalized_edits(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ReplayError("edit argument is not valid JSON") from error
    if not isinstance(raw, list):
        raise ReplayError("edit argument must be a list")
    if not all(isinstance(entry, dict) for entry in raw):
        raise ReplayError("every edit entry must be an object")
    return raw


def frozen_tool_environment(event: dict[str, Any]) -> dict[str, str]:
    """Freeze common time-sensitive tools to the recorded call completion."""
    raw = event.get("result_timestamp_ms")
    if raw is None:
        raw = event.get("assistant_timestamp_ms")
    if raw is None:
        return {}
    try:
        if isinstance(raw, (int, float)):
            seconds = float(raw)
            if seconds > 100_000_000_000:
                seconds /= 1000
            moment = datetime.fromtimestamp(seconds, tz=timezone.utc)
        else:
            value = str(raw)
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            moment = datetime.fromisoformat(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            moment = moment.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError) as error:
        raise ReplayError(f"invalid tool timestamp: {raw!r}") from error
    git_date = moment.strftime("%Y-%m-%dT%H:%M:%S %z")
    return {
        "GIT_AUTHOR_DATE": git_date,
        "GIT_COMMITTER_DATE": git_date,
        "SOURCE_DATE_EPOCH": str(int(moment.timestamp())),
        "TZ": "UTC",
    }


def replay_event(
    docker: DockerCLI,
    container: str,
    event: dict[str, Any],
    *,
    workdir: str,
) -> dict[str, Any]:
    name = str(event["name"])
    arguments = event["arguments"]
    expected_error = bool(event.get("expected_error"))

    if expected_error and event.get("mutation_status") == "none":
        return {
            "call_id": event["call_id"],
            "name": name,
            "status": "skipped_failed_no_mutation",
        }

    if name in READ_ONLY_TOOLS:
        return {
            "call_id": event["call_id"],
            "name": name,
            "status": "skipped_read_only",
        }
    if name not in MUTATING_TOOLS:
        raise ReplayError(f"unsupported tool may mutate state: {name}")

    started = time.perf_counter()
    if name == "bash":
        command = arguments.get("command")
        if not isinstance(command, str):
            raise ReplayError(f"bash command is not a string: {event['call_id']}")
        process = docker.exec_bash(
            container,
            command,
            workdir=workdir,
            environment=frozen_tool_environment(event),
        )
        actual_error = process.returncode != 0
        if actual_error != expected_error:
            raise ReplayError(
                f"bash outcome differs for {event['call_id']}: "
                f"expected_error={expected_error}, return_code={process.returncode}"
            )
        return {
            "call_id": event["call_id"],
            "name": name,
            "status": "replayed",
            "return_code": process.returncode,
            "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
            "wall_time_s": time.perf_counter() - started,
        }

    if expected_error:
        if event.get("mutation_status") == "none":
            return {
                "call_id": event["call_id"],
                "name": name,
                "status": "skipped_failed_no_mutation",
            }
        raise ReplayError(
            f"failed {name} call has ambiguous partial-mutation semantics: "
            f"{event['call_id']}"
        )
    path = arguments.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ReplayError(f"{name} path must be absolute: {event['call_id']}")
    if name == "write":
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ReplayError(f"write content is not a string: {event['call_id']}")
        docker.write_text(container, path, content)
    else:
        docker.edit_text(
            container,
            path,
            normalized_edits(arguments.get("edits")),
        )
    return {
        "call_id": event["call_id"],
        "name": name,
        "status": "replayed",
        "wall_time_s": time.perf_counter() - started,
    }


def build_manifest(
    docker: DockerCLI,
    *,
    session_path: Path,
    image: str,
    source_container: str | None,
    task_id: str,
    workdir: str,
    state_roots: Sequence[str],
    max_tool_calls: int | None,
    checkpoint_id: str | None,
    model_prefix_sha256: str | None,
) -> dict[str, Any]:
    events = extract_pi_tool_events(
        session_path,
        max_tool_calls=max_tool_calls,
    )
    unsupported = sorted(
        {
            str(event["name"])
            for event in events
            if event["name"] not in READ_ONLY_TOOLS | MUTATING_TOOLS
        }
    )
    if unsupported:
        raise ReplayError(f"unsupported tools in transcript: {unsupported}")
    expected_state: dict[str, Any] = {}
    sealed_from: dict[str, str] | None = None
    if source_container is not None:
        source = docker.inspect(source_container)
        source_image = str(source.get("Image") or "")
        image_identity = docker.image_identity(image)
        if source_image != image_identity["image_id"]:
            raise ReplayError(
                "source container image does not match the requested base image"
            )
        expected_state = {
            "filesystem_sha256": docker.filesystem_digest(
                source_container,
                state_roots,
            ),
            "process_fingerprint": docker.process_fingerprint(source_container),
        }
        sealed_from = {
            "container_id": str(source.get("Id") or source_container),
            "container_name": str(source.get("Name") or "").lstrip("/"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": "deterministic_pi_tool_replay_v1",
        "task_id": task_id,
        "base_environment": {
            **docker.image_identity(image),
            "workdir": workdir,
            "state_roots": list(state_roots),
            "network": "none",
            "command": ["sleep", "infinity"],
        },
        "transcript": {
            "path": str(session_path.resolve()),
            "sha256": file_sha256(session_path),
            "completed_tool_calls": len(events),
            "events_sha256": canonical_json_sha256(events),
        },
        "model_prefix": {
            "checkpoint_id": checkpoint_id,
            "sha256": model_prefix_sha256,
        },
        "sealed_from": sealed_from,
        "lineage": {
            "particle_id": uuid.uuid4().hex,
            "parent_particle_id": None,
            "generation": 0,
        },
        "tool_events": events,
        "expected_state": expected_state,
    }


def fork_manifest(
    docker: DockerCLI,
    manifest: dict[str, Any],
    *,
    name: str,
    network: str | None = None,
) -> dict[str, Any]:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ReplayError("unsupported particle manifest schema")
    if manifest.get("backend") != "deterministic_pi_tool_replay_v1":
        raise ReplayError("unsupported particle backend")
    environment = manifest["base_environment"]
    image_id = str(environment["image_id"])
    if docker.image_identity(image_id)["image_id"] != image_id:
        raise ReplayError("base image digest changed")
    workdir = str(environment["workdir"])
    effective_network = network or str(environment.get("network") or "none")
    lineage = manifest["lineage"]
    child_id = uuid.uuid4().hex
    labels = {
        "smcsd.terminal-particle": "true",
        "smcsd.particle-id": child_id,
        "smcsd.parent-particle-id": str(lineage["particle_id"]),
    }
    container_id = docker.create(
        name=name,
        image=image_id,
        workdir=workdir,
        network=effective_network,
        command=list(environment.get("command") or ["sleep", "infinity"]),
        labels=labels,
    )

    calls: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for event in manifest["tool_events"]:
            calls.append(
                replay_event(
                    docker,
                    name,
                    event,
                    workdir=workdir,
                )
            )
        time.sleep(0.1)
        roots = list(environment["state_roots"])
        filesystem = docker.filesystem_digest(
            name,
            roots,
            bool(environment.get("ignore_runtime_caches", False)),
        )
        processes = docker.process_fingerprint(name)
        expected = manifest.get("expected_state") or {}
        if expected.get("filesystem_sha256") not in (None, filesystem):
            raise ReplayError(
                "filesystem equivalence failed: "
                f"expected {expected['filesystem_sha256']}, got {filesystem}"
            )
        if expected.get("process_fingerprint") not in (None, processes):
            raise ReplayError("surviving-process equivalence failed")
    except Exception:
        docker.remove(name)
        raise

    return {
        "schema_version": SCHEMA_VERSION,
        "backend": manifest["backend"],
        "container_id": container_id,
        "container_name": name,
        "task_id": manifest["task_id"],
        "lineage": {
            "particle_id": child_id,
            "parent_particle_id": lineage["particle_id"],
            "generation": int(lineage["generation"]) + 1,
        },
        "model_prefix": manifest["model_prefix"],
        "transcript_sha256": manifest["transcript"]["sha256"],
        "events_sha256": manifest["transcript"]["events_sha256"],
        "state": {
            "filesystem_sha256": filesystem,
            "process_fingerprint": processes,
        },
        "cost": {
            "replay_wall_time_s": time.perf_counter() - started,
            "tool_calls_replayed": sum(
                row["status"] == "replayed" for row in calls
            ),
            "read_only_calls_skipped": sum(
                row["status"] == "skipped_read_only" for row in calls
            ),
            "failed_no_mutation_calls_skipped": sum(
                row["status"] == "skipped_failed_no_mutation"
                for row in calls
            ),
        },
        "calls": calls,
    }


def synthetic_events() -> list[dict[str, Any]]:
    def event(index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "call_id": f"smoke-{index}",
            "name": name,
            "arguments": arguments,
            "expected_error": False,
            "result_sha256": hashlib.sha256(b"").hexdigest(),
        }

    return [
        event(
            1,
            "bash",
            {
                "command": (
                    "mkdir -p /app/smcsd-particle-smoke && "
                    "printf 'alpha\\n' > /app/smcsd-particle-smoke/data.txt"
                )
            },
        ),
        event(
            2,
            "write",
            {
                "path": "/app/smcsd-particle-smoke/note.txt",
                "content": "particle-state\\n",
            },
        ),
        event(
            3,
            "edit",
            {
                "path": "/app/smcsd-particle-smoke/data.txt",
                "edits": [{"oldText": "alpha", "newText": "beta"}],
            },
        ),
        event(
            4,
            "bash",
            {
                "command": (
                    "nohup bash -c 'exec -a smcsd-heartbeat sleep 600' "
                    ">/tmp/smcsd-heartbeat.log 2>&1 </dev/null &"
                )
            },
        ),
    ]


def run_smoke(
    docker: DockerCLI,
    *,
    image: str,
    output: Path | None,
) -> dict[str, Any]:
    suffix = uuid.uuid4().hex[:10]
    source = f"smcsd-particle-source-{suffix}"
    child = f"smcsd-particle-child-{suffix}"
    identity = docker.image_identity(image)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "backend": "deterministic_pi_tool_replay_v1",
        "task_id": "synthetic-fork-equivalence",
        "base_environment": {
            **identity,
            "workdir": "/app",
            "state_roots": ["/app/smcsd-particle-smoke"],
            "network": "none",
            "command": ["sleep", "infinity"],
        },
        "transcript": {
            "path": None,
            "sha256": hashlib.sha256(b"synthetic").hexdigest(),
            "completed_tool_calls": 4,
            "events_sha256": canonical_json_sha256(synthetic_events()),
        },
        "model_prefix": {
            "checkpoint_id": "synthetic-checkpoint",
            "sha256": hashlib.sha256(b"synthetic-prefix").hexdigest(),
        },
        "lineage": {
            "particle_id": uuid.uuid4().hex,
            "parent_particle_id": None,
            "generation": 0,
        },
        "tool_events": synthetic_events(),
        "expected_state": {},
    }
    docker.create(
        name=source,
        image=identity["image_id"],
        workdir="/app",
        network="none",
        command=["sleep", "infinity"],
        labels={"smcsd.terminal-particle-smoke": "source"},
    )
    try:
        for event in manifest["tool_events"]:
            replay_event(docker, source, event, workdir="/app")
        time.sleep(0.1)
        manifest["expected_state"] = {
            "filesystem_sha256": docker.filesystem_digest(
                source,
                manifest["base_environment"]["state_roots"],
            ),
            "process_fingerprint": docker.process_fingerprint(source),
        }
        result = fork_manifest(docker, manifest, name=child)
        report = {
            "status": "pass",
            "capabilities": docker.capabilities(),
            "manifest": manifest,
            "fork": result,
        }
        if output is not None:
            atomic_write_json(output, report)
        return report
    finally:
        docker.remove(source)
        docker.remove(child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-binary", default="docker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser("capabilities")
    capabilities.set_defaults(action="capabilities")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--session", type=Path, required=True)
    extract.add_argument("--image", required=True)
    extract.add_argument(
        "--source-container",
        help="seal expected filesystem/process state from a quiescent container",
    )
    extract.add_argument("--task-id", required=True)
    extract.add_argument("--workdir", default="/app")
    extract.add_argument("--state-root", action="append")
    extract.add_argument("--max-tool-calls", type=int)
    extract.add_argument("--checkpoint-id")
    extract.add_argument("--model-prefix-sha256")
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(action="extract")

    fork = subparsers.add_parser("fork")
    fork.add_argument("--manifest", type=Path, required=True)
    fork.add_argument("--name", required=True)
    fork.add_argument("--network")
    fork.add_argument("--ledger-output", type=Path)
    fork.add_argument("--cleanup", action="store_true")
    fork.set_defaults(action="fork")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--image", default="alexgshaw/fix-git:20260403")
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(action="smoke")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    docker = DockerCLI(args.docker_binary)

    if args.action == "capabilities":
        print(json.dumps(docker.capabilities(), indent=2, sort_keys=True))
        return 0
    if args.action == "extract":
        roots = args.state_root or [args.workdir]
        manifest = build_manifest(
            docker,
            session_path=args.session,
            image=args.image,
            source_container=args.source_container,
            task_id=args.task_id,
            workdir=args.workdir,
            state_roots=roots,
            max_tool_calls=args.max_tool_calls,
            checkpoint_id=args.checkpoint_id,
            model_prefix_sha256=args.model_prefix_sha256,
        )
        atomic_write_json(args.output, manifest)
        print(json.dumps(manifest["transcript"], indent=2, sort_keys=True))
        return 0
    if args.action == "fork":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = fork_manifest(
            docker,
            manifest,
            name=args.name,
            network=args.network,
        )
        if args.ledger_output is not None:
            atomic_write_json(args.ledger_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.cleanup:
            docker.remove(args.name)
        return 0
    if args.action == "smoke":
        report = run_smoke(docker, image=args.image, output=args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "capabilities": report["capabilities"],
                    "state": report["fork"]["state"],
                    "cost": report["fork"]["cost"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())

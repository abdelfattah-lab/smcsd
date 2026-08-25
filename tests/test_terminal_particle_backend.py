import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_backend():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "terminal_particle_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "terminal_particle_backend_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_session(path: Path) -> None:
    rows = [
        {
            "type": "message",
            "id": "assistant-1",
            "timestamp": 10,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-bash",
                        "name": "bash",
                        "arguments": {"command": "printf ok"},
                    },
                    {
                        "type": "toolCall",
                        "id": "call-read",
                        "name": "read",
                        "arguments": {"path": "/app/value"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "result-bash",
            "timestamp": 11,
            "message": {
                "role": "toolResult",
                "toolCallId": "call-bash",
                "toolName": "bash",
                "isError": False,
                "content": [{"type": "text", "text": "ok"}],
            },
        },
        {
            "type": "message",
            "id": "result-read",
            "timestamp": 12,
            "message": {
                "role": "toolResult",
                "toolCallId": "call-read",
                "toolName": "read",
                "isError": False,
                "content": [{"type": "text", "text": "value"}],
            },
        },
        {
            "type": "message",
            "id": "assistant-incomplete",
            "timestamp": 13,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call-incomplete",
                        "name": "write",
                        "arguments": {
                            "path": "/app/incomplete",
                            "content": "no result",
                        },
                    }
                ],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_extract_pi_tool_events_keeps_completed_calls_in_order(tmp_path):
    backend = load_backend()
    session = tmp_path / "session.jsonl"
    write_session(session)

    events = backend.extract_pi_tool_events(session)

    assert [event["call_id"] for event in events] == [
        "call-bash",
        "call-read",
    ]
    assert events[0]["expected_error"] is False
    assert events[0]["assistant_timestamp_ms"] == 10
    assert events[0]["result_timestamp_ms"] == 11
    assert len(events[0]["result_sha256"]) == 64
    assert backend.extract_pi_tool_events(
        session,
        max_tool_calls=1,
    ) == events[:1]


def test_extract_pi_tool_events_rejects_orphan_results(tmp_path):
    backend = load_backend()
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "message",
                "id": "orphan",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "missing",
                    "content": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(backend.ReplayError, match="no preceding call"):
        backend.extract_pi_tool_events(session)


class FakeDocker:
    def __init__(self):
        self.created = []
        self.removed = []
        self.writes = []
        self.edits = []
        self.commands = []

    def image_identity(self, image):
        assert image == "sha256:" + "a" * 64
        return {"reference": image, "image_id": image}

    def inspect(self, reference):
        assert reference == "source"
        return {
            "Id": "source-container-id",
            "Name": "/source",
            "Image": "sha256:" + "a" * 64,
        }

    def create(self, **kwargs):
        self.created.append(kwargs)
        return "container-id"

    def remove(self, container):
        self.removed.append(container)

    def exec_bash(self, container, command, *, workdir, environment=None):
        self.commands.append((container, command, workdir, environment))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def write_text(self, container, path, content):
        self.writes.append((container, path, content))

    def edit_text(self, container, path, edits):
        self.edits.append((container, path, edits))

    def filesystem_digest(
        self,
        container,
        roots,
        ignore_runtime_caches=False,
    ):
        return "b" * 64

    def process_fingerprint(self, container):
        return ["root\tS\tservice\tservice --foreground"]


class RecordingDocker:
    def __init__(self, backend):
        class Recorder(backend.DockerCLI):
            def __init__(self):
                self.commands = []
                self.copied = None

            def run(self, arguments, **kwargs):
                self.commands.append(arguments)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def copy_to(self, source, container, destination):
                self.copied = (
                    source.read_text(encoding="utf-8"),
                    container,
                    destination,
                )

        self.instance = Recorder()


def test_write_text_creates_missing_parent_before_copy():
    backend = load_backend()
    docker = RecordingDocker(backend).instance

    docker.write_text("particle", "/app/missing/value", "payload")

    assert docker.commands == [
        ["exec", "particle", "mkdir", "-p", "/app/missing"]
    ]
    assert docker.copied == (
        "payload",
        "particle",
        "/app/missing/value",
    )


def sample_manifest(backend):
    events = [
        {
            "call_id": "bash-1",
            "name": "bash",
            "arguments": {"command": "touch /app/value"},
            "expected_error": False,
        },
        {
            "call_id": "read-1",
            "name": "read",
            "arguments": {"path": "/app/value"},
            "expected_error": False,
        },
    ]
    return {
        "schema_version": backend.SCHEMA_VERSION,
        "backend": "deterministic_pi_tool_replay_v1",
        "task_id": "task",
        "base_environment": {
            "reference": "task:tag",
            "image_id": "sha256:" + "a" * 64,
            "workdir": "/app",
            "state_roots": ["/app"],
            "network": "none",
            "command": ["sleep", "infinity"],
        },
        "transcript": {
            "sha256": "c" * 64,
            "events_sha256": backend.canonical_json_sha256(events),
        },
        "model_prefix": {
            "checkpoint_id": "checkpoint",
            "sha256": "d" * 64,
        },
        "lineage": {
            "particle_id": "parent",
            "parent_particle_id": None,
            "generation": 2,
        },
        "tool_events": events,
        "expected_state": {
            "filesystem_sha256": "b" * 64,
            "process_fingerprint": [
                "root\tS\tservice\tservice --foreground"
            ],
        },
    }


def test_build_manifest_can_seal_expected_source_state(tmp_path):
    backend = load_backend()
    docker = FakeDocker()
    session = tmp_path / "session.jsonl"
    write_session(session)

    manifest = backend.build_manifest(
        docker,
        session_path=session,
        image="sha256:" + "a" * 64,
        source_container="source",
        task_id="task",
        workdir="/app",
        state_roots=["/app"],
        max_tool_calls=None,
        checkpoint_id="checkpoint",
        model_prefix_sha256="d" * 64,
    )

    assert manifest["sealed_from"] == {
        "container_id": "source-container-id",
        "container_name": "source",
    }
    assert manifest["expected_state"] == {
        "filesystem_sha256": "b" * 64,
        "process_fingerprint": [
            "root\tS\tservice\tservice --foreground"
        ],
    }


def test_fork_manifest_binds_lineage_prefix_and_state():
    backend = load_backend()
    docker = FakeDocker()
    manifest = sample_manifest(backend)

    result = backend.fork_manifest(
        docker,
        manifest,
        name="particle-child",
    )

    assert result["container_id"] == "container-id"
    assert result["lineage"]["parent_particle_id"] == "parent"
    assert result["lineage"]["generation"] == 3
    assert result["model_prefix"]["checkpoint_id"] == "checkpoint"
    assert result["state"]["filesystem_sha256"] == "b" * 64
    assert result["cost"]["tool_calls_replayed"] == 1
    assert result["cost"]["read_only_calls_skipped"] == 1
    assert docker.commands == [
        ("particle-child", "touch /app/value", "/app", {})
    ]
    assert docker.removed == []


def test_fork_manifest_fails_closed_and_removes_divergent_clone():
    backend = load_backend()
    docker = FakeDocker()
    manifest = sample_manifest(backend)
    manifest["expected_state"]["filesystem_sha256"] = "e" * 64

    with pytest.raises(backend.ReplayError, match="filesystem equivalence"):
        backend.fork_manifest(
            docker,
            manifest,
            name="particle-diverged",
        )

    assert docker.removed == ["particle-diverged"]


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            {
                "call_id": "unknown",
                "name": "browser",
                "arguments": {},
                "expected_error": False,
            },
            "unsupported tool",
        ),
        (
            {
                "call_id": "failed-write",
                "name": "write",
                "arguments": {"path": "/app/x", "content": "x"},
                "expected_error": True,
            },
            "ambiguous partial-mutation",
        ),
    ],
)
def test_replay_event_rejects_ambiguous_mutations(event, message):
    backend = load_backend()

    with pytest.raises(backend.ReplayError, match=message):
        backend.replay_event(
            FakeDocker(),
            "particle",
            event,
            workdir="/app",
        )


def test_frozen_tool_environment_normalizes_iso_and_epoch_milliseconds():
    backend = load_backend()
    iso = backend.frozen_tool_environment(
        {"result_timestamp_ms": "2026-08-24T16:52:06.126Z"}
    )
    epoch = backend.frozen_tool_environment(
        {"result_timestamp_ms": 1787590326126}
    )

    assert iso == epoch
    assert iso["GIT_AUTHOR_DATE"] == "2026-08-24T16:52:06 +0000"
    assert iso["GIT_COMMITTER_DATE"] == iso["GIT_AUTHOR_DATE"]
    assert iso["SOURCE_DATE_EPOCH"] == "1787590326"
    assert iso["TZ"] == "UTC"


def test_replay_skips_proven_failed_no_mutation_event():
    backend = load_backend()
    event = {
        "call_id": "safe-failed-edit",
        "name": "edit",
        "arguments": {
            "path": "/app/value",
            "edits": [{"oldText": "missing", "newText": "value"}],
        },
        "expected_error": True,
        "mutation_status": "none",
    }

    result = backend.replay_event(
        FakeDocker(),
        "particle",
        event,
        workdir="/app",
    )

    assert result["status"] == "skipped_failed_no_mutation"


def test_replay_skips_unknown_tool_when_controller_proved_no_mutation():
    backend = load_backend()
    event = {
        "call_id": "malformed-unknown",
        "name": "hallucinated_tool",
        "arguments": {"raw_arguments": "{"},
        "expected_error": True,
        "mutation_status": "none",
    }

    result = backend.replay_event(
        FakeDocker(),
        "particle",
        event,
        workdir="/app",
    )

    assert result["status"] == "skipped_failed_no_mutation"

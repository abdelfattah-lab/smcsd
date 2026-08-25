import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_controller():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "terminal_particle_controller.py"
    )
    spec = importlib.util.spec_from_file_location(
        "terminal_particle_controller_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeDocker:
    def __init__(self, controller):
        self.controller = controller
        self.files = {}
        self.removed = []
        self.counter = 0

    def image_identity(self, image):
        assert image == "sha256:" + "a" * 64
        return {"reference": image, "image_id": image}

    def create(self, *, name, **kwargs):
        self.counter += 1
        self.files[name] = {}
        return f"container-{self.counter}"

    def remove(self, container):
        self.removed.append(container)
        self.files.pop(container, None)

    def write_text(self, container, path, content):
        self.files[container][path] = content

    def copy_from(self, container, source, destination):
        destination.write_text(self.files[container][source], encoding="utf-8")

    def copy_to(self, source, container, destination):
        self.files[container][destination] = source.read_text(encoding="utf-8")

    def edit_text(self, container, path, edits):
        value = self.files[container][path]
        for edit in edits:
            value = value.replace(edit["oldText"], edit["newText"], 1)
        self.files[container][path] = value

    def run(self, arguments, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def exec_bash(self, container, command, *, workdir, environment=None):
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    def filesystem_digest(
        self,
        container,
        roots,
        ignore_runtime_caches=False,
    ):
        return self.controller.backend.canonical_json_sha256(
            self.files[container]
        )

    def process_fingerprint(self, container):
        return ["root\tSS\tsleep\tsleep infinity"]


def sample_manifest(controller):
    events = []
    return {
        "schema_version": controller.backend.SCHEMA_VERSION,
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
            "events_sha256": controller.backend.canonical_json_sha256(events),
            "completed_tool_calls": 0,
        },
        "model_prefix": {
            "checkpoint_id": "checkpoint",
            "sha256": None,
        },
        "lineage": {
            "particle_id": "root",
            "parent_particle_id": None,
            "generation": 0,
        },
        "tool_events": events,
        "expected_state": {},
    }


def request_template():
    return {
        "model": "model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "write a value"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }


def test_load_captured_checkpoint_and_validate_binding(tmp_path):
    controller = load_controller()
    payload = request_template()
    payload["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "call-checkpoint",
            "content": "done",
        }
    )
    capture = tmp_path / "capture.jsonl"
    capture.write_text(
        json.dumps({"t": 1, "payload": payload}) + "\n",
        encoding="utf-8",
    )
    manifest = sample_manifest(controller)
    manifest["tool_events"].append(
        {
            "call_id": "call-checkpoint",
            "name": "bash",
            "arguments": {"command": "true"},
            "expected_error": False,
        }
    )

    loaded = controller.load_captured_checkpoint(
        capture,
        after_tool_call_id="call-checkpoint",
    )
    controller.validate_checkpoint_binding(
        manifest,
        loaded,
        after_tool_call_id="call-checkpoint",
    )

    with pytest.raises(controller.ControllerError, match="different tool"):
        controller.validate_checkpoint_binding(
            manifest,
            loaded,
            after_tool_call_id="other",
        )


def test_format_bash_result_and_simultaneous_edit(tmp_path):
    controller = load_controller()
    docker = FakeDocker(controller)
    docker.files["particle"] = {"/app/value": "one two"}

    assert controller.format_bash_result("", "", 0) == "(no output)"
    assert controller.format_bash_result("bad\n", "", 2) == (
        "bad\n\nCommand exited with code 2"
    )

    controller.edit_text_tool(
        docker,
        "particle",
        "/app/value",
        [
            {"oldText": "one", "newText": "two"},
            {"oldText": "two", "newText": "three"},
        ],
    )
    assert docker.files["particle"]["/app/value"] == "two three"


def test_resample_copies_environment_messages_lineage_and_cost(tmp_path):
    controller = load_controller()
    docker = FakeDocker(controller)

    def model_caller(payload, timeout):
        seed = payload["seed"]
        value = "chosen" if seed == 1 else "dead"
        return {
            "content": f"writing {value}",
            "tool_calls": [
                {
                    "id": f"call-{seed}",
                    "name": "write",
                    "arguments": json.dumps(
                        {"path": "/app/value", "content": value}
                    ),
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "finish_reason": "tool_calls",
            "latency_s": 0.25,
        }

    runtime = controller.TerminalParticleController(
        docker,
        request_template=request_template(),
        base_url="http://unused",
        temperature=0.8,
        top_p=0.95,
        max_tokens=64,
        timeout_s=1,
        model_caller=model_caller,
    )
    manifest = sample_manifest(controller)
    messages = request_template()["messages"]
    survivor = runtime.spawn(manifest, messages, slot=0, name="p0")
    replaced = runtime.spawn(manifest, messages, slot=1, name="p1")

    runtime.advance(survivor, seed=1)
    runtime.advance(replaced, seed=2)
    assert survivor.state != replaced.state

    clone = runtime.resample(
        survivor=survivor,
        replaced=replaced,
        name="p1-clone",
    )

    assert clone.state == survivor.state
    assert clone.messages == survivor.messages
    assert clone.model_prefix_sha256 == survivor.model_prefix_sha256
    assert clone.lineage["parent_particle_id"] == survivor.particle_id
    assert clone.resampled_from == survivor.particle_id
    assert "p1" in docker.removed
    assert runtime.cost["model_calls"] == 2
    assert runtime.cost["prompt_tokens"] == 20
    assert runtime.cost["environment_replays"] == 3
    assert runtime.cost["tool_calls"] == 2

    report = tmp_path / "report.json"
    paths = controller.write_particle_checkpoints(
        report,
        [survivor, clone],
    )
    checkpoint = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert checkpoint["restorable_by_replay"] is True
    assert checkpoint["messages"] == clone.messages
    assert checkpoint["manifest"]["expected_state"] == clone.state
    assert checkpoint["message_prefix_sha256"] == clone.model_prefix_sha256
    loaded = controller.load_particle_checkpoint(Path(paths[1]))
    assert loaded["messages"] == clone.messages
    restored = controller.restore_particle_checkpoint(
        docker,
        checkpoint_path=Path(paths[1]),
        name="restored",
    )
    assert restored["fork"]["state"] == clone.state
    assert restored["message_prefix_sha256"] == clone.model_prefix_sha256
    docker.remove("restored")
    runtime.remove_particles([survivor, clone])

def test_failed_edit_validation_is_a_replayable_noop():
    controller = load_controller()
    docker = FakeDocker(controller)
    docker.files["particle"] = {"/app/value": "original"}
    call = {
        "id": "failed-edit",

        "function": {
            "name": "edit",
            "arguments": json.dumps(
                {
                    "path": "/app/value",
                    "edits": [{"oldText": "missing", "newText": "changed"}],
                }
            ),
        },
    }

    result, event = controller.execute_live_tool(
        docker,
        "particle",
        call,
        workdir="/app",
    )

    assert result.is_error is True
    assert event["expected_error"] is True
    assert event["mutation_status"] == "none"
    assert docker.files["particle"]["/app/value"] == "original"

    call["id"] = "failed-edit-schema"
    call["function"]["arguments"] = json.dumps(
        {
            "path": "/app/value",
            "edits": {"oldText": "original", "newText": "changed"},
        }
    )
    result, event = controller.execute_live_tool(
        docker,
        "particle",
        call,
        workdir="/app",
    )
    assert result.is_error is True
    assert event["mutation_status"] == "none"
    assert docker.files["particle"]["/app/value"] == "original"

    unknown_call = {
        "id": "unknown-tool",
        "function": {"name": "browser", "arguments": "{}"},
    }
    result, event = controller.execute_live_tool(
        docker,
        "particle",
        unknown_call,
        workdir="/app",
    )
    assert result.is_error is True
    assert event["mutation_status"] == "none"


def test_malformed_model_tool_json_becomes_replayable_error():
    controller = load_controller()
    docker = FakeDocker(controller)

    def model_caller(payload, timeout):
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "malformed-edit",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": '{"path": "/app/value"',
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "latency_s": 0.1,
        }

    runtime = controller.TerminalParticleController(
        docker,
        request_template=request_template(),
        base_url="http://unused",
        temperature=0.7,
        top_p=0.95,
        max_tokens=64,
        timeout_s=1,
        model_caller=model_caller,
    )
    current = runtime.spawn(
        sample_manifest(controller),
        request_template()["messages"],
        slot=0,
        name="particle",
    )
    before = current.state

    transition = runtime.advance(current, seed=0)

    event = current.manifest["tool_events"][-1]
    assert transition["finished"] is False
    assert event["call_id"] == "malformed-edit"
    assert event["expected_error"] is True
    assert event["mutation_status"] == "none"
    assert event["validation_error"] == "invalid_json_arguments"
    assert event["arguments"]["raw_arguments"] == '{"path": "/app/value"'
    normalized = json.loads(
        current.messages[-2]["tool_calls"][0]["function"]["arguments"]
    )
    assert normalized == {
        "_smcsd_invalid_json": '{"path": "/app/value"',
    }
    assert current.messages[-1]["role"] == "tool"
    assert "invalid JSON arguments" in current.messages[-1]["content"]
    assert current.state == before
    assert current.generated_tokens == 5
    runtime.remove_particles([current])


def test_length_capped_response_continues_until_a_real_stop():
    controller = load_controller()
    docker = FakeDocker(controller)
    payloads = []
    responses = iter(
        [
            {
                "content": "partial reasoning",
                "tool_calls": [],
                "finish_reason": "length",
                "usage": {"prompt_tokens": 10, "completion_tokens": 64},
                "latency_s": 0.1,
            },
            {
                "content": "done",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                "latency_s": 0.1,
            },
        ]
    )

    def model_caller(payload, timeout):
        payloads.append(payload)
        return next(responses)

    runtime = controller.TerminalParticleController(
        docker,
        request_template=request_template(),
        base_url="http://unused",
        temperature=0.7,
        top_p=0.95,
        max_tokens=64,
        timeout_s=1,
        model_caller=model_caller,
    )
    current = runtime.spawn(
        sample_manifest(controller),
        request_template()["messages"],
        slot=0,
        name="particle-length",
    )

    first = runtime.advance(current, seed=1)
    assert first["finish_reason"] == "length"
    assert first["continued_after_length"] is True
    assert first["finished"] is False
    assert current.messages[-1] == {
        "role": "assistant",
        "content": "partial reasoning",
    }

    second = runtime.advance(current, seed=2)
    assert payloads[1]["messages"][-1] == current.messages[-2]
    assert second["continued_after_length"] is False
    assert second["finished"] is True
    assert current.generated_tokens == 65
    runtime.remove_particles([current])

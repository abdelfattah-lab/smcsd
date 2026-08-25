import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]


def load_benchmark():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "terminal_semantic_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "terminal_semantic_benchmark_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeDocker:
    def __init__(self):
        self.calls = []

    def image_identity(self, reference):
        return {
            "reference": reference,
            "image_id": "sha256:pinned",
        }

    def inspect(self, image_id):
        assert image_id == "sha256:pinned"
        return {"Config": {"WorkingDir": "/app"}}

    def run(self, args, *, check=True, timeout=None):
        self.calls.append((list(args), check, timeout))
        if args[-1] == "/logs/verifier/reward.txt":
            return subprocess.CompletedProcess(args, 0, "1\n", "")
        if args[-2:] == ["bash", "/tests/test.sh"]:
            return subprocess.CompletedProcess(args, 0, "official pass", "")
        return subprocess.CompletedProcess(args, 0, "", "")


def test_fresh_task_uses_pinned_image_and_exact_initial_request(tmp_path):
    module = load_benchmark()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """
[environment]
docker_image = "example/task:pin"

[metadata]
difficulty = "hard"
category = "software-engineering"
""".strip()
    )
    capture = tmp_path / "requests-1.jsonl"
    initial = {
        "model": "generator",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "tools": [],
    }
    capture.write_text(json.dumps({"payload": initial}) + "\n")

    task = module.fresh_task(
        FakeDocker(),
        {
            "id": "example",
            "path": str(task_dir),
            "capture_glob": str(capture),
            "state_roots": ["/app"],
        },
    )

    assert task["checkpoint"] == initial
    assert task["image_id"] == "sha256:pinned"
    assert task["workdir"] == "/app"
    assert task["manifest"]["tool_events"] == []
    assert task["manifest"]["expected_state"] == {}
    assert task["manifest"]["transcript"]["completed_tool_calls"] == 0


def test_official_tests_are_materialized_only_inside_grader(tmp_path):
    module = load_benchmark()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")
    task = {
        "task_id": "example",
        "task_dir": tmp_path,
        "workdir": "/app",
    }
    docker = FakeDocker()
    grader, metadata = module.make_official_grader(
        task,
        timeout_s=30,
        workers=1,
    )
    particle = SimpleNamespace(
        slot=0,
        particle_id="particle-0",
        container_name="container-0",
        manifest={"base_environment": {"network": "bridge"}},
    )

    rows, _ = grader(docker, [particle])

    commands = [call[0] for call in docker.calls]
    copy_index = next(
        index for index, command in enumerate(commands)
        if command[0] == "cp"
    )
    test_index = next(
        index for index, command in enumerate(commands)
        if command[-2:] == ["bash", "/tests/test.sh"]
    )
    reward_index = next(
        index for index, command in enumerate(commands)
        if command[-1] == "/logs/verifier/reward.txt"
    )
    assert copy_index < test_index < reward_index
    assert rows[0]["reward"] == 1.0
    assert metadata["tests_copied_after_terminal_semantic_scoring"] is True
    assert metadata["reward_isolated"] is True


def test_selected_repetition_ids_preserve_frozen_indices():
    module = load_benchmark()

    assert module.selected_repetition_ids(None, 3) == [0, 1, 2]
    assert module.selected_repetition_ids(
        "2,1,2", 3
    ) == [1, 2]


def test_frozen_plan_has_expected_primary_matrix():
    plan = json.loads(
        (
            REPO
            / "configs"
            / "terminal_bench"
            / "semantic_allocation_holdout_v1.json"
        ).read_text()
    )

    assert len(plan["tasks"]) == 8
    assert len(plan["methods"]) == 4

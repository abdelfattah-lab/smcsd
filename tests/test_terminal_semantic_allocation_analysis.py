import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_analyzer():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "analyze_semantic_allocation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_semantic_allocation_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pass_result(task, method, repetition, selected, rewards):
    population_rate = sum(reward > 0 for reward in rewards) / len(rewards)
    return {
        "status": "pass",
        "experiment_id": f"{task}-{method}-{repetition}",
        "repetition": repetition,
        "task": {"task_id": task},
        "configuration": {"method_id": method},
        "selection": {"reward": selected},
        "selection_baselines": {
            "ar1_slot0": {"reward": rewards[0]},
            "random_particle_expected_reward": population_rate,
            "pass_at_n": float(any(rewards)),
        },
        "grader": {
            "particles": [
                {"slot": slot, "reward": reward}
                for slot, reward in enumerate(rewards)
            ],
            "success_rate": population_rate,
            "wall_time_s": 4.0,
        },
        "generator_tool_replay_cost": {
            "model_calls": 3,
            "prompt_tokens": 30,
            "completion_tokens": 6,
            "tool_calls": 2,
            "environment_replays": 1,
            "environment_replay_wall_time_s": 0.5,
        },
        "semantic_verifier_cost": {
            "physical_verifier_calls": 2,
            "logical_verifier_calls": 2,
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "inference_wall_time_s": 0.25,
        },
        "resampling_count": 1,
        "wall_time_s": 5.0,
    }


def error_result(task, method, repetition):
    return {
        "status": "error",
        "experiment_id": f"{task}-{method}-{repetition}",
        "repetition": repetition,
        "task": {"task_id": task},
        "configuration": {"method_id": method},
        "error": "ReplayError: synthetic",
    }


def write_report(path, method, results, allocated_seconds):
    report = {
        "status": (
            "pass"
            if all(result["status"] == "pass" for result in results)
            else "partial"
        ),
        "repetitions": 2,
        "selected_repetitions": [0, 1],
        "selected_methods": [method],
        "results": results,
        "allocation_accounting": {
            "runner_wall_time_s": allocated_seconds / 2,
            "runner_allocated_accelerator_seconds": allocated_seconds,
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_analysis_merges_recovery_and_reports_paired_gate(tmp_path):
    module = load_analyzer()
    baseline_path = tmp_path / "baseline.json"
    method_path = tmp_path / "method.json"
    recovery_path = tmp_path / "recovery.json"
    write_report(
        baseline_path,
        "terminal_bon",
        [
            pass_result("a", "terminal_bon", 0, 1, [1, 0]),
            pass_result("a", "terminal_bon", 1, 0, [0, 0]),
            pass_result("b", "terminal_bon", 0, 0, [0, 0]),
            pass_result("b", "terminal_bon", 1, 0, [0, 0]),
        ],
        10,
    )
    write_report(
        method_path,
        "semantic_smc_ess25",
        [
            error_result("a", "semantic_smc_ess25", 0),
            pass_result("a", "semantic_smc_ess25", 1, 1, [1, 0]),
            pass_result("b", "semantic_smc_ess25", 0, 1, [1, 1]),
            pass_result("b", "semantic_smc_ess25", 1, 0, [0, 0]),
        ],
        20,
    )
    write_report(
        recovery_path,
        "semantic_smc_ess25",
        [pass_result("a", "semantic_smc_ess25", 0, 0, [0, 0])],
        2,
    )

    report = module.analyze(
        [baseline_path, method_path, recovery_path],
        baseline="terminal_bon",
        bootstrap_samples=100,
        seed=7,
    )

    assert report["coverage"] == {
        "tasks": ["a", "b"],
        "methods": ["semantic_smc_ess25", "terminal_bon"],
        "repetitions": [0, 1],
        "valid_cells": 8,
        "official_particle_gradings": 16,
        "executed_result_records": 9,
        "executed_error_records": 1,
        "recovered_error_cells": 1,
    }
    metrics = report["metrics"]
    assert metrics["terminal_bon"]["mean_selected_reward"] == 0.25
    assert metrics["semantic_smc_ess25"]["mean_selected_reward"] == 0.5
    assert (
        metrics["semantic_smc_ess25"][
            "executed_allocated_accelerator_seconds"
        ]
        == 22
    )
    paired = report["paired_vs_terminal_bon"][
        "semantic_smc_ess25_minus_terminal_bon"
    ]
    assert paired["selected_reward_difference"] == 0.25
    assert (paired["wins"], paired["losses"], paired["ties"]) == (2, 1, 1)
    assert report["decision_gate"]["passed"] is True


def test_extract_artifact_infers_legacy_repetition_order(tmp_path):
    module = load_analyzer()
    path = tmp_path / "legacy.json"
    report = {
        "status": "pass",
        "repetitions": 3,
        "selected_repetitions": [1, 2],
        "selected_methods": ["terminal_bon"],
        "results": [
            {
                **pass_result("a", "terminal_bon", 1, 0, [0, 0]),
            },
            {
                **pass_result("a", "terminal_bon", 2, 0, [0, 0]),
            },
        ],
    }
    for result in report["results"]:
        result.pop("repetition")
    path.write_text(json.dumps(report), encoding="utf-8")

    records, _ = module.extract_artifact(path)

    assert [record["repetition"] for record in records] == [1, 2]


def test_merge_rejects_duplicate_successful_cells():
    module = load_analyzer()
    row = {
        "key": ("a", "terminal_bon", 0),
        "task_id": "a",
        "method_id": "terminal_bon",
        "repetition": 0,
        "status": "pass",
        "source": "one.json",
    }
    duplicate = {**row, "source": "two.json"}

    with pytest.raises(ValueError, match="ambiguous duplicate"):
        module.merge_records([row, duplicate])

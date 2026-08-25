import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = REPO_ROOT / "scripts/analyze_olympiadbench_semantic_smc_holdout.py"
    spec = importlib.util.spec_from_file_location("holdout_analysis_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_script()


def candidate(problem_id, sample_id, *, correct, answer, position):
    return {
        "problem_id": problem_id,
        "sample_id": sample_id,
        "correct": correct,
        "extracted_answer": answer,
        "gold_answer": "2" if problem_id == "p0" else "5",
        "dataset_index": position + 100,
        "selection_position": position,
        "selection_seed": 0,
        "subfield": "Algebra",
    }


def test_frozen_split_rejects_development_overlap():
    rows = [candidate("p0", 0, correct=True, answer="2", position=0)]
    config = {
        "dataset": {
            "selection_positions": [0, 0],
            "selection_seed": 0,
            "excluded_development_positions": [0, 49],
        }
    }

    with pytest.raises(ValueError, match="overlaps excluded development"):
        analysis.validate_frozen_split(rows, config)


def test_aligned_methods_use_expected_candidates():
    trajectories = [
        candidate("p0", 0, correct=False, answer="3", position=50),
        candidate("p0", 1, correct=True, answer="2", position=50),
        candidate("p1", 0, correct=True, answer="5", position=51),
        candidate("p1", 1, correct=False, answer="6", position=51),
    ]
    groups = analysis.group_trajectories(
        trajectories, expected_problems=2, expected_n=2
    )
    scores_a = {
        ("p0", 0): {"score": 0.2},
        ("p0", 1): {"score": 0.9},
        ("p1", 0): {"score": 0.4},
        ("p1", 1): {"score": 0.8},
    }
    scores_b = {
        ("p0", 0): {"score": 1.0},
        ("p0", 1): {"score": 0.0},
        ("p1", 0): {"score": 0.9},
        ("p1", 1): {"score": 0.1},
    }
    smc = {
        "p0": {
            "selected_correct": True,
            "oracle_correct": True,
            "selected_slot": 1,
        },
        "p1": {
            "selected_correct": False,
            "oracle_correct": True,
            "selected_slot": 0,
        },
    }

    rows = analysis.analyze_outcomes(groups, scores_a, scores_b, smc)

    assert [row["ar_at_1"] for row in rows] == [False, True]
    assert [row["self_consistency_at_8"] for row in rows] == [False, True]
    assert [row["terminal_semantic_bon_at_8"] for row in rows] == [True, False]
    assert [row["terminal_multimodel_ensemble_at_8"] for row in rows] == [
        False,
        True,
    ]
    assert [row["semantic_smc_at_8"] for row in rows] == [True, False]


def test_bootstrap_reports_paired_wins_and_losses():
    outcomes = [
        {"a": True, "b": False},
        {"a": False, "b": True},
        {"a": True, "b": True},
        {"a": True, "b": False},
    ]

    metrics, paired = analysis.bootstrap_metrics(
        outcomes, ("a", "b"), samples=100, seed=7
    )

    assert metrics["a"]["accuracy"] == 0.75
    assert metrics["b"]["accuracy"] == 0.5
    assert paired["a"]["b"]["accuracy_difference"] == 0.25
    assert paired["a"]["b"]["wins"] == 2
    assert paired["a"]["b"]["losses"] == 1
    assert paired["a"]["b"]["ties"] == 1

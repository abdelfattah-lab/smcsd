import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name="offline_recoverability_test", path="scripts/offline_recoverability.py"):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


semantic = load_script()
math500 = load_script("accuracy_test_math500_test", "scripts/accuracy_test_math500.py")
policy = load_script("offline_policy_sim_test", "scripts/offline_policy_sim.py")
pairwise = load_script(
    "offline_pairwise_verifier_test", "scripts/offline_pairwise_verifier.py"
)
listwise = load_script(
    "offline_listwise_verifier_test", "scripts/offline_listwise_verifier.py"
)
error_audit = load_script(
    "offline_pointwise_error_audit_test",
    "scripts/offline_pointwise_error_audit.py",
)
distill = load_script(
    "offline_pairwise_distill_test", "scripts/offline_pairwise_distill.py"
)
knockout = load_script(
    "offline_knockout_verifier_test", "scripts/offline_knockout_verifier.py"
)
prefix_pairwise = load_script(
    "offline_prefix_pairwise_verifier_test",
    "scripts/offline_prefix_pairwise_verifier.py",
)
dual_audit = load_script(
    "offline_dual_semantic_audit_test", "scripts/offline_dual_semantic_audit.py"
)
semantic_ensemble = load_script(
    "offline_semantic_ensemble_test", "scripts/offline_semantic_ensemble.py"
)
particle_scale = load_script(
    "online_semantic_particlescale_test",
    "scripts/online_semantic_particlescale.py",
)
method_comparison = load_script(
    "offline_semantic_method_comparison_test",
    "scripts/offline_semantic_method_comparison.py",
)
likelihood_runner = load_script(
    "online_likelihood_smc_olympiadbench_test",
    "scripts/online_likelihood_smc_olympiadbench.py",
)
likelihood_semantic = load_script(
    "offline_likelihood_semantic_comparison_test",
    "scripts/offline_likelihood_semantic_comparison.py",
)
olympiad = load_script(
    "accuracy_test_olympiadbench_test",
    "scripts/accuracy_test_olympiadbench.py",
)
olympiad_judge = load_script(
    "olympiadbench_judge_test", "scripts/olympiadbench_judge.py"
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(text)]

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "/".join(str(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize
        assert add_generation_prompt
        return messages[0]["content"]


def trajectory(problem_id="p0", sample_id=0, correct=True):
    return {
        "problem_id": problem_id,
        "problem": "What is 1+1?",
        "sample_id": sample_id,
        "generator_model": "generator",
        "generator_output_ids": list(range(1, 21)),
        "full_text": "work; answer 2",
        "extracted_answer": "2" if correct else "3",
        "gold_answer": "2",
        "correct": correct,
    }


def test_score_tokens_are_unique_single_tokens():
    assert semantic.resolve_score_token_ids(FakeTokenizer(), ["A", "B", "C"]) == [
        65,
        66,
        67,
    ]


def test_expected_score_uses_exact_requested_token_logprobs():
    output = {
        "meta_info": {
            "output_token_ids_logprobs": [
                [
                    (math.log(0.05), 65),
                    (math.log(0.10), 66),
                    (math.log(0.35), 67),
                ]
            ],
            "prompt_tokens": 17,
            "completion_tokens": 1,
        }
    }

    result = semantic.expected_score_from_output(
        output, [65, 66, 67], [0.0, 0.5, 1.0]
    )

    assert result["score"] == pytest.approx(0.8)
    assert result["score_token_mass"] == pytest.approx(0.5)
    assert result["logprob_source"] == "selected"
    assert result["prompt_tokens"] == 17


def test_missing_requested_score_token_is_an_error():
    output = {"meta_info": {"output_token_ids_logprobs": [[(-0.1, 65)]]}}
    with pytest.raises(ValueError, match="missing 1 configured score tokens"):
        semantic.expected_score_from_output(output, [65, 66], [0.0, 1.0])


def test_prefixes_are_cut_from_saved_generator_token_ids():
    jobs = semantic.make_prefix_jobs(
        [trajectory()], FakeTokenizer(), [0.25, 0.5, 0.75, 1.0], min_prefix_tokens=1
    )

    assert [job["token_position"] for job in jobs] == [5, 10, 15, 20]
    assert jobs[1]["prefix"] == "1/2/3/4/5/6/7/8/9/10"
    assert [job["terminal"] for job in jobs] == [False, False, False, True]


def test_predictiveness_metrics_reward_perfect_ordering():
    assert semantic.binary_auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1
    assert semantic.average_precision(
        [0.9, 0.8, 0.2, 0.1], [True, True, False, False]
    ) == 1

    records = [
        {"problem_id": "p0", "sample_id": 0, "score": 0.9, "correct": True},
        {"problem_id": "p0", "sample_id": 1, "score": 0.1, "correct": False},
        {"problem_id": "p1", "sample_id": 0, "score": 0.8, "correct": True},
        {"problem_id": "p1", "sample_id": 1, "score": 0.2, "correct": False},
    ]
    summary = semantic.summarize_checkpoint(records)
    assert summary["within_problem_ranking_accuracy"] == 1
    assert summary["top_half_correct_survival"] == 1
    assert summary["top_half_survival_lift"] == pytest.approx(0.5)
    assert summary["top1_eventual_correct_rate"] == 1


def test_selection_baselines_are_computed_from_same_samples():
    trajectories = []
    terminal = []
    specifications = {
        "p0": [("2", True, 0.9), ("2", True, 0.8), ("3", False, 0.1)],
        "p1": [("3", True, 0.9), ("4", False, 0.2), ("4", False, 0.1)],
    }
    for problem_id, samples in specifications.items():
        for sample_id, (answer, correct, score) in enumerate(samples):
            trajectories.append(
                {
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "extracted_answer": answer,
                    "gold_answer": "2" if problem_id == "p0" else "3",
                    "correct": correct,
                }
            )
            terminal.append(
                {
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "score": score,
                    "correct": correct,
                }
            )

    result = semantic.selection_baselines(trajectories, terminal)
    assert result["self_consistency_accuracy"] == 0.5
    assert result["terminal_pointwise_bon_accuracy"] == 1
    assert result["oracle_pass_at_n"] == 1


def test_schema_validation_rejects_inconsistent_correctness():
    row = trajectory(correct=True)
    row["extracted_answer"] = "3"
    with pytest.raises(ValueError, match="inconsistent correctness"):
        semantic.validate_trajectory_rows([row])


def test_math_scalar_equation_normalizes_to_its_value():
    assert math500.normalize_answer("x = 5") == "5"
    assert math500.normalize_answer("5.0") == "5"


def test_offline_policy_loader_accepts_schema_v2(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    scores = tmp_path / "scores.jsonl"
    trajectories.write_text(
        json.dumps(
            {
                "problem_id": "p0",
                "sample_id": 0,
                "extracted_answer": "2",
                "gold_answer": "2",
            }
        )
        + "\n"
    )
    scores.write_text(
        json.dumps(
            {
                "problem_id": "p0",
                "sample_id": 0,
                "requested_fraction": 0.25,
                "score": 0.7,
                "trajectory_tokens": 10,
                "token_position": 3,
                "prompt_tokens": 100,
            }
        )
        + "\n"
    )

    qids, values, ntok, positions, verifier_tokens, answers = policy.load(
        trajectories, scores
    )

    assert qids == ["p0"]
    assert values[("p0", 0)][0.25] == 0.7
    assert ntok[("p0", 0)] == 10
    assert positions[("p0", 0)][0.25] == 3
    assert verifier_tokens[("p0", 0)][0.25] == 100
    assert answers[("p0", 0)] == ("2", "2")


def pairwise_trajectory(sample_id, correct):
    row = trajectory(sample_id=sample_id, correct=correct)
    row["full_text"] = f"solution-{sample_id}"
    return row


def orientation(pair_id, low, high, candidate_a, probability_a, correct_low, correct_high):
    low_is_a = candidate_a == low
    return {
        "pair_id": pair_id,
        "problem_id": "p0",
        "sample_low": low,
        "sample_high": high,
        "candidate_a": candidate_a,
        "candidate_b": high if low_is_a else low,
        "correct_a": correct_low if low_is_a else correct_high,
        "correct_b": correct_high if low_is_a else correct_low,
        "probability_a": probability_a,
        "probability_b": 1 - probability_a,
        "prompt_tokens": 10,
        "completion_tokens": 1,
    }


def test_pairwise_jobs_include_every_pair_in_both_orders():
    rows = [
        pairwise_trajectory(0, True),
        pairwise_trajectory(1, False),
        pairwise_trajectory(2, False),
    ]

    jobs = pairwise.make_jobs(rows, FakeTokenizer())

    assert len(jobs) == 6
    assert [(jobs[0]["candidate_a"], jobs[0]["candidate_b"]),
            (jobs[1]["candidate_a"], jobs[1]["candidate_b"])] == [(0, 1), (1, 0)]
    assert "Solution A:\nsolution-0" in jobs[0]["prompt"]
    assert "Solution B:\nsolution-1" in jobs[0]["prompt"]


def test_pairwise_probability_uses_exact_a_b_logprobs():
    output = {
        "output_ids": [65],
        "meta_info": {
            "output_token_ids_logprobs": [
                [(math.log(0.6), 65), (math.log(0.2), 66)]
            ],
            "prompt_tokens": 42,
            "completion_tokens": 1,
        },
    }

    result = pairwise.choice_probability(output, [65, 66])

    assert result["probability_a"] == pytest.approx(0.75)
    assert result["choice_token_mass"] == pytest.approx(0.8)
    assert result["generated_token_id"] == 65


def test_order_swapped_pairwise_scores_rank_the_best_candidate():
    orientations = [
        orientation("p0:0:1", 0, 1, 0, 0.8, True, False),
        orientation("p0:0:1", 0, 1, 1, 0.2, True, False),
        orientation("p0:0:2", 0, 2, 0, 0.7, True, False),
        orientation("p0:0:2", 0, 2, 2, 0.3, True, False),
        orientation("p0:1:2", 1, 2, 1, 0.6, False, False),
        orientation("p0:1:2", 1, 2, 2, 0.4, False, False),
    ]
    rows = [
        pairwise_trajectory(0, True),
        pairwise_trajectory(1, False),
        pairwise_trajectory(2, False),
    ]

    pairs = pairwise.combine_orientations(orientations)
    rankings = pairwise.aggregate_rankings(rows, pairs)

    assert pairs[0]["probability_low"] == pytest.approx(0.8)
    assert pairs[0]["probability_correct"] == pytest.approx(0.8)
    assert pairs[0]["hard_order_agreement"] is True
    assert rankings[0]["ranking"] == [0, 1, 2]
    assert rankings[0]["selected_correct"] is True


def test_pairwise_analysis_attaches_pointwise_terminal_choice():
    problems = [{"problem_id": "p0"}]
    scores = [
        {"problem_id": "p0", "sample_id": 0, "terminal": True, "score": 0.2, "correct": True},
        {"problem_id": "p0", "sample_id": 1, "terminal": True, "score": 0.8, "correct": False},
    ]

    pairwise.attach_pointwise_outcomes(problems, scores)

    assert problems[0]["pointwise_terminal_selected_correct"] is False


def test_knockout_uses_seven_order_swapped_matches_for_eight_candidates():
    rows = [pairwise_trajectory(index, index == 0) for index in range(8)]
    orientations = []
    for low in range(8):
        for high in range(low + 1, 8):
            pair_id = f"p0:{low}:{high}"
            orientations.extend(
                [
                    orientation(pair_id, low, high, low, 0.8, low == 0, False),
                    orientation(pair_id, low, high, high, 0.2, low == 0, False),
                ]
            )
    pairs = pairwise.combine_orientations(orientations)
    problems = pairwise.aggregate_rankings(rows, pairs)

    pairwise.attach_knockout_outcomes(problems, rows, pairs, orientations)

    assert problems[0]["knockout_selected_sample_id"] == 0
    assert problems[0]["knockout_selected_correct"] is True
    assert len(problems[0]["knockout_matches"]) == 7
    assert problems[0]["knockout_verifier_calls"] == 14


def test_listwise_jobs_randomize_and_label_every_candidate_once():
    rows = [pairwise_trajectory(index, index == 0) for index in range(4)]

    first = listwise.make_jobs(rows, FakeTokenizer(), seed=7)
    second = listwise.make_jobs(rows, FakeTokenizer(), seed=7)

    assert first[0]["label_to_sample"] == second[0]["label_to_sample"]
    assert set(first[0]["label_to_sample"]) == {"A", "B", "C", "D"}
    assert set(first[0]["label_to_sample"].values()) == {0, 1, 2, 3}
    for label, sample_id in first[0]["label_to_sample"].items():
        assert f"Solution {label}:\nsolution-{sample_id}" in first[0]["prompt"]


def test_listwise_distribution_uses_all_exact_label_logprobs():
    output = {
        "output_ids": [66],
        "meta_info": {
            "output_token_ids_logprobs": [
                [
                    (math.log(0.1), 65),
                    (math.log(0.6), 66),
                    (math.log(0.2), 67),
                ]
            ],
            "prompt_tokens": 99,
            "completion_tokens": 1,
        },
    }

    result = listwise.choice_distribution(output, ["A", "B", "C"], [65, 66, 67])

    assert result["probabilities"]["B"] == pytest.approx(2 / 3)
    assert result["choice_token_mass"] == pytest.approx(0.9)
    assert result["generated_label"] == "B"
    assert result["prompt_tokens"] == 99


def test_listwise_probability_ranking_selects_mapped_candidate():
    rows = [pairwise_trajectory(index, index == 0) for index in range(3)]
    jobs = [
        {
            "job_id": "p0",
            "problem_id": "p0",
            "label_to_sample": {"A": 2, "B": 0, "C": 1},
        }
    ]
    scores = [
        {
            "job_id": "p0",
            "probabilities": {"A": 0.1, "B": 0.8, "C": 0.1},
            "generated_label": "B",
        }
    ]

    problems = listwise.aggregate_rankings(rows, jobs, scores)

    assert problems[0]["ranking"] == [0, 2, 1]
    assert problems[0]["selected_correct"] is True


def test_error_audit_prompt_is_pointwise_and_error_focused():
    prompt = error_audit.build_error_audit_prompt(
        FakeTokenizer(),
        problem="Compute 2+2.",
        prefix="2+2=5",
    )

    assert "one partial solution" in prompt
    assert "substantive error" in prompt
    assert "A=0%" in prompt
    assert "E=100%" in prompt
    assert "Solution B" not in prompt


def test_error_audit_fixed_token_jobs_share_online_horizons():
    jobs = error_audit.make_fixed_token_jobs(
        [trajectory()],
        FakeTokenizer(),
        [5, 10],
        include_terminal=True,
        min_prefix_tokens=1,
    )

    assert [job["checkpoint"] for job in jobs] == [
        "token_5",
        "token_10",
        "terminal",
    ]
    assert [job["token_position"] for job in jobs] == [5, 10, 20]
    assert [job["terminal"] for job in jobs] == [False, False, True]
    assert jobs[0]["prefix"] == "1/2/3/4/5"


def test_prefix_pairwise_jobs_use_only_cross_outcome_equal_horizon_pairs():
    rows = [
        pairwise_trajectory(0, True),
        pairwise_trajectory(1, False),
        pairwise_trajectory(2, False),
    ]
    for row in rows:
        row["generator_output_ids"] = list(range(1, 21))

    jobs = prefix_pairwise.make_jobs(
        rows, FakeTokenizer(), FakeTokenizer(), [5, 10]
    )

    assert len(jobs) == 8
    assert {job["checkpoint"] for job in jobs} == {"token_5", "token_10"}
    assert all(job["correct_a"] != job["correct_b"] for job in jobs)
    assert all("stopped after exactly" in job["prompt"] for job in jobs)
    assert "Partial solution A:\n1/2/3/4/5" in jobs[0]["prompt"]
    assert "Partial solution B:\n1/2/3/4/5" in jobs[0]["prompt"]


def test_prefix_pairwise_combines_orders_and_aligns_pointwise_scores():
    orientations = [
        {
            **orientation("token_5:p0:0:1", 0, 1, 0, 0.8, True, False),
            "checkpoint": "token_5",
            "token_position": 5,
        },
        {
            **orientation("token_5:p0:0:1", 0, 1, 1, 0.2, True, False),
            "checkpoint": "token_5",
            "token_position": 5,
        },
    ]
    scores = [
        {"problem_id": "p0", "sample_id": 0, "checkpoint": "token_5", "score": 0.9},
        {"problem_id": "p0", "sample_id": 1, "checkpoint": "token_5", "score": 0.1},
    ]

    pairs = prefix_pairwise.combine_orientations(orientations)
    prefix_pairwise.attach_pointwise_scores(pairs, scores)

    assert pairs[0]["probability_correct"] == pytest.approx(0.8)
    assert pairs[0]["correct_sample_id"] == 0
    assert pairs[0]["pointwise_margin_correct"] == pytest.approx(0.8)


def test_prefix_pairwise_attaches_incorrect_trajectory_outcome():
    pairs = [
        {
            "problem_id": "p0",
            "correct_sample_id": 0,
            "incorrect_sample_id": 1,
        }
    ]
    trajectories = [
        {"problem_id": "p0", "sample_id": 0, "extracted_answer": "2", "finish_reason": {"type": "stop"}},
        {"problem_id": "p0", "sample_id": 1, "extracted_answer": None, "finish_reason": {"type": "length"}},
    ]

    prefix_pairwise.attach_trajectory_outcomes(pairs, trajectories)

    assert pairs[0]["incorrect_has_answer"] is False
    assert pairs[0]["incorrect_at_length_cap"] is True


def test_budgeted_progress_prompt_separates_progress_from_correctness():
    prompt = dual_audit.build_progress_prompt(
        FakeTokenizer(),
        problem="A hard problem",
        prefix="Trying an approach",
        token_position=1024,
        total_budget=16384,
    )

    assert "15360 additional tokens" in prompt
    assert "progress audit, not a correctness audit" in prompt
    assert "A=0%" in prompt and "E=100%" in prompt


def test_semantic_pairwise_model_learns_validity_times_progress_direction():
    rows = []
    for problem_index in range(10):
        for sample_id, correct, validity, progress in (
            (0, True, 0.9, 0.8),
            (1, False, 0.8, 0.2),
            (2, False, 0.2, 0.9),
        ):
            rows.append(
                {
                    "problem_id": f"p{problem_index}",
                    "sample_id": sample_id,
                    "correct": correct,
                    "features": {"validity": validity, "progress": progress},
                }
            )

    predictions, models = semantic_ensemble.cross_validated_scores(
        rows,
        ["validity", "progress"],
        folds=5,
        seed=3,
        ridge=1.0,
    )

    for problem_index in range(10):
        assert predictions[(f"p{problem_index}", 0)] > predictions[(f"p{problem_index}", 1)]
        assert predictions[(f"p{problem_index}", 0)] > predictions[(f"p{problem_index}", 2)]
    assert all(
        model["n_train_problems"] == 8
        for fold, model in models.items()
        if fold != "all_data_deployment_fit_not_used_for_metrics"
    )


def test_problem_folds_never_split_a_problem():
    problem_ids = [f"p{index // 3}" for index in range(30)]

    folds = distill.make_problem_folds(problem_ids, folds=5, seed=3)

    assert len(folds) == 10
    assert set(folds.values()) == set(range(5))
    assert all(folds[problem_id] == folds[problem_id] for problem_id in problem_ids)


def test_out_of_fold_pairwise_teacher_distillation_learns_audit_direction():
    rows = []
    for problem_index in range(10):
        for sample_id, audit_score, teacher_rating in (
            (0, 0.9, 0.8),
            (1, 0.1, 0.2),
        ):
            rows.append(
                {
                    "problem_id": f"p{problem_index}",
                    "sample_id": sample_id,
                    "requested_fraction": 1.0,
                    "token_position": 10,
                    "recoverability_score": 0.5,
                    "error_audit_score": audit_score,
                    "pairwise_teacher_rating": teacher_rating,
                }
            )

    fused, models = distill.cross_validated_distillation(
        rows, folds=5, ridge=1.0, seed=0
    )

    by_problem = {}
    for row in fused:
        by_problem.setdefault(row["problem_id"], {})[row["sample_id"]] = row
    assert len(fused) == len(rows)
    assert all(group[0]["score"] > group[1]["score"] for group in by_problem.values())
    assert all(
        model["n_train_problems"] == 8
        for fold, model in models["1.0"].items()
        if fold != "all_data_deployment_fit_not_used_for_metrics"
    )


def test_stagewise_knockout_builds_only_current_round_calls():
    rows = [pairwise_trajectory(index, index == 0) for index in range(8)]
    groups = knockout.group_trajectories(rows)
    alive = {"p0": list(range(8))}

    jobs = knockout.make_round_jobs(
        alive, groups, FakeTokenizer(), round_index=0
    )

    assert len(jobs) == 8
    assert {job["round"] for job in jobs} == {0}
    assert {job["pair_id"] for job in jobs} == {
        "p0:0:1",
        "p0:2:3",
        "p0:4:5",
        "p0:6:7",
    }


def test_stagewise_knockout_advances_symmetrized_winners():
    alive = {"p0": [0, 1, 2, 3]}
    pairs = [
        {
            "pair_id": "p0:0:1",
            "problem_id": "p0",
            "sample_low": 0,
            "sample_high": 1,
            "probability_low": 0.8,
            "probability_high": 0.2,
        },
        {
            "pair_id": "p0:2:3",
            "problem_id": "p0",
            "sample_low": 2,
            "sample_high": 3,
            "probability_low": 0.4,
            "probability_high": 0.6,
        },
    ]

    survivors, matches = knockout.advance_round(alive, pairs, round_index=0)

    assert survivors == {"p0": [0, 3]}
    assert [row["winner"] for row in matches] == [0, 3]


def test_olympiadbench_extracts_nested_and_multiple_boxed_answers():
    text = r"Work. \boxed{\frac{1}{2}} and \boxed{(3,4)}"

    assert olympiad.extract_boxed_answers(text) == r"\frac{1}{2},(3,4)"
    assert olympiad.extract_boxed_answers("no boxed answer") is None
    assert olympiad.extract_boxed_answers(r"\boxed{1") is None


def test_olympiadbench_judge_matches_symbolic_and_unordered_answers():
    judge = olympiad_judge.OlympiadBenchJudge()

    assert judge.judge(r"$\frac{1}{2}$", r"\boxed{\frac{2}{4}}")
    assert judge.judge(r"$69$,$84$", r"\boxed{84,69}")
    assert judge.judge(r"$[0,1)$", r"\boxed{[0,1)}")
    assert not judge.judge(r"$\frac{1}{2}$", r"\boxed{\frac{3}{4}}")


def test_olympiadbench_answer_clusters_preserve_schema_equality():
    judge = olympiad_judge.OlympiadBenchJudge()
    gold = r"$\frac{1}{2}$"

    answers, correctness = olympiad.cluster_answers(
        [r"\frac{2}{4}", "3", "3.0", None], gold, 1e-8, judge
    )

    assert correctness == [True, False, False, False]
    assert answers == [gold, "3", "3", None]


def test_olympiadbench_frozen_permutation_is_reproducible():
    first = olympiad.frozen_indices(20, 7)
    second = olympiad.frozen_indices(20, 7)

    assert first == second
    assert sorted(first) == list(range(20))
    assert first != olympiad.frozen_indices(20, 8)


def test_semantic_particle_resamplers_are_fixed_shape():
    rng = __import__("random").Random(0)

    assert particle_scale.systematic_indices([1.0, 0.0, 0.0, 0.0], rng) == [
        0,
        0,
        0,
        0,
    ]
    assert particle_scale.deterministic_fork_indices([0.1, 0.9, 0.2, 0.8]) == [
        1,
        1,
        3,
        3,
    ]


def test_semantic_particle_clone_propagates_score_state_and_resets_weight():
    particles = [
        {
            "particle_id": index,
            "ancestry": [],
            "semantic_score": 0.1 * index,
            "last_semantic_score": 0.05 * index,
            "rubric_scores": {"validity": 0.1 * index},
            "log_weight": 10.0 + index,
        }
        for index in range(4)
    ]

    clones, next_id = particle_scale.clone_population(
        particles, [3, 3, 1, 0], next_particle_id=10
    )

    assert next_id == 14
    assert [row["parent_particle_id"] for row in clones] == [3, 3, 1, 0]
    assert [row["semantic_score"] for row in clones] == pytest.approx(
        [0.3, 0.3, 0.1, 0.0]
    )
    assert all(row["log_weight"] == 0 for row in clones)
    assert clones[0]["ancestry"] == [3]


def test_semantic_method_comparison_uses_paired_problem_bootstrap():
    outcomes = {
        "p0": {"a": True, "b": False},
        "p1": {"a": True, "b": True},
        "p2": {"a": False, "b": False},
        "p3": {"a": True, "b": False},
    }

    metrics, paired = method_comparison.bootstrap(
        outcomes, ["a", "b"], samples=200, seed=3
    )

    assert metrics["a"]["accuracy"] == 0.75
    assert metrics["b"]["accuracy"] == 0.25
    assert paired["a"]["b"]["accuracy_difference"] == 0.5
    assert paired["a"]["b"]["wins"] == 2
    assert paired["a"]["b"]["losses"] == 0


def test_likelihood_weighted_vote_aggregates_equivalent_answers():
    answers = ["a", "b", "a", None]
    weights = [0.0, 0.5, 0.0, 100.0]

    assert likelihood_runner.weighted_vote(answers, weights) == "a"
    assert likelihood_semantic.weighted_vote(answers, weights) == "a"


def test_hybrid_weighted_vote_is_stable_on_ties_and_nonfinite_weights():
    assert likelihood_semantic.weighted_vote(["a", "b"], [0.0, 0.0]) == "a"
    assert likelihood_semantic.weighted_vote(["a", "b"], [-math.inf, 0.0]) == "b"
    assert likelihood_semantic.weighted_vote([None, None], [0.0, 1.0]) is None


def test_frozen_v1_manifest_matches_study_contract():
    manifest = json.loads(
        (REPO_ROOT / "configs/semantic/semantic_verifier_v1.json").read_text()
    )
    assert manifest["dataset"] == {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "start_index": 0,
        "num_problems": 200,
        "held_out": False,
    }
    assert manifest["generator"]["samples_per_problem"] == 8
    assert manifest["generator"]["thinking"] == "off"
    assert manifest["prefixes"]["fractions"] == [0.25, 0.5, 0.75, 1.0]
    assert len(manifest["verifier"]["score_labels"]) == 20

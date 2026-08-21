import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_likelihood_manifest_expands_to_39_jobs() -> None:
    runner = load_script(
        "run_likelihood_matrix_test",
        "scripts/terminal_bench/run_likelihood_matrix.py",
    )
    manifest = json.loads(
        (REPO_ROOT / "configs/terminal_bench/likelihood_dev_v1.json").read_text()
    )

    specs = runner.expand_specs(manifest)

    assert len(manifest["benchmark"]["tasks"]) == 12
    assert len(specs) == 39
    assert sum(spec.method == "ar" for spec in specs) == 3
    assert sum(spec.method == "smcsd" for spec in specs) == 36


def test_likelihood_matrix_filters_one_matched_pair() -> None:
    runner = load_script(
        "run_likelihood_matrix_filter_test",
        "scripts/terminal_bench/run_likelihood_matrix.py",
    )
    manifest = json.loads(
        (REPO_ROOT / "configs/terminal_bench/likelihood_dev_v1.json").read_text()
    )
    args = runner.build_parser().parse_args(
        [
            "--methods",
            "ar,smcsd",
            "--particles",
            "4",
            "--gamma",
            "4",
            "--seeds",
            "0",
        ]
    )

    specs = runner.filter_specs(runner.expand_specs(manifest), args)

    assert [spec.setting for spec in specs] == ["ar", "smcsd-n4-g4"]


def test_no_resample_manifest_expands_to_four_seed_zero_jobs() -> None:
    runner = load_script(
        "run_no_resample_matrix_test",
        "scripts/terminal_bench/run_likelihood_matrix.py",
    )
    manifest = json.loads(
        (REPO_ROOT / "configs/terminal_bench/no_resample_dev_v1.json").read_text()
    )

    specs = runner.expand_specs(manifest)

    assert [spec.setting for spec in specs] == [
        "smcsd-n16-g4-r0",
        "smcsd-n16-g8-r0",
        "smcsd-n32-g4-r0",
        "smcsd-n32-g8-r0",
    ]
    assert all(spec.resample_threshold == 0.0 for spec in specs)
    assert manifest["server"]["mem_fraction_static"] == 0.4


def test_prometheus_parser_sums_labeled_series(tmp_path: Path) -> None:
    summary = load_script(
        "summarize_pi_jobs_test",
        "scripts/terminal_bench/summarize_pi_jobs.py",
    )
    metrics = tmp_path / "metrics.prom"
    metrics.write_text(
        "# HELP sglang:num_requests_total requests\n"
        'sglang:num_requests_total{is_streaming="true"} 2\n'
        'sglang:num_requests_total{is_streaming="false"} 3\n'
        'sglang:e2e_request_latency_seconds_sum{model_name="target"} 1.25\n'
        'sglang:e2e_request_latency_seconds_count{model_name="target"} 5\n'
    )

    parsed = summary.parse_prometheus(metrics)

    assert parsed["sglang:num_requests_total"] == 5
    assert (
        summary.average_histogram(parsed, "sglang:e2e_request_latency_seconds") == 0.25
    )


def test_summary_counts_errored_trials_as_failed_work(tmp_path: Path) -> None:
    summary = load_script(
        "summarize_pi_jobs_error_accounting_test",
        "scripts/terminal_bench/summarize_pi_jobs.py",
    )
    metadata = {
        "job_name": "screen-ar-seed0",
        "run_tag": "screen-seed0",
        "tasks": ["task-ok", "task-timeout"],
        "spec": {"method": "ar", "seed": 0},
        "measurement_started_at": "2026-01-01T00:00:00Z",
        "measurement_finished_at": "2026-01-01T00:01:00Z",
    }
    (tmp_path / "experiment.json").write_text(json.dumps(metadata))
    (tmp_path / "prometheus_before.prom").write_text("")
    (tmp_path / "prometheus_after.prom").write_text("")

    cases = [
        ("task-ok", 1.0, None, 10, 100, 20),
        (
            "task-timeout",
            None,
            {"exception_type": "AgentTimeoutError"},
            20,
            200,
            30,
        ),
    ]
    for task, reward, exception, seconds, input_tokens, output_tokens in cases:
        trial_dir = tmp_path / task
        trial_dir.mkdir()
        result = {
            "task_name": task,
            "trial_name": task,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
            "agent_execution": {
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": f"2026-01-01T00:00:{seconds:02d}Z",
            },
            "agent_result": {
                "n_input_tokens": input_tokens,
                "n_output_tokens": output_tokens,
            },
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": exception,
        }
        (trial_dir / "result.json").write_text(json.dumps(result))

    aggregate, trials = summary.aggregate_job(tmp_path)

    assert len(trials) == 2
    assert aggregate["n_expected_trials"] == 2
    assert aggregate["n_completed"] == 2
    assert aggregate["n_non_error"] == 1
    assert aggregate["n_errors"] == 1
    assert aggregate["reward_sum"] == 1.0
    assert aggregate["pass_rate"] == 0.5
    assert aggregate["agent_wall_time_s"] == 30.0
    assert aggregate["correct_tasks_per_agent_gpu_hour"] == 120.0
    assert aggregate["agent_input_tokens"] == 300
    assert aggregate["agent_output_tokens"] == 50

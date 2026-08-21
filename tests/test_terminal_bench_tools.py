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

#!/usr/bin/env python3
"""Run the frozen target-AR versus likelihood-only SM-CSD agent matrix.

The default is a dry run. Pass --execute to launch one server at a time, warm
its tool-call contract, snapshot Prometheus counters, run Harbor, and store
the resolved experiment metadata alongside the Harbor job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs/terminal_bench/likelihood_dev_v1.json"


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    particles: int | None = None
    gamma: int | None = None

    @property
    def setting(self) -> str:
        if self.method == "ar":
            return "ar"
        return f"smcsd-n{self.particles}-g{self.gamma}"


def _csv_ints(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _csv_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def expand_specs(manifest: dict[str, Any]) -> list[RunSpec]:
    matrix = manifest["matrix"]
    specs: list[RunSpec] = []
    for seed in matrix["seeds"]:
        if "ar" in matrix["methods"]:
            specs.append(RunSpec(method="ar", seed=seed))
        if "smcsd" in matrix["methods"]:
            for particles in matrix["particles"]:
                for gamma in matrix["gamma"]:
                    specs.append(
                        RunSpec(
                            method="smcsd",
                            seed=seed,
                            particles=particles,
                            gamma=gamma,
                        )
                    )
    return specs


def filter_specs(specs: Iterable[RunSpec], args: argparse.Namespace) -> list[RunSpec]:
    methods = _csv_strings(args.methods)
    particles = _csv_ints(args.particles)
    gammas = _csv_ints(args.gamma)
    seeds = _csv_ints(args.seeds)
    selected = [
        spec
        for spec in specs
        if (methods is None or spec.method in methods)
        and (seeds is None or spec.seed in seeds)
        and (
            spec.method == "ar"
            or (
                (particles is None or spec.particles in particles)
                and (gammas is None or spec.gamma in gammas)
            )
        )
    ]
    if args.max_configs is not None:
        selected = selected[: args.max_configs]
    return selected


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def validate_external_checkout(
    path: Path, expected: str, label: str, allow_mismatch: bool
) -> str:
    if not path.is_dir():
        raise RuntimeError(f"{label} checkout does not exist: {path}")
    actual = git_head(path)
    if actual != expected and not allow_mismatch:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, found {actual}; "
            "use --allow-revision-mismatch only for non-paper debugging"
        )
    return actual


def fetch_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def endpoint_is_live(url: str) -> bool:
    try:
        fetch_text(url, timeout=1.0)
    except (OSError, urllib.error.URLError):
        return False
    return True


def wait_for_server(url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "server not ready"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"server exited before readiness with status {code}")
        try:
            fetch_text(url, timeout=2.0)
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server readiness timed out after {timeout_s}s: {last_error}")


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--methods", help="Comma-separated subset: ar,smcsd")
    parser.add_argument("--particles", help="Comma-separated SM-CSD N values")
    parser.add_argument("--gamma", help="Comma-separated SM-CSD gamma values")
    parser.add_argument("--seeds", help="Comma-separated seeds")
    parser.add_argument("--tasks", help="Comma-separated frozen task subset")
    parser.add_argument(
        "--run-tag",
        help="Artifact namespace, e.g. full or pilot-a; inferred safely if omitted.",
    )
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--substrate-root",
        type=Path,
        default=Path(os.environ.get("SUBSTRATE_ROOT", "~/agentbench/substrate")),
    )
    parser.add_argument(
        "--tb21-root",
        type=Path,
        default=Path(os.environ.get("TB21_ROOT", "~/agentbench/terminal-bench-2-1")),
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path(os.environ.get("JOBS_DIR", "~/agentbench/jobs")),
    )
    parser.add_argument("--skip-contract", action="store_true")
    parser.add_argument("--allow-revision-mismatch", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def run_one(
    spec: RunSpec,
    tasks: list[str],
    manifest: dict[str, Any],
    args: argparse.Namespace,
    revisions: dict[str, str],
) -> None:
    experiment_id = manifest["experiment_id"]
    job_name = f"{experiment_id}__{args.run_tag}__{spec.setting}__seed{spec.seed}"
    jobs_dir = args.jobs_dir.expanduser().resolve()
    job_dir = jobs_dir / job_name
    if job_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing job: {job_dir}")

    host_base = f"http://127.0.0.1:{args.port}"
    models_url = f"{host_base}/v1/models"
    if endpoint_is_live(models_url):
        raise RuntimeError(f"port {args.port} already serves an OpenAI endpoint")

    logs_dir = jobs_dir / ".server-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job_name}.log"
    failure_path = logs_dir / f"{job_name}.failed.json"
    launcher = (
        REPO_ROOT
        / "scripts/terminal_bench"
        / (
            "launch_ar_pi_docker.sh"
            if spec.method == "ar"
            else "launch_smcsd_pi_docker.sh"
        )
    )
    server_env = os.environ.copy()
    server_env.update(
        {
            "GPU_DEVICE": args.gpu,
            "PORT": str(args.port),
            "TARGET_MODEL": manifest["models"]["target"],
            "MAX_RUNNING_REQUESTS": str(manifest["server"]["max_running_requests"]),
            "RANDOM_SEED": str(spec.seed),
            "ENABLE_METRICS": "true",
            "DISABLE_FLASHINFER_AUTOTUNE": "false",
        }
    )
    if spec.method == "smcsd":
        server_env.update(
            {
                "DRAFT_MODEL": manifest["models"]["draft"],
                "PARTICLES": str(spec.particles),
                "GAMMA": str(spec.gamma),
                "DISABLE_CUDA_GRAPH": "false",
            }
        )

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_tag": args.run_tag,
        "job_name": job_name,
        "status": "starting",
        "spec": {
            "method": spec.method,
            "particles": spec.particles,
            "gamma": spec.gamma,
            "seed": spec.seed,
        },
        "tasks": tasks,
        "models": manifest["models"],
        "revisions": revisions,
        "gpu_device": args.gpu,
        "server_log": str(log_path),
        "started_at": utc_now(),
    }

    process: subprocess.Popen[str] | None = None
    server_log = log_path.open("w")
    try:
        launch_started = time.monotonic()
        process = subprocess.Popen(
            [str(launcher)],
            cwd=REPO_ROOT,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_server(models_url, process, args.server_timeout)
        metadata["server_startup_s"] = time.monotonic() - launch_started

        if not args.skip_contract:
            contract = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/terminal_bench/smcsd_pi_contract.py"),
                    "--base-url",
                    host_base,
                    "--model",
                    manifest["models"]["target"],
                    "--timeout",
                    "300",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            metadata["contract"] = json.loads(contract.stdout)

        metrics_before = fetch_text(f"{host_base}/metrics", timeout=30.0)
        metadata["measurement_started_at"] = utc_now()
        harbor_env = os.environ.copy()
        harbor_env.update(
            {
                "SUBSTRATE_ROOT": str(args.substrate_root),
                "TB21_ROOT": str(args.tb21_root),
                "JOBS_DIR": str(jobs_dir),
                "JOB_NAME": job_name,
                "TASK_IDS": ",".join(tasks),
                "TRIALS": str(manifest["agent"]["attempts_per_seed"]),
                "CONCURRENCY": str(manifest["agent"]["concurrency"]),
                "PI_VERSION": manifest["agent"]["pi_version"],
                "SMC_MODEL": manifest["models"]["target"],
                "SMC_BASE_URL": f"http://172.17.0.1:{args.port}/v1",
            }
        )
        subprocess.run(
            [str(REPO_ROOT / "scripts/terminal_bench/run_pi_smoke.sh")],
            cwd=REPO_ROOT,
            env=harbor_env,
            check=True,
        )
        metadata["measurement_finished_at"] = utc_now()
        metrics_after = fetch_text(f"{host_base}/metrics", timeout=30.0)

        if not job_dir.is_dir():
            raise RuntimeError(f"Harbor finished but did not create {job_dir}")
        (job_dir / "prometheus_before.prom").write_text(metrics_before)
        (job_dir / "prometheus_after.prom").write_text(metrics_after)
        metadata["status"] = "complete"
        metadata["finished_at"] = utc_now()
        write_json(job_dir / "experiment.json", metadata)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["finished_at"] = utc_now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if job_dir.is_dir():
            write_json(job_dir / "experiment.json", metadata)
        else:
            write_json(failure_path, metadata)
        raise
    finally:
        if process is not None:
            stop_server(process)
        server_log.close()


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("only experiment schema_version=1 is supported")

    specs = filter_specs(expand_specs(manifest), args)
    frozen_tasks = [entry["id"] for entry in manifest["benchmark"]["tasks"]]
    requested_tasks = _csv_strings(args.tasks)
    tasks = [
        task
        for task in frozen_tasks
        if requested_tasks is None or task in requested_tasks
    ]
    unknown_tasks = (requested_tasks or set()) - set(frozen_tasks)
    if unknown_tasks:
        raise RuntimeError(
            f"tasks are not in the frozen manifest: {sorted(unknown_tasks)}"
        )
    if not tasks:
        raise RuntimeError("no tasks selected")
    if not specs:
        raise RuntimeError("no matrix configurations selected")
    if args.run_tag is None:
        if tasks == frozen_tasks:
            args.run_tag = "full"
        else:
            digest = hashlib.sha256(",".join(tasks).encode()).hexdigest()[:8]
            args.run_tag = f"subset-{len(tasks)}-{digest}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_tag) is None:
        raise RuntimeError(
            "--run-tag must start with an alphanumeric and contain only "
            "letters, numbers, dot, underscore, or dash"
        )

    preview = [
        {
            "method": spec.method,
            "particles": spec.particles,
            "gamma": spec.gamma,
            "seed": spec.seed,
            "tasks": tasks,
        }
        for spec in specs
    ]
    if not args.execute:
        print(
            json.dumps(
                {"run_tag": args.run_tag, "n_jobs": len(preview), "jobs": preview},
                indent=2,
            )
        )
        return 0

    args.substrate_root = args.substrate_root.expanduser().resolve()
    args.tb21_root = args.tb21_root.expanduser().resolve()
    args.jobs_dir = args.jobs_dir.expanduser().resolve()
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    revisions = {
        "smcsd": git_head(REPO_ROOT),
        "substrate": validate_external_checkout(
            args.substrate_root,
            manifest["agent"]["adapter_commit"],
            "tb-with-pi",
            args.allow_revision_mismatch,
        ),
        "terminal_bench": validate_external_checkout(
            args.tb21_root,
            manifest["benchmark"]["dataset_commit"],
            "Terminal-Bench",
            args.allow_revision_mismatch,
        ),
    }

    failures = 0
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.setting} seed={spec.seed}", flush=True)
        try:
            run_one(spec, tasks, manifest, args, revisions)
        except BaseException as exc:
            failures += 1
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if not args.keep_going:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run matched semantic-allocation methods on held-out Terminal-Bench tasks.

The driver starts every method from a fresh pinned task image and the exact
initial Pi provider request. Official task tests are copied into each container
only after generation and terminal semantic scoring are complete, so neither
the generator nor semantic verifier can observe rewards or hidden tests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tomllib
from typing import Any, Callable, Sequence
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import terminal_particle_backend as backend
import terminal_particle_controller as live
import terminal_semantic_smc as smc


SCHEMA_VERSION = 1
METHOD_POLICIES = {"semantic_smc", "terminal_bon", "particle_scale"}


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def resolve_initial_capture(pattern: str) -> tuple[Path, dict[str, Any]]:
    expanded = str(Path(pattern).expanduser())
    matches = [Path(value) for value in sorted(glob.glob(expanded))]
    if not matches:
        raise FileNotFoundError(f"capture glob matched nothing: {pattern}")
    failures: list[str] = []
    for path in matches:
        try:
            return path, smc.load_initial_captured_checkpoint(path)
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")
    raise live.ControllerError(
        "no capture contained one unambiguous initial request: "
        + "; ".join(failures[:3])
    )


def fresh_task(
    docker: backend.DockerCLI,
    task_row: dict[str, Any],
) -> dict[str, Any]:
    task_dir = Path(task_row["path"]).expanduser().resolve()
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    task_id = str(task_row.get("id") or task_dir.name)
    capture_path, checkpoint = resolve_initial_capture(
        str(task_row["capture_glob"])
    )
    environment = config["environment"]
    image_reference = str(environment["docker_image"])
    identity = docker.image_identity(image_reference)
    image = docker.inspect(identity["image_id"])
    workdir = str(
        task_row.get("workdir")
        or (image.get("Config") or {}).get("WorkingDir")
        or "/"
    )
    state_roots = [
        str(value)
        for value in task_row.get("state_roots", [workdir])
    ]
    messages = checkpoint["messages"]
    prefix_sha256 = backend.canonical_json_sha256(messages)
    empty_events_sha256 = backend.canonical_json_sha256([])
    manifest = {
        "schema_version": backend.SCHEMA_VERSION,
        "backend": "deterministic_pi_tool_replay_v1",
        "task_id": task_id,
        "base_environment": {
            "reference": identity["reference"],
            "image_id": identity["image_id"],
            "workdir": workdir,
            "state_roots": state_roots,
            "ignore_runtime_caches": bool(
                task_row.get("ignore_runtime_caches", True)
            ),
            "network": str(task_row.get("particle_network", "bridge")),
            "command": ["sleep", "infinity"],
        },
        "transcript": {
            "source_session": None,
            "sha256": prefix_sha256,
            "events_sha256": empty_events_sha256,
            "completed_tool_calls": 0,
        },
        "model_prefix": {
            "checkpoint_id": "initial-provider-request",
            "sha256": prefix_sha256,
            "representation": "openai_messages_v1",
            "kv_cache_handle": None,
        },
        "lineage": {
            "particle_id": uuid.uuid4().hex,
            "parent_particle_id": None,
            "generation": 0,
        },
        "tool_events": [],
        "expected_state": {},
        "sealed_from": None,
    }
    metadata = config.get("metadata") or {}
    return {
        "task_id": task_id,
        "task_dir": task_dir,
        "capture_path": capture_path,
        "checkpoint": checkpoint,
        "manifest": manifest,
        "image_reference": image_reference,
        "image_id": identity["image_id"],
        "workdir": workdir,
        "state_roots": state_roots,
        "particle_network": manifest["base_environment"]["network"],
        "difficulty": task_row.get("difficulty") or metadata.get("difficulty"),
        "category": task_row.get("category") or metadata.get("category"),
    }


def output_tail(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[-limit:]


def make_official_grader(
    task: dict[str, Any],
    *,
    timeout_s: float,
    workers: int,
) -> tuple[
    Callable[..., tuple[list[dict[str, Any]], float]],
    dict[str, Any],
]:
    tests_dir = task["task_dir"] / "tests"
    tests_digest = sha256_tree(tests_dir)

    def grade_all(
        docker: backend.DockerCLI,
        particles: Sequence[live.Particle],
    ) -> tuple[list[dict[str, Any]], float]:
        overall_started = time.perf_counter()

        def grade(particle: live.Particle) -> dict[str, Any]:
            began = time.perf_counter()
            container = particle.container_name
            network_added = False
            timed_out = False
            error_text: str | None = None
            process: subprocess.CompletedProcess[str] | None = None
            docker.run(
                [
                    "exec",
                    container,
                    "mkdir",
                    "-p",
                    "/tests",
                    "/logs/verifier",
                ]
            )
            docker.run(
                [
                    "cp",
                    str(tests_dir) + "/.",
                    f"{container}:/tests",
                ]
            )
            if particle.manifest["base_environment"]["network"] == "none":
                connected = docker.run(
                    ["network", "connect", "bridge", container],
                    check=False,
                )
                network_added = connected.returncode == 0
                if not network_added:
                    error_text = (
                        connected.stderr.strip()
                        or connected.stdout.strip()
                        or "failed to attach grader network"
                    )
            try:
                if error_text is None:
                    try:
                        process = docker.run(
                            [
                                "exec",
                                "--workdir",
                                task["workdir"],
                                container,
                                "bash",
                                "/tests/test.sh",
                            ],
                            check=False,
                            timeout=timeout_s,
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        error_text = f"official verifier exceeded {timeout_s}s"
                        docker.run(
                            [
                                "exec",
                                container,
                                "pkill",
                                "-f",
                                "/tests/test.sh|pytest|uvx",
                            ],
                            check=False,
                        )
                reward_process = docker.run(
                    [
                        "exec",
                        container,
                        "cat",
                        "/logs/verifier/reward.txt",
                    ],
                    check=False,
                )
                try:
                    reward = float(reward_process.stdout.strip())
                except ValueError:
                    reward = 0.0
                    if error_text is None:
                        error_text = (
                            reward_process.stderr.strip()
                            or "official verifier did not write reward.txt"
                        )
            finally:
                if network_added:
                    docker.run(
                        ["network", "disconnect", "bridge", container],
                        check=False,
                    )
            stdout = "" if process is None else process.stdout
            stderr = "" if process is None else process.stderr
            return {
                "slot": particle.slot,
                "particle_id": particle.particle_id,
                "reward": reward,
                "wall_time_s": time.perf_counter() - began,
                "timed_out": timed_out,
                "error": error_text,
                "test_exit_code": None if process is None else process.returncode,
                "test_stdout_tail": output_tail(stdout),
                "test_stderr_tail": output_tail(stderr),
                "test_stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "test_stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            }

        with ThreadPoolExecutor(
            max_workers=min(workers, len(particles))
        ) as executor:
            rows = list(executor.map(grade, particles))
        return (
            sorted(rows, key=lambda row: row["slot"]),
            time.perf_counter() - overall_started,
        )

    return (
        grade_all,
        {
            "name": "terminal-bench-official-test-sh-v1",
            "task_id": task["task_id"],
            "tests_path": str(tests_dir),
            "tests_sha256": tests_digest,
            "timeout_s": timeout_s,
            "workers": workers,
            "tests_copied_after_terminal_semantic_scoring": True,
            "reward_isolated": True,
        },
    )


def selected_metrics(result: dict[str, Any]) -> dict[str, float]:
    baselines = result["selection_baselines"]
    return {
        "selected_reward": float(result["selection"]["reward"]),
        "ar1_reward": float(baselines["ar1_slot0"]["reward"]),
        "terminal_semantic_reward": float(
            baselines["terminal_semantic"]["reward"]
        ),
        "pass_at_n": float(baselines["pass_at_n"]),
        "random_particle_expected_reward": float(
            baselines["random_particle_expected_reward"]
        ),
        "population_success_rate": float(result["grader"]["success_rate"]),
    }


def summarize(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "pass":
            continue
        method_id = str(result["configuration"]["method_id"])
        groups.setdefault(method_id, []).append(result)
    rows: list[dict[str, Any]] = []
    for method_id, values in sorted(groups.items()):
        metrics = [selected_metrics(result) for result in values]
        generator = [result["generator_tool_replay_cost"] for result in values]
        semantic_cost = [result["semantic_verifier_cost"] for result in values]
        rows.append(
            {
                "method_id": method_id,
                "policy": values[0]["configuration"]["method"],
                "runs": len(values),
                "tasks": len({value["task"]["task_id"] for value in values}),
                **{
                    key: statistics.fmean(row[key] for row in metrics)
                    for key in metrics[0]
                },
                "mean_wall_time_s": statistics.fmean(
                    float(value["wall_time_s"]) for value in values
                ),
                "mean_generator_calls": statistics.fmean(
                    float(value["model_calls"]) for value in generator
                ),
                "mean_physical_verifier_calls": statistics.fmean(
                    float(value["physical_verifier_calls"])
                    for value in semantic_cost
                ),
                "mean_grader_wall_time_s": statistics.fmean(
                    float(value["grader"]["wall_time_s"]) for value in values
                ),
            }
        )
    return rows


def csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {piece.strip() for piece in value.split(",") if piece.strip()}
    return result or None


def write_report(
    path: Path,
    *,
    plan_path: Path,
    plan: dict[str, Any],
    results: list[dict[str, Any]],
    status: str,
    error: str | None,
    started: float,
    verifier: smc.OnlineSemanticVerifier | None,
    verifier_startup_s: float,
    selected_tasks: Sequence[str],
    selected_methods: Sequence[str],
    repetitions: int,
) -> dict[str, Any]:
    wall_time = time.perf_counter() - started
    allocation = plan["allocation"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "terminal-semantic-allocation-benchmark-v1",
        "created_at": live.utc_now(),
        "status": status,
        "error": error,
        "plan": str(plan_path.resolve()),
        "scientific_scope": plan["scientific_scope"],
        "selected_tasks": list(selected_tasks),
        "selected_methods": list(selected_methods),
        "repetitions": repetitions,
        "models": plan["models"],
        "verifier": {
            "startup_wall_time_s": verifier_startup_s,
            "cost": None if verifier is None else verifier.cost,
        },
        "allocation_accounting": {
            "runner_wall_time_s": wall_time,
            "generator_accelerators": allocation["generator_accelerators"],
            "verifier_accelerators": allocation["verifier_accelerators"],
            "runner_allocated_accelerator_seconds": wall_time
            * (
                allocation["generator_accelerators"]
                + allocation["verifier_accelerators"]
            ),
            "external_model_server_startup_excluded": True,
            "official_grader_wall_time_charged_while_models_reserved": True,
        },
        "results": results,
        "summary": summarize(results),
    }
    backend.atomic_write_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--generator-base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--verifier-base-url", default="http://127.0.0.1:30001")
    parser.add_argument("--task-id")
    parser.add_argument("--method-id")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--num-particles", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-containers", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    task_filter = csv_filter(args.task_id)
    method_filter = csv_filter(args.method_id)
    tasks = [
        (index, row)
        for index, row in enumerate(plan["tasks"])
        if task_filter is None or row["id"] in task_filter
    ]
    methods = [
        row
        for row in plan["methods"]
        if method_filter is None or row["id"] in method_filter
    ]
    if not tasks:
        raise ValueError("no tasks selected")
    if not methods:
        raise ValueError("no methods selected")
    if any(row["policy"] not in METHOD_POLICIES for row in methods):
        raise ValueError("plan contains an unsupported method policy")
    repetitions = int(args.repetitions or plan["repetitions"])
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    started = time.perf_counter()
    verifier: smc.OnlineSemanticVerifier | None = None
    verifier_startup_s = 0.0
    results: list[dict[str, Any]] = []
    run_error: str | None = None
    status = "running"
    checkpoint_dir = args.output.parent / f"{args.output.stem}.particles"
    selected_tasks = [row["id"] for _, row in tasks]
    selected_methods = [row["id"] for row in methods]
    docker = backend.DockerCLI()
    try:
        verifier, verifier_startup_s = smc.OnlineSemanticVerifier.from_model(
            scorer_model=plan["models"]["semantic_verifier"]["model"],
            base_url=args.verifier_base_url,
            timeout_s=float(plan["timeouts"]["model_request_s"]),
            base_gpu_id=int(plan["allocation"]["generator_accelerators"]),
            tp_size=int(plan["allocation"]["verifier_accelerators"]),
            mem_fraction_static=float(
                plan["verifier_engine"]["mem_fraction_static"]
            ),
            max_running_requests=int(
                plan["verifier_engine"]["max_running_requests"]
            ),
            max_mamba_cache_size=int(
                plan["verifier_engine"]["max_mamba_cache_size"]
            ),
            seed=int(plan["seed_base"]),
            batch_size=int(plan["verifier_engine"]["batch_size"]),
            transcript_max_chars=int(plan["semantic"]["transcript_max_chars"]),
            tool_output_max_chars=int(
                plan["semantic"]["tool_output_max_chars"]
            ),
        )
        for task_index, task_row in tasks:
            task = fresh_task(docker, task_row)
            grader, grader_metadata = make_official_grader(
                task,
                timeout_s=float(
                    task_row.get(
                        "grader_timeout_s",
                        plan["timeouts"]["official_grader_s"],
                    )
                ),
                workers=int(plan["grader_workers"]),
            )
            for method_row in methods:
                settings = {
                    **plan["defaults"],
                    **method_row.get("overrides", {}),
                }
                if args.num_particles is not None:
                    settings["num_particles"] = args.num_particles
                if args.max_rounds is not None:
                    settings["max_rounds"] = args.max_rounds
                for repetition in range(repetitions):
                    seed_base = (
                        int(plan["seed_base"])
                        + task_index * 1_000_000
                        + repetition * 100_000
                    )
                    experiment_id = smc.semantic.stable_id(
                        "terminal-semantic-allocation-benchmark-v1",
                        task["task_id"],
                        method_row["id"],
                        repetition,
                        seed_base,
                        backend.canonical_json_sha256(
                            task["checkpoint"]["messages"]
                        ),
                        smc.semantic.canonical_json(settings),
                    )
                    print(
                        f"start task={task['task_id']} "
                        f"method={method_row['id']} repeat={repetition} "
                        f"id={experiment_id}",
                        flush=True,
                    )
                    try:
                        result = smc.run_configuration(
                            docker,
                            verifier,
                            manifest=task["manifest"],
                            checkpoint=task["checkpoint"],
                            experiment_id=experiment_id,
                            num_particles=int(settings["num_particles"]),
                            checkpoint_interval=int(
                                settings["checkpoint_interval_tokens"]
                            ),
                            verifier_calls=int(
                                settings["verifier_calls_per_checkpoint"]
                            ),
                            beta=float(settings["semantic_beta"]),
                            ess_threshold=float(
                                settings["ess_threshold_fraction"]
                            ),
                            generator_base_url=args.generator_base_url,
                            generator_temperature=float(
                                plan["generator"]["temperature"]
                            ),
                            generator_top_p=float(plan["generator"]["top_p"]),
                            generator_max_tokens=int(
                                plan["generator"]["max_tokens_per_turn"]
                            ),
                            timeout_s=float(
                                plan["timeouts"]["model_request_s"]
                            ),
                            max_rounds=int(settings["max_rounds"]),
                            seed_base=seed_base,
                            materialize_workers=int(
                                plan["materialize_workers"]
                            ),
                            name_prefix=(
                                f"semantic-bench-{task['task_id']}-"
                                f"{method_row['id']}"
                            ),
                            checkpoint_dir=checkpoint_dir,
                            keep_containers=args.keep_containers,
                            method=method_row["policy"],
                            grader=grader,
                            grader_name=grader_metadata["name"],
                        )
                        result["configuration"]["method_id"] = method_row["id"]
                        result["task"] = {
                            key: (
                                str(value)
                                if isinstance(value, Path)
                                else value
                            )
                            for key, value in task.items()
                            if key
                            in {
                                "task_id",
                                "task_dir",
                                "capture_path",
                                "image_reference",
                                "image_id",
                                "workdir",
                                "state_roots",
                                "particle_network",
                                "difficulty",
                                "category",
                            }
                        }
                        result["official_grader"] = grader_metadata
                    except BaseException as error:
                        result = {
                            "schema_version": SCHEMA_VERSION,
                            "experiment_id": experiment_id,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                            "task": {
                                "task_id": task["task_id"],
                                "difficulty": task["difficulty"],
                                "category": task["category"],
                            },
                            "configuration": {
                                "method": method_row["policy"],
                                "method_id": method_row["id"],
                                **settings,
                            },
                        }
                        if not plan.get("continue_on_error", True):
                            results.append(result)
                            raise
                    results.append(result)
                    write_report(
                        args.output,
                        plan_path=args.plan,
                        plan=plan,
                        results=results,
                        status="running",
                        error=None,
                        started=started,
                        verifier=verifier,
                        verifier_startup_s=verifier_startup_s,
                        selected_tasks=selected_tasks,
                        selected_methods=selected_methods,
                        repetitions=repetitions,
                    )
                    print(
                        json.dumps(
                            {
                                "id": experiment_id,
                                "status": result["status"],
                                "selected_reward": (
                                    result.get("selection") or {}
                                ).get("reward"),
                                "population_success": (
                                    result.get("grader") or {}
                                ).get("success_rate"),
                                "wall_time_s": result.get("wall_time_s"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        status = (
            "pass"
            if all(result["status"] == "pass" for result in results)
            else "partial"
        )
    except BaseException as error:
        run_error = f"{type(error).__name__}: {error}"
        status = "error"
        raise
    finally:
        if verifier is not None:
            verifier.shutdown()
        report = write_report(
            args.output,
            plan_path=args.plan,
            plan=plan,
            results=results,
            status=status,
            error=run_error,
            started=started,
            verifier=verifier,
            verifier_startup_s=verifier_startup_s,
            selected_tasks=selected_tasks,
            selected_methods=selected_methods,
            repetitions=repetitions,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "runs": len(results),
                "summary": report["summary"],
                "allocated_accelerator_seconds": report[
                    "allocation_accounting"
                ]["runner_allocated_accelerator_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

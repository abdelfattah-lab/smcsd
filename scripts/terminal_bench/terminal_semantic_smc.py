#!/usr/bin/env python3
"""Run semantic-only SMC over live, coupled Terminal-Bench particles.

The generator is an OpenAI-compatible server.  The semantic verifier is a
separate SGLang engine using the frozen leakage-safe recoverability prompt.
Every particle keeps its transcript and Docker state coupled through the
terminal particle controller.  Semantic score differences update log weights;
ESS gates systematic resampling; copied prefixes are scored only once.

This runner currently provides an exact reward-isolated grader for ``fix-git``.
It is an online quality/cost scaling harness, not a substitute for a multi-task
held-out Terminal-Bench evaluation.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_semantic_checkpoints as semantic
import terminal_particle_backend as backend
import terminal_particle_controller as live


SCHEMA_VERSION = 1
VERIFIER_VARIANTS = (
    "",
    "Focus especially on concrete evidence of task progress in tool results.",
    "Focus especially on whether the current plan is technically sound.",
    "Focus especially on irreversible mistakes and remaining recoverability.",
    "Focus especially on whether observed state supports the agent's beliefs.",
    "Focus especially on how much required work remains before completion.",
    "Focus especially on efficient next actions rather than verbal confidence.",
    "Take a skeptical overall view and account explicitly for uncertainty.",
)


def parse_csv(value: str, cast: Any) -> list[Any]:
    values = [cast(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return list(dict.fromkeys(values))


def softmax(log_weights: Sequence[float]) -> list[float]:
    if not log_weights:
        raise ValueError("cannot normalize an empty population")
    maximum = max(log_weights)
    values = [math.exp(value - maximum) for value in log_weights]
    total = sum(values)
    return [value / total for value in values]


def effective_sample_size(probabilities: Sequence[float]) -> float:
    if not probabilities or any(value < 0 for value in probabilities):
        raise ValueError("probabilities must be non-empty and non-negative")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"probabilities sum to {total}, not one")
    return 1.0 / sum(value * value for value in probabilities)


def systematic_indices(
    probabilities: Sequence[float],
    rng: random.Random,
) -> list[int]:
    count = len(probabilities)
    if count == 0:
        raise ValueError("cannot resample an empty population")
    cumulative: list[float] = []
    running = 0.0
    for probability in probabilities:
        running += probability
        cumulative.append(running)
    if not math.isclose(running, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("resampling probabilities must sum to one")
    cumulative[-1] = 1.0
    offset = rng.random() / count
    indices: list[int] = []
    ancestor = 0
    for sample in range(count):
        position = offset + sample / count
        while position > cumulative[ancestor]:
            ancestor += 1
        indices.append(ancestor)
    return indices


def load_initial_captured_checkpoint(capture_path: Path) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for raw in capture_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw).get("payload")
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        if messages[-1].get("role") != "user":
            continue
        if any(message.get("role") == "tool" for message in messages):
            continue
        matches.append(copy.deepcopy(payload))
    if len(matches) != 1:
        raise live.ControllerError(
            f"expected one initial provider request, found {len(matches)}"
        )
    return matches[0]


def initial_manifest(
    source: dict[str, Any],
    messages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    manifest = copy.deepcopy(source)
    manifest["tool_events"] = []
    manifest["expected_state"] = {}
    manifest["sealed_from"] = None
    manifest["transcript"]["completed_tool_calls"] = 0
    manifest["transcript"]["events_sha256"] = backend.canonical_json_sha256([])
    manifest["model_prefix"] = {
        "checkpoint_id": "initial-provider-request",
        "sha256": backend.canonical_json_sha256(list(messages)),
        "representation": "openai_messages_v1",
        "kv_cache_handle": None,
    }
    manifest["lineage"] = {
        "particle_id": uuid.uuid4().hex,
        "parent_particle_id": None,
        "generation": 0,
    }
    return manifest


def build_variant_prompt(
    tokenizer: Any,
    *,
    task: str,
    transcript: str,
    labels: Sequence[str],
    variant_index: int,
) -> str:
    if not 0 <= variant_index < len(VERIFIER_VARIANTS):
        raise ValueError(f"unsupported verifier variant: {variant_index}")
    if variant_index == 0:
        return semantic.build_verifier_prompt(
            tokenizer,
            task=task,
            transcript=transcript,
            labels=labels,
        )
    _, scale = semantic.score_scale(labels)
    content = semantic.VERIFIER_TEMPLATE.format(
        task=task,
        transcript=transcript,
        scale=scale,
    )
    content = content.replace(
        "Return exactly one score label and no other text.",
        VERIFIER_VARIANTS[variant_index]
        + "\n\nReturn exactly one score label and no other text.",
    )
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            **kwargs,
        )


class HTTPGenerateEngine:
    """Small native SGLang /generate client with the Engine.generate shape."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        payload = {
            "text": list(prompts),
            "sampling_params": sampling_params,
            **kwargs,
        }
        request = urllib.request.Request(
            self.base_url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"verifier HTTP {error.code}: {detail}"
            ) from error
        return data

    def shutdown(self) -> None:
        return None


class OnlineSemanticVerifier:
    """Batch and deduplicate leakage-safe semantic score requests."""

    def __init__(
        self,
        *,
        scorer_model: str,
        tokenizer: Any,
        engine: Any,
        score_labels: Sequence[str],
        score_token_ids: Sequence[int],
        batch_size: int,
        transcript_max_chars: int,
        tool_output_max_chars: int,
    ) -> None:
        self.scorer_model = scorer_model
        self.tokenizer = tokenizer
        self.engine = engine
        self.score_labels = list(score_labels)
        self.score_token_ids = list(score_token_ids)
        self.score_values, _ = semantic.score_scale(self.score_labels)
        self.batch_size = batch_size
        self.transcript_max_chars = transcript_max_chars
        self.tool_output_max_chars = tool_output_max_chars
        self.cost: dict[str, float | int] = {
            "physical_verifier_calls": 0,
            "logical_verifier_calls": 0,
            "deduplicated_verifier_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "inference_wall_time_s": 0.0,
        }

    @classmethod
    def from_model(
        cls,
        *,
        scorer_model: str,
        base_url: str | None,
        timeout_s: float,
        base_gpu_id: int,
        tp_size: int,
        mem_fraction_static: float,
        max_running_requests: int,
        max_mamba_cache_size: int,
        seed: int,
        batch_size: int,
        transcript_max_chars: int,
        tool_output_max_chars: int,
    ) -> tuple["OnlineSemanticVerifier", float]:
        from transformers import AutoTokenizer

        labels = list(semantic.DEFAULT_SCORE_LABELS)
        tokenizer = AutoTokenizer.from_pretrained(
            scorer_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        token_ids = semantic.resolve_score_token_ids(tokenizer, labels)
        if base_url:
            engine = HTTPGenerateEngine(
                base_url,
                timeout_s,
            )
            startup = 0.0
        else:
            import sglang as sgl

            kwargs: dict[str, Any] = {
                "model_path": scorer_model,
                "trust_remote_code": True,
                "attention_backend": "triton",
                "mem_fraction_static": mem_fraction_static,
                "base_gpu_id": base_gpu_id,
                "random_seed": seed,
                "max_running_requests": max_running_requests,
                "max_mamba_cache_size": max_mamba_cache_size,
            }
            if tp_size > 1:
                kwargs.update(
                    tp_size=tp_size,
                    disable_custom_all_reduce=True,
                    enforce_disable_flashinfer_allreduce_fusion=True,
                )
            started = time.perf_counter()
            engine = sgl.Engine(**kwargs)
            startup = time.perf_counter() - started
        return (
            cls(
                scorer_model=scorer_model,
                tokenizer=tokenizer,
                engine=engine,
                score_labels=labels,
                score_token_ids=token_ids,
                batch_size=batch_size,
                transcript_max_chars=transcript_max_chars,
                tool_output_max_chars=tool_output_max_chars,
            ),
            startup,
        )

    def snapshot_cost(self) -> dict[str, float | int]:
        return copy.deepcopy(self.cost)

    def score(
        self,
        particles: Sequence[live.Particle],
        *,
        calls_per_checkpoint: int,
        experiment_id: str,
        checkpoint_index: int,
    ) -> dict[str, Any]:
        if not 1 <= calls_per_checkpoint <= len(VERIFIER_VARIANTS):
            raise ValueError(
                f"calls_per_checkpoint must be in [1, {len(VERIFIER_VARIANTS)}]"
            )
        unique: dict[str, live.Particle] = {}
        for particle in sorted(particles, key=lambda item: item.slot):
            unique.setdefault(particle.model_prefix_sha256, particle)

        jobs: list[dict[str, Any]] = []
        compactions: dict[str, dict[str, int]] = {}
        for prefix, particle in unique.items():
            checkpoint = {
                "checkpoint_id": f"{experiment_id}:{checkpoint_index}:{prefix}",
                "semantic_input": {"messages": particle.messages},
            }
            task, transcript, compaction = semantic.checkpoint_transcript(
                checkpoint,
                transcript_max_chars=self.transcript_max_chars,
                tool_output_max_chars=self.tool_output_max_chars,
            )
            compactions[prefix] = compaction
            for call_index in range(calls_per_checkpoint):
                jobs.append(
                    {
                        "prefix": prefix,
                        "call_index": call_index,
                        "call_id": semantic.stable_id(
                            experiment_id,
                            checkpoint_index,
                            prefix,
                            self.scorer_model,
                            semantic.CRITERION_VERSION,
                            call_index,
                        ),
                        "prompt": build_variant_prompt(
                            self.tokenizer,
                            task=task,
                            transcript=transcript,
                            labels=self.score_labels,
                            variant_index=call_index,
                        ),
                    }
                )

        records: list[dict[str, Any]] = []
        inference_wall = 0.0
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start : start + self.batch_size]
            began = time.perf_counter()
            outputs = self.engine.generate(
                [job["prompt"] for job in batch],
                {"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=self.score_token_ids,
            )
            batch_wall = time.perf_counter() - began
            inference_wall += batch_wall
            if not isinstance(outputs, list):
                outputs = [outputs]
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"verifier returned {len(outputs)} outputs for {len(batch)} jobs"
                )
            for job, output in zip(batch, outputs):
                result = semantic.expected_score_from_output(
                    output,
                    self.score_token_ids,
                    self.score_values,
                )
                records.append(
                    {
                        "verifier_call_id": job["call_id"],
                        "prefix_sha256": job["prefix"],
                        "call_index": job["call_index"],
                        "variant": VERIFIER_VARIANTS[job["call_index"]],
                        "batch_wall_time_s": batch_wall,
                        **result,
                    }
                )

        scores_by_prefix: dict[str, list[float]] = {
            prefix: [] for prefix in unique
        }
        for record in records:
            scores_by_prefix[record["prefix_sha256"]].append(
                float(record["score"])
            )
        aggregated = {
            particle.particle_id: statistics.fmean(
                scores_by_prefix[particle.model_prefix_sha256]
            )
            for particle in particles
        }
        logical_calls = len(particles) * calls_per_checkpoint
        physical_calls = len(records)
        additions: dict[str, float | int] = {
            "physical_verifier_calls": physical_calls,
            "logical_verifier_calls": logical_calls,
            "deduplicated_verifier_calls": logical_calls - physical_calls,
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in records),
            "completion_tokens": sum(
                int(row["completion_tokens"]) for row in records
            ),
            "inference_wall_time_s": inference_wall,
        }
        for key, value in additions.items():
            self.cost[key] += value
        masses = [float(row["score_token_mass"]) for row in records]
        return {
            "checkpoint_index": checkpoint_index,
            "scorer_model": self.scorer_model,
            "criterion": semantic.CRITERION_VERSION,
            "calls_per_checkpoint": calls_per_checkpoint,
            "unique_prefixes": len(unique),
            "physical_calls": physical_calls,
            "logical_calls": logical_calls,
            "deduplicated_calls": logical_calls - physical_calls,
            "inference_wall_time_s": inference_wall,
            "prompt_tokens": additions["prompt_tokens"],
            "completion_tokens": additions["completion_tokens"],
            "mean_score_token_mass": statistics.fmean(masses) if masses else None,
            "compactions": compactions,
            "scores_by_particle": aggregated,
            "calls": records,
            "reward_isolated": True,
        }

    def shutdown(self) -> None:
        self.engine.shutdown()


def apply_semantic_scores(
    particles: Sequence[live.Particle],
    scoring: dict[str, Any],
    *,
    initialize: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calls = int(scoring["calls_per_checkpoint"])
    for particle in particles:
        current = float(scoring["scores_by_particle"][particle.particle_id])
        previous = current if initialize else particle.semantic_score
        delta = 0.0 if initialize else current - previous
        particle.last_semantic_score = previous
        particle.semantic_score = current
        particle.semantic_log_weight += delta
        particle.last_semantic_checkpoint_tokens = particle.generated_tokens
        particle.semantic_call_count += calls
        rows.append(
            {
                "slot": particle.slot,
                "particle_id": particle.particle_id,
                "generated_tokens": particle.generated_tokens,
                "previous_score": previous,
                "score": current,
                "delta": delta,
                "semantic_log_weight": particle.semantic_log_weight,
            }
        )
    return rows


def population_weights(
    particles: Sequence[live.Particle],
    beta: float,
) -> tuple[list[float], float]:
    probabilities = softmax(
        [beta * particle.semantic_log_weight for particle in particles]
    )
    return probabilities, effective_sample_size(probabilities)


def _copy_particle_progress(
    child: live.Particle,
    ancestor: live.Particle,
) -> None:
    child.rounds = ancestor.rounds
    child.finished = ancestor.finished
    child.generated_tokens = ancestor.generated_tokens
    child.semantic_score = ancestor.semantic_score
    child.last_semantic_score = ancestor.last_semantic_score
    child.semantic_log_weight = 0.0
    child.last_semantic_checkpoint_tokens = (
        ancestor.last_semantic_checkpoint_tokens
    )
    child.semantic_call_count = ancestor.semantic_call_count


def _collect_materialized(
    futures: Sequence[Any],
    controller: live.TerminalParticleController,
) -> list[live.Particle]:
    particles: list[live.Particle] = []
    first_error: BaseException | None = None
    for future in futures:
        try:
            particles.append(future.result())
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        controller.remove_particles(particles)
        raise first_error
    return particles


def materialize_population(
    controller: live.TerminalParticleController,
    manifest: dict[str, Any],
    messages: Sequence[dict[str, Any]],
    *,
    count: int,
    name_prefix: str,
    workers: int,
) -> list[live.Particle]:
    def spawn(slot: int) -> live.Particle:
        return controller.spawn(
            manifest,
            messages,
            slot=slot,
            name=f"{name_prefix}-p{slot}",
        )

    with ThreadPoolExecutor(max_workers=min(workers, count)) as executor:
        futures = [executor.submit(spawn, slot) for slot in range(count)]
        particles = _collect_materialized(futures, controller)
    return sorted(particles, key=lambda particle: particle.slot)


def resample_population(
    controller: live.TerminalParticleController,
    particles: Sequence[live.Particle],
    ancestor_indices: Sequence[int],
    *,
    name_prefix: str,
    workers: int,
) -> list[live.Particle]:
    if len(particles) != len(ancestor_indices):
        raise ValueError("resampling must preserve population size")
    unique_ancestors = {
        particles[index].particle_id: particles[index]
        for index in ancestor_indices
    }
    for ancestor in unique_ancestors.values():
        controller.seal(ancestor)

    def spawn(item: tuple[int, int]) -> live.Particle:
        slot, ancestor_index = item
        ancestor = particles[ancestor_index]
        child = controller.spawn(
            ancestor.manifest,
            ancestor.messages,
            slot=slot,
            name=f"{name_prefix}-p{slot}",
            resampled_from=ancestor.particle_id,
        )
        _copy_particle_progress(child, ancestor)
        if child.model_prefix_sha256 != ancestor.model_prefix_sha256:
            controller.docker.remove(child.container_name)
            raise live.ControllerError("population resampling copied wrong prefix")
        if child.state != ancestor.state:
            controller.docker.remove(child.container_name)
            raise live.ControllerError("population resampling copied wrong state")
        return child

    items = list(enumerate(ancestor_indices))
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = [executor.submit(spawn, item) for item in items]
        children = _collect_materialized(futures, controller)
    controller.remove_particles(particles)
    return sorted(children, key=lambda particle: particle.slot)


FIX_GIT_GRADER = r"""
import hashlib
import json
paths = {
    "/app/personal-site/_includes/about.md": "0273104059c6bf524e767b8847b22946",
    "/app/personal-site/_layouts/default.html": "0f879389f66640f45316e393a71c5f2f",
}
actual = {}
for path, expected in paths.items():
    try:
        with open(path, "rb") as stream:
            value = hashlib.md5(stream.read().strip()).hexdigest()
    except OSError:
        value = None
    actual[path] = {"actual": value, "expected": expected, "pass": value == expected}
reward = float(all(row["pass"] for row in actual.values()))
result = {"reward": reward, "checks": actual}
print(json.dumps(result, sort_keys=True))
"""


def grade_fix_git(
    docker: backend.DockerCLI,
    particles: Sequence[live.Particle],
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()

    def grade(particle: live.Particle) -> dict[str, Any]:
        began = time.perf_counter()
        process = docker.run(
            ["exec", particle.container_name, "python3", "-c", FIX_GIT_GRADER],
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"fix-git grader failed for slot {particle.slot}: {process.stderr}"
            )
        result = json.loads(process.stdout)
        return {
            "slot": particle.slot,
            "particle_id": particle.particle_id,
            "reward": float(result["reward"]),
            "checks": result["checks"],
            "wall_time_s": time.perf_counter() - began,
        }

    with ThreadPoolExecutor(max_workers=min(16, len(particles))) as executor:
        rows = list(executor.map(grade, particles))
    return sorted(rows, key=lambda row: row["slot"]), time.perf_counter() - started


def numeric_delta(
    after: dict[str, float | int],
    before: dict[str, float | int],
) -> dict[str, float | int]:
    return {key: after[key] - before.get(key, 0) for key in after}


def run_configuration(
    docker: backend.DockerCLI,
    verifier: OnlineSemanticVerifier,
    *,
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    experiment_id: str,
    num_particles: int,
    checkpoint_interval: int,
    verifier_calls: int,
    beta: float,
    ess_threshold: float,
    generator_base_url: str,
    generator_temperature: float,
    generator_top_p: float,
    generator_max_tokens: int,
    timeout_s: float,
    max_rounds: int,
    seed_base: int,
    materialize_workers: int,
    name_prefix: str,
    checkpoint_dir: Path,
    keep_containers: bool,
) -> dict[str, Any]:
    controller = live.TerminalParticleController(
        docker,
        request_template=checkpoint,
        base_url=generator_base_url,
        temperature=generator_temperature,
        top_p=generator_top_p,
        max_tokens=generator_max_tokens,
        timeout_s=timeout_s,
    )
    config_started = time.perf_counter()
    verifier_before = verifier.snapshot_cost()
    particles: list[live.Particle] = []
    checkpoint_events: list[dict[str, Any]] = []
    generation_rounds: list[dict[str, Any]] = []
    rng = random.Random(seed_base)
    next_checkpoint_tokens = checkpoint_interval
    checkpoint_index = 0
    suffix = uuid.uuid4().hex[:8]
    try:
        particles = materialize_population(
            controller,
            manifest,
            checkpoint["messages"],
            count=num_particles,
            name_prefix=f"{name_prefix}-{suffix}-initial",
            workers=materialize_workers,
        )
        initial_states = {backend.canonical_json_sha256(p.state) for p in particles}
        initial_prefixes = {p.model_prefix_sha256 for p in particles}
        if len(initial_states) != 1 or len(initial_prefixes) != 1:
            raise live.ControllerError("initial semantic SMC population is not coupled")

        baseline = verifier.score(
            particles,
            calls_per_checkpoint=verifier_calls,
            experiment_id=experiment_id,
            checkpoint_index=checkpoint_index,
        )
        baseline_rows = apply_semantic_scores(
            particles,
            baseline,
            initialize=True,
        )
        checkpoint_events.append(
            {
                "kind": "common_baseline",
                "target_generated_tokens": 0,
                "scoring": baseline,
                "particles": baseline_rows,
                "ess": float(num_particles),
                "resampled": False,
            }
        )
        checkpoint_index += 1

        for round_index in range(max_rounds):
            active = [particle for particle in particles if not particle.finished]
            if not active:
                break
            rows = controller.advance_many(
                active,
                seeds=[
                    seed_base + round_index * 10_000 + particle.slot
                    for particle in active
                ],
            )
            generation_rounds.append(
                {
                    "round": round_index,
                    "active_particles": len(active),
                    "transitions": rows,
                    "generated_tokens": [
                        particle.generated_tokens for particle in particles
                    ],
                }
            )
            ready = all(
                particle.finished
                or particle.generated_tokens >= next_checkpoint_tokens
                for particle in particles
            )
            if not ready or all(particle.finished for particle in particles):
                continue

            scoring = verifier.score(
                particles,
                calls_per_checkpoint=verifier_calls,
                experiment_id=experiment_id,
                checkpoint_index=checkpoint_index,
            )
            score_rows = apply_semantic_scores(
                particles,
                scoring,
                initialize=False,
            )
            probabilities, ess = population_weights(particles, beta)
            should_resample = ess < ess_threshold * num_particles
            event: dict[str, Any] = {
                "kind": "token_interval",
                "target_generated_tokens": next_checkpoint_tokens,
                "actual_generated_tokens": [
                    particle.generated_tokens for particle in particles
                ],
                "checkpoint_alignment": "completed_assistant_turn_at_or_after_target",
                "scoring": scoring,
                "particles": score_rows,
                "probabilities": probabilities,
                "ess": ess,
                "ess_threshold": ess_threshold * num_particles,
                "resampled": should_resample,
            }
            if should_resample:
                ancestors = systematic_indices(probabilities, rng)
                event["ancestor_slots"] = ancestors
                event["unique_ancestors"] = len(set(ancestors))
                particles = resample_population(
                    controller,
                    particles,
                    ancestors,
                    name_prefix=(
                        f"{name_prefix}-{suffix}-checkpoint-{checkpoint_index}"
                    ),
                    workers=materialize_workers,
                )
            checkpoint_events.append(event)
            checkpoint_index += 1
            next_checkpoint_tokens += checkpoint_interval
            completed_tokens = [
                particle.generated_tokens
                for particle in particles
                if not particle.finished
            ]
            if completed_tokens:
                while min(completed_tokens) >= next_checkpoint_tokens:
                    next_checkpoint_tokens += checkpoint_interval

        terminal = verifier.score(
            particles,
            calls_per_checkpoint=verifier_calls,
            experiment_id=experiment_id,
            checkpoint_index=checkpoint_index,
        )
        terminal_rows = apply_semantic_scores(
            particles,
            terminal,
            initialize=False,
        )
        probabilities, terminal_ess = population_weights(particles, beta)
        checkpoint_events.append(
            {
                "kind": "terminal",
                "target_generated_tokens": None,
                "scoring": terminal,
                "particles": terminal_rows,
                "probabilities": probabilities,
                "ess": terminal_ess,
                "resampled": False,
            }
        )

        grades, grader_wall = grade_fix_git(docker, particles)
        grades_by_id = {row["particle_id"]: row for row in grades}
        selected = max(
            particles,
            key=lambda particle: (particle.semantic_score, -particle.slot),
        )
        selected_grade = grades_by_id[selected.particle_id]
        checkpoint_path = checkpoint_dir / f"{experiment_id}-selected.json"
        backend.atomic_write_json(
            checkpoint_path,
            live.particle_checkpoint(selected),
        )
        verifier_after = verifier.snapshot_cost()
        wall_time = time.perf_counter() - config_started
        success_count = sum(row["reward"] >= 1.0 for row in grades)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": "pass",
            "configuration": {
                "num_particles": num_particles,
                "checkpoint_interval_tokens": checkpoint_interval,
                "verifier_calls_per_checkpoint": verifier_calls,
                "verifier_call_variants": list(
                    VERIFIER_VARIANTS[:verifier_calls]
                ),
                "semantic_beta": beta,
                "ess_threshold_fraction": ess_threshold,
                "generator_max_tokens_per_completed_turn": generator_max_tokens,
                "max_rounds": max_rounds,
            },
            "checkpoint_semantics": {
                "requested_interval_tokens": checkpoint_interval,
                "exact_partial_assistant_checkpoint": False,
                "implemented_alignment": (
                    "first completed assistant/tool turn at or after the target"
                ),
                "reason": (
                    "the OpenAI chat controller has serialized messages but no "
                    "persistent mid-assistant KV continuation handle"
                ),
            },
            "initial_coupling": {
                "unique_environment_states": len(initial_states),
                "unique_model_prefixes": len(initial_prefixes),
            },
            "generation_rounds": generation_rounds,
            "semantic_checkpoints": checkpoint_events,
            "resampling_count": sum(
                bool(event["resampled"]) for event in checkpoint_events
            ),
            "final_particles": [live.particle_summary(p) for p in particles],
            "grader": {
                "name": "terminal-bench-fix-git-file-hashes-v1",
                "reward_isolated_from_semantic_verifier": True,
                "wall_time_s": grader_wall,
                "particles": grades,
                "success_count": success_count,
                "success_rate": success_count / num_particles,
                "any_success": success_count > 0,
            },
            "selection": {
                "policy": "best_terminal_semantic_score",
                "slot": selected.slot,
                "particle_id": selected.particle_id,
                "semantic_score": selected.semantic_score,
                "reward": selected_grade["reward"],
                "selected_checkpoint": str(checkpoint_path.resolve()),
            },
            "generator_tool_replay_cost": copy.deepcopy(controller.cost),
            "semantic_verifier_cost": numeric_delta(
                verifier_after,
                verifier_before,
            ),
            "wall_time_s": wall_time,
            "population_finished": all(particle.finished for particle in particles),
            "round_limit_reached": any(
                not particle.finished for particle in particles
            ),
        }
    finally:
        if particles and not keep_containers:
            controller.remove_particles(particles)


def configurations(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.axis == "particles":
        values = parse_csv(args.values, int)
    elif args.axis in {"interval", "verifier_calls"}:
        values = parse_csv(args.values, int)
    else:
        values = parse_csv(args.values, float)
    points: list[dict[str, Any]] = []
    for value in values:
        point = {
            "num_particles": args.num_particles,
            "checkpoint_interval": args.checkpoint_interval,
            "verifier_calls": args.verifier_calls,
            "beta": args.beta,
            "ess_threshold": args.ess_threshold,
        }
        mapping = {
            "particles": "num_particles",
            "interval": "checkpoint_interval",
            "verifier_calls": "verifier_calls",
            "beta": "beta",
            "ess": "ess_threshold",
        }
        point[mapping[args.axis]] = value
        point["axis_value"] = value
        if int(point["num_particles"]) < 2:
            raise ValueError("semantic SMC requires at least two particles")
        if int(point["checkpoint_interval"]) <= 0:
            raise ValueError("checkpoint interval must be positive")
        if not 1 <= int(point["verifier_calls"]) <= len(VERIFIER_VARIANTS):
            raise ValueError("verifier calls must be in [1, 8]")
        if not 0 < float(point["ess_threshold"]) <= 1:
            raise ValueError("ESS threshold must be in (0, 1]")
        points.append(point)
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--after-tool-call-id")
    parser.add_argument("--generator-base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--generator-temperature", type=float, default=0.7)
    parser.add_argument("--generator-top-p", type=float, default=0.95)
    parser.add_argument("--generator-max-tokens", type=int, default=256)
    parser.add_argument("--generator-gpus", type=int, default=1)
    parser.add_argument("--verifier-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--verifier-base-url")
    parser.add_argument("--verifier-base-gpu-id", type=int, default=1)
    parser.add_argument("--verifier-tp", type=int, default=1)
    parser.add_argument("--verifier-mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--verifier-max-running-requests", type=int, default=64)
    parser.add_argument("--verifier-max-mamba-cache-size", type=int, default=64)
    parser.add_argument("--verifier-batch-size", type=int, default=64)
    parser.add_argument("--transcript-max-chars", type=int, default=30000)
    parser.add_argument("--tool-output-max-chars", type=int, default=4000)
    parser.add_argument(
        "--axis",
        choices=("particles", "interval", "verifier_calls", "beta", "ess"),
        default="particles",
    )
    parser.add_argument("--values", default="4,8,16,32,64")
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--verifier-calls", type=int, default=1)
    parser.add_argument("--beta", type=float, default=12.0)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--materialize-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--name-prefix", default="semantic-smc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-containers", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.after_tool_call_id:
        checkpoint = live.load_captured_checkpoint(
            args.capture,
            after_tool_call_id=args.after_tool_call_id,
        )
        live.validate_checkpoint_binding(
            source_manifest,
            checkpoint,
            after_tool_call_id=args.after_tool_call_id,
        )
        manifest = source_manifest
        source_kind = "captured_post_tool_checkpoint"
    else:
        checkpoint = load_initial_captured_checkpoint(args.capture)
        manifest = initial_manifest(source_manifest, checkpoint["messages"])
        source_kind = "captured_initial_provider_request"

    points = configurations(args)
    run_started = time.perf_counter()
    verifier: OnlineSemanticVerifier | None = None
    engine_startup = 0.0
    results: list[dict[str, Any]] = []
    run_error: str | None = None
    checkpoint_dir = args.output.parent / f"{args.output.stem}.particles"
    try:
        verifier, engine_startup = OnlineSemanticVerifier.from_model(
            scorer_model=args.verifier_model,
            base_url=args.verifier_base_url,
            timeout_s=args.timeout,
            base_gpu_id=args.verifier_base_gpu_id,
            tp_size=args.verifier_tp,
            mem_fraction_static=args.verifier_mem_fraction_static,
            max_running_requests=args.verifier_max_running_requests,
            max_mamba_cache_size=args.verifier_max_mamba_cache_size,
            seed=args.seed_base,
            batch_size=args.verifier_batch_size,
            transcript_max_chars=args.transcript_max_chars,
            tool_output_max_chars=args.tool_output_max_chars,
        )
        for point in points:
            for repetition in range(args.repetitions):
                axis_value = point["axis_value"]
                experiment_id = semantic.stable_id(
                    "terminal-semantic-smc-v1",
                    backend.canonical_json_sha256(checkpoint["messages"]),
                    args.axis,
                    semantic.canonical_json(point),
                    repetition,
                    args.seed_base,
                )
                print(
                    f"start axis={args.axis} value={axis_value} "
                    f"repeat={repetition} id={experiment_id}",
                    flush=True,
                )
                result = run_configuration(
                    backend.DockerCLI(),
                    verifier,
                    manifest=manifest,
                    checkpoint=checkpoint,
                    experiment_id=experiment_id,
                    num_particles=int(point["num_particles"]),
                    checkpoint_interval=int(point["checkpoint_interval"]),
                    verifier_calls=int(point["verifier_calls"]),
                    beta=float(point["beta"]),
                    ess_threshold=float(point["ess_threshold"]),
                    generator_base_url=args.generator_base_url,
                    generator_temperature=args.generator_temperature,
                    generator_top_p=args.generator_top_p,
                    generator_max_tokens=args.generator_max_tokens,
                    timeout_s=args.timeout,
                    max_rounds=args.max_rounds,
                    seed_base=args.seed_base + repetition * 100_000,
                    materialize_workers=args.materialize_workers,
                    name_prefix=args.name_prefix,
                    checkpoint_dir=checkpoint_dir,
                    keep_containers=args.keep_containers,
                )
                results.append(result)
                backend.atomic_write_json(
                    args.output,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "running",
                        "results": results,
                    },
                )
                print(
                    json.dumps(
                        {
                            "id": experiment_id,
                            "success_rate": result["grader"]["success_rate"],
                            "selected_reward": result["selection"]["reward"],
                            "resampling_count": result["resampling_count"],
                            "wall_time_s": result["wall_time_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    except BaseException as error:
        run_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        if verifier is not None:
            verifier.shutdown()
        total_wall = time.perf_counter() - run_started
        report = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "terminal-semantic-smc-v1",
            "created_at": live.utc_now(),
            "status": "pass" if run_error is None else "error",
            "error": run_error,
            "source": {
                "kind": source_kind,
                "manifest": str(args.manifest.resolve()),
                "capture": str(args.capture.resolve()),
                "after_tool_call_id": args.after_tool_call_id,
                "task_id": manifest["task_id"],
                "generator_model": checkpoint["model"],
                "model_prefix_sha256": backend.canonical_json_sha256(
                    checkpoint["messages"]
                ),
            },
            "sweep": {
                "axis": args.axis,
                "values": [point["axis_value"] for point in points],
                "repetitions": args.repetitions,
                "one_factor_at_a_time": True,
            },
            "verifier": {
                "model": args.verifier_model,
                "criterion": semantic.CRITERION_VERSION,
                "base_url": args.verifier_base_url,
                "external_server": args.verifier_base_url is not None,
                "engine_startup_wall_time_s": engine_startup,
                "tensor_parallel_size": args.verifier_tp,
                "cost": verifier.cost if verifier is not None else None,
            },
            "allocation_accounting": {
                "runner_wall_time_s": total_wall,
                "generator_accelerators": args.generator_gpus,
                "verifier_accelerators": args.verifier_tp,
                "runner_allocated_accelerator_seconds": total_wall
                * (args.generator_gpus + args.verifier_tp),
                "includes_verifier_engine_startup_and_shutdown": (
                    args.verifier_base_url is None
                ),
                "includes_external_generator_engine_startup": False,
                "includes_external_verifier_engine_startup": False,
                "external_generator_server": True,
            },
            "results": results,
        }
        backend.atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.resolve()),
                "points": len(results),
                "wall_time_s": report["allocation_accounting"]["runner_wall_time_s"],
                "allocated_accelerator_seconds": report[
                    "allocation_accounting"
                ]["runner_allocated_accelerator_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

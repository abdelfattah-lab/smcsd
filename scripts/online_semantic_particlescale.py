"""Run true branched semantic particle allocation on saved equal prefixes.

The initial particles are exact generator-token prefixes from a common saved
Best-of-N pool.  After each semantic checkpoint, this runner either performs
systematic SMC resampling from incremental semantic weights or deterministic
top-half/fork-two allocation.  Selected copies are continued independently by
the generator, so this is an online branching experiment rather than the
fixed-trajectory top-m replay in ``offline_policy_sim.py``.

No generator or target-model likelihood is used.  Correctness is assigned only
after generation with the benchmark's symbolic answer judge.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

try:
    from scripts import offline_dual_semantic_audit as dual_audit
    from scripts import offline_pointwise_error_audit as error_audit
    from scripts.accuracy_test_olympiadbench import (
        cluster_answers,
        extract_boxed_answers,
    )
    from scripts.olympiadbench_judge import OlympiadBenchJudge
except ModuleNotFoundError:
    import offline_dual_semantic_audit as dual_audit
    import offline_pointwise_error_audit as error_audit
    from accuracy_test_olympiadbench import cluster_answers, extract_boxed_answers
    from olympiadbench_judge import OlympiadBenchJudge


METHODS = ("smc", "deterministic_fork")


def parse_csv(value: str) -> list[str]:
    values = list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_weights(value: str, rubrics: Sequence[str]) -> dict[str, float]:
    pieces = parse_csv(value)
    if all("=" not in piece for piece in pieces):
        if len(pieces) != len(rubrics):
            raise ValueError("Positional rubric weights must match --rubrics.")
        weights = {rubric: float(weight) for rubric, weight in zip(rubrics, pieces)}
    else:
        weights = {}
        for piece in pieces:
            if "=" not in piece:
                raise ValueError("Use either all named or all positional rubric weights.")
            rubric, weight = piece.split("=", 1)
            weights[rubric] = float(weight)
        if set(weights) != set(rubrics):
            raise ValueError("Named rubric weights must exactly match --rubrics.")
    total = sum(abs(weight) for weight in weights.values())
    if total <= 0:
        raise ValueError("At least one rubric weight must be nonzero.")
    return {rubric: weight / total for rubric, weight in weights.items()}


def finish_type(output: dict) -> str | None:
    reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(reason, dict):
        return reason.get("type")
    return str(reason) if reason is not None else None


def softmax(log_weights: Sequence[float]) -> list[float]:
    maximum = max(log_weights)
    values = [math.exp(value - maximum) for value in log_weights]
    total = sum(values)
    return [value / total for value in values]


def effective_sample_size(probabilities: Sequence[float]) -> float:
    return 1.0 / sum(value * value for value in probabilities)


def systematic_indices(
    probabilities: Sequence[float], rng: random.Random
) -> list[int]:
    """Draw len(probabilities) systematic-resampling ancestor indices."""
    count = len(probabilities)
    cumulative = []
    running = 0.0
    for probability in probabilities:
        running += probability
        cumulative.append(running)
    cumulative[-1] = 1.0
    offset = rng.random() / count
    result = []
    index = 0
    for sample in range(count):
        position = offset + sample / count
        while position > cumulative[index]:
            index += 1
        result.append(index)
    return result


def deterministic_fork_indices(scores: Sequence[float]) -> list[int]:
    """Keep the top half and fork each survivor twice, with stable ties."""
    count = len(scores)
    if count < 2 or count % 2:
        raise ValueError("deterministic_fork requires an even particle count >= 2.")
    survivors = sorted(range(count), key=lambda index: (-scores[index], index))[
        : count // 2
    ]
    return [index for index in survivors for _ in range(2)]


def clone_population(
    particles: Sequence[dict],
    ancestor_indices: Sequence[int],
    *,
    next_particle_id: int,
) -> tuple[list[dict], int]:
    clones = []
    for slot, ancestor_index in enumerate(ancestor_indices):
        ancestor = particles[ancestor_index]
        clone = copy.deepcopy(ancestor)
        clone["parent_particle_id"] = int(ancestor["particle_id"])
        clone["particle_id"] = next_particle_id
        next_particle_id += 1
        clone["slot"] = slot
        clone["log_weight"] = 0.0
        clone["ancestry"] = list(ancestor["ancestry"]) + [int(ancestor["particle_id"])]
        clones.append(clone)
    return clones, next_particle_id


def make_particle_groups(
    trajectories: Sequence[dict], initial_prefix_tokens: int
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    next_particle_id = 0
    for row in trajectories:
        output_ids = list(row["generator_output_ids"])
        prefix_ids = output_ids[:initial_prefix_tokens]
        finished = len(output_ids) <= initial_prefix_tokens
        grouped[str(row["problem_id"])].append(
            {
                "particle_id": next_particle_id,
                "parent_particle_id": None,
                "root_sample_id": int(row["sample_id"]),
                "slot": int(row["sample_id"]),
                "problem_id": str(row["problem_id"]),
                "prompt": row["prompt"],
                "problem": row["problem"],
                "gold_answer": row["gold_answer"],
                "grading_precision": float(row["grading_precision"]),
                "output_ids": prefix_ids,
                "finished": finished,
                "finish_reason": row.get("finish_reason") if finished else None,
                "rubric_scores": {},
                "semantic_score": 0.0,
                "last_semantic_score": 0.0,
                "log_weight": 0.0,
                "ancestry": [],
            }
        )
        next_particle_id += 1
    for problem_id, particles in grouped.items():
        particles.sort(key=lambda row: int(row["root_sample_id"]))
        expected = list(range(len(particles)))
        observed = [int(row["root_sample_id"]) for row in particles]
        if observed != expected:
            raise ValueError(f"Expected sample IDs {expected} for {problem_id}; got {observed}.")
    return dict(grouped)


def score_population(
    verifier_engine,
    verifier_tokenizer,
    generator_tokenizer,
    groups: dict[str, list[dict]],
    *,
    rubrics: Sequence[str],
    rubric_weights: dict[str, float],
    total_budget: int,
    score_token_ids: Sequence[int],
    score_values: Sequence[float],
    batch_size: int,
) -> dict:
    unique_jobs = {}
    particle_keys = {}
    for particles in groups.values():
        for particle in particles:
            decoded = generator_tokenizer.decode(
                particle["output_ids"], skip_special_tokens=True
            )
            prefix_key = (particle["problem_id"], tuple(particle["output_ids"]))
            for rubric in rubrics:
                key = (*prefix_key, rubric)
                particle_keys[(int(particle["particle_id"]), rubric)] = key
                if key in unique_jobs:
                    continue
                if rubric == "validity":
                    prompt = error_audit.build_error_audit_prompt(
                        verifier_tokenizer,
                        problem=particle["problem"],
                        prefix=decoded,
                        labels=dual_audit.LABELS,
                    )
                elif rubric == "progress":
                    token_position = min(len(particle["output_ids"]), total_budget - 1)
                    prompt = dual_audit.build_progress_prompt(
                        verifier_tokenizer,
                        problem=particle["problem"],
                        prefix=decoded,
                        token_position=token_position,
                        total_budget=total_budget,
                        labels=dual_audit.LABELS,
                    )
                else:
                    raise ValueError(f"Unsupported rubric {rubric!r}.")
                unique_jobs[key] = {"prompt": prompt, "rubric": rubric}

    keys = list(unique_jobs)
    results = {}
    inference_started = time.perf_counter()
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        outputs = verifier_engine.generate(
            [unique_jobs[key]["prompt"] for key in batch_keys],
            {"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            top_logprobs_num=0,
            token_ids_logprob=list(score_token_ids),
        )
        if not isinstance(outputs, list):
            outputs = [outputs]
        for key, output in zip(batch_keys, outputs):
            results[key] = error_audit.expected_score_from_output(
                output, score_token_ids, score_values
            )
    inference_time = time.perf_counter() - inference_started

    for particles in groups.values():
        for particle in particles:
            previous = float(particle["semantic_score"])
            scores = {
                rubric: float(results[particle_keys[(int(particle["particle_id"]), rubric)]]["score"])
                for rubric in rubrics
            }
            current = sum(rubric_weights[rubric] * scores[rubric] for rubric in rubrics)
            particle["rubric_scores"] = scores
            particle["last_semantic_score"] = previous
            particle["semantic_score"] = current
            particle["log_weight"] += current - previous

    return {
        "inference_wall_time_s": inference_time,
        "calls": len(keys),
        "prompt_tokens": sum(int(result["prompt_tokens"]) for result in results.values()),
        "completion_tokens": sum(
            int(result["completion_tokens"]) for result in results.values()
        ),
        "mean_score_token_mass": statistics.fmean(
            float(result["score_token_mass"]) for result in results.values()
        ),
        "selected_logprob_coverage": statistics.fmean(
            float(result["logprob_source"] == "selected") for result in results.values()
        ),
    }


def allocate_population(
    groups: dict[str, list[dict]],
    *,
    method: str,
    beta: float,
    ess_threshold: float,
    rng: random.Random,
    next_particle_id: int,
) -> tuple[dict[str, list[dict]], list[dict], int]:
    next_groups = {}
    events = []
    for problem_id, particles in groups.items():
        semantic_log_weights = [beta * float(row["log_weight"]) for row in particles]
        probabilities = softmax(semantic_log_weights)
        ess = effective_sample_size(probabilities)
        if method == "smc":
            resampled = ess < ess_threshold * len(particles)
            ancestor_indices = (
                systematic_indices(probabilities, rng)
                if resampled
                else list(range(len(particles)))
            )
        elif method == "deterministic_fork":
            resampled = True
            ancestor_indices = deterministic_fork_indices(
                [float(row["semantic_score"]) for row in particles]
            )
        else:
            raise ValueError(f"Unknown method {method!r}.")

        if resampled:
            next_particles, next_particle_id = clone_population(
                particles, ancestor_indices, next_particle_id=next_particle_id
            )
        else:
            next_particles = particles
        next_groups[problem_id] = next_particles
        events.append(
            {
                "problem_id": problem_id,
                "ess": ess,
                "resampled": resampled,
                "ancestor_slots": ancestor_indices,
                "unique_selected_particles": len(set(ancestor_indices)),
                "semantic_scores": [float(row["semantic_score"]) for row in particles],
                "probabilities": probabilities,
                "active_particles": sum(not row["finished"] for row in particles),
            }
        )
    return next_groups, events, next_particle_id


def continue_population(
    generator_engine,
    generator_tokenizer,
    groups: dict[str, list[dict]],
    *,
    additional_tokens: int,
    temperature: float,
) -> dict:
    active = [
        particle
        for particles in groups.values()
        for particle in particles
        if not particle["finished"]
    ]
    if not active:
        return {"inference_wall_time_s": 0.0, "output_tokens": 0, "requests": 0}
    inputs = [
        generator_tokenizer.encode(particle["prompt"], add_special_tokens=False)
        + list(particle["output_ids"])
        for particle in active
    ]
    inference_started = time.perf_counter()
    outputs = generator_engine.generate(
        input_ids=inputs,
        sampling_params={
            "max_new_tokens": additional_tokens,
            "temperature": temperature,
        },
    )
    inference_time = time.perf_counter() - inference_started
    output_tokens = 0
    for particle, output in zip(active, outputs):
        new_ids = list(output["output_ids"])
        particle["output_ids"].extend(new_ids)
        output_tokens += len(new_ids)
        kind = finish_type(output)
        particle["finish_reason"] = output.get("meta_info", {}).get("finish_reason")
        particle["finished"] = kind != "length"
    return {
        "inference_wall_time_s": inference_time,
        "output_tokens": output_tokens,
        "requests": len(active),
    }


def finalize(
    groups: dict[str, list[dict]], generator_tokenizer
) -> tuple[list[dict], list[dict], dict]:
    judge = OlympiadBenchJudge()
    particle_rows = []
    problem_rows = []
    for problem_id, particles in groups.items():
        texts = [
            generator_tokenizer.decode(row["output_ids"], skip_special_tokens=True)
            for row in particles
        ]
        raw_answers = [extract_boxed_answers(text) for text in texts]
        canonical, correctness = cluster_answers(
            raw_answers,
            particles[0]["gold_answer"],
            float(particles[0]["grading_precision"]),
            judge,
        )
        for particle, text, raw_answer, answer, correct in zip(
            particles, texts, raw_answers, canonical, correctness
        ):
            particle_rows.append(
                {
                    **particle,
                    "full_text": text,
                    "raw_extracted_answer": raw_answer,
                    "extracted_answer": answer,
                    "correct": bool(correct),
                    "completion_tokens": len(particle["output_ids"]),
                }
            )
        winner_index = min(
            range(len(particles)),
            key=lambda index: (-float(particles[index]["semantic_score"]), index),
        )
        votes = [answer for answer in canonical if answer is not None]
        majority = Counter(votes).most_common(1)[0][0] if votes else None
        problem_rows.append(
            {
                "problem_id": problem_id,
                "selected_slot": winner_index,
                "selected_correct": bool(correctness[winner_index]),
                "selected_answer": canonical[winner_index],
                "selected_semantic_score": float(particles[winner_index]["semantic_score"]),
                "particle_majority_correct": majority == particles[0]["gold_answer"],
                "oracle_correct": any(correctness),
                "unique_root_samples": len(
                    {int(row["root_sample_id"]) for row in particles}
                ),
                "unique_final_token_sequences": len(
                    {tuple(row["output_ids"]) for row in particles}
                ),
            }
        )
    summary = {
        "accuracy": statistics.fmean(float(row["selected_correct"]) for row in problem_rows),
        "particle_majority_accuracy": statistics.fmean(
            float(row["particle_majority_correct"]) for row in problem_rows
        ),
        "oracle_pass_at_n": statistics.fmean(
            float(row["oracle_correct"]) for row in problem_rows
        ),
        "mean_unique_root_samples": statistics.fmean(
            row["unique_root_samples"] for row in problem_rows
        ),
        "mean_unique_final_token_sequences": statistics.fmean(
            row["unique_final_token_sequences"] for row in problem_rows
        ),
    }
    return particle_rows, problem_rows, summary


def write_json(path: str | Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--scorer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--rubrics", default="validity")
    parser.add_argument("--rubric-weights", default="1")
    parser.add_argument("--initial-prefix-tokens", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=2048)
    parser.add_argument("--total-budget", type=int, default=16384)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--generator-base-gpu-id", type=int, default=0)
    parser.add_argument("--generator-tp", type=int, default=1)
    parser.add_argument("--generator-mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--verifier-base-gpu-id", type=int, default=1)
    parser.add_argument("--verifier-dp", type=int, default=1)
    parser.add_argument("--verifier-tp", type=int, default=1)
    parser.add_argument("--verifier-mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--verifier-batch-size", type=int, default=512)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument(
        "--generator-cost-summary",
        default="work_dirs/semantic_olympiadbench_pilot_v1/pilot_16k_summary.json",
    )
    parser.add_argument("--save-particles", required=True)
    parser.add_argument("--save-events", required=True)
    parser.add_argument("--save-problems", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = build_parser().parse_args(argv)
    trajectories = error_audit.load_jsonl(args.trajectories)
    error_audit.validate_trajectory_rows(trajectories)
    problem_order = list(dict.fromkeys(str(row["problem_id"]) for row in trajectories))
    if args.max_problems is not None:
        allowed = set(problem_order[: args.max_problems])
        trajectories = [row for row in trajectories if str(row["problem_id"]) in allowed]
    generator = args.generator or trajectories[0]["generator_model"]
    rubrics = parse_csv(args.rubrics)
    if any(rubric not in dual_audit.VALID_RUBRICS for rubric in rubrics):
        raise ValueError(f"Rubrics must be drawn from {dual_audit.VALID_RUBRICS}.")
    rubric_weights = parse_weights(args.rubric_weights, rubrics)
    if args.initial_prefix_tokens <= 0 or args.initial_prefix_tokens >= args.total_budget:
        raise ValueError("--initial-prefix-tokens must be in (0, total budget).")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")

    from transformers import AutoTokenizer

    generator_tokenizer = AutoTokenizer.from_pretrained(generator)
    verifier_tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    score_values, _ = error_audit.score_scale(dual_audit.LABELS)
    score_token_ids = error_audit.resolve_score_token_ids(
        verifier_tokenizer, dual_audit.LABELS
    )
    groups = make_particle_groups(trajectories, args.initial_prefix_tokens)
    particle_counts = {len(group) for group in groups.values()}
    if len(particle_counts) != 1:
        raise ValueError(f"Unequal particle counts: {sorted(particle_counts)}")
    n_particles = next(iter(particle_counts))
    if args.method == "deterministic_fork" and n_particles % 2:
        raise ValueError("deterministic_fork requires an even number of particles.")
    if args.dry_run:
        result = {
            "problems": len(groups),
            "particles_per_problem": n_particles,
            "method": args.method,
            "rubrics": rubrics,
            "rubric_weights": rubric_weights,
            "score_token_ids": score_token_ids,
        }
        print(json.dumps(result, indent=2))
        return result

    import sglang as sgl

    generator_kwargs = dict(
        model_path=generator,
        trust_remote_code=True,
        attention_backend="triton",
        mem_fraction_static=args.generator_mem_fraction_static,
        random_seed=args.seed,
        base_gpu_id=args.generator_base_gpu_id,
        tp_size=args.generator_tp,
        disable_custom_all_reduce=args.generator_tp > 1,
        enforce_disable_flashinfer_allreduce_fusion=args.generator_tp > 1,
        disable_cuda_graph=True,
    )
    verifier_kwargs = dict(
        model_path=args.scorer,
        trust_remote_code=True,
        attention_backend="triton",
        mem_fraction_static=args.verifier_mem_fraction_static,
        random_seed=args.seed + 1,
        base_gpu_id=args.verifier_base_gpu_id,
        max_running_requests=args.max_running_requests,
        max_mamba_cache_size=args.max_running_requests,
        disable_cuda_graph=True,
    )
    if args.verifier_dp > 1:
        verifier_kwargs.update(
            dp_size=args.verifier_dp,
            load_balance_method="total_tokens",
        )
    if args.verifier_tp > 1:
        verifier_kwargs.update(
            tp_size=args.verifier_tp,
            disable_custom_all_reduce=True,
            enforce_disable_flashinfer_allreduce_fusion=True,
        )

    initialization_started = time.perf_counter()
    generator_engine = sgl.Engine(**generator_kwargs)
    verifier_engine = sgl.Engine(**verifier_kwargs)
    initialization_time = time.perf_counter() - initialization_started
    events = []
    generator_costs = []
    verifier_costs = []
    next_particle_id = len(trajectories)
    rng = random.Random(args.seed)
    algorithm_started = time.perf_counter()
    try:
        checkpoint = args.initial_prefix_tokens
        while True:
            verifier_cost = score_population(
                verifier_engine,
                verifier_tokenizer,
                generator_tokenizer,
                groups,
                rubrics=rubrics,
                rubric_weights=rubric_weights,
                total_budget=args.total_budget,
                score_token_ids=score_token_ids,
                score_values=score_values,
                batch_size=args.verifier_batch_size,
            )
            verifier_cost["checkpoint"] = checkpoint
            verifier_costs.append(verifier_cost)
            print(
                f"checkpoint={checkpoint} verifier_calls={verifier_cost['calls']} "
                f"active={sum(not p['finished'] for g in groups.values() for p in g)}",
                flush=True,
            )
            if checkpoint >= args.total_budget or not any(
                not particle["finished"]
                for particles in groups.values()
                for particle in particles
            ):
                break
            groups, allocation_events, next_particle_id = allocate_population(
                groups,
                method=args.method,
                beta=args.beta,
                ess_threshold=args.ess_threshold,
                rng=rng,
                next_particle_id=next_particle_id,
            )
            for event in allocation_events:
                event["checkpoint"] = checkpoint
                event["method"] = args.method
            events.extend(allocation_events)
            additional = min(args.checkpoint_interval, args.total_budget - checkpoint)
            generation_cost = continue_population(
                generator_engine,
                generator_tokenizer,
                groups,
                additional_tokens=additional,
                temperature=args.temperature,
            )
            generation_cost["from_checkpoint"] = checkpoint
            generation_cost["to_checkpoint"] = checkpoint + additional
            generator_costs.append(generation_cost)
            checkpoint += additional
        algorithm_wall_time = time.perf_counter() - algorithm_started
    finally:
        verifier_engine.shutdown()
        generator_engine.shutdown()

    particles, problems, quality = finalize(groups, generator_tokenizer)
    initial_tokens = sum(
        min(len(row["generator_output_ids"]), args.initial_prefix_tokens)
        for row in trajectories
    )
    generated_tokens = sum(int(row["output_tokens"]) for row in generator_costs)
    generation_inference = sum(
        float(row["inference_wall_time_s"]) for row in generator_costs
    )
    verifier_inference = sum(
        float(row["inference_wall_time_s"]) for row in verifier_costs
    )
    verifier_gpus = args.verifier_dp * args.verifier_tp
    generator_gpus = args.generator_tp
    reference = json.loads(Path(args.generator_cost_summary).read_text())
    generator_rate = (
        float(reference["cost"]["allocated_generator_gpu_seconds"])
        / float(reference["cost"]["total_output_tokens"])
    )
    initial_active_gpu_seconds = initial_tokens * generator_rate
    active_gpu_seconds = (
        initial_active_gpu_seconds
        + generation_inference * generator_gpus
        + verifier_inference * verifier_gpus
    )
    estimated_initial_wall = initial_active_gpu_seconds / generator_gpus
    static_reserved_gpu_seconds = (
        estimated_initial_wall + algorithm_wall_time
    ) * (generator_gpus + verifier_gpus)
    summary = {
        "schema_version": 1,
        "experiment": {
            "method": args.method,
            "trajectories": args.trajectories,
            "generator": generator,
            "scorer": args.scorer,
            "rubrics": rubrics,
            "rubric_weights": rubric_weights,
            "uses_generator_likelihood": False,
            "initial_prefix_tokens": args.initial_prefix_tokens,
            "checkpoint_interval": args.checkpoint_interval,
            "total_budget": args.total_budget,
            "particles_per_problem": n_particles,
            "beta": args.beta,
            "ess_threshold": args.ess_threshold,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "dataset": {
            "n_problems": len(groups),
            "n_initial_trajectories": len(trajectories),
        },
        "quality": quality,
        "allocation": {
            "checkpoints": sorted({int(row["checkpoint"]) for row in verifier_costs}),
            "problem_checkpoint_events": len(events),
            "resampled_events": sum(bool(row["resampled"]) for row in events),
            "mean_ess": statistics.fmean(float(row["ess"]) for row in events),
            "mean_unique_selected_particles": statistics.fmean(
                float(row["unique_selected_particles"]) for row in events
            ),
        },
        "cost": {
            "engine_initialization_time_s": initialization_time,
            "algorithm_wall_time_s": algorithm_wall_time,
            "initial_generator_tokens": initial_tokens,
            "new_generator_tokens": generated_tokens,
            "total_generator_tokens": initial_tokens + generated_tokens,
            "generator_inference_wall_time_s": generation_inference,
            "verifier_inference_wall_time_s": verifier_inference,
            "verifier_calls": sum(int(row["calls"]) for row in verifier_costs),
            "verifier_prompt_tokens": sum(
                int(row["prompt_tokens"]) for row in verifier_costs
            ),
            "verifier_completion_tokens": sum(
                int(row["completion_tokens"]) for row in verifier_costs
            ),
            "active_component_gpu_seconds": active_gpu_seconds,
            "active_component_gpu_seconds_per_problem": active_gpu_seconds / len(groups),
            "static_reserved_gpu_seconds": static_reserved_gpu_seconds,
            "static_reserved_gpu_seconds_per_problem": static_reserved_gpu_seconds / len(groups),
            "generator_gpus": generator_gpus,
            "verifier_gpus": verifier_gpus,
            "initial_generator_cost_estimated_from_saved_rate": True,
            "model_initialization_excluded_from_gpu_seconds": True,
        },
    }
    write_jsonl(args.save_particles, particles)
    write_jsonl(args.save_events, events)
    write_jsonl(args.save_problems, problems)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

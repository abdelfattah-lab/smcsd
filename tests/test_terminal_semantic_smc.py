import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_semantic_smc():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "terminal_semantic_smc.py"
    )
    spec = importlib.util.spec_from_file_location(
        "terminal_semantic_smc_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_analysis():
    path = (
        REPO
        / "scripts"
        / "terminal_bench"
        / "analyze_terminal_semantic_smc.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_terminal_semantic_smc_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class FakeEngine:
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, *args, **kwargs):
        self.prompts.extend(prompts)
        return [
            {
                "meta_info": {
                    "output_token_ids_logprobs": [
                        [(-2.0, 10), (-0.25, 11)]
                    ],
                    "prompt_tokens": 20,
                    "completion_tokens": 1,
                }
            }
            for _ in prompts
        ]


def particle(module, particle_id, messages):
    return module.live.Particle(
        slot=int(particle_id[-1]),
        container_name=f"container-{particle_id}",
        container_id=particle_id,
        manifest={},
        messages=messages,
        lineage={
            "particle_id": particle_id,
            "parent_particle_id": None,
            "generation": 0,
        },
        state={"filesystem_sha256": "a", "process_fingerprint": []},
        replay_cost={},
    )


def test_weight_normalization_ess_and_systematic_resampling():
    module = load_semantic_smc()
    probabilities = module.softmax([0.0, math.log(3.0)])
    assert probabilities == pytest.approx([0.25, 0.75])
    assert module.effective_sample_size(probabilities) == pytest.approx(1.6)
    assert module.systematic_indices(
        probabilities,
        random.Random(0),
    ) == [1, 1]


def test_initial_capture_and_manifest_are_exact_and_empty(tmp_path):
    module = load_semantic_smc()
    initial = {
        "model": "generator",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "tools": [],
    }
    later = {
        **initial,
        "messages": initial["messages"]
        + [{"role": "tool", "content": "result"}],
    }
    capture = tmp_path / "requests.jsonl"
    capture.write_text(
        "\n".join(
            json.dumps({"payload": payload}) for payload in (initial, later)
        )
        + "\n"
    )
    assert module.load_initial_captured_checkpoint(capture) == initial
    source = {
        "tool_events": [{"call_id": "old"}],
        "expected_state": {"filesystem_sha256": "old"},
        "sealed_from": {"container_id": "old"},
        "transcript": {
            "completed_tool_calls": 1,
            "events_sha256": "old",
        },
        "model_prefix": {},
        "lineage": {},
    }
    manifest = module.initial_manifest(source, initial["messages"])
    assert manifest["tool_events"] == []
    assert manifest["expected_state"] == {}
    assert manifest["transcript"]["completed_tool_calls"] == 0
    assert manifest["model_prefix"]["sha256"] == (
        module.backend.canonical_json_sha256(initial["messages"])
    )


def test_verifier_deduplicates_prefixes_and_uses_independent_variants():
    module = load_semantic_smc()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "working"},
    ]
    particles = [
        particle(module, "p0", messages),
        particle(module, "p1", messages),
    ]
    engine = FakeEngine()
    verifier = module.OnlineSemanticVerifier(
        scorer_model="verifier",
        tokenizer=FakeTokenizer(),
        engine=engine,
        score_labels=["A", "B"],
        score_token_ids=[10, 11],
        batch_size=8,
        transcript_max_chars=1000,
        tool_output_max_chars=100,
    )
    result = verifier.score(
        particles,
        calls_per_checkpoint=2,
        experiment_id="experiment",
        checkpoint_index=1,
    )
    assert result["unique_prefixes"] == 1
    assert result["physical_calls"] == 2
    assert result["logical_calls"] == 4
    assert result["deduplicated_calls"] == 2
    assert result["reward_isolated"] is True
    assert result["scores_by_particle"]["p0"] == pytest.approx(
        result["scores_by_particle"]["p1"]
    )
    assert engine.prompts[0] != engine.prompts[1]


def test_score_differences_and_resampling_copy_semantic_state():
    module = load_semantic_smc()
    messages = [
        {"role": "user", "content": "task"},
    ]
    particles = [
        particle(module, "p0", messages),
        particle(module, "p1", messages + [{"role": "assistant", "content": "x"}]),
    ]
    initial = {
        "calls_per_checkpoint": 1,
        "scores_by_particle": {"p0": 0.4, "p1": 0.4},
    }
    module.apply_semantic_scores(particles, initial, initialize=True)
    updated = {
        "calls_per_checkpoint": 1,
        "scores_by_particle": {"p0": 0.7, "p1": 0.2},
    }
    rows = module.apply_semantic_scores(
        particles,
        updated,
        initialize=False,
    )
    assert [row["delta"] for row in rows] == pytest.approx([0.3, -0.2])
    probabilities, ess = module.population_weights(particles, beta=4.0)
    assert probabilities[0] > probabilities[1]
    assert ess < 2.0

    ancestor = particles[0]
    ancestor.rounds = 3
    ancestor.finished = True
    ancestor.generated_tokens = 256
    child = particle(module, "p2", messages)
    module._copy_particle_progress(child, ancestor)
    assert child.rounds == 3
    assert child.finished is True
    assert child.generated_tokens == 256
    assert child.semantic_score == pytest.approx(0.7)
    assert child.semantic_log_weight == 0.0
    assert child.semantic_call_count == 2


def test_binary_auc_handles_order_ties_and_single_class():
    analysis = load_analysis()

    assert analysis.binary_auc(
        [(0.9, 1.0), (0.8, 1.0), (0.2, 0.0)]
    ) == pytest.approx(1.0)
    assert analysis.binary_auc(
        [(0.5, 1.0), (0.5, 0.0)]
    ) == pytest.approx(0.5)
    assert analysis.binary_auc(
        [(0.1, 1.0), (0.9, 0.0)]
    ) == pytest.approx(0.0)
    assert analysis.binary_auc(
        [(0.9, 1.0), (0.8, 1.0)]
    ) is None


def test_collect_materialized_drains_and_cleans_successes_after_failure():
    module = load_semantic_smc()
    first = object()
    later = object()

    class Result:
        def __init__(self, value=None, error=None):
            self.value = value
            self.error = error
            self.called = False

        def result(self):
            self.called = True
            if self.error is not None:
                raise self.error
            return self.value

    class Controller:
        def __init__(self):
            self.removed = []

        def remove_particles(self, particles):
            self.removed.extend(particles)

    futures = [
        Result(value=first),
        Result(error=RuntimeError("replay failed")),
        Result(value=later),
    ]
    controller = Controller()

    with pytest.raises(RuntimeError, match="replay failed"):
        module._collect_materialized(futures, controller)

    assert all(future.called for future in futures)
    assert controller.removed == [first, later]


def test_particle_scale_top_half_is_deterministic_and_balanced():
    module = load_semantic_smc()
    particles = [
        particle(module, "p0", [{"role": "user", "content": "task"}]),
        particle(module, "p1", [{"role": "user", "content": "task"}]),
        particle(module, "p2", [{"role": "user", "content": "task"}]),
        particle(module, "p3", [{"role": "user", "content": "task"}]),
    ]
    for item, score in zip(particles, [0.1, 0.9, 0.8, 0.2]):
        item.semantic_score = score

    assert module.top_half_indices(particles) == [1, 2, 1, 2]
    particles[2].semantic_score = 0.9

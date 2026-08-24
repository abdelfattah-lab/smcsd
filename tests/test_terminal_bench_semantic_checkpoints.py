import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_exporter():
    path = REPO_ROOT / "scripts/terminal_bench/export_semantic_checkpoints.py"
    spec = importlib.util.spec_from_file_location(
        "export_semantic_checkpoints_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()
scorer = None


def load_scorer():
    global scorer
    if scorer is None:
        path = REPO_ROOT / "scripts/terminal_bench/score_semantic_checkpoints.py"
        spec = importlib.util.spec_from_file_location(
            "score_semantic_checkpoints_test", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        scorer = module
    return scorer


def load_analyzer():
    path = (
        REPO_ROOT
        / "scripts/terminal_bench/analyze_semantic_actionability.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_semantic_actionability_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    """Tiny chat tokenizer whose generated span is content + EOS.

    The chat template appends one whitespace token after EOS. This mirrors the
    Qwen template behavior that the exporter must trim to Pi's usage count.
    """

    all_special_tokens = ["<eos>"]

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        **kwargs,
    ):
        assert tokenize
        del tools, kwargs
        if add_generation_prompt:
            return [42] * (10 + len(messages))
        assistant = messages[-1]
        prompt = [42] * (10 + len(messages[:-1]))
        content = assistant.get("content") or ""
        return prompt + [ord(char) for char in content] + [999, 10]

    def decode(self, token_ids, skip_special_tokens=False):
        assert not skip_special_tokens
        pieces = []
        for token_id in token_ids:
            if token_id == 999:
                pieces.append("<eos>")
            elif token_id == 10:
                pieces.append("\n")
            else:
                pieces.append(chr(token_id))
        return "".join(pieces)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_fixture(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job-a"
    trial_dir = job_dir / "task-a__trial"
    session_dir = trial_dir / "agent/pi/sessions"
    capture_dir = trial_dir / "agent/pi-capture"
    session_dir.mkdir(parents=True)
    capture_dir.mkdir(parents=True)

    initial_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
    ]
    first_openai_assistant = {
        "role": "assistant",
        "content": "a" * 199,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": "true"}),
                },
            }
        ],
    }
    first_pi_assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "a" * 199},
            {
                "type": "toolCall",
                "id": "call-1",
                "name": "bash",
                "arguments": {"command": "true"},
            },
        ],
        "usage": {"input": 10, "output": 200},
    }
    tool_result = {
        "role": "toolResult",
        "toolCallId": "call-1",
        "toolName": "bash",
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
    }
    second_pi_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "b" * 119}],
        "usage": {"input": 20, "output": 120},
    }
    second_messages = initial_messages + [
        first_openai_assistant,
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]
    write_jsonl(
        capture_dir / "requests-1.jsonl",
        [
            {
                "t": 1767225601000,
                "pid": 1,
                "payload": {
                    "model": "generator",
                    "messages": initial_messages,
                    "tools": [],
                },
            },
            {
                "t": 1767225603000,
                "pid": 1,
                "payload": {
                    "model": "generator",
                    "messages": second_messages,
                    "tools": [],
                },
            },
        ],
    )
    write_jsonl(
        session_dir / "session.jsonl",
        [
            {"type": "message", "message": {"role": "user", "content": []}},
            {"type": "message", "message": first_pi_assistant},
            {"type": "message", "message": tool_result},
            {"type": "message", "message": second_pi_assistant},
        ],
    )
    write_json(
        trial_dir / "result.json",
        {
            "task_name": "terminal-bench/task-a",
            "trial_name": "task-a__trial",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:06Z",
            "agent_execution": {
                "started_at": "2026-01-01T00:00:01Z",
                "finished_at": "2026-01-01T00:00:05Z",
            },
            "agent_result": {
                "n_input_tokens": 30,
                "n_output_tokens": 320,
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
        },
    )
    write_json(
        job_dir / "experiment.json",
        {
            "status": "complete",
            "job_name": "job-a",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:06Z",
            "tasks": ["task-a"],
            "spec": {"method": "ar", "seed": 7},
            "models": {"target": "generator"},
            "revisions": {"terminal_bench": "dataset-rev"},
        },
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "dataset-a",
        "benchmark": {"tasks": [{"id": "task-a"}]},
        "source_jobs": [{"job_name": "job-a"}],
        "tokenizer": {"model": "generator"},
        "checkpoint_policy": {
            "base_interval_tokens": 256,
            "derived_interval_tokens": [512, 2048, 8192],
        },
        "online_scaling_axes": {
            "checkpoint_interval_tokens": [256, 512, 2048, 8192],
            "particles": [4, 8, 16, 32, 64],
            "verifier_calls_per_checkpoint": [1, 2, 4, 8],
        },
        "labels": {"success_threshold": 1.0},
        "cost_accounting": {"accelerators_allocated": 1},
        "smoke": {"expected_trajectories": 1},
    }
    return jobs_dir, manifest


def test_frozen_smoke_manifest_includes_expanded_axes():
    manifest = json.loads(
        (
            REPO_ROOT
            / "configs/terminal_bench/semantic_actionability_smoke_v1.json"
        ).read_text()
    )

    assert manifest["checkpoint_policy"]["base_interval_tokens"] == 256
    assert manifest["online_scaling_axes"]["checkpoint_interval_tokens"] == [
        256,
        512,
        2048,
        8192,
    ]
    assert manifest["online_scaling_axes"]["particles"][-1] == 64
    assert manifest["online_scaling_axes"]["verifier_calls_per_checkpoint"][-1] == 8
    assert manifest["smoke"]["expected_trajectories"] == 4


def test_frozen_dev_manifest_pins_verifiers_and_actionability_gate():
    manifest = json.loads(
        (
            REPO_ROOT
            / "configs/terminal_bench/semantic_actionability_dev_v1.json"
        ).read_text()
    )

    assert manifest["collection"]["expected_trajectories"] == 288
    verifier_models = manifest["semantic_verifiers"]["models"]
    assert [entry["id"] for entry in verifier_models] == [
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3-32B",
    ]
    gate = manifest["offline_actionability_gate"]
    assert gate["group_balanced_ranking_accuracy_min"] == 0.60
    assert gate["ranking_bootstrap_95_ci_lower_strictly_above"] == 0.50
    assert gate["top_half_correct_trajectory_survival_lift_min"] == 0.10
    assert gate["mixed_tasks_min"] == 4
    assert gate["pairwise_comparisons_min"] == 100


def test_stable_ids_are_reproducible_and_namespaced():
    first = exporter.stable_id("dataset", "trajectory", "token", 256)
    assert first == exporter.stable_id("dataset", "trajectory", "token", 256)
    assert first != exporter.stable_id("dataset", "trajectory", "token", 512)
    assert first != exporter.stable_id("other", "trajectory", "token", 256)




def test_generation_tokens_trim_declared_template_terminator():
    tokenizer = FakeTokenizer()
    request = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "tools": [],
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": "abc"}],
    }

    _, output_ids, alignment_exact = exporter.generation_token_ids(
        tokenizer, request, assistant, expected_tokens=3
    )

    assert output_ids == [ord("a"), ord("b"), ord("c")]
    assert alignment_exact is True


def test_structural_suffix_rejects_non_template_text():
    tokenizer = FakeTokenizer()

    assert exporter.structural_template_suffix(tokenizer, "<eos>\n")
    assert exporter.structural_template_suffix(
        tokenizer, ">\n</function>\n</tool_call><eos>\n"
    )
    assert not exporter.structural_template_suffix(
        tokenizer, "answer</tool_call><eos>\n"
    )




def test_tool_use_prefix_rewrite_is_marked_nonalignable():
    class PrefixRewriteTokenizer(FakeTokenizer):
        def apply_chat_template(
            self,
            messages,
            *,
            tools,
            tokenize,
            add_generation_prompt,
            **kwargs,
        ):
            token_ids = super().apply_chat_template(
                messages,
                tools=tools,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return token_ids + [777] if add_generation_prompt else token_ids

    request = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "tools": [],
    }
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "abc"},
            {"type": "toolCall", "id": "call-a", "name": "bash", "arguments": {}},
        ],
        "stopReason": "toolUse",
    }

    prompt_ids, output_ids, alignment_exact = exporter.generation_token_ids(
        PrefixRewriteTokenizer(), request, assistant, expected_tokens=3
    )

    assert prompt_ids[-1] == 777
    assert output_ids == []
    assert alignment_exact is False

def test_tool_use_length_mismatches_are_marked_nonalignable():
    tokenizer = FakeTokenizer()
    request = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "tools": [],
    }
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "abc"},
            {
                "type": "toolCall",
                "id": "call-a",
                "name": "bash",
                "arguments": {"command": "true"},
            },
        ],
        "stopReason": "toolUse",
    }
    prompt_ids = exporter.chat_token_ids(
        tokenizer, request["messages"], [], add_generation_prompt=True
    )
    complete_ids = exporter.chat_token_ids(
        tokenizer,
        request["messages"] + [exporter.pi_assistant_to_openai(assistant)],
        [],
        add_generation_prompt=False,
    )
    rendered_tokens = len(complete_ids) - len(prompt_ids)

    _, output_ids, alignment_exact = exporter.generation_token_ids(
        tokenizer, request, assistant, expected_tokens=rendered_tokens + 1
    )

    assert len(output_ids) == rendered_tokens
    assert alignment_exact is False

    _, output_ids, alignment_exact = exporter.generation_token_ids(
        tokenizer, request, assistant, expected_tokens=1
    )

    assert output_ids == []


def test_export_separates_labels_and_builds_exact_resumable_prefix(tmp_path):
    jobs_dir, manifest = make_fixture(tmp_path)
    output_dir = tmp_path / "output"

    summary = exporter.export_dataset(
        manifest,
        jobs_dir,
        output_dir,
        tokenizer=FakeTokenizer(),
    )

    checkpoints = [
        json.loads(line)
        for line in (output_dir / "checkpoints.jsonl").read_text().splitlines()
    ]
    trajectories = [
        json.loads(line)
        for line in (output_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    labels = [
        json.loads(line)
        for line in (output_dir / "labels.jsonl").read_text().splitlines()
    ]
    by_kind = {row["trigger"]["kind"]: row for row in checkpoints}

    assert summary["counts"] == {
        "tasks": 1,
        "trajectories": 1,
        "checkpoints": 2,
        "token_interval_checkpoints": 1,
        "post_tool_checkpoints": 1,
        "errored_trajectories": 0,
        "incomplete_model_requests_dropped": 0,
        "trajectories_with_nonalignable_turns": 0,
        "nonalignable_model_turns": 0,
        "token_checkpoints_skipped_for_alignment": 0,
        "errored_trajectories_with_token_accounting_mismatch": 0,
        "session_minus_agent_result_input_tokens": 0,
        "session_minus_agent_result_output_tokens": 0,
    }
    token_checkpoint = by_kind["token_interval"]
    assert token_checkpoint["trigger"]["target_generated_tokens"] == 256
    assert token_checkpoint["trigger"]["actual_generated_tokens"] == 256
    assert token_checkpoint["resume_state"]["turn_index"] == 1
    assert token_checkpoint["resume_state"]["turn_generated_tokens"] == 56
    assert len(token_checkpoint["resume_state"]["partial_generation_token_ids"]) == 56
    assert token_checkpoint["costs_so_far"]["generator"] == {
        "calls": 2,
        "input_tokens": 30,
        "output_tokens": 256,
        "active_accelerator_seconds": None,
        "active_accelerator_seconds_status": "not_available_per_request",
    }

    tool_checkpoint = by_kind["post_tool"]
    assert tool_checkpoint["trigger"]["generated_tokens"] == 200
    assert tool_checkpoint["semantic_input"]["messages"][-1]["role"] == "tool"
    assert tool_checkpoint["costs_so_far"]["tools"]["calls"] == 1

    assert exporter.label_keys(checkpoints) == []
    assert '"final_reward"' not in (output_dir / "checkpoints.jsonl").read_text()
    assert labels[0]["final_reward"] == 1.0
    assert labels[0]["eventual_success"] is True
    assert trajectories[0]["costs"]["generator"]["input_tokens"] == 30
    assert trajectories[0]["costs"]["generator"]["output_tokens"] == 320
    assert trajectories[0]["costs"]["semantic_verifier"]["calls"] == 0
    assert trajectories[0]["costs"]["total"]["agent_wall_time_s"] == 4.0
    assert trajectories[0]["costs"]["total"]["allocated_accelerator_seconds"] is None
    assert trajectories[0]["costs"]["total"][
        "allocated_accelerator_seconds_status"
    ] == "unavailable_per_trajectory_under_concurrent_source_serving"
    assert summary["source_generation_cost"]["allocated_accelerator_seconds"] == 6.0
    assert (
        summary["source_generation_cost"]["covers_exported_dataset_exactly"] is True
    )
    assert trajectories[0]["resume_capability"]["serialized_model_prefix"] is True
    assert (
        trajectories[0]["resume_capability"]["terminal_environment_cloneable"]
        is False
    )
    assert summary["validation"]["exact_token_checkpoint_alignment"] is True
    assert summary["validation"]["checkpoint_records_have_no_label_fields"] is True


def test_semantic_transcript_keeps_task_latest_state_and_partial_generation():
    semantic = load_scorer()
    checkpoint = {
        "checkpoint_id": "checkpoint-a",
        "semantic_input": {
            "messages": [
                {"role": "system", "content": "do not include this"},
                {"role": "user", "content": "repair the service"},
                {"role": "assistant", "content": "old plan" * 40},
                {
                    "role": "tool",
                    "tool_call_id": "call-a",
                    "content": "old output " * 100,
                },
                {"role": "assistant", "content": "latest diagnosis"},
            ],
            "partial_assistant_rendered": "run the focused test",
        },
    }

    task, transcript, stats = semantic.checkpoint_transcript(
        checkpoint,
        transcript_max_chars=220,
        tool_output_max_chars=60,
    )

    assert task == "repair the service"
    assert "do not include this" not in transcript
    assert "latest diagnosis" in transcript
    assert "run the focused test" in transcript
    assert stats["omitted_blocks"] > 0
    assert stats["tool_outputs_compacted"] == 1


def test_semantic_expected_score_uses_requested_label_distribution():
    semantic = load_scorer()
    output = {
        "meta_info": {
            "output_token_ids_logprobs": [[(-2.0, 10), (-1.0, 11)]],
            "prompt_tokens": 50,
            "completion_tokens": 1,
        }
    }

    result = semantic.expected_score_from_output(output, [10, 11], [0.0, 1.0])

    assert result["score"] > 0.5
    assert result["prompt_tokens"] == 50
    assert result["completion_tokens"] == 1
    assert result["logprob_source"] == "selected"


def test_verifier_call_ids_separate_models_and_are_reproducible():
    semantic = load_scorer()
    first = semantic.stable_id(
        "dataset", "checkpoint", "verifier-a", semantic.CRITERION_VERSION, 0
    )

    assert first == semantic.stable_id(
        "dataset", "checkpoint", "verifier-a", semantic.CRITERION_VERSION, 0
    )
    assert first != semantic.stable_id(
        "dataset", "checkpoint", "verifier-b", semantic.CRITERION_VERSION, 0
    )


def test_actionability_metrics_rank_successful_trajectories_within_groups():
    analyzer = load_analyzer()
    records = [
        {
            "task_id": task,
            "group_id": f"{task}:token:256",
            "trajectory_id": f"{task}-{label}",
            "eventual_success": label,
            "score": score,
        }
        for task in ("task-a", "task-b")
        for label, score in ((True, 0.9), (False, 0.1))
    ]

    metrics = analyzer.group_metrics(records)

    assert metrics["group_balanced_ranking_accuracy"] == 1.0
    assert metrics["pooled_pairwise_ranking_accuracy"] == 1.0
    assert metrics["pairwise_comparisons"] == 2
    assert metrics["mixed_tasks"] == 2
    assert metrics["top_half_correct_trajectory_survival"] == 1.0
    assert metrics["top_half_survival_lift"] == 0.5


def test_actionability_views_filter_terminal_and_subsample_intervals():
    analyzer = load_analyzer()
    records = [
        {
            "trigger": {"kind": "token_interval"},
            "position": position,
            "preterminal": preterminal,
        }
        for position, preterminal in ((256, True), (512, True), (1024, False))
    ]
    records.append(
        {
            "trigger": {"kind": "post_tool"},
            "position": 1,
            "preterminal": True,
        }
    )

    token_256 = analyzer.view_records(records, "token_256")
    token_512 = analyzer.view_records(records, "token_512")
    post_tool = analyzer.view_records(records, "post_tool")

    assert [row["position"] for row in token_256] == [256, 512]
    assert [row["position"] for row in token_512] == [512]


def test_export_drops_only_trailing_incomplete_request_on_timeout(tmp_path):
    jobs_dir, manifest = make_fixture(tmp_path)
    trial_dir = jobs_dir / "job-a/task-a__trial"
    requests_path = trial_dir / "agent/pi-capture/requests-1.jsonl"
    request_rows = [
        json.loads(line) for line in requests_path.read_text().splitlines()
    ]
    incomplete = dict(request_rows[-1])
    incomplete["t"] += 1000
    write_jsonl(requests_path, [*request_rows, incomplete])

    result_path = trial_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["exception_info"] = {
        "exception_type": "AgentTimeoutError",
        "exception_message": "timed out",
    }
    result["verifier_result"]["rewards"]["reward"] = 0.0
    result["agent_result"]["n_input_tokens"] = 29
    result["agent_result"]["n_output_tokens"] = 319
    write_json(result_path, result)

    output_dir = tmp_path / "timeout-output"
    summary = exporter.export_dataset(
        manifest,
        jobs_dir,
        output_dir,
        tokenizer=FakeTokenizer(),
    )
    trajectory = json.loads(
        (output_dir / "trajectories.jsonl").read_text().strip()
    )
    label = json.loads((output_dir / "labels.jsonl").read_text().strip())

    assert summary["counts"]["errored_trajectories"] == 1
    assert summary["counts"]["incomplete_model_requests_dropped"] == 1
    assert trajectory["counts"]["incomplete_model_requests_dropped"] == 1
    assert summary["counts"][
        "errored_trajectories_with_token_accounting_mismatch"
    ] == 1
    assert summary["counts"]["session_minus_agent_result_input_tokens"] == 1
    assert summary["counts"]["session_minus_agent_result_output_tokens"] == 1
    assert trajectory["counts"]["model_turns"] == 2
    assert label["errored"] is True
    assert trajectory["counts"]["agent_result_token_accounting_mismatch"] is True
    assert trajectory["costs"]["generator"]["token_accounting_status"] == (
        "session_sum_preserved_agent_result_differs_on_errored_trajectory"
    )


def test_score_summary_counts_full_gpu_allocation_across_resumed_runs(tmp_path):
    semantic = load_scorer()
    scores_path = tmp_path / "scores.jsonl"
    run_costs_path = tmp_path / "scores.runs.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(
        scores_path,
        [
            {
                "run_id": "run-a",
                "batch_index": 0,
                "batch_wall_time_s": 2.0,
                "checkpoint_id": "checkpoint-a",
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "score_token_mass": 0.9,
            },
            {
                "run_id": "run-b",
                "batch_index": 0,
                "batch_wall_time_s": 1.0,
                "checkpoint_id": "checkpoint-b",
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "score_token_mass": 0.8,
            },
        ],
    )
    write_jsonl(
        run_costs_path,
        [
            {
                "allocation_wall_time_s": 10.0,
                "engine_startup_wall_time_s": 4.0,
            },
            {
                "allocation_wall_time_s": 20.0,
                "engine_startup_wall_time_s": 5.0,
            },
        ],
    )

    summary = semantic.summarize_scores(
        scores_path,
        summary_path,
        run_costs_path,
        scorer_model="verifier",
        criterion="criterion",
        tp_size=2,
    )

    assert summary["counts"]["scorer_process_runs"] == 2
    assert summary["cost"]["inference_wall_time_s"] == 3.0
    assert summary["cost"]["allocation_wall_time_s"] == 30.0
    assert summary["cost"]["engine_startup_wall_time_s"] == 9.0
    assert summary["cost"]["allocated_accelerator_seconds"] == 60.0

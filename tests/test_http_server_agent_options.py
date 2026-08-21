import argparse

import pytest

from smcsd.http_server import (
    _json_object,
    build_cli_parser,
    server_args_from_cli,
)


def test_json_object_accepts_mapping() -> None:
    assert _json_object('{"enable_thinking": false}') == {"enable_thinking": False}


@pytest.mark.parametrize("value", ["[]", '"text"', "not-json"])
def test_json_object_rejects_non_object(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _json_object(value)


def test_agentic_cli_options_reach_server_args(monkeypatch) -> None:
    args = build_cli_parser().parse_args(
        [
            "--model",
            "Qwen/Qwen3.5-9B",
            "--draft-model",
            "Qwen/Qwen3.5-2B",
            "--served-model-name",
            "local/Qwen3.5-9B-SMC",
            "--tool-call-parser",
            "auto",
            "--reasoning-parser",
            "auto",
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--disable-cuda-graph",
            "--disable-flashinfer-autotune",
            "--random-seed",
            "17",
            "--enable-metrics",
            "-N",
            "4",
            "-g",
            "8",
        ]
    )

    captured = {}

    def fake_build_smc_server_args(**kwargs):
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(
        "smcsd.http_server.build_smc_server_args", fake_build_smc_server_args
    )
    server_args = server_args_from_cli(args)

    assert server_args is captured
    assert captured["served_model_name"] == "local/Qwen3.5-9B-SMC"
    assert captured["tool_call_parser"] == "auto"
    assert captured["reasoning_parser"] == "auto"
    assert captured["default_chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["n_particles"] == 4
    assert captured["gamma"] == 8
    assert captured["disable_cuda_graph"] is True
    assert captured["disable_flashinfer_autotune"] is True
    assert captured["random_seed"] == 17
    assert captured["enable_metrics"] is True

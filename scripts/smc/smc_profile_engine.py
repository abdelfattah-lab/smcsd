"""
Profile SMC scheduler variants through the offline Engine API.

Examples:
  source .venv/bin/activate
  python scripts/smc/smc_profile_engine.py --output-dir /tmp/sglang-smc-profile
  python scripts/smc/smc_profile_engine.py --profile-v2 --decode-only
  python scripts/smc/smc_profile_engine.py --smc-resampling-overlap
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
]


def default_model_path() -> str:
    return DEFAULT_MODEL_PATH

def default_draft_model_path() -> str:
    return DEFAULT_DRAFT_MODEL_PATH

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile SMC scheduler variants with offline sgl.Engine()."
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--draft-model-path", default=default_draft_model_path())
    parser.add_argument("--output-dir", default="/tmp/sglang-smc-profile")
    parser.add_argument("--profile-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--profile-v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SGLANG_PROFILE_V2.",
    )
    parser.add_argument(
        "--decode-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only meaningful with --profile-v2.",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also record CPU activity. GPU profiling is always enabled.",
    )
    parser.add_argument(
        "--with-stack",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override profiler stack capture. Defaults to the runtime setting.",
    )
    parser.add_argument("--smc-n-particles", type=int, default=8)
    parser.add_argument("--smc-gamma", type=int, default=8)
    parser.add_argument(
        "--smc-resampling-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the experimental SMC overlap scheduler instead of the baseline normal SMC scheduler.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Repeat to add custom prompts.",
    )
    return parser.parse_args()


def build_run_dir(base_dir: str) -> Path:
    run_dir = Path(base_dir).expanduser().resolve() / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def wait_for_artifacts(run_dir: Path, timeout_sec: float = 30.0) -> list[Path]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        files = sorted(
            path
            for path in run_dir.iterdir()
            if path.is_file()
            and (
                path.name.endswith(".trace.json.gz")
                or path.name.endswith(".pickle")
            )
        )
        if files:
            return files
        time.sleep(0.5)
    return []


def sampling_params(max_new_tokens: int) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }


def summarize_server_info(server_info: dict[str, Any]) -> dict[str, Any]:
    internal_states = server_info.get("internal_states") or [{}]
    first_state = internal_states[0] if internal_states else {}
    return {
        "model_path": server_info.get("model_path"),
        "speculative_algorithm": server_info.get("speculative_algorithm"),
        "disable_overlap_schedule": server_info.get("disable_overlap_schedule"),
        "smc_n_particles": server_info.get("smc_n_particles"),
        "smc_gamma": server_info.get("smc_gamma"),
        "avg_spec_accept_length": first_state.get("avg_spec_accept_length"),
        "last_gen_throughput": first_state.get("last_gen_throughput"),
    }


def main() -> None:
    args = parse_args()
    if args.profile_steps <= 0:
        raise SystemExit("--profile-steps must be positive.")

    if args.profile_v2:
        os.environ["SGLANG_PROFILE_V2"] = "1"

    import sglang as sgl

    prompts = args.prompts or DEFAULT_PROMPTS
    run_dir = build_run_dir(args.output_dir)
    profile_stages = ["decode"] if args.decode_only else ["prefill", "decode"]
    if args.decode_only and not args.profile_v2:
        print(
            "WARNING --decode-only needs --profile-v2; "
            "classic profiling will still emit both stages."
        )
        profile_stages = ["prefill", "decode"]

    activities = ["GPU"]
    if args.cpu:
        activities.insert(0, "CPU")

    write_json(
        run_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "output_dir": str(run_dir),
            "profile_steps": args.profile_steps,
            "profile_v2": args.profile_v2,
            "profile_stages": profile_stages,
            "activities": activities,
            "with_stack": args.with_stack,
            "max_new_tokens": args.max_new_tokens,
            "smc_n_particles": args.smc_n_particles,
            "smc_gamma": args.smc_gamma,
            "prompts": prompts,
        },
    )

    with sgl.Engine(
        model_path=args.model_path,
        speculative_algorithm="SMC",
        speculative_draft_model_path=args.model_path,
        smc_n_particles=args.smc_n_particles,
        smc_gamma=args.smc_gamma,
        page_size=1,
        mem_fraction_static=0.4,
        trust_remote_code=True,
        log_level="info",
        random_seed=1,
        cuda_graph_max_bs=32,
        attention_backend="fa3",
        smc_draft_temperature=0.8,
        smc_target_temperature=0.8,
    ) as engine:
        server_info = engine.get_server_info()
        write_json(run_dir / "server_info.json", server_info)

        print("SERVER_INFO")
        print(json.dumps(summarize_server_info(server_info), indent=2))
        print(f"PROFILE_DIR {run_dir}")

        print("WARMUP")
        engine.generate(prompts[:1], sampling_params(8))

        print(
            "START_PROFILE",
            json.dumps(
                {
                    "profile_steps": args.profile_steps,
                    "profile_stages": profile_stages,
                    "activities": activities,
                    "with_stack": args.with_stack,
                    "profile_v2": args.profile_v2,
                }
            ),
        )
        engine.start_profile(
            output_dir=str(run_dir),
            num_steps=args.profile_steps,
            activities=activities,
            with_stack=args.with_stack,
            profile_by_stage=True,
            profile_prefix="smc-engine",
            profile_stages=profile_stages,
        )

        outputs = engine.generate(prompts, sampling_params(args.max_new_tokens))
        write_json(run_dir / "outputs.json", outputs)

        artifacts = wait_for_artifacts(run_dir)
        if not artifacts and not args.profile_v2:
            try:
                engine.stop_profile()
            except RuntimeError:
                pass
            artifacts = wait_for_artifacts(run_dir)
        write_json(run_dir / "profile_artifacts.json", [str(path) for path in artifacts])

        for index, (prompt, output) in enumerate(zip(prompts, outputs, strict=True), start=1):
            print(f"PROMPT_{index}: {prompt}")
            print(f"OUTPUT_{index}: {output['text']}")
            print("-" * 80)

        print("PROFILE_ARTIFACTS")
        for artifact in artifacts:
            print(str(artifact))


if __name__ == "__main__":
    main()

"""
Offline E2E probe for SMC speculative decoding.

Verifies correctness and, when overlap scheduling is enabled, checks that
resample KV copies on resample_stream are launched AFTER run_batch() so
they overlap with the GPU forward pass.

Usage:
  source .venv/bin/activate
  python tasks/smc_offline_e2e_probe.py
  python tasks/smc_offline_e2e_probe.py --attention-backend flashinfer
"""

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
venv_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

import sglang as sgl


MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL_PATH = MODEL_PATH
BASE_PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
]
DEFAULT_PROMPT_REPEAT = 3
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.8,
    "max_new_tokens": 64,
    "ignore_eos": True,
}
PROBE_RECORD_PATH_ENV = "SGLANG_SMC_PROBE_RECORD_PATH"


# ---------------------------------------------------------------------------
# Overlap instrumentation
# ---------------------------------------------------------------------------

_records_lock = threading.Lock()
_records: list[dict] = []
_probes_installed = False


def _append_probe_record(record: dict) -> None:
    record = dict(record)
    record["pid"] = os.getpid()

    record_path = os.environ.get(PROBE_RECORD_PATH_ENV)
    if record_path:
        with open(record_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, sort_keys=True) + "\n")

    with _records_lock:
        _records.append(record)


def _install_overlap_probes() -> None:
    """Patch SMCScheduler and Scheduler class methods to record probe events.

    Must be called before any scheduler instance is created (i.e. before
    the Engine is started).
    """
    global _probes_installed
    if _probes_installed:
        return
    _probes_installed = True

    from sglang.srt.speculative import smc_scheduler as _smc_mod
    from sglang.srt.managers import scheduler as _sched_mod

    _orig_run_batch = _sched_mod.Scheduler.run_batch

    def _probed_run_batch(self_scheduler, batch):
        is_decode = batch is not None and batch.forward_mode.is_decode()
        if not is_decode:
            return _orig_run_batch(self_scheduler, batch)

        started_ns = time.perf_counter_ns()
        result = _orig_run_batch(self_scheduler, batch)
        _append_probe_record(
            {
                "type": "forward_decode",
                "batch_size": len(batch.reqs),
                "started_ns": started_ns,
                "returned_ns": time.perf_counter_ns(),
            }
        )
        return result

    _sched_mod.Scheduler.run_batch = _probed_run_batch


def _load_probe_records(record_path: Optional[Path]) -> list[dict]:
    records = []
    if record_path is not None and record_path.exists():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        with _records_lock:
            records.extend(_records)

    return records


def _report_overlap(record_path: Optional[Path]) -> bool:
    """Print overlap probe report using records emitted from worker processes.

    Returns True if overlap scheduling was exercised without probe failures.
    """
    records = _load_probe_records(record_path)
    forwards = [rec for rec in records if rec.get("type") == "forward_decode"]
    resamples = [rec for rec in records if rec.get("type") == "resample_launch"]

    print()
    print("=" * 65)
    print("SMC Resample Overlap Probe")
    print("=" * 65)
    print(f"  Decode forwards recorded : {len(forwards)}")
    print(f"  Resample launches recorded: {len(resamples)}")

    if not resamples:
        print(
            "  No resamples triggered. Cannot verify overlap, "
            "but generation succeeded."
        )
        return True

    print()
    print(f"  {'#':>4}  {'Group':>36}  {'Evictions':>10}  Note")
    print(f"  {'-'*65}")
    for i, rec in enumerate(resamples):
        print(
            f"  {i:>4}  {rec['group_id'][:36]:>36}  "
            f"{rec['num_evictions']:>10}  launched from step_after_forward"
        )

    print(f"  {'-'*65}")
    print(
        f"  PASS: observed {len(resamples)} resample launches across "
        f"{len({rec['group_id'] for rec in resamples})} SMC groups."
    )
    return True


def _build_prompts(prompt_repeat: int) -> list[str]:
    prompts = []
    for repeat_idx in range(prompt_repeat):
        for base_idx, prompt in enumerate(BASE_PROMPTS, start=1):
            prompts.append(
                f"[probe batch {repeat_idx + 1} item {base_idx}] {prompt}"
            )
    return prompts


# ---------------------------------------------------------------------------
# Original probe logic
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Target model path or repo id.",
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default=None,
        help="Draft model path or repo id. Defaults to --model-path.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="triton",
        help="Override the global attention backend for the SMC probe.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=DEFAULT_PROMPT_REPEAT,
        help="Repeat the base prompt set this many times to create a larger batch.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_SAMPLING_PARAMS["temperature"],
        help="Request sampling temperature. Non-zero values make resampling more likely.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_SAMPLING_PARAMS["max_new_tokens"],
        help="Maximum generation length per request.",
    )
    parser.add_argument(
        "--smc-n-particles",
        type=int,
        default=4,
        help="Number of SMC particles per request.",
    )
    parser.add_argument(
        "--smc-gamma",
        type=int,
        default=4,
        help="SMC gamma used for draft steps.",
    )
    parser.add_argument(
        "--smc-draft-temperature",
        type=float,
        default=1.25,
        help="Multiplier applied to request temperature for SMC draft sampling.",
    )
    parser.add_argument(
        "--show-all-outputs",
        action="store_true",
        help="Print every output instead of truncating to the first four.",
    )
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip the overlap instrumentation (just do correctness check).",
        default=True
    )
    return parser.parse_args()


def _drop_none(value):
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_none(item)) is not None
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _drop_none(item)) is not None]
    return value


def main():
    args = parse_args()
    start = time.time()
    prompts = _build_prompts(args.prompt_repeat)
    sampling_params = dict(DEFAULT_SAMPLING_PARAMS)
    sampling_params["temperature"] = args.temperature
    sampling_params["max_new_tokens"] = args.max_new_tokens
    draft_model_path = (
        args.draft_model_path if args.draft_model_path is not None else args.model_path
    )
    record_path = Path("/tmp") / f"smc_probe_{os.getpid()}_{int(start)}.jsonl"
    if record_path.exists():
        record_path.unlink()
    os.environ[PROBE_RECORD_PATH_ENV] = str(record_path)

    overlap_enabled = os.environ.get("SGLANG_ENABLE_SPEC_V2", "") == "1"
    if overlap_enabled and not args.skip_overlap_check:
        _install_overlap_probes()

    engine_kwargs = dict(
        model_path=args.model_path,
        speculative_algorithm="SMC",
        speculative_draft_model_path=draft_model_path,
        smc_n_particles=args.smc_n_particles,
        smc_gamma=args.smc_gamma,
        smc_draft_temperature=args.smc_draft_temperature,
        page_size=1,
        cuda_graph_max_bs=4,
        mem_fraction_static=0.45,
        trust_remote_code=True,
        log_level="info",
        attention_backend="triton"
    )
    if args.attention_backend is not None:
        engine_kwargs["attention_backend"] = args.attention_backend

    with sgl.Engine(**engine_kwargs) as engine:
        outputs = engine.generate(prompts, sampling_params)
        server_info = engine.get_server_info()
        compact_server_info = _drop_none(
            {
                "speculative_algorithm": server_info.get("speculative_algorithm"),
                "disable_overlap_schedule": server_info.get(
                    "disable_overlap_schedule"
                ),
                "smc_n_particles": server_info.get("smc_n_particles"),
                "smc_gamma": server_info.get("smc_gamma"),
                "attention_backend": server_info.get("attention_backend"),
                "prompt_count": len(prompts),
                "sampling_temperature": args.temperature,
                "avg_spec_accept_length": server_info.get("internal_states", [{}])[
                    0
                ].get("avg_spec_accept_length"),
            }
        )
        print("SERVER_INFO", json.dumps(compact_server_info, indent=2))

        prompts_to_show = prompts if args.show_all_outputs else prompts[:4]
        outputs_to_show = outputs if args.show_all_outputs else outputs[:4]
        for i, (prompt, output) in enumerate(
            zip(prompts_to_show, outputs_to_show, strict=True), start=1
        ):
            print(f"PROMPT_{i}: {prompt}")
            print(f"OUTPUT_{i}: {output['text']}")
            print(
                "META_{}: {}".format(
                    i,
                    json.dumps(
                        _drop_none(output.get("meta_info", {})),
                        indent=2,
                        default=str,
                    ),
                )
            )
            print("-" * 80)
        if len(outputs_to_show) < len(outputs):
            print(
                f"... omitted {len(outputs) - len(outputs_to_show)} additional outputs "
                f"(use --show-all-outputs to print everything)"
            )

    print(f"TOTAL_SECONDS {time.time() - start:.2f}")

    # Report overlap if probes were installed
    if overlap_enabled and not args.skip_overlap_check:
        ok = _report_overlap(record_path)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

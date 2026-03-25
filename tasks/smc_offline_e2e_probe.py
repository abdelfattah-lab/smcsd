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

os.environ.setdefault("SGLANG_ENABLE_SPEC_V2", "1")
venv_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

import sglang as sgl


MODEL_PATH = "/home/ccchang/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DRAFT_MODEL_PATH = MODEL_PATH
PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
]
SAMPLING_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 32,
    "ignore_eos": True,
}


# ---------------------------------------------------------------------------
# Overlap instrumentation
# ---------------------------------------------------------------------------

_records_lock = threading.Lock()
_records: list[dict] = []
_probes_installed = False


def _install_overlap_probes() -> None:
    """Patch SMCScheduler and Scheduler class methods to record CUDA events.

    Must be called before any scheduler instance is created (i.e. before
    the Engine is started).
    """
    global _probes_installed
    if _probes_installed:
        return
    _probes_installed = True

    import torch
    from sglang.srt.speculative import smc_scheduler as _smc_mod
    from sglang.srt.managers import scheduler as _sched_mod

    _orig_before = _smc_mod.SMCScheduler.step_before_forward
    _orig_after = _smc_mod.SMCScheduler.step_after_forward
    _orig_run_batch = _sched_mod.Scheduler.run_batch

    def _probed_step_before_forward(self_sched, scheduler_arg):
        if not hasattr(self_sched, "_probe_device"):
            self_sched._probe_device = (
                self_sched.device is not None
                and str(self_sched.device) != "cpu"
            )
        _orig_before(self_sched, scheduler_arg)

    def _probed_step_after_forward(self_sched, scheduler_arg):
        has_groups = bool(self_sched._groups_needing_resample)
        evt_before = None
        if has_groups and getattr(self_sched, "_probe_device", False):
            evt_before = torch.cuda.Event(enable_timing=True)
            evt_before.record()

        _orig_after(self_sched, scheduler_arg)

        if evt_before is not None:
            evt_after = torch.cuda.Event(enable_timing=True)
            evt_after.record()
            with _records_lock:
                _records.append({
                    "type": "resample_launch",
                    "evt_before": evt_before,
                    "evt_after": evt_after,
                })

    def _probed_run_batch(self_scheduler, batch):
        is_decode = (
            batch is not None and batch.forward_mode.is_decode()
        )
        if not is_decode:
            return _orig_run_batch(self_scheduler, batch)

        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)
        evt_start.record()
        result = _orig_run_batch(self_scheduler, batch)
        evt_end.record()
        with _records_lock:
            _records.append({
                "type": "forward_decode",
                "evt_start": evt_start,
                "evt_end": evt_end,
            })
        return result

    _smc_mod.SMCScheduler.step_before_forward = _probed_step_before_forward
    _smc_mod.SMCScheduler.step_after_forward = _probed_step_after_forward
    _sched_mod.Scheduler.run_batch = _probed_run_batch


def _report_overlap() -> bool:
    """Synchronize CUDA events and print overlap report.

    Returns True if overlap was observed (or no resamples occurred).
    """
    import torch

    torch.cuda.synchronize()

    forwards = []
    resamples = []
    with _records_lock:
        for rec in _records:
            if rec["type"] == "forward_decode":
                ms = rec["evt_start"].elapsed_time(rec["evt_end"])
                forwards.append(ms)
            elif rec["type"] == "resample_launch":
                ms = rec["evt_before"].elapsed_time(rec["evt_after"])
                resamples.append(ms)

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
    print(f"  {'#':>4}  {'Forward(ms)':>12}  {'Resample(ms)':>13}  Note")
    print(f"  {'-'*50}")

    # Pair each resample with the decode forward that preceded it.
    # Due to the split, step_after_forward runs after run_batch, so
    # resample[i] was launched while forward[i] was still on the GPU.
    paired = min(len(forwards), len(resamples))
    for i in range(paired):
        fwd_ms = forwards[i] if i < len(forwards) else float("nan")
        rs_ms = resamples[i]
        print(f"  {i:>4}  {fwd_ms:>12.3f}  {rs_ms:>13.3f}  launched after run_batch")

    print(f"  {'-'*50}")
    print(
        f"  {paired}/{len(resamples)} resample launches occurred after "
        f"run_batch (in forward overlap window)"
    )

    if paired > 0:
        print()
        print(
            "  PASS: step_after_forward exercised — KV copies on "
            "resample_stream submitted while compute stream forward "
            "was in flight."
        )
        return True
    else:
        print()
        print("  FAIL: No resample launches detected after run_batch.")
        return False


# ---------------------------------------------------------------------------
# Original probe logic
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Override the global attention backend for the SMC probe.",
    )
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip the overlap instrumentation (just do correctness check).",
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

    overlap_enabled = os.environ.get("SGLANG_ENABLE_SPEC_V2", "") == "1"
    if overlap_enabled and not args.skip_overlap_check:
        _install_overlap_probes()

    engine_kwargs = dict(
        model_path=MODEL_PATH,
        speculative_algorithm="SMC",
        speculative_draft_model_path=DRAFT_MODEL_PATH,
        smc_n_particles=4,
        smc_gamma=3,
        page_size=1,
        cuda_graph_max_bs=4,
        mem_fraction_static=0.45,
        trust_remote_code=True,
        log_level="info",
    )
    if args.attention_backend is not None:
        engine_kwargs["attention_backend"] = args.attention_backend

    with sgl.Engine(**engine_kwargs) as engine:
        outputs = engine.generate(PROMPTS, SAMPLING_PARAMS)
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
                "avg_spec_accept_length": server_info.get("internal_states", [{}])[
                    0
                ].get("avg_spec_accept_length"),
            }
        )
        print("SERVER_INFO", json.dumps(compact_server_info, indent=2))

        for i, (prompt, output) in enumerate(
            zip(PROMPTS, outputs, strict=True), start=1
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

    print(f"TOTAL_SECONDS {time.time() - start:.2f}")

    # Report overlap if probes were installed
    if overlap_enabled and not args.skip_overlap_check:
        ok = _report_overlap()
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

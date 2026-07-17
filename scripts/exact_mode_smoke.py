"""Smoke test for exact mode: one engine per mode, same prompts.

Usage:
    python scripts/exact_mode_smoke.py [--mode exact|smc|both] [--gpu 2]

Checks that mode="exact" boots, decodes coherent text end-to-end through
draft AR -> TARGET_VERIFY -> multi-draft accept -> collapse/rollback, and
that mode="smc" is unaffected.
"""

import argparse
import os

os.environ.setdefault("HF_HOME", "/dev/shm/hf")

PROMPTS = [
    "Question: What is 13 + 29? Answer with just the number.\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


def run(mode: str, gpu: int):
    from smcsd import SMCEngine

    print(f"\n=== mode={mode} ===", flush=True)
    eng = SMCEngine(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        mode=mode,
        n_particles=4,
        gamma=4,
        draft_temperature=0.7,
        target_temperature=0.7,
        mem_fraction_static=0.4,
        base_gpu_id=gpu,
        max_running_requests=4,
        log_level="info",
    )
    try:
        outs = eng.generate(
            PROMPTS, sampling_params={"max_new_tokens": 48}
        )
        for prompt, out in zip(PROMPTS, outs):
            print(f"--- prompt: {prompt[:40]!r}")
            print(f"    completion_tokens={out['completion_tokens']}")
            print(f"    text: {out['text'][:200]!r}", flush=True)
    finally:
        eng.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["exact", "smc", "both"])
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    modes = ["exact", "smc"] if args.mode == "both" else [args.mode]
    for m in modes:
        run(m, args.gpu)
    print("SMOKE_OK", flush=True)

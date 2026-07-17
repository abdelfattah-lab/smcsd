"""Smoke test for the unified exact/SMC modes: one engine per mode.

Usage:
    python scripts/exact_mode_smoke.py [--mode exact|smc|mixed|all] [--gpu 2]

* exact — every request verified by multi-draft rejection sampling.
* smc   — regression check for the default fast path.
* mixed — three requests in one engine: default SMC, per-request exact
  (custom_params {"smc_mode": "exact"}), and a mid-sequence switch plan
  ({"smc_mode_plan": [[16, "exact"]]}); prints each request's mode stats.
"""

import argparse

PROMPTS = [
    "Question: What is 13 + 29? Answer with just the number.\nAnswer:",
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
]


def make_engine(mode: str, gpu: int):
    from smcsd import SMCEngine

    return SMCEngine(
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


def run_single_mode(mode: str, gpu: int):
    print(f"\n=== mode={mode} ===", flush=True)
    eng = make_engine(mode, gpu)
    try:
        outs = eng.generate(PROMPTS, sampling_params={"max_new_tokens": 48})
        for prompt, out in zip(PROMPTS, outs):
            print(f"--- prompt: {prompt[:40]!r}")
            print(f"    completion_tokens={out['completion_tokens']}")
            print(f"    text: {out['text'][:200]!r}", flush=True)
    finally:
        eng.shutdown()


def run_mixed(gpu: int):
    print("\n=== mode=mixed (per-request + mid-sequence switch) ===",
          flush=True)
    eng = make_engine("mixed", gpu)
    labels = ["smc(default)", "exact(request)", "plan(smc->exact@16)"]
    params = [
        {"max_new_tokens": 48},
        {"max_new_tokens": 48, "custom_params": {"smc_mode": "exact"}},
        {"max_new_tokens": 48,
         "custom_params": {"smc_mode_plan": [[16, "exact"]]}},
    ]
    try:
        outs = eng.generate([PROMPTS[1]] * 3, sampling_params=params)
        for label, out in zip(labels, outs):
            print(f"--- {label}")
            print(f"    completion_tokens={out['completion_tokens']}")
            print(f"    mode_stats={out.get('smc_mode_stats')}")
            print(f"    text: {out['text'][:160]!r}", flush=True)
    finally:
        eng.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="all", choices=["exact", "smc", "mixed", "all"]
    )
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    modes = (
        ["exact", "smc", "mixed"] if args.mode == "all" else [args.mode]
    )
    for m in modes:
        if m == "mixed":
            run_mixed(args.gpu)
        else:
            run_single_mode(m, args.gpu)
    print("SMOKE_OK", flush=True)

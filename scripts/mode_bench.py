"""Throughput comparison across verification modes / cycle schedules.

Single-stream decode (one group), fixed output length via ignore_eos, so
tok/s differences isolate the per-cycle cost and the committed-tokens-per-
cycle of each mode.

Usage:
    python scripts/mode_bench.py [--gpu 2] [--tokens 512] [--n 8] [--gamma 8]
"""

import argparse
import time

PROMPT = (
    "Explain, step by step, how a two-stage rocket reaches low earth "
    "orbit, covering thrust, staging, and orbital insertion."
)


def bench(mode, gpu, tokens, n, gamma, mode_cycles=None, label=None):
    from smcsd import SMCEngine

    label = label or mode
    eng = SMCEngine(
        model_path="Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        mode=mode,
        mode_cycles=mode_cycles,
        n_particles=n,
        gamma=gamma,
        draft_temperature=0.7,
        target_temperature=0.7,
        mem_fraction_static=0.4,
        base_gpu_id=gpu,
        max_running_requests=4,
        log_level="error",
    )
    sp = {"max_new_tokens": tokens, "ignore_eos": True}
    try:
        eng.generate(PROMPT, {"max_new_tokens": 64, "ignore_eos": True})  # warmup
        t0 = time.perf_counter()
        out = eng.generate(PROMPT, sp)
        dt = time.perf_counter() - t0
        ct = out["completion_tokens"]
        stats = out.get("smc_mode_stats")
        print(
            f"[{label:>14}] {ct} tok in {dt:.2f}s = {ct / dt:7.1f} tok/s"
            f"   stats={stats}",
            flush=True,
        )
    finally:
        eng.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument(
        "--only", default=None,
        choices=[None, "smc", "exact", "sched42", "sched-mixedgraph"],
    )
    args = ap.parse_args()

    runs = {
        "smc": lambda: bench("smc", args.gpu, args.tokens, args.n, args.gamma),
        "exact": lambda: bench(
            "exact", args.gpu, args.tokens, args.n, args.gamma
        ),
        "sched42": lambda: bench(
            "mixed", args.gpu, args.tokens, args.n, args.gamma,
            mode_cycles=[("exact", 4), ("smc", 2)], label="exact4/smc2",
        ),
    }
    for name, fn in runs.items():
        if args.only in (None, name):
            fn()
    print("BENCH_DONE", flush=True)

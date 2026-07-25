"""Kernel-level breakdown of an SMC decode cycle at a chosen context length.

Uses the scheduler-side torch profiler (the scheduler is a subprocess, so a
profiler in the driver sees nothing), then aggregates the resulting trace by
kernel name so we can see how much of a long-context cycle is draft-decode
attention vs verify attention vs GEMMs.

  python scripts/ctx_kernel_breakdown.py --gpu 4 --ctx 16384
"""

import argparse
import glob
import gzip
import json
import os
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="exact", choices=["smc", "exact", "auto"])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--outdir", default="/tmp/smcprof")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for f in glob.glob(os.path.join(args.outdir, "*")):
        os.remove(f)
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = args.outdir

    from smcsd import SMCEngine

    eng = SMCEngine(
        model_path=args.model,
        draft_model_path=args.draft,
        mode=args.mode,
        n_particles=args.n,
        gamma=args.gamma,
        draft_temperature=0.7,
        target_temperature=0.7,
        mem_fraction_static=0.75,
        base_gpu_id=args.gpu,
        max_running_requests=2,
        context_length=args.ctx + args.tokens + 512,
        log_level="error",
    )
    sp = {"ignore_eos": True, "temperature": 0.7}
    prompt = "The quick brown fox jumps over the lazy dog. " * (args.ctx // 10)
    try:
        eng.generate([prompt], {**sp, "max_new_tokens": 16})  # warm + prefill
        eng.start_profile(output_dir=args.outdir, activities=["GPU"])
        t0 = time.perf_counter()
        out = eng.generate([prompt], {**sp, "max_new_tokens": args.tokens})
        dt = time.perf_counter() - t0
        eng.stop_profile()
        time.sleep(5)
        st = out[0].get("smc_mode_stats") or {}
        cyc = (st.get("smc_cycles") or 0) + (st.get("exact_cycles") or 0)
        print(f"ctx={args.ctx} {args.tokens} tok in {dt:.3f}s, cycles={cyc}")
    finally:
        eng.shutdown()

    traces = sorted(glob.glob(os.path.join(args.outdir, "*.trace.json*")))
    print(f"traces: {[os.path.basename(t) for t in traces]}")
    if not traces:
        return
    agg = defaultdict(lambda: [0.0, 0])
    for tr in traces:
        op = gzip.open if tr.endswith(".gz") else open
        with op(tr, "rt") as fh:
            data = json.load(fh)
        for ev in data.get("traceEvents", []):
            if ev.get("ph") != "X":
                continue
            cat = ev.get("cat", "")
            if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            name = ev["name"]
            agg[name][0] += ev.get("dur", 0)
            agg[name][1] += 1
    total = sum(v[0] for v in agg.values())
    print(f"\ntotal GPU kernel time in window: {total / 1000:.1f} ms "
          f"over {cyc} cycles = {total / 1000 / max(cyc, 1):.2f} ms/cycle\n")
    print(f"{'us total':>10} {'%':>6} {'calls':>7}  kernel")
    for name, (dur, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:28]:
        print(f"{dur:>10.0f} {100 * dur / total:>5.1f}% {n:>7}  {name[:88]}")


if __name__ == "__main__":
    main()

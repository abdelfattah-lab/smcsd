"""How does one SMC decode cycle scale with context length?

Generates a fixed number of tokens from prompts of increasing length and
reports ms/cycle.  If the cycle cost grows ~linearly in context, decode is
KV-bandwidth bound and the N-fold per-particle re-read of the shared prefix
is the thing to attack; if it is flat, attention is not the bottleneck.

  python scripts/ctx_scaling_probe.py --gpu 4 --mode exact
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smc", choices=["smc", "exact", "auto"])
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[256, 1024, 4096, 8192, 16384])
    args = ap.parse_args()

    from smcsd import SMCEngine

    kwargs = dict(
        model_path=args.model,
        draft_model_path=args.draft,
        n_particles=args.n,
        gamma=args.gamma,
        draft_temperature=0.7,
        target_temperature=0.7,
        mem_fraction_static=0.75,
        base_gpu_id=args.gpu,
        max_running_requests=2,
        context_length=max(args.ctx) + args.tokens + 512,
        log_level="error",
    )
    # `mode` only exists on the exact/approx branches; main is SMC-only.
    try:
        eng = SMCEngine(mode=args.mode, **kwargs)
    except TypeError:
        if args.mode != "smc":
            raise SystemExit(
                f"--mode {args.mode} needs a branch that supports it; "
                "this checkout is SMC-only"
            )
        eng = SMCEngine(**kwargs)
    sp = {"ignore_eos": True, "temperature": 0.7}
    try:
        eng.generate(["Count slowly:"], {**sp, "max_new_tokens": 32})  # warm
        # Two generation lengths per context: differencing them cancels the
        # one-off prompt prefill, which otherwise dominates the long-context
        # rows (a 16k prefill amortized over ~15 cycles is worth >10ms/cycle
        # of pure artifact).
        short, long_ = args.tokens, args.tokens * 4
        print(f"{'ctx':>7} {'ms/cycle':>9} {'ms/1k tok':>10} {'tok/s':>8} "
              f"{'prefill s':>10} {'tok/cycle':>9}")
        base = None
        for ctx in args.ctx:
            # A prompt of ~ctx tokens (word-ish tokens keep this close enough).
            prompt = "The quick brown fox jumps over the lazy dog. " * (ctx // 10)
            pts = []
            for toks in (short, long_):
                t0 = time.perf_counter()
                out = eng.generate([prompt], {**sp, "max_new_tokens": toks})
                dt = time.perf_counter() - t0
                st = out[0].get("smc_mode_stats") or {}
                cyc = (st.get("smc_cycles") or 0) + (st.get("exact_cycles") or 0)
                pts.append((dt, cyc, toks))
            (d0, c0, t0n), (d1, c1, t1n) = pts
            # Cycle counters only exist on branches that expose
            # smc_mode_stats; ms/1k-token is the mode-independent fallback
            # and is what the base ratio is computed from.
            ms = 1000 * (d1 - d0) / (c1 - c0) if c1 > c0 else float("nan")
            per_1k = 1000 * (d1 - d0) / (t1n - t0n)
            prefill = d0 - (c0 * ms / 1000) if c1 > c0 else d0 - (
                t0n * per_1k / 1000)
            tps = (t1n - t0n) / (d1 - d0)
            if base is None:
                base = per_1k
            print(f"{ctx:>7} {ms:>9.2f} {per_1k:>10.1f} {tps:>8.1f} "
                  f"{prefill:>10.2f} "
                  f"{((t1n - t0n) / (c1 - c0) if c1 > c0 else 0):>9.2f}"
                  f"   ({per_1k / base:.2f}x base)")
    finally:
        eng.shutdown()
    print("PROBE_DONE")


if __name__ == "__main__":
    main()

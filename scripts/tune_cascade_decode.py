"""Sweep cascade-decode launch config against the stock grouped decode.

The cascade kernel wins big at long context but LOSES below ~2k, which is
what forces regime-keyed dual CUDA-graph capture (dispatch cannot branch
inside a captured graph).  If a single config never loses, the draft loop
can capture one graph unconditionally and the integration gets much simpler.

  python scripts/tune_cascade_decode.py --gpu 5
"""

import argparse
import itertools
import time

import torch

from smcsd.core.kernels.cascade_decode import cascade_decode_fwd
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_normal,
)

# (H_q, H_kv, d, layers, label) — the draft models we actually run.
SHAPES = [
    (32, 8, 64, 16, "Llama-3.2-1B"),
    (16, 8, 128, 28, "Qwen3-0.6B"),
]
CTXS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_graph(fn, per_graph=10, iters=20, warmup=15):
    """Time ``fn`` as it actually runs in production: inside a CUDA graph.

    Eager timing charges every call ~10us of launch dispatch, which at short
    context is larger than the attention itself and makes any multi-launch
    kernel look bad for reasons that vanish under capture.  ``per_graph``
    calls per graph amortize the replay overhead the same way the real
    cycle graph does (one replay covers all layers).
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(per_graph):
            fn()
    torch.cuda.synchronize()

    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / per_graph * 1e6


def make_case(N, h_q, h_kv, d, L0, suffix, n_groups, dev, scatter=True):
    """Build one decode case.

    ``scatter`` models what page_size=1 actually gives you: the group's
    shared prefix is a list of L0 pool rows scattered over a fragmented
    pool, not a contiguous run.  Benching against a contiguous range makes
    both kernels look bandwidth-bound when the real one is gather-latency
    bound, and badly overstates cascade's win.
    """
    bs = n_groups * N
    S = L0 + suffix
    pool = bs * S + 16
    k = torch.randn(pool, h_kv, d, dtype=torch.bfloat16, device=dev)
    v = torch.randn(pool, h_kv, d, dtype=torch.bfloat16, device=dev)
    q = torch.randn(bs, h_q, d, dtype=torch.bfloat16, device=dev)
    o = torch.empty(bs, h_q, d, dtype=torch.bfloat16, device=dev)
    seq_lens = torch.full((bs,), S, dtype=torch.int32, device=dev)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(seq_lens, 0)
    idx = torch.empty(bs * S, dtype=torch.int32, device=dev)
    perm = (torch.randperm(pool, device=dev).to(torch.int32) if scatter
            else torch.arange(pool, dtype=torch.int32, device=dev))
    for g in range(n_groups):
        base_g = g * (L0 + N * suffix)
        shared = perm[base_g: base_g + L0]
        for n in range(N):
            b = g * N + n
            idx[b * S: b * S + L0] = shared
            base = base_g + L0 + n * suffix
            idx[b * S + L0: (b + 1) * S] = perm[base: base + suffix]
    shared_lens = torch.full((bs,), L0, dtype=torch.int32, device=dev)
    return dict(q=q, o=o, k=k, v=v, kv_indptr=kv_indptr, idx=idx,
                shared_lens=shared_lens, bs=bs, sm=d ** -0.5, h_q=h_q, d=d)


def stock_fn(c, dev, n_splits=16):
    attn_logits = torch.empty((c["bs"], c["h_q"], n_splits, c["d"]),
                              dtype=torch.float32, device=dev)
    attn_lse = torch.empty((c["bs"], c["h_q"], n_splits),
                           dtype=torch.float32, device=dev)
    num_kv_splits = torch.full((c["bs"],), n_splits, dtype=torch.int32,
                               device=dev)

    def run():
        decode_attention_fwd_normal(
            c["q"], c["k"], c["v"], c["o"], c["kv_indptr"], c["idx"],
            attn_logits, attn_lse, num_kv_splits, n_splits, c["sm"], 1.0, 0.0)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--suffix", type=int, default=8)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--contiguous", action="store_true",
                    help="unrealistic contiguous shared range (overstates it)")
    ap.add_argument("--eager", action="store_true",
                    help="time eager launches instead of graph replay")
    args = ap.parse_args()
    timer = bench if args.eager else bench_graph
    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(dev)

    # splits+1 must be pow2 for the stage-2 merge.
    configs = [
        {"splits": s, "block_n": bn, "num_warps": w, "num_stages": st}
        for s, bn, w, st in itertools.product(
            (7, 15, 31, 63), (32, 64, 128), (4, 8), (2, 3))
    ]

    for h_q, h_kv, d, layers, label in SHAPES:
        print(f"\n{'=' * 78}\n{label}: H_q={h_q} H_kv={h_kv} d={d} "
              f"N={args.n} suffix={args.suffix} groups={args.groups} "
          f"[{'eager' if args.eager else 'cuda-graph'}]\n"
              f"{'=' * 78}")
        cases, stock_us = {}, {}
        for L0 in CTXS:
            c = make_case(args.n, h_q, h_kv, d, L0, args.suffix,
                          args.groups, dev, scatter=not args.contiguous)
            cases[L0] = c
            stock_us[L0] = timer(stock_fn(c, dev))
        print("stock:  " + "  ".join(f"{L0}:{stock_us[L0]:.0f}us"
                                     for L0 in CTXS))

        results = []
        for cfg in configs:
            us = {}
            ok = True
            for L0 in CTXS:
                c = cases[L0]
                try:
                    us[L0] = timer(lambda c=c, cfg=cfg: cascade_decode_fwd(
                        c["q"], c["o"], c["k"], c["v"], c["kv_indptr"],
                        c["idx"], c["shared_lens"], args.n, c["sm"], **cfg))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            speedups = [stock_us[L0] / us[L0] for L0 in CTXS]
            results.append((min(speedups), speedups, cfg, us))

        results.sort(key=lambda r: -r[0])
        print(f"\n{'worst':>6} " + " ".join(f"{L0:>7}" for L0 in CTXS)
              + "   config")
        for worst, sp, cfg, us in results[:8]:
            row = " ".join(f"{s:>6.2f}x" for s in sp)
            print(f"{worst:>5.2f}x {row}   "
                  f"S={cfg['splits']} BN={cfg['block_n']} "
                  f"W={cfg['num_warps']} St={cfg['num_stages']}")
        best = results[0]
        print(f"\nbest worst-case: {best[0]:.2f}x with {best[2]}")
        print("  us: " + "  ".join(f"{L0}:{best[3][L0]:.0f}" for L0 in CTXS))


if __name__ == "__main__":
    main()

"""Multi-group batched throughput benchmark.

Sends G identical long prompts concurrently to the SMC engine.  Each
group has N particles, so the active particle count is G*N.  This is the
regime where cascade attention's "shared prefix read once" wins over
fa3, which has no cross-request prefix sharing.

Compares fa3 vs flashinfer+cascade in eager mode (cuda-graph capture for
multi-group cascade is a separate work-item; this benchmark isolates the
kernel-level comparison).

Usage:
  python scripts/bench_multi_group.py --groups 4 --target-len 8192
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from smcsd.engine import SMCEngine
from bench_long_prompt import PASSAGE, build_prompt


def run_one(
    *, model: str, draft: str, prompts: List[str], cascade: bool,
    n_particles: int, gamma: int, max_new: int, mem_fraction: float, seed: int,
    disable_cuda_graph: bool,
) -> Tuple[float, int]:
    backend = "flashinfer" if cascade else "fa3"
    n_groups = len(prompts)
    mode_str = "eager" if disable_cuda_graph else "graphs"
    print(f"\n--- Booting cascade={cascade} attn={backend} {mode_str}, G={n_groups} ---")
    torch.manual_seed(seed)
    engine = SMCEngine(
        model_path=model,
        draft_model_path=draft,
        n_particles=n_particles,
        gamma=gamma,
        draft_temperature=0.7,
        target_temperature=0.7,
        resample_threshold=0.5,
        shared_prefix_attn=cascade,
        attention_backend=backend,
        mem_fraction_static=mem_fraction,
        # Allow up to G concurrent groups (each with N particles).
        max_running_requests=n_groups,
        disable_cuda_graph=disable_cuda_graph,
    )
    sp = {"temperature": 0.7, "max_new_tokens": max_new}
    # warmup
    _ = engine.generate(prompts, sampling_params=sp)
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params=sp)
    dt = time.perf_counter() - t0
    if not isinstance(outputs, list):
        outputs = [outputs]
    total_new = sum(o["completion_tokens"] for o in outputs)
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return dt, total_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--target-len", type=int, default=4096)
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--particles", type=int, default=12)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-cuda-graph", action="store_true",
                    help="run both backends in eager mode (apples-to-apples kernel)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    base_prompt = build_prompt(
        tok, args.target_len,
        question="What three pillars supported Roman dominance?",
    )
    actual_len = len(tok.encode(base_prompt))
    print(f"base prompt actual_len: {actual_len} tokens (target {args.target_len})")
    print(f"settings: N={args.particles} γ={args.gamma} max_new={args.max_new} "
          f"groups={args.groups} mode={'eager' if args.disable_cuda_graph else 'graphs'}")

    rows = []
    for G in args.groups:
        prompts = [base_prompt] * G
        t_fa3, n_fa3 = run_one(
            model=args.model, draft=args.draft, prompts=prompts, cascade=False,
            n_particles=args.particles, gamma=args.gamma, max_new=args.max_new,
            mem_fraction=args.mem_fraction, seed=args.seed,
            disable_cuda_graph=args.disable_cuda_graph,
        )
        t_cas, n_cas = run_one(
            model=args.model, draft=args.draft, prompts=prompts, cascade=True,
            n_particles=args.particles, gamma=args.gamma, max_new=args.max_new,
            mem_fraction=args.mem_fraction, seed=args.seed,
            disable_cuda_graph=args.disable_cuda_graph,
        )
        rows.append((G, n_fa3, t_fa3, n_cas, t_cas))

    print("\n" + "=" * 80)
    print(f"  Multi-group throughput  (L≈{actual_len}, N={args.particles}, "
          f"γ={args.gamma}, mode={'eager' if args.disable_cuda_graph else 'graphs'})")
    print("=" * 80)
    print(f"  {'G':>2}  {'fa3 tok/s':>10}  {'cas tok/s':>10}  {'speedup':>8}  "
          f"{'fa3 wall':>9}  {'cas wall':>9}")
    print("  " + "-" * 70)
    for G, n_fa3, t_fa3, n_cas, t_cas in rows:
        sp = (n_cas / t_cas) / (n_fa3 / t_fa3)
        marker = "🚀" if sp >= 1.10 else ("≈" if sp >= 0.95 else "  ")
        print(f"  {G:>2}  {n_fa3 / t_fa3:>10.1f}  {n_cas / t_cas:>10.1f}  "
              f"{sp:>7.2f}x  {t_fa3:>8.2f}s  {t_cas:>8.2f}s  {marker}")
    print("=" * 80)


if __name__ == "__main__":
    main()

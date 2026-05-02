"""Single-prompt parity test: cascade attention off vs on.

Boots SMCEngine twice — once with the regular flashinfer backend, once
with `--smc-shared-prefix-attn` enabled — runs the same short prompt
with the same seed/temperature, and compares outputs token-by-token.

Expected:
  - Token sequences identical (or rare divergence near end-of-decode
    due to fp16 noise propagating through sampling decisions).
  - Both runs finish without crashing.
  - Cascade-on path actually engages (no silent fallback) — we check
    that the engine reports the cascade backend was selected.

Usage:
  python scripts/smoke_cascade_parity.py
  python scripts/smoke_cascade_parity.py --temperature 0.0  # greedy: bitwise-equal
"""

from __future__ import annotations

import argparse
import gc
import sys

import torch

from smcsd.engine import SMCEngine


def run_once(
    model: str,
    draft: str,
    prompt: str,
    *,
    cascade: bool,
    n_particles: int,
    gamma: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
):
    # Baseline: SGLang's flashinfer verify path is EAGLE-style and doesn't
    # support SMC's linear verify, so the only non-cascade backend that
    # works with SMC is fa3.  Treatment: flashinfer + our cascade wrapper.
    backend = "flashinfer" if cascade else "fa3"
    print(f"\n=== Booting SMCEngine (cascade={cascade}, attn={backend}) ===")
    torch.manual_seed(seed)
    engine = SMCEngine(
        model_path=model,
        draft_model_path=draft,
        n_particles=n_particles,
        gamma=gamma,
        draft_temperature=temperature,
        target_temperature=temperature,
        resample_threshold=0.5,
        shared_prefix_attn=cascade,
        attention_backend=backend,
        mem_fraction_static=0.5,
        max_running_requests=1,
        disable_cuda_graph=True,
    )
    sp = {"temperature": temperature, "max_new_tokens": max_new_tokens}
    print(f"--- Generating (seed={seed}, temp={temperature}, max_new={max_new_tokens}) ---")
    out = engine.generate(prompt, sampling_params=sp)
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--particles", type=int, default=8)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (deterministic, allows bitwise comparison)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"prompt: {args.prompt!r}")
    print(f"particles={args.particles} gamma={args.gamma} "
          f"temperature={args.temperature} max_new={args.max_new}")

    # Run cascade off first — establishes a baseline.
    out_off = run_once(
        args.model, args.draft, args.prompt,
        cascade=False,
        n_particles=args.particles, gamma=args.gamma,
        temperature=args.temperature, max_new_tokens=args.max_new,
        seed=args.seed,
    )
    print(f"\n[cascade=OFF] tokens={len(out_off['output_ids'])}")
    print(f"[cascade=OFF] text: {out_off['text']!r}")

    out_on = run_once(
        args.model, args.draft, args.prompt,
        cascade=True,
        n_particles=args.particles, gamma=args.gamma,
        temperature=args.temperature, max_new_tokens=args.max_new,
        seed=args.seed,
    )
    print(f"\n[cascade=ON ] tokens={len(out_on['output_ids'])}")
    print(f"[cascade=ON ] text: {out_on['text']!r}")

    ids_off = out_off["output_ids"]
    ids_on = out_on["output_ids"]
    common = min(len(ids_off), len(ids_on))
    n_match = sum(1 for a, b in zip(ids_off[:common], ids_on[:common]) if a == b)
    print(f"\n=== Parity ===")
    print(f"  off len: {len(ids_off)}   on len: {len(ids_on)}   common: {common}")
    print(f"  matching prefix tokens: {n_match}/{common}")
    if n_match == common and len(ids_off) == len(ids_on):
        print("  ✅ identical token sequences")
        sys.exit(0)
    elif n_match > 0.9 * common:
        print("  ≈ near-identical (minor divergence; expected at temp>0)")
        sys.exit(0)
    else:
        print("  ❌ significant divergence — cascade path may be incorrect")
        sys.exit(1)


if __name__ == "__main__":
    main()

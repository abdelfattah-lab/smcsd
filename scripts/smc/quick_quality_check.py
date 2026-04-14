"""Quick output quality check — vanilla vs SMC.

Usage:
  python scripts/smc/quick_quality_check.py
  python scripts/smc/quick_quality_check.py --temperature 0.8
  python scripts/smc/quick_quality_check.py --temperature 0.0 --mode smc
  python scripts/smc/quick_quality_check.py --mode both
"""

import argparse
import os
import sglang as sgl

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
PROMPTS = [
    "The capital of France is",
    "Write one sentence about why speculative decoding matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
    "What is 1+1?",
]


def run_vanilla(prompts, sampling_params):
    print("=" * 60)
    print("VANILLA (no spec decode)")
    print("=" * 60)
    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        mem_fraction_static=0.45,
        attention_backend="triton",
    )
    results = engine.generate(prompts, sampling_params)
    for i, r in enumerate(results):
        print(f"  OUTPUT_{i+1}: {r['text'][:200]}")
    engine.shutdown()
    print()


def run_smc(prompts, sampling_params, args):
    print("=" * 60)
    print(
        f"SMC (particles={args.particles}, gamma={args.gamma})"
    )
    print("=" * 60)
    engine = sgl.Engine(
        model_path=TARGET_MODEL,
        speculative_algorithm="SMC",
        speculative_draft_model_path=DRAFT_MODEL,
        smc_n_particles=args.particles,
        smc_gamma=args.gamma,
        smc_draft_temperature=max(args.temperature, 0.01),
        smc_target_temperature=max(args.temperature, 0.01),
        mem_fraction_static=0.45,
        disable_piecewise_cuda_graph=False,
        cuda_graph_max_bs=16,
        attention_backend="triton",
    )
    results = engine.generate(prompts, sampling_params)
    for i, r in enumerate(results):
        print(f"  OUTPUT_{i+1}: {r['text']}")
    engine.shutdown()
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mode", choices=["vanilla", "smc", "both"], default="both")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        #"ignore_eos": True,
    }
    print(f"Sampling: temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")
    print()

    if args.mode in ("vanilla", "both"):
        run_vanilla(PROMPTS, sampling_params)

    if args.mode in ("smc", "both"):
        run_smc(PROMPTS, sampling_params, args)


if __name__ == "__main__":
    main()

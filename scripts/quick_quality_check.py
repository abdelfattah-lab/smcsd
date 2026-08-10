"""Quick output quality check — vanilla vs SMC.

Usage:
  python scripts/smc/quick_quality_check.py
  python scripts/smc/quick_quality_check.py --temperature 0.8
  python scripts/smc/quick_quality_check.py --temperature 0.0 --mode smc
  python scripts/smc/quick_quality_check.py --mode both
"""

import argparse
import sglang as sgl
from smcsd import SMCEngine

DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
PROMPTS = [
    "The capital of France is",
    "Write one sentence about why overlap scheduling matters for inference systems.",
    "List two prime numbers and one composite number.",
    "In one short paragraph, explain speculative decoding.",
    "What is 1+1?",
]


def run_vanilla(prompts, sampling_params, args):
    print("=" * 60)
    print("VANILLA (no spec decode)")
    print("=" * 60)
    engine = sgl.Engine(
        model_path=args.model_path,
        mem_fraction_static=0.45,
        attention_backend="triton",
        tp_size=args.tp,
        base_gpu_id=args.base_gpu_id,
    )
    results = engine.generate(prompts, sampling_params)
    for i, r in enumerate(results):
        print(f"  OUTPUT_{i+1}: {r['text'][:200]}")
    engine.shutdown()
    print()


def run_smc(prompts, sampling_params, args):
    print("=" * 60)
    print(f"SMC (particles={args.particles}, gamma={args.gamma})")
    print("=" * 60)
    engine = SMCEngine(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=max(args.temperature, 0.01),
        target_temperature=max(args.temperature, 0.01),
        mem_fraction_static=0.45,
        cuda_graph_max_bs=16,
        attention_backend="triton",
        tp_size=args.tp,
        base_gpu_id=args.base_gpu_id,
        cross_tokenizer=args.cross_tokenizer,
        draft_tokenizer_path=args.draft_tokenizer_path,
        cross_tokenizer_artifact_path=args.cross_tokenizer_artifact_path,
        cross_tokenizer_mode=args.cross_tokenizer_mode,
    )
    results = engine.generate(prompts, sampling_params)
    for i, r in enumerate(results):
        print(f"  OUTPUT_{i+1}: {r['text']}")
    engine.shutdown()
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--draft-model-path", default=DEFAULT_DRAFT_MODEL_PATH)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mode", choices=["vanilla", "smc", "both"], default="both")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--cross-tokenizer", action="store_true")
    parser.add_argument("--draft-tokenizer-path", default=None)
    parser.add_argument("--cross-tokenizer-artifact-path", default=None)
    parser.add_argument("--cross-tokenizer-mode", choices=["live", "hybrid"], default="hybrid")
    args = parser.parse_args()

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        #"ignore_eos": True,
    }
    print(f"Sampling: temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")
    print()

    if args.mode in ("vanilla", "both"):
        run_vanilla(PROMPTS, sampling_params, args)

    if args.mode in ("smc", "both"):
        run_smc(PROMPTS, sampling_params, args)


if __name__ == "__main__":
    main()

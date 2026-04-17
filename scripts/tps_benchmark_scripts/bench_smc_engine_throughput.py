"""Throughput micro-benchmark for SMC v2 via the dedicated SMCEngine.

A drop-in replacement for the old:
    python -m sglang.bench_offline_throughput --speculative-algorithm SMC ...

That code path went away with the v1 retirement (no SMC branch in
SpeculativeAlgorithm.create_worker anymore), so this script wraps
``SMCEngine.generate(...)`` directly.

Generates ``--num-prompts`` random-token inputs of length
``--random-input-len``, asks for ``--random-output-len`` new tokens each
(``ignore_eos=True`` for stable throughput), then prints the same
"Output token throughput: <X> tok/s" line that the old harness emitted
so the calling shell scripts can keep grepping for it.

Example:
    python -O bench_smc_engine_throughput.py \\
        --model-path meta-llama/Llama-3.1-8B-Instruct \\
        --draft-model-path meta-llama/Llama-3.2-1B-Instruct \\
        --smc-n-particles 8 --smc-gamma 8 \\
        --smc-draft-temperature 0.7 --smc-target-temperature 0.7 \\
        --attention-backend triton \\
        --mem-fraction-static 0.6 \\
        --max-running-requests 64 --cuda-graph-max-bs 64 \\
        --random-input-len 256 --random-output-len 512 --num-prompts 8
"""

import argparse
import time

import numpy as np

from smcsd.engine import SMCEngine


def _make_random_token_inputs(
    num_prompts: int, input_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Sample ``num_prompts`` token-id sequences of length ``input_len``.

    `bench_offline_throughput`'s random dataset uses uniform sampling over
    the vocab, which is good enough for measuring decode TPS — the prompt
    content doesn't affect throughput once warmup is done.
    """
    rng = np.random.default_rng(seed)
    return [
        rng.integers(low=1, high=vocab_size, size=input_len, dtype=np.int64).tolist()
        for _ in range(num_prompts)
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-model-path", required=True)
    # SMC sweep knobs (mirrors --smc-* flags accepted by sgl.Engine)
    p.add_argument("--smc-n-particles", type=int, required=True)
    p.add_argument("--smc-gamma", type=int, required=True)
    p.add_argument("--smc-draft-temperature", type=float, default=0.7)
    p.add_argument("--smc-target-temperature", type=float, default=0.7)
    p.add_argument(
        "--smc-resample-threshold", type=float, default=None,
        help="Override SMCEngine resample_threshold (default: engine default).",
    )
    p.add_argument(
        "--smc-fast-resample", action="store_true",
        help="Enable the fused-collect/fused-KV resample path.",
    )
    # Engine / runtime knobs
    p.add_argument("--attention-backend", choices=["triton", "fa3"], default="triton")
    p.add_argument("--mem-fraction-static", type=float, default=0.6)
    p.add_argument("--max-running-requests", type=int, default=None)
    p.add_argument("--cuda-graph-max-bs", type=int, default=None)
    p.add_argument("--tp", type=int, default=1, dest="tp_size",
                   help="Tensor-parallel size for the target model.")
    # Workload
    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--random-output-len", type=int, default=512)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="Used only for random token sampling; default 32000.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine_kwargs: dict = dict(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        n_particles=args.smc_n_particles,
        gamma=args.smc_gamma,
        draft_temperature=args.smc_draft_temperature,
        target_temperature=args.smc_target_temperature,
        attention_backend=args.attention_backend,
        page_size=1,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=args.seed,
        tp_size=args.tp_size,
        trust_remote_code=True,
    )
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.smc_resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.smc_resample_threshold
    if args.smc_fast_resample:
        engine_kwargs["smc_fast_resample"] = True

    input_ids = _make_random_token_inputs(
        num_prompts=args.num_prompts,
        input_len=args.random_input_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    sampling_params = {
        "max_new_tokens": args.random_output_len,
        # Force fixed output length for stable throughput numbers.
        "ignore_eos": True,
        "temperature": args.smc_target_temperature,
    }

    print(
        f"[bench_smc_engine_throughput] num_prompts={args.num_prompts} "
        f"input_len={args.random_input_len} output_len={args.random_output_len} "
        f"n_particles={args.smc_n_particles} gamma={args.smc_gamma} "
        f"attention={args.attention_backend} fast_resample={args.smc_fast_resample}"
    )

    with SMCEngine(**engine_kwargs) as engine:
        # Warmup: a single short prompt amortises CUDA graph capture and JIT
        # compilation so they don't pollute the measured wall time.
        _ = engine.generate(
            input_ids=input_ids[0],
            sampling_params={"max_new_tokens": 16, "ignore_eos": True},
        )

        tic = time.perf_counter()
        outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
        latency = time.perf_counter() - tic

    if not isinstance(outputs, list):
        outputs = [outputs]
    total_output_tokens = sum(o["completion_tokens"] for o in outputs)
    tps = total_output_tokens / latency if latency > 0 else 0.0

    print()
    print("=" * 60)
    print(f"  SMCEngine throughput sweep")
    print("=" * 60)
    print(f"  Number of prompts:       {args.num_prompts}")
    print(f"  Input length:            {args.random_input_len}")
    print(f"  Output length (max):     {args.random_output_len}")
    print(f"  Total output tokens:     {total_output_tokens}")
    print(f"  Wall time:               {latency:.2f}s")
    # NB: keep this exact phrasing — the calling shell scripts grep for
    # "Output token throughput" to extract the TPS value.
    print(f"  Output token throughput: {tps:.2f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()

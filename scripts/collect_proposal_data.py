"""Collect SMC rollouts for proposal (draft-model) finetuning.

Runs the offline SMCEngine over a prompt set and dumps, per request, the
full particle collection the engine already emits on its side channel:
every particle's output token ids, the final per-particle log-weights
(log w_tilde), the unbiased log Z_hat estimate, and the decode-cycle
diagnostics (n_cycles / n_resamples / mean ESS).

The dump is consumed by ``scripts/train_proposal.py``, which finetunes the
draft model toward the tempered target distribution the SMC weights are
computed against.  Collect with the SAME SMC config (N, gamma, temperatures,
power alpha) you deploy with, so the trajectories match the proposal's
deployment distribution.

Output format (JSONL):
  line 0:   {"meta": {...engine + dataset config...}}
  line 1..: {"i": int, "question": str, "prompt_ids": [int],
             "particle_output_ids": [[int] x N], "log_w_tilde": [float x N],
             "log_Z_hat": float, "n_cycles": int, "n_resamples": int,
             "mean_ess": float|null, "picked_output_ids": [int]}

Usage:
  python scripts/collect_proposal_data.py \
      --output /data/proposal_data/gsm8k_train_N8g8.jsonl \
      -N 8 -g 8 --num-prompts 2000 --batch-size 16
"""

import argparse
import json
import os
import sys
import time

# Mean-ESS tracking is the point of collection; enable it before the engine
# forks the scheduler subprocess (which inherits the environment).
os.environ.setdefault("SMC_TRACK_ESS", "1")

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_test_gsm8k import format_instruction  # noqa: E402

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_prompts(tokenizer, args):
    """Build chat-template prompts.  GSM8K *train* split by default — the
    eval harness uses the test split, so training data stays disjoint."""
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split=args.split)
        n = min(args.num_prompts, len(dataset))
        questions = [s["question"] for s in dataset.select(range(n))]
        instructions = [format_instruction(q) for q in questions]
    else:  # jsonl file with {"question": ...} or {"prompt": ...} per line
        questions, instructions = [], []
        with open(args.dataset) as f:
            for line in f:
                if len(questions) >= args.num_prompts:
                    break
                rec = json.loads(line)
                q = rec.get("question") or rec["prompt"]
                questions.append(q)
                instructions.append(format_instruction(q))

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ins}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ins in instructions
    ]
    return questions, prompts


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions, prompts = load_prompts(tokenizer, args)
    # Encode here exactly like SMCEngine.generate does for str prompts, and
    # pass input_ids through — the recorded prompt_ids are then guaranteed
    # to be the ids the engine conditioned on.
    prompt_ids = [tokenizer.encode(p) for p in prompts]
    print(f"Collecting over {len(prompts)} prompts "
          f"({args.dataset}/{args.split}), N={args.particles} γ={args.gamma}")

    from smcsd.engine import SMCEngine

    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.draft_temperature,
        target_temperature=args.target_temperature,
        power_alpha=args.power_alpha,
        resample_threshold=args.resample_threshold,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        base_gpu_id=args.base_gpu_id,
        max_running_requests=args.max_running_requests,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.disable_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True

    meta = {
        "model": args.model,
        "draft_model": args.draft_model,
        "n_particles": args.particles,
        "gamma": args.gamma,
        "draft_temperature": args.draft_temperature,
        "target_temperature": args.target_temperature,
        "power_alpha": args.power_alpha,
        "resample_threshold": args.resample_threshold,
        "max_new_tokens": args.max_new_tokens,
        "dataset": args.dataset,
        "split": args.split,
        "num_prompts": len(prompts),
        "seed": args.seed,
    }

    sampling_params = {"max_new_tokens": args.max_new_tokens}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_written = 0
    sum_resample_rate = 0.0
    sum_ess = 0.0
    n_ess = 0
    total_particle_tokens = 0

    with SMCEngine(**engine_kwargs) as engine, open(args.output, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            ids_batch = prompt_ids[start : start + args.batch_size]
            outputs = engine.generate(
                input_ids=ids_batch, sampling_params=sampling_params
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, out in enumerate(outputs):
                qi = start + i
                if out.get("smc_particle_output_ids") is None:
                    print(f"\nWARNING: no particle collection for prompt {qi} "
                          "(aborted?) — skipping", flush=True)
                    continue
                rec = {
                    "i": qi,
                    "question": questions[qi],
                    "prompt_ids": ids_batch[i],
                    "particle_output_ids": out["smc_particle_output_ids"],
                    "log_w_tilde": out["smc_log_w_tilde"],
                    "log_Z_hat": out["smc_log_Z_hat"],
                    "n_cycles": out["smc_n_cycles"],
                    "n_resamples": out["smc_n_resamples"],
                    "mean_ess": out["smc_mean_ess"],
                    "picked_output_ids": out["output_ids"],
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                if rec["n_cycles"] > 0:
                    sum_resample_rate += rec["n_resamples"] / rec["n_cycles"]
                if rec["mean_ess"] is not None:
                    sum_ess += rec["mean_ess"]
                    n_ess += 1
                total_particle_tokens += sum(
                    len(p) for p in rec["particle_output_ids"]
                )
            f.flush()
            elapsed = time.perf_counter() - tic
            mean_rr = sum_resample_rate / max(n_written, 1)
            mean_ess = sum_ess / n_ess if n_ess else float("nan")
            print(
                f"\r[{min(start + args.batch_size, len(prompts))}/{len(prompts)}] "
                f"resample_rate={mean_rr:.3f} "
                f"mean_ess={mean_ess:.2f}/{args.particles} "
                f"particle_tok={total_particle_tokens} "
                f"elapsed={elapsed:.0f}s",
                end="",
                flush=True,
            )

    print(f"\nWrote {n_written} records to {args.output}")
    print(f"  mean resample rate : {sum_resample_rate / max(n_written, 1):.3f}")
    if n_ess:
        print(f"  mean ESS           : {sum_ess / n_ess:.2f} / N={args.particles}")
    print(f"  particle tokens    : {total_particle_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="output JSONL path")

    smc = parser.add_argument_group("SMC parameters (match deployment!)")
    smc.add_argument("--particles", "-N", type=int, default=8)
    smc.add_argument("--gamma", "-g", type=int, default=8)
    smc.add_argument("--draft-temperature", type=float, default=0.7)
    smc.add_argument("--target-temperature", type=float, default=0.7)
    smc.add_argument("--power-alpha", type=float, default=1.0)
    smc.add_argument("--resample-threshold", type=float, default=0.5)

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default="gsm8k",
                      help="'gsm8k' or a JSONL file with question/prompt fields")
    data.add_argument("--split", type=str, default="train",
                      help="dataset split (default: train — eval uses test)")
    data.add_argument("--num-prompts", type=int, default=1000)
    data.add_argument("--max-new-tokens", type=int, default=512)
    data.add_argument("--batch-size", type=int, default=16)
    data.add_argument("--seed", type=int, default=42)

    eng = parser.add_argument_group("engine")
    eng.add_argument("--attention-backend", type=str, default="triton",
                     choices=["triton", "fa3"])
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--max-running-requests", type=int, default=32)
    eng.add_argument("--base-gpu-id", type=int, default=0)
    eng.add_argument("--disable-cuda-graph", action="store_true", default=False)

    main(parser.parse_args())

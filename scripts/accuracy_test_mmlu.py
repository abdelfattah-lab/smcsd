"""MMLU benchmark for SMC speculative decoding.

Mirrors the structure of accuracy_test_gsm8k.py. Two modes:
  - smc_engine: SMC via the dedicated SMCEngine (offline, no tokenizer manager)
  - baseline:   vanilla generation (no speculative decoding)

Generation-based eval: the model is prompted to output a single letter
(A/B/C/D) on its final line. We extract the last A-D in the output. The
default split is `cais/mmlu` `all` test (14k questions); pass --num-questions
to subsample.

Usage:
  # baseline
  python scripts/accuracy_test_mmlu.py --mode baseline --num-questions 100

  # SMC
  python scripts/accuracy_test_mmlu.py --mode smc_engine -N 8 -g 8
"""

import argparse
import re
import time
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

CHOICES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------


def extract_choice(text: str) -> Optional[str]:
    """Extract A/B/C/D from model output.

    Order of preference:
      1. `Answer: X` line (case-insensitive)
      2. `#### X` line
      3. last standalone A-D in the text
    """
    m = re.search(r"[Aa]nswer\s*[:\-]\s*\(?([ABCD])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"####\s*\(?([ABCD])\)?", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([ABCD])\b", text)
    return matches[-1].upper() if matches else None


def format_instruction(question: str, choices: list[str]) -> str:
    a, b, c, d = choices
    return (
        "Answer the following multiple-choice question. "
        "Think step by step, then on the final line output ONLY the letter "
        "of the correct choice in the format:\n"
        "Answer: <A|B|C|D>\n\n"
        f"Question: {question}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n"
    )


def load_mmlu(tokenizer, num_questions: int, subject: Optional[str] = None, seed: int = 0):
    """Load MMLU test split and build chat-template prompts + gold letters."""
    config = subject or "all"
    print(f"Loading MMLU dataset (config={config}) ...")
    dataset = load_dataset("cais/mmlu", config, split="test")

    # Deterministic subsample so repeated runs are comparable.
    if num_questions < len(dataset):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(dataset), size=num_questions, replace=False)
        idx.sort()
        dataset = dataset.select(idx.tolist())

    prompts, labels = [], []
    for sample in dataset:
        instruction = format_instruction(sample["question"], sample["choices"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(CHOICES[int(sample["answer"])])
    return prompts, labels


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


def run_smc_engine_eval(args, prompts, labels):
    from smcsd.engine import SMCEngine

    draft_model = args.draft_model or DEFAULT_DRAFT_MODEL
    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.resample_threshold
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    else:
        engine_kwargs["max_running_requests"] = max(args.particles + 4, 16)
    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with SMCEngine(**engine_kwargs) as engine:
        preds, total_output_tokens = [], 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens, gold={labels[qi]}) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_choice(output["text"]))
                total_output_tokens += output["completion_tokens"]
            elapsed = time.perf_counter() - tic
            correct = sum(p == l for p, l in zip(preds, labels[: len(preds)]))
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    return preds, total_output_tokens, latency


def run_baseline_eval(args, prompts, labels):
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        trust_remote_code=True,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    with sgl.Engine(**engine_kwargs) as engine:
        preds, total_output_tokens = [], 0
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            for i, output in enumerate(outputs):
                qi = start + i
                if qi < 3:
                    ntok = output["meta_info"]["completion_tokens"]
                    print(f"--- Q{qi} ({ntok} tokens, gold={labels[qi]}) ---")
                    print(output["text"][:400])
                    print()
                preds.append(extract_choice(output["text"]))
                total_output_tokens += output["meta_info"]["completion_tokens"]
            elapsed = time.perf_counter() - tic
            correct = sum(p == l for p, l in zip(preds, labels[: len(preds)]))
            print(
                f"\r[{len(preds)}/{len(prompts)}] "
                f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
                f"tps={total_output_tokens / elapsed:.0f} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        latency = time.perf_counter() - tic

    return preds, total_output_tokens, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    mode_label = {
        "smc_engine": "SMCEngine (dedicated offline)",
        "baseline": "Baseline (vanilla)",
    }
    print(f"Mode: {mode_label[args.mode]} | Model: {args.model}")
    if args.mode == "smc_engine":
        draft = args.draft_model or DEFAULT_DRAFT_MODEL
        print(
            f"  particles={args.particles}, gamma={args.gamma}, "
            f"temperature={args.temperature}, draft={draft}"
        )
    print(
        f"  num_questions={args.num_questions}, max_new_tokens={args.max_new_tokens}, "
        f"subject={args.subject or 'all'}, ignore_eos={args.ignore_eos}"
    )
    print()

    if args.seed is not None:
        np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, labels = load_mmlu(
        tokenizer, args.num_questions, subject=args.subject, seed=args.seed or 0
    )

    if args.mode == "smc_engine":
        preds, total_tokens, latency = run_smc_engine_eval(args, prompts, labels)
    else:
        preds, total_tokens, latency = run_baseline_eval(args, prompts, labels)

    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    print(f"\n{'=' * 55}")
    print(f"  {mode_label[args.mode]} | MMLU ({args.subject or 'all'})")
    if args.mode == "smc_engine":
        print(f"  N={args.particles}, γ={args.gamma}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:           {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Output throughput: {total_tokens / latency:.1f} tok/s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {latency:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", choices=["baseline", "smc_engine"], default="baseline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", type=str, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--subject", type=str, default=None,
                        help="MMLU subject config (default: 'all')")

    smc_grp = parser.add_argument_group("SMC parameters")
    smc_grp.add_argument("--particles", "-N", type=int, default=4)
    smc_grp.add_argument("--gamma", "-g", type=int, default=4)
    smc_grp.add_argument("--temperature", type=float, default=0.7)
    smc_grp.add_argument("--seed", type=int, default=0)
    smc_grp.add_argument("--resample-threshold", type=float, default=None)

    bench = parser.add_argument_group("benchmark")
    bench.add_argument("--num-questions", type=int, default=100)
    bench.add_argument("--max-new-tokens", type=int, default=512)
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False)

    eng = parser.add_argument_group("engine overrides")
    eng.add_argument("--attention-backend", type=str, default="triton",
                     choices=["triton", "fa3"])
    eng.add_argument("--mem-fraction-static", type=float, default=0.4)
    eng.add_argument("--cuda-graph-max-bs", type=int, default=128)
    eng.add_argument("--max-running-requests", type=int, default=16)

    args = parser.parse_args()
    main(args)

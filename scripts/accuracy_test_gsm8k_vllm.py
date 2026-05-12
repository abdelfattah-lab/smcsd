"""GSM8K accuracy test for SMCVLLMEngine (vLLM backend).

Tests draft-model generation quality on GSM8K.
(Target model verification and resampling not yet implemented — this
measures raw draft accuracy through the SMC vLLM scheduling path.)

Usage:
  python scripts/accuracy_test_gsm8k_vllm.py
  python scripts/accuracy_test_gsm8k_vllm.py -N 8 -g 8
  python scripts/accuracy_test_gsm8k_vllm.py --num-questions 100 --max-tokens 512
  python scripts/accuracy_test_gsm8k_vllm.py --temperature 0.7
"""

import argparse
import re
import time
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from smcsd.vllm_backend.engine import SMCVLLMEngine

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ---------------------------------------------------------------------------
# GSM8K helpers (unchanged from accuracy_test_gsm8k.py)
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return numbers[-1].replace(",", "") if numbers else None


def format_instruction(question: str) -> str:
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k(tokenizer, num_questions: int):
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    prompts, labels = [], []
    for sample in dataset.select(range(num_questions)):
        instruction = format_instruction(sample["question"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        labels.append(extract_answer(sample["answer"]))
    assert all(l is not None for l in labels), "Some gold labels could not be parsed"
    return prompts, labels


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_eval(args, prompts, labels):
    engine = SMCVLLMEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        tp_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    sampling_params = {
        "draft_temperature": args.temperature,
        "target_temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    preds = []
    total_output_tokens = 0
    tic = time.perf_counter()

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        outputs = engine.generate(prompt=batch, sampling_params=sampling_params)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for i, output in enumerate(outputs):
            qi = start + i
            if qi < 3:
                ntok = output["completion_tokens"]
                print(f"--- Q{qi} ({ntok} tokens) ---")
                print(output["text"][:400])
                print()
            preds.append(extract_answer(output["text"]))
            total_output_tokens += output["completion_tokens"]

        elapsed = time.perf_counter() - tic
        correct = sum(p == l for p, l in zip(preds, labels[: len(preds)]))
        print(
            f"\r[{len(preds)}/{len(prompts)}] "
            f"acc={correct}/{len(preds)} ({correct / len(preds):.1%}) "
            f"elapsed={elapsed:.0f}s",
            end="",
            flush=True,
        )

    print()
    latency = time.perf_counter() - tic
    engine.shutdown()
    return preds, total_output_tokens, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--particles", "-N", type=int, default=1)
    parser.add_argument("--gamma", "-g", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.4)
    args = parser.parse_args()

    print(f"Model:       {args.model}")
    print(f"Draft model: {args.draft_model}")
    print(f"N={args.particles}  γ={args.gamma}  temp={args.temperature}")
    print(f"num_questions={args.num_questions}  max_tokens={args.max_tokens}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    prompts, labels = load_gsm8k(tokenizer, args.num_questions)

    preds, total_tokens, latency = run_eval(args, prompts, labels)

    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    print(f"\n{'=' * 55}")
    print(f"  SMCVLLMEngine — draft-only (no target verify yet)")
    print(f"  N={args.particles}, γ={args.gamma}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:      {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:       {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Wall time:     {latency:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

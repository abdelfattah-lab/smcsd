"""GSM8K accuracy baseline using plain vLLM LLM on the draft model.

Run this to get a reference accuracy for the draft model (1B) alone,
then compare against accuracy_test_gsm8k_vllm.py (SMC vLLM engine).

Usage:
  python scripts/accuracy_test_gsm8k_vllm_baseline.py
  python scripts/accuracy_test_gsm8k_vllm_baseline.py --num-questions 100
  python scripts/accuracy_test_gsm8k_vllm_baseline.py --temperature 0.7
"""

import argparse
import re
import time
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


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


def run_eval(args, prompts, labels):
    llm = LLM(
        model=args.draft_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        async_scheduling=False,
    )
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    preds = []
    total_output_tokens = 0
    tic = time.perf_counter()

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        raw_outputs = llm.generate(batch, sp)
        for i, out in enumerate(raw_outputs):
            qi = start + i
            text = out.outputs[0].text
            ntok = len(out.outputs[0].token_ids)
            if qi < 3:
                print(f"--- Q{qi} ({ntok} tokens) ---")
                print(text[:400])
                print()
            preds.append(extract_answer(text))
            total_output_tokens += ntok

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
    return preds, total_output_tokens, latency


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Model: {args.draft_model}")
    print(f"temp={args.temperature}  num_questions={args.num_questions}  max_tokens={args.max_tokens}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    prompts, labels = load_gsm8k(tokenizer, args.num_questions)

    preds, total_tokens, latency = run_eval(args, prompts, labels)

    correct = sum(p == l for p, l in zip(preds, labels))
    invalid = sum(p is None for p in preds)
    n = len(preds)

    print(f"\n{'=' * 55}")
    print(f"  vLLM baseline — draft model only")
    print(f"  model={args.draft_model.split('/')[-1]}, temp={args.temperature}")
    print(f"{'=' * 55}")
    print(f"  Accuracy:      {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Invalid:       {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Wall time:     {latency:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()

"""Target-only baseline for cross-tokenizer SMC research.

Runs the same few-shot GSM8K prompt through ``model.generate()`` with
*just the target* (no draft, no SMC). Lets us separate "Mistral-base is
weak at GSM8K" from "SMC layer is corrupting things".
"""

from __future__ import annotations

import argparse
import time

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cross_tokenizer_smc import format_fewshot_prompt, extract_answer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Target-only baseline: target={args.target}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    torch.manual_seed(args.seed)
    dataset = load_dataset("gsm8k", "main", split="test")

    correct = 0
    total_tokens = 0
    tic = time.perf_counter()
    for i, sample in enumerate(dataset.select(range(args.num_questions))):
        prompt = format_fewshot_prompt(sample["question"])
        gold = extract_answer(sample["answer"])
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tok.pad_token_id,
            )
        completion_ids = out[0, ids.shape[1]:].tolist()
        text = tok.decode(completion_ids, skip_special_tokens=True)
        cut = text.find("\nQ:")
        if cut != -1:
            text = text[:cut]
        pred = extract_answer(text)
        ok = (pred == gold)
        correct += int(ok)
        total_tokens += len(completion_ids)
        elapsed = time.perf_counter() - tic
        print(
            f"[{i+1}/{args.num_questions}] {'OK' if ok else 'X '} "
            f"acc={correct}/{i+1} ({correct/(i+1):.1%}) "
            f"ntok={len(completion_ids)} tps={total_tokens/elapsed:.1f} "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )
        if i < 2:
            print(f"--- Q{i} pred={pred} gold={gold} ---")
            print(text[:400])
            print()

    elapsed = time.perf_counter() - tic
    print(f"\nBaseline accuracy: {correct}/{args.num_questions} ({correct/args.num_questions:.1%})")
    print(f"Total tokens: {total_tokens}; Wall time: {elapsed:.1f}s; TPS: {total_tokens/elapsed:.1f}")


if __name__ == "__main__":
    main()

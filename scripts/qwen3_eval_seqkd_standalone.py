"""Evaluate a Qwen3 draft on a question-disjoint SeqKD split.

This is a standalone-quality guardrail for SMCSD draft selection. It evaluates
only the user turns from provenance-checked GSM8K-train SeqKD rows, deduplicated
by question ID, and never reads GSM8K test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import sglang as sgl
from transformers import AutoTokenizer

from smcsd.qwen3_seqkd import normalize_numeric_answer, strict_terminal_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument("--attention-backend", choices=["triton", "fa3"], default="triton")
    parser.add_argument(
        "--repair-invalid-tail",
        action="store_true",
        help=(
            "For responses without an exact terminal answer, make one "
            "regex-constrained repair call and append its terminal line."
        ),
    )
    return parser.parse_args()


def read_questions(path: Path, limit: int | None) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            question_id = row.get("question_id")
            messages = row.get("sft_messages")
            gold = row.get("gold_answer")
            if question_id in seen:
                continue
            if (
                not isinstance(question_id, str)
                or not isinstance(gold, str)
                or not isinstance(messages, list)
                or not messages
                or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)
            ):
                raise ValueError(f"Malformed SeqKD row at line {line_number}")
            seen.add(question_id)
            questions.append(
                {
                    "question_id": question_id,
                    "gold_answer": gold,
                    "user_content": messages[0]["content"],
                }
            )
            if limit is not None and len(questions) >= limit:
                break
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def render_prompt(tokenizer: Any, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def render_repair_prompt(
    tokenizer: Any,
    *,
    user_content: str,
    response: str,
) -> str:
    return render_prompt(
        tokenizer,
        (
            "Extract the final numeric answer from the proposed solution below. "
            "Output exactly one line in the form `#### <number>` and nothing else.\n\n"
            f"Problem:\n{user_content}\n\nProposed solution:\n{response}"
        ),
    )


def main() -> None:
    args = parse_args()
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    print("[startup] loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("[startup] tokenizer loaded", flush=True)
    questions = read_questions(args.input, args.max_questions)
    prompts = [render_prompt(tokenizer, row["user_content"]) for row in questions]
    print(f"[startup] rendered {len(prompts)} prompts", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    engine_kwargs = {
        "model_path": args.model,
        "trust_remote_code": True,
        "attention_backend": args.attention_backend,
        "random_seed": args.seed,
        "mem_fraction_static": args.mem_fraction_static,
        "max_total_tokens": args.max_total_tokens,
    }
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    repair_sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 24,
        "regex": r"#### -?\d+(?:,\d{3})*(?:\.\d+)?",
    }
    completed = correct = generated_tokens = repair_tokens = 0
    invalid_before = invalid_after = wrong_tail = 0
    started = time.perf_counter()
    print("[startup] creating SGLang engine", flush=True)
    with sgl.Engine(**engine_kwargs) as engine, args.output.open(
        "w", encoding="utf-8"
    ) as handle:
        print("[startup] SGLang engine ready", flush=True)
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            outputs = engine.generate(batch, sampling_params)
            batch_questions = questions[start : start + len(batch)]
            strict_before = [
                strict_terminal_answer(output["text"]) for output in outputs
            ]
            repairs: dict[int, dict[str, Any]] = {}
            invalid_indices = [
                index
                for index, prediction in enumerate(strict_before)
                if prediction is None
            ]
            if args.repair_invalid_tail and invalid_indices:
                repair_prompts = [
                    render_repair_prompt(
                        tokenizer,
                        user_content=batch_questions[index]["user_content"],
                        response=outputs[index]["text"],
                    )
                    for index in invalid_indices
                ]
                repair_outputs = engine.generate(
                    repair_prompts, repair_sampling_params
                )
                repairs = dict(zip(invalid_indices, repair_outputs))
            for index, (question, output) in enumerate(
                zip(batch_questions, outputs)
            ):
                text = output["text"]
                prediction_before = strict_before[index]
                repair_text = None
                if index in repairs:
                    repair_text = repairs[index]["text"]
                    repair_tokens += int(
                        repairs[index]["meta_info"]["completion_tokens"]
                    )
                    text = text.rstrip() + "\n\n" + repair_text.strip()
                prediction = strict_terminal_answer(text)
                gold = normalize_numeric_answer(question["gold_answer"])
                is_correct = prediction == gold
                generated_tokens += int(output["meta_info"]["completion_tokens"])
                correct += int(is_correct)
                invalid_before += int(prediction_before is None)
                invalid_after += int(prediction is None)
                wrong_tail += int(prediction is not None and prediction != gold)
                completed += 1
                handle.write(
                    json.dumps(
                        {
                            "question_id": question["question_id"],
                            "gold_answer": question["gold_answer"],
                            "prediction": prediction,
                            "prediction_before_repair": prediction_before,
                            "correct": is_correct,
                            "response": text,
                            "repair_text": repair_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            elapsed = time.perf_counter() - started
            print(
                f"[{completed}/{len(questions)}] accuracy={correct / completed:.3%} "
                f"tps={generated_tokens / max(elapsed, 1e-6):.1f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    summary = {
        "model": args.model,
        "input": str(args.input.resolve()),
        "questions": completed,
        "correct": correct,
        "accuracy": correct / completed,
        "generated_tokens": generated_tokens,
        "repair_tokens": repair_tokens,
        "total_generated_tokens": generated_tokens + repair_tokens,
        "invalid_before": invalid_before,
        "invalid_after": invalid_after,
        "invalid_rate_before": invalid_before / completed,
        "invalid_rate_after": invalid_after / completed,
        "wrong_tail": wrong_tail,
        "wrong_tail_rate": wrong_tail / completed,
        "repair_invalid_tail": args.repair_invalid_tail,
        "latency_seconds": elapsed,
        "tokens_per_second": generated_tokens / max(elapsed, 1e-6),
        "seed": args.seed,
    }
    summary_path = args.output.with_suffix(f"{args.output.suffix}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

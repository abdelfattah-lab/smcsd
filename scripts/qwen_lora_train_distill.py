#!/usr/bin/env python3
"""Train a cross-tokenizer draft with response-only target SeqKD and LoRA.

Teacher responses are text, not target-token IDs. Each conversation is rendered
and tokenized with the draft tokenizer; user/system/padding tokens are masked so
only assistant-response tokens contribute to cross-entropy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_DRAFT = "Qwen/Qwen3-4B-Instruct-2507"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def read_jsonl(paths: Iterable[Path], limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def messages_for(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("sft_messages")
    if isinstance(messages, list) and len(messages) >= 2:
        return messages
    teacher_text = row.get("teacher_text")
    prompt_messages = row.get("messages")
    if isinstance(teacher_text, str) and teacher_text and isinstance(prompt_messages, list):
        return [*prompt_messages, {"role": "assistant", "content": teacher_text}]
    return None


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, generating: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generating,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, **kwargs, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class SeqKDDataset(Dataset):
    """Draft-tokenized conversations with prompt-masked response labels."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        for row in rows:
            messages = messages_for(row)
            if not messages:
                continue
            prompt_text = render_chat(tokenizer, messages[:-1], generating=True)
            full_text = render_chat(tokenizer, messages, generating=False)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "The draft chat template did not produce a token-prefix "
                    "stable assistant boundary; refusing to guess the "
                    "response-only label offset."
                )
            if tokenizer.eos_token_id is not None and (
                not full_ids or full_ids[-1] != tokenizer.eos_token_id
            ):
                full_ids.append(tokenizer.eos_token_id)
            full_ids = full_ids[:max_length]
            label_start = min(len(prompt_ids), len(full_ids))
            if label_start >= len(full_ids):
                continue
            self.examples.append(
                {
                    "input_ids": [int(token_id) for token_id in full_ids],
                    "attention_mask": [1] * len(full_ids),
                    "labels": [-100] * label_start
                    + [int(token_id) for token_id in full_ids[label_start:]],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class SeqKDCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-jsonl", nargs="+", type=Path)
    parser.add_argument("--model", default=DEFAULT_DRAFT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-output-dir", type=Path)
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        help="Continue training an existing PEFT adapter.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SeqKDDataset(
        read_jsonl(args.train_jsonl, args.limit),
        tokenizer,
        args.max_length,
    )
    eval_dataset = (
        SeqKDDataset(
            read_jsonl(args.eval_jsonl, args.eval_limit),
            tokenizer,
            args.max_length,
        )
        if args.eval_jsonl
        else None
    )
    if not train_dataset:
        raise RuntimeError("No training examples remain after tokenization.")
    if args.eval_jsonl and not eval_dataset:
        raise RuntimeError("No evaluation examples remain after tokenization.")

    load_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=load_dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    if args.resume_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, args.resume_adapter, is_trainable=True
        )
    else:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(LORA_TARGET_MODULES),
            ),
        )
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=args.dtype == "bfloat16",
        tf32=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SeqKDCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    # Gradient checkpointing requires cache-free training, but persisted
    # adapters/merged checkpoints are inference artifacts and should retain KV
    # caching by default.
    model.config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    if args.merged_output_dir:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        args.merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(args.merged_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output_dir)


if __name__ == "__main__":
    main()

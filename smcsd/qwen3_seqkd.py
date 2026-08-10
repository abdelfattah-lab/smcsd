"""Pure helpers for Qwen3 SeqKD GSM8K dataset preprocessing.

The builder intentionally stores message records instead of rendered token text:
the Qwen3 chat template is applied only by the trainer that owns the tokenizer.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


INSTRUCTION_PREFIX = (
    "Solve this math problem step by step.\n"
    "At the very end, output ONLY the final numeric answer on a new line in the exact format:\n"
    "#### <number>\n\n"
    "Problem:\n"
)
STRICT_FINAL_RE = re.compile(r"(?:^|\n)#### (-?\d+(?:,\d+)*(?:\.\d+)?)\Z")


def normalize_numeric_answer(value: str | None) -> str | None:
    """Normalize equivalent decimal spellings without accepting non-numeric values."""

    if value is None:
        return None
    try:
        decimal = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not decimal.is_finite():
        return None
    if decimal == decimal.to_integral_value():
        return format(decimal.to_integral_value(), "f")
    return format(decimal.normalize(), "f")


def canonical_question(question: str) -> str:
    """Stable question identity that is insensitive to unicode/space drift."""

    return " ".join(unicodedata.normalize("NFKC", question).split())


def question_id(question: str) -> str:
    """Return the stable ID used for question-level split assignment."""

    digest = hashlib.sha256(canonical_question(question).encode("utf-8")).hexdigest()
    return f"gsm8k-main-train-{digest}"


def expected_user_content(question: str) -> str:
    return f"{INSTRUCTION_PREFIX}{question}\n"


def strict_terminal_answer(text: str) -> str | None:
    """Return a normalized answer only for an exact terminal ``#### <number>`` line."""

    match = STRICT_FINAL_RE.search(text)
    return normalize_numeric_answer(match.group(1)) if match else None


def normalized_reasoning(text: str) -> str | None:
    """Normalize only for duplicate detection, retaining original text in output."""

    match = STRICT_FINAL_RE.search(text)
    if not match:
        return None
    reasoning = text[: match.start()].rstrip()
    return " ".join(unicodedata.normalize("NFKC", reasoning).casefold().split())


@dataclass(frozen=True)
class SplitConfig:
    """Percentages in the fixed train/dev/intrinsic/end-to-end order."""

    train: int = 80
    dev: int = 10
    intrinsic: int = 5
    end_to_end_development: int = 5
    salt: str = "qwen3-smcsd-seqkd-v1"

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.train,
                self.dev,
                self.intrinsic,
                self.end_to_end_development,
            )
        ):
            raise ValueError("Split percentages must be non-negative.")
        if (
            self.train + self.dev + self.intrinsic + self.end_to_end_development
            != 100
        ):
            raise ValueError("Split percentages must sum to 100.")


def assign_split(question_identifier: str, config: SplitConfig = SplitConfig()) -> str:
    """Assign every reasoning path for a question to one deterministic split."""

    bucket = int.from_bytes(
        hashlib.sha256(f"{config.salt}\0{question_identifier}".encode("utf-8")).digest()[:8],
        "big",
    ) % 100
    train_cutoff = config.train
    dev_cutoff = train_cutoff + config.dev
    intrinsic_cutoff = dev_cutoff + config.intrinsic
    if bucket < train_cutoff:
        return "train"
    if bucket < dev_cutoff:
        return "dev"
    if bucket < intrinsic_cutoff:
        return "intrinsic"
    return "end_to_end_development"


def split_question_ids(
    identifiers: Iterable[str], config: SplitConfig = SplitConfig()
) -> dict[str, list[str]]:
    """Partition unique IDs in sorted order; no question can span splits."""

    splits = {
        "train": [],
        "dev": [],
        "intrinsic": [],
        "end_to_end_development": [],
    }
    for identifier in sorted(set(identifiers)):
        splits[assign_split(identifier, config)].append(identifier)
    return splits


def validation_error(
    raw: dict[str, Any], train_questions: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    """Validate one source row and return (reason, question_id, normalized_gold)."""

    if raw.get("source") != "gsm8k":
        return "source_not_gsm8k", None, None
    if raw.get("correct") is not True:
        return "source_row_not_marked_correct", None, None
    messages = raw.get("sft_messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or any(not isinstance(message, dict) for message in messages)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[0].get("content"), str)
        or not isinstance(messages[1].get("content"), str)
    ):
        return "invalid_sft_messages", None, None

    user_content = messages[0]["content"]
    if not user_content.startswith(INSTRUCTION_PREFIX) or not user_content.endswith("\n"):
        return "unexpected_user_prompt", None, None
    question = user_content[len(INSTRUCTION_PREFIX) : -1]
    identifier = question_id(question)
    gold = train_questions.get(identifier)
    if gold is None:
        return "question_not_in_openai_gsm8k_main_train", identifier, None
    if user_content != expected_user_content(question):
        return "noncanonical_user_prompt", identifier, gold

    final_answer = strict_terminal_answer(messages[1]["content"])
    if final_answer is None:
        return "assistant_missing_exact_terminal_final", identifier, gold
    if final_answer != gold:
        return "assistant_final_disagrees_with_gsm8k_gold", identifier, gold
    if normalized_reasoning(messages[1]["content"]) is None:
        return "assistant_reasoning_unparseable", identifier, gold
    return None, identifier, gold

"""Build a provenance-verified Qwen3 SeqKD dataset from 70B GSM8K paths.

The resulting JSONL holds compact ``sft_messages`` records.  It deliberately
does not call a tokenizer or render a chat template: Qwen3 formatting is
rendered later by the trainer using the exact tokenizer/model selected there.

Only prompts that exactly reconstruct an ``openai/gsm8k`` ``main/train``
question and whose strict final ``#### <number>`` equals its gold answer are
emitted.  The train split is the only GSM8K split loaded; therefore no test
examples or labels can enter any output split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smcsd.qwen3_seqkd import (
    SplitConfig,
    assign_split,
    normalize_numeric_answer,
    normalized_reasoning,
    question_id,
    validation_error,
)


DEFAULT_INPUT = Path("artifacts/qwen_lora_distill/multipath/mp_train_70b.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/qwen3_seqkd")
SPLIT_NAMES = ("train", "dev", "intrinsic", "end_to_end_development")


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                rows.append((line_number, {"__parse_error__": str(error)}))
                continue
            if not isinstance(value, dict):
                rows.append((line_number, {"__parse_error__": "row_is_not_an_object"}))
                continue
            rows.append((line_number, value))
    return rows


def read_jsonls(paths: list[Path]) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Read independently durable source shards with source-level provenance."""
    rows: list[tuple[int, dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"SeqKD source does not exist: {path}")
        shard_rows = read_jsonl(path)
        # Make line-number diagnostics unambiguous across shards.
        rows.extend(
            [
                (source_index * 10_000_000 + line_number, row)
                for line_number, row in shard_rows
            ]
        )
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": len(shard_rows),
            }
        )
    return rows, sources


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_compact_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_compact_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def load_openai_gsm8k_train() -> tuple[dict[str, str], dict[str, Any]]:
    """Load only the approved source split and normalize its gold markers."""

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    questions: dict[str, str] = {}
    for row in dataset:
        identifier = question_id(row["question"])
        gold = normalize_numeric_answer(row["answer"].rsplit("####", maxsplit=1)[-1].strip())
        if gold is None:
            raise RuntimeError(f"Could not parse GSM8K gold answer for {identifier}.")
        if identifier in questions:
            raise RuntimeError(f"Question-ID collision in GSM8K train: {identifier}.")
        questions[identifier] = gold
    metadata = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "test_split_loaded": False,
    }
    return questions, metadata


def parse_split_percentages(value: str) -> SplitConfig:
    try:
        train, dev, intrinsic, end_to_end_development = (
            int(part.strip()) for part in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected four comma-separated integer percentages: train,dev,intrinsic,end_to_end."
        ) from error
    try:
        return SplitConfig(
            train=train,
            dev=dev,
            intrinsic=intrinsic,
            end_to_end_development=end_to_end_development,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_rows(
    source_rows: list[tuple[int, dict[str, Any]]],
    train_questions: dict[str, str],
    split_config: SplitConfig,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str], list[dict[str, Any]]]:
    """Validate and deduplicate records before deterministically splitting questions."""

    rejection_counts: Counter[str] = Counter()
    rejection_examples: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for line_number, raw in source_rows:
        if "__parse_error__" in raw:
            reason = str(raw["__parse_error__"])
            rejection_counts[reason] += 1
            if len(rejection_examples) < 20:
                rejection_examples.append({"line": line_number, "reason": reason})
            continue
        reason, identifier, gold = validation_error(raw, train_questions)
        if reason is not None:
            rejection_counts[reason] += 1
            if len(rejection_examples) < 20:
                rejection_examples.append(
                    {"line": line_number, "reason": reason, "question_id": identifier}
                )
            continue

        messages = raw["sft_messages"]
        reasoning_key = normalized_reasoning(messages[1]["content"])
        assert identifier is not None and gold is not None and reasoning_key is not None
        row = {
            "question_id": identifier,
            "gold_answer": gold,
            # Preserve the unrendered, source message content byte-for-byte.
            "sft_messages": messages,
        }
        candidate_key = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        existing = candidates[identifier].get(reasoning_key)
        if existing is not None:
            rejection_counts["duplicate_normalized_reasoning"] += 1
        if existing is None or candidate_key < existing["_candidate_key"]:
            row["_candidate_key"] = candidate_key
            candidates[identifier][reasoning_key] = row

    split_rows = {name: [] for name in SPLIT_NAMES}
    for identifier in sorted(candidates):
        split_name = assign_split(identifier, split_config)
        for reasoning_key in sorted(candidates[identifier]):
            row = candidates[identifier][reasoning_key].copy()
            row.pop("_candidate_key")
            split_rows[split_name].append(row)
    return split_rows, rejection_counts, rejection_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[DEFAULT_INPUT],
        help="One or more correctness-filtered SeqKD JSONL source shards.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-percentages",
        default="80,10,5,5",
        help="Question-level train,dev,intrinsic,end_to_end percentages (must sum to 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_config = parse_split_percentages(args.split_percentages)
    train_questions, gsm8k_metadata = load_openai_gsm8k_train()
    source_rows, source_manifest = read_jsonls(args.input)
    split_rows, rejection_counts, rejection_examples = build_rows(
        source_rows, train_questions, split_config
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, Any] = {}
    emitted_question_ids: set[str] = set()
    for split_name in SPLIT_NAMES:
        rows = split_rows[split_name]
        path = args.output_dir / f"seqkd_{split_name}.jsonl"
        write_compact_jsonl(path, rows)
        ids = sorted({row["question_id"] for row in rows})
        emitted_question_ids.update(ids)
        split_manifest[split_name] = {
            "file": path.name,
            "rows": len(rows),
            "questions": len(ids),
            "question_ids_sha256": hashlib.sha256(
                "\n".join(ids).encode("utf-8")
            ).hexdigest(),
            "jsonl_sha256": file_sha256(path),
        }

    question_split_counts = Counter()
    for split_name, rows in split_rows.items():
        question_split_counts[split_name] = len({row["question_id"] for row in rows})
    overlap_count = sum(question_split_counts.values()) - len(emitted_question_ids)
    if overlap_count:
        raise RuntimeError("Question ID split overlap detected.")
    if not emitted_question_ids.issubset(train_questions):
        raise RuntimeError("A non-training GSM8K question reached output.")

    summary = {
        "input_rows": len(source_rows),
        "emitted_rows": sum(len(rows) for rows in split_rows.values()),
        "emitted_questions": len(emitted_question_ids),
        "rejected_rows": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "split_rows": {name: len(split_rows[name]) for name in SPLIT_NAMES},
        "split_questions": {
            name: question_split_counts[name] for name in SPLIT_NAMES
        },
        "no_question_overlap_across_splits": True,
        "no_gsm8k_test_leakage": True,
    }
    manifest = {
        "schema": "qwen3-seqkd-v1",
        "qwen_chat_template_rendering": "deferred_to_trainer",
        "source": source_manifest,
        "gold_provenance": gsm8k_metadata,
        "validation": {
            "required_source": "gsm8k",
            "required_correct_flag": True,
            "required_prompt": "exact GSM8K instruction + question",
            "required_final_format": "terminal exact `#### <number>`",
            "gold_match": "normalized numeric equality against openai/gsm8k main train",
            "reasoning_deduplication": "NFKC + whitespace + casefold, per question",
        },
        "split": {
            "method": "sha256(salt + NUL + question_id) modulo 100",
            "salt": split_config.salt,
            "percentages": {
                "train": split_config.train,
                "dev": split_config.dev,
                "intrinsic": split_config.intrinsic,
                "end_to_end_development": split_config.end_to_end_development,
            },
            "unit": "question_id; all reasoning paths remain together",
        },
        "output": split_manifest,
    }
    write_compact_json(args.output_dir / "summary.json", summary)
    write_compact_json(args.output_dir / "manifest.json", manifest)
    write_compact_json(args.output_dir / "split_manifest.json", split_manifest)
    if rejection_examples:
        write_compact_json(args.output_dir / "rejection_examples.json", rejection_examples)
    print(
        f"Validated {summary['emitted_rows']}/{summary['input_rows']} rows across "
        f"{summary['emitted_questions']} GSM8K-train questions -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()

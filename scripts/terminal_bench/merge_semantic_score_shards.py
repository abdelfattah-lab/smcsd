#!/usr/bin/env python3
"""Validate and atomically merge sharded semantic-verifier score artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from score_semantic_checkpoints import (
    CRITERION_VERSION,
    canonical_json,
    iter_checkpoints,
    summarize_scores,
)


def load_score_parts(
    paths: list[Path], *, scorer_model: str, criterion: str
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    by_checkpoint: dict[str, dict[str, Any]] = {}
    call_ids: set[str] = set()
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("scorer_model") != scorer_model:
                    raise ValueError(
                        f"wrong scorer model at {path}:{line_number}: "
                        f"{row.get('scorer_model')}"
                    )
                if row.get("criterion") != criterion:
                    raise ValueError(
                        f"wrong criterion at {path}:{line_number}: "
                        f"{row.get('criterion')}"
                    )
                checkpoint_id = str(row["checkpoint_id"])
                call_id = str(row["verifier_call_id"])
                if checkpoint_id in by_checkpoint:
                    raise ValueError(f"duplicate checkpoint score: {checkpoint_id}")
                if call_id in call_ids:
                    raise ValueError(f"duplicate verifier call: {call_id}")
                by_checkpoint[checkpoint_id] = row
                call_ids.add(call_id)
    return by_checkpoint, call_ids


def load_run_cost_parts(
    paths: list[Path], *, scorer_model: str, criterion: str, tp_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("scorer_model") != scorer_model:
                    raise ValueError(
                        f"wrong run-cost scorer at {path}:{line_number}"
                    )
                if row.get("criterion") != criterion:
                    raise ValueError(
                        f"wrong run-cost criterion at {path}:{line_number}"
                    )
                if int(row.get("tensor_parallel_size") or 0) != tp_size:
                    raise ValueError(
                        f"wrong tensor parallel size at {path}:{line_number}"
                    )
                run_id = str(row["run_id"])
                if run_id in run_ids:
                    raise ValueError(f"duplicate run cost: {run_id}")
                run_ids.add(run_id)
                rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            for row in rows:
                stream.write(canonical_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--score-part", type=Path, action="append", required=True)
    parser.add_argument("--run-cost-part", type=Path, action="append", required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    parser.add_argument("--run-cost-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--criterion", default=CRITERION_VERSION)
    parser.add_argument("--tp", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scores, _ = load_score_parts(
        args.score_part, scorer_model=args.scorer, criterion=args.criterion
    )
    run_costs = load_run_cost_parts(
        args.run_cost_part,
        scorer_model=args.scorer,
        criterion=args.criterion,
        tp_size=args.tp,
    )
    ordered_checkpoint_ids = [
        str(row["checkpoint_id"]) for row in iter_checkpoints(args.checkpoints)
    ]
    expected = set(ordered_checkpoint_ids)
    actual = set(scores)
    if actual != expected:
        raise ValueError(
            "score coverage mismatch: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    score_run_ids = {str(row["run_id"]) for row in scores.values()}
    cost_run_ids = {str(row["run_id"]) for row in run_costs}
    if not score_run_ids <= cost_run_ids:
        raise ValueError(
            f"missing run-cost records for {len(score_run_ids - cost_run_ids)} runs"
        )

    atomic_write_jsonl(
        args.scores_output, (scores[checkpoint_id] for checkpoint_id in ordered_checkpoint_ids)
    )
    atomic_write_jsonl(args.run_cost_output, run_costs)
    summary = summarize_scores(
        args.scores_output,
        args.summary_output,
        args.run_cost_output,
        scorer_model=args.scorer,
        criterion=args.criterion,
        tp_size=args.tp,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

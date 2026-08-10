#!/usr/bin/env python3
"""Audit hybrid/live byte and proposal-accounting equivalence on fixed cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from smcsd.cross_tokenizer.artifacts import load_artifact
from smcsd.cross_tokenizer.mapper import TokenMapper


CASES = [
    "16",
    "1 + 2 = 3",
    "\n#### 42",
    "  leading and trailing  ",
    "café",
    "π ≈ 3.14",
]


def exact_decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-tokenizer", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    draft = AutoTokenizer.from_pretrained(
        args.draft_tokenizer, trust_remote_code=True
    )
    target = AutoTokenizer.from_pretrained(
        args.target_tokenizer, trust_remote_code=True
    )
    artifact = load_artifact(args.artifact)
    hybrid = TokenMapper(
        draft_tokenizer=draft,
        target_tokenizer=target,
        artifact=artifact,
        mode="hybrid",
    )
    live = TokenMapper(
        draft_tokenizer=draft,
        target_tokenizer=target,
        mode="live",
    )

    records = []
    for text in CASES:
        draft_ids = draft.encode(text, add_special_tokens=False)
        logq = [-(index + 1) / 10 for index in range(len(draft_ids))]
        hybrid_result = hybrid.map_tokens(draft_ids, logq)
        live_result = live.map_tokens(draft_ids, logq)
        draft_text = exact_decode(draft, draft_ids)
        hybrid_text = exact_decode(target, hybrid_result.target_ids)
        live_text = exact_decode(target, live_result.target_ids)
        records.append(
            {
                "case": text,
                "draft_ids": draft_ids,
                "hybrid_target_ids": hybrid_result.target_ids,
                "live_target_ids": live_result.target_ids,
                "draft_logq_sum": sum(logq),
                "hybrid_logq_sum": sum(hybrid_result.target_logq),
                "live_logq_sum": sum(live_result.target_logq),
                "hybrid_segments": [
                    {
                        "draft_start": segment.draft_start,
                        "draft_end": segment.draft_end,
                        "target_start": segment.target_start,
                        "target_end": segment.target_end,
                        "draft_logq": segment.draft_logq,
                    }
                    for segment in hybrid_result.segments
                ],
                "live_segments": [
                    {
                        "draft_start": segment.draft_start,
                        "draft_end": segment.draft_end,
                        "target_start": segment.target_start,
                        "target_end": segment.target_end,
                        "draft_logq": segment.draft_logq,
                    }
                    for segment in live_result.segments
                ],
                "bytes_equal": (
                    draft_text.encode("utf-8")
                    == hybrid_text.encode("utf-8")
                    == live_text.encode("utf-8")
                ),
                "proposal_charge_equal": (
                    abs(sum(logq) - sum(hybrid_result.target_logq)) < 1e-9
                    and abs(sum(logq) - sum(live_result.target_logq)) < 1e-9
                ),
            }
        )

    draft_eos = getattr(draft, "eos_token_id", None)
    if draft_eos is not None:
        hybrid_eos = hybrid.map_tokens([int(draft_eos)], [-0.25])
        live_eos = live.map_tokens([int(draft_eos)], [-0.25])
        target_eos = int(target.eos_token_id)
        records.append(
            {
                "case": "special_eos",
                "draft_ids": [int(draft_eos)],
                "hybrid_target_ids": hybrid_eos.target_ids,
                "live_target_ids": live_eos.target_ids,
                "bytes_equal": (
                    hybrid_eos.target_ids == [target_eos]
                    and live_eos.target_ids == [target_eos]
                ),
                "proposal_charge_equal": (
                    abs(sum(hybrid_eos.target_logq) + 0.25) < 1e-9
                    and abs(sum(live_eos.target_logq) + 0.25) < 1e-9
                ),
            }
        )

    payload = {
        "all_pass": all(
            row["bytes_equal"] and row["proposal_charge_equal"]
            for row in records
        ),
        "cases": records,
        "hybrid_stats": hybrid.stats.snapshot(),
        "live_stats": live.stats.snapshot(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

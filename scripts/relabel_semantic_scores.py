"""Relabel saved semantic scores after trajectories receive longer outcomes."""

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trajectories = load_jsonl(args.trajectories)
    outcomes = {
        (str(row["problem_id"]), int(row["sample_id"])): row
        for row in trajectories
    }
    scores = load_jsonl(args.scores)
    for row in scores:
        key = (str(row["problem_id"]), int(row["sample_id"]))
        if key not in outcomes:
            raise ValueError(f"Missing continued trajectory for {key}")
        outcome = outcomes[key]
        row.update(
            {
                "correct": bool(outcome["correct"]),
                "extracted_answer": outcome["extracted_answer"],
                "gold_answer": outcome["gold_answer"],
                "trajectory_tokens": int(outcome["completion_tokens"]),
                "actual_fraction": (
                    int(row["token_position"]) / int(outcome["completion_tokens"])
                ),
                "outcome_trajectory_cap": int(outcome["max_new_tokens"]),
                "outcome_relabelled": True,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"relabelled {len(scores)} scores from {len(outcomes)} trajectories")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

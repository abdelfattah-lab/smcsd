"""Phase 2a/2b offline study: does a scorer's recoverability estimate on a
PARTIAL trajectory predict eventual correctness?

Input: trajectory JSONL from accuracy_test_math500.py --dump-trajectories
(independent majority samples with per-sample correctness labels).

For each trajectory, truncate the generated text at fractions of its token
length, and score the prefix two ways:

  design B (clean re-prompt): chat-templated judge question about the
      partial response, answered with a single score token 1-5.
  design A (suffix-form): the ORIGINAL generation context + prefix,
      continued with a fixed assessment suffix -- the shape a KV-fork
      scorer would use online.

Score = expectation over the {1..5} token distribution at the answer
position. Outputs AUROC (score vs eventual correctness) per design x
truncation fraction, plus within-question ranking accuracy.
"""

import argparse
import json
import math
import sys
from collections import defaultdict


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank for ties
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


JUDGE_INSTRUCTION = (
    "You are grading an UNFINISHED solution to a math problem.\n\n"
    "Problem:\n{problem}\n\n"
    "Partial solution (unfinished, cut off mid-way):\n{prefix}\n\n"
    "Estimate the probability that this partial solution can still be "
    "completed into a fully correct solution, taking into account any "
    "irreversible logical, factual, or arithmetic errors already present. "
    "Answer with a single digit from 1 (certainly cannot) to 5 (certainly "
    "can). Answer: "
)

SUFFIX_FORM = (
    "\n\n[Self-assessment: on a scale of 1-5, the probability that this "
    "solution-in-progress reaches a fully correct final answer is: "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--problems-dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--scorer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--fractions", default="0.25,0.5,0.75")
    ap.add_argument("--max-trajectories", type=int, default=800)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--probe", action="store_true",
                    help="score 2 prompts, print meta structure, exit")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import sglang as sgl

    ds = load_dataset(args.problems_dataset, split="test")
    problems = [r["problem"] for r in ds]
    tok = AutoTokenizer.from_pretrained(args.scorer)
    score_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(1, 6)]
    assert all(len(x) == 1 for x in score_ids), f"score tokens not single: {score_ids}"
    score_ids = [x[0] for x in score_ids]

    rows = []
    with open(args.trajectories) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows = rows[: args.max_trajectories]
    fracs = [float(x) for x in args.fractions.split(",")]

    # Build all scoring prompts.
    jobs = []  # (design, frac, row_idx, prompt)
    for ri, r in enumerate(rows):
        ids = tok.encode(r["text"], add_special_tokens=False)
        problem = problems[r["qid"]]
        for f in fracs:
            prefix = tok.decode(ids[: max(8, int(len(ids) * f))])
            b_prompt = tok.apply_chat_template(
                [{"role": "user",
                  "content": JUDGE_INSTRUCTION.format(problem=problem, prefix=prefix)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            jobs.append(("B", f, ri, b_prompt))
            gen_ctx = tok.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            jobs.append(("A", f, ri, gen_ctx + prefix + SUFFIX_FORM))
    if args.probe:
        jobs = jobs[:2]
    print(f"{len(jobs)} scoring prompts", flush=True)

    eng = sgl.Engine(model_path=args.scorer, trust_remote_code=True,
                     attention_backend="triton",
                     mem_fraction_static=args.mem_fraction_static)
    try:
        outs = eng.generate(
            [j[3] for j in jobs],
            {"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            top_logprobs_num=20,
        )
    finally:
        eng.shutdown()

    if args.probe:
        for o in outs:
            mi = o.get("meta_info", {})
            print("meta keys:", list(mi.keys()))
            print("output_top_logprobs:", str(mi.get("output_top_logprobs"))[:400])
        return

    def expected_score(o):
        mi = o.get("meta_info", {})
        top = mi.get("output_top_logprobs")
        if not top:
            return None
        # top[0] = list of (logprob, token_id, text?) entries for 1st token
        entries = top[0]
        p = {}
        for e in entries:
            lp, tid = e[0], e[1]
            if tid in score_ids:
                p[score_ids.index(tid) + 1] = math.exp(lp)
        if not p:
            return None
        z = sum(p.values())
        return sum(k * v for k, v in p.items()) / z

    per = defaultdict(lambda: ([], []))  # (design, frac) -> (scores, labels)
    perq = defaultdict(lambda: defaultdict(list))  # (design, frac) -> qid -> [(s, corr)]
    skipped = 0
    for (design, f, ri, _), o in zip(jobs, outs):
        s = expected_score(o)
        if s is None:
            skipped += 1
            continue
        r = rows[ri]
        per[(design, f)][0].append(s)
        per[(design, f)][1].append(1 if r["correct"] else 0)
        perq[(design, f)][r["qid"]].append((s, r["correct"]))

    print(f"skipped (no score-token mass in top-20): {skipped}/{len(jobs)}")
    print(f"{'design':6s} {'frac':5s} {'AUROC':>6s} {'rank-acc':>8s} {'n':>5s}")
    for (design, f), (scores, labels) in sorted(per.items()):
        # within-question ranking: fraction of (correct, incorrect) pairs
        # where the correct sample scores higher
        wins = tot = 0
        for q, lst in perq[(design, f)].items():
            cs = [s for s, c in lst if c]
            ws = [s for s, c in lst if not c]
            for a in cs:
                for b in ws:
                    tot += 1
                    wins += 1 if a > b else (0.5 if a == b else 0)
        ra = wins / tot if tot else float("nan")
        print(f"{design:6s} {f:<5.2f} {auroc(scores, labels):6.3f} {ra:8.3f} {len(labels):5d}")


if __name__ == "__main__":
    main()

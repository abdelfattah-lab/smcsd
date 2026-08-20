"""Offline reallocation-policy simulations over dumped trajectories + judge scores.

Inputs: --trajectories (accuracy_test_math500.py --dump-trajectories JSONL)
and --scores (offline_recoverability.py --save-scores JSONL).

Policies simulated at matched generation-token budgets:
  majority@k        : uniform k-vote baseline (mean over subsample trials)
  pruned f,m        : run all n to fraction f, judge-score, keep top-m to
                      completion, vote among survivors
  judge-adaptive    : per-question budget (2 or 8 votes) chosen by mean judge
                      score of two prefixes at f=0.25
  agreement-adaptive: 2 votes; stop if they agree, else escalate to 8
                      (judge-free control)

Findings on MATH500-100 / 9B / 27B judge (2026-08-20): pruning sits at or
below the uniform majority curve at matched budget; both adaptive policies
reach maj@8 accuracy at ~30% fewer tokens, and the judge-free agreement
control matches the judge -- the verifier's unique value must come from
acting BEFORE completion (long-horizon tasks) or from serving economics.
"""

import argparse
import json
import random
from collections import Counter, defaultdict


def load(traj_path, scores_path, design="A"):
    scores = defaultdict(dict)
    ntok = {}
    for line in open(scores_path):
        r = json.loads(line)
        if r["design"] != design:
            continue
        scores[(r["qid"], r["sample"])][r["frac"]] = r["score"]
        ntok[(r["qid"], r["sample"])] = r["ntok"]
    answers = {}
    for line in open(traj_path):
        r = json.loads(line)
        answers[(r["qid"], r["sample"])] = (r["answer"], r["gold"])
    qids = sorted({q for q, _ in scores})
    return qids, scores, ntok, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--design", default="A")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    qids, scores, ntok, answers = load(args.trajectories, args.scores, args.design)
    rng = random.Random(0)
    nq = len(qids)

    def vote(q, ss):
        vs = [answers[(q, s)][0] for s in ss if answers[(q, s)][0]]
        return Counter(vs).most_common(1)[0][0] if vs else None

    def gold(q):
        return answers[(q, 0)][1]

    print("majority@k (mean over trials):")
    for k in [2, 3, 4, 5, 6, 8]:
        accs, buds = [], []
        for _ in range(args.trials):
            acc = bud = 0
            for q in qids:
                ss = rng.sample(range(8), k)
                acc += vote(q, ss) == gold(q)
                bud += sum(ntok[(q, s)] for s in ss)
            accs.append(acc)
            buds.append(bud)
        print(f"  maj@{k}: acc={sum(accs)/args.trials:.1f}/{nq} tok/q={sum(buds)/args.trials/nq:.0f}")

    print("pruned (all 8 to f, keep top-m):")
    for f in [0.25, 0.5]:
        for m in [1, 2, 3, 4]:
            acc = bud = judge = 0
            for q in qids:
                sc = sorted(((scores[(q, s)].get(f, 0), s) for s in range(8)), reverse=True)
                keep = [s for _, s in sc[:m]]
                acc += vote(q, keep) == gold(q)
                bud += sum(int(ntok[(q, s)] * f) for s in range(8))
                bud += sum(ntok[(q, s)] - int(ntok[(q, s)] * f) for s in keep)
                judge += sum(int(ntok[(q, s)] * f) for s in range(8))
            print(f"  f={f} m={m}: acc={acc}/{nq} tok/q={bud/nq:.0f} judge/q={judge/nq:.0f}")

    qscore = {q: (scores[(q, 0)].get(0.25, 3) + scores[(q, 1)].get(0.25, 3)) / 2 for q in qids}
    lo = set(sorted(qids, key=lambda q: qscore[q])[: nq // 2])
    acc = bud = 0
    for q in qids:
        ss = list(range(8)) if q in lo else [0, 1]
        acc += vote(q, ss) == gold(q)
        bud += sum(ntok[(q, s)] for s in ss)
    print(f"judge-adaptive: acc={acc}/{nq} tok/q={bud/nq:.0f}")

    accs, buds = [], []
    for _ in range(args.trials):
        acc = bud = 0
        for q in qids:
            ss = rng.sample(range(8), 8)
            a0, a1 = answers[(q, ss[0])][0], answers[(q, ss[1])][0]
            use = ss[:2] if (a0 is not None and a0 == a1) else ss
            acc += vote(q, use) == gold(q)
            bud += sum(ntok[(q, s)] for s in use)
        accs.append(acc)
        buds.append(bud)
    print(f"agreement-adaptive: acc={sum(accs)/args.trials:.1f}/{nq} tok/q={sum(buds)/args.trials/nq:.0f}")


if __name__ == "__main__":
    main()

"""Offline reallocation-policy simulations over dumped trajectories + judge scores.

Inputs: schema-v2 trajectories from a stock benchmark dump and semantic scores
from offline_recoverability.py. Legacy score/dump files remain supported.

Policies simulated at matched generation-token budgets:
  majority@k        : uniform k-vote baseline (mean over subsample trials)
  pruned f,m        : run all n to fraction f, judge-score, keep top-m to
                      completion, vote among survivors
  judge-adaptive    : per-question budget (2 or 8 votes) chosen by mean judge
                      score of two prefixes at f=0.25
  agreement-adaptive: 2 votes; stop if they agree, else escalate to 8
                      (judge-free control)

This is an offline top-m pruning ablation, not online ParticleScale: it cannot
branch a promising prefix or replenish particles after pruning.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load(traj_path, scores_path, design=None):
    scores = defaultdict(dict)
    ntok = {}
    positions = defaultdict(dict)
    verifier_prompt_tokens = defaultdict(dict)
    for line in open(scores_path):
        r = json.loads(line)
        if design is not None and r.get("design") not in (None, design):
            continue
        qid = r.get("problem_id", r.get("qid"))
        sample = int(r.get("sample", r.get("sample_id")))
        fraction = float(r.get("frac", r.get("requested_fraction")))
        key = (qid, sample)
        scores[key][fraction] = float(r["score"])
        ntok[key] = int(r.get("ntok", r.get("trajectory_tokens")))
        positions[key][fraction] = int(
            r.get("token_position", int(ntok[key] * fraction))
        )
        verifier_prompt_tokens[key][fraction] = int(r.get("prompt_tokens", 0))
    answers = {}
    for line in open(traj_path):
        r = json.loads(line)
        qid = r.get("problem_id", r.get("qid"))
        sample = int(r.get("sample", r.get("sample_id")))
        answers[(qid, sample)] = (
            r.get("answer", r.get("extracted_answer")),
            r.get("gold", r.get("gold_answer")),
        )
    qids = sorted({q for q, _ in scores})
    return qids, scores, ntok, positions, verifier_prompt_tokens, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--design", default=None)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument(
        "--adaptive-calibration-scores",
        default=None,
        help="Freeze the judge-adaptive threshold from this separate score file.",
    )
    ap.add_argument("--summary-output", default=None)
    ap.add_argument(
        "--generator-full-inference-wall-time",
        type=float,
        default=None,
        help=(
            "Measured steady-state inference seconds for generating all eight "
            "trajectories. Enables token-scaled GPU-time estimates."
        ),
    )
    ap.add_argument(
        "--adaptive-verifier-summary",
        default=None,
        help=(
            "Summary from a direct two-prefix verifier run; its inference time "
            "is scaled by verifier prompt-token count."
        ),
    )
    args = ap.parse_args()
    qids, scores, ntok, positions, verifier_tokens, answers = load(
        args.trajectories, args.scores, args.design
    )
    rng = random.Random(0)
    nq = len(qids)
    summary = {
        "schema_version": 1,
        "n_problems": nq,
        "trajectories": args.trajectories,
        "scores": args.scores,
        "trials": args.trials,
        "majority": {},
        "pruned": {},
    }

    def vote(q, ss):
        vs = [answers[(q, s)][0] for s in sorted(ss) if answers[(q, s)][0]]
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
        accuracy = sum(accs) / args.trials / nq
        tokens_per_problem = sum(buds) / args.trials / nq
        summary["majority"][str(k)] = {
            "accuracy": accuracy,
            "generator_tokens_per_problem": tokens_per_problem,
        }
        print(
            f"  maj@{k}: acc={accuracy * nq:.1f}/{nq} "
            f"tok/q={tokens_per_problem:.0f}"
        )

    print("pruned (all 8 to f, keep top-m):")
    for f in [0.25, 0.5]:
        for m in [1, 2, 3, 4]:
            acc = bud = verifier = 0
            for q in qids:
                sc = sorted(((scores[(q, s)].get(f, 0), s) for s in range(8)), reverse=True)
                keep = [s for _, s in sc[:m]]
                acc += vote(q, keep) == gold(q)
                bud += sum(positions[(q, s)][f] for s in range(8))
                bud += sum(
                    ntok[(q, s)] - positions[(q, s)][f] for s in keep
                )
                verifier += sum(verifier_tokens[(q, s)][f] for s in range(8))
            print(
                f"  f={f} m={m}: acc={acc}/{nq} gen_tok/q={bud/nq:.0f} "
                f"verifier_prompt_tok/q={verifier/nq:.0f}"
            )
            summary["pruned"][f"{f}:{m}"] = {
                "fraction": f,
                "survivors": m,
                "accuracy": acc / nq,
                "generator_tokens_per_problem": bud / nq,
                "verifier_prompt_tokens_per_problem": verifier / nq,
            }

    qscore = {
        q: (scores[(q, 0)].get(0.25, 0.5) + scores[(q, 1)].get(0.25, 0.5)) / 2
        for q in qids
    }
    if args.adaptive_calibration_scores:
        calibration_qids, calibration_scores, _, _, _, _ = load(
            args.trajectories,
            args.adaptive_calibration_scores,
            args.design,
        )
        calibration_values = sorted(
            (
                calibration_scores[(q, 0)].get(0.25, 0.5)
                + calibration_scores[(q, 1)].get(0.25, 0.5)
            )
            / 2
            for q in calibration_qids
        )
        calibration_source = args.adaptive_calibration_scores
    else:
        calibration_values = sorted(qscore.values())
        calibration_source = args.scores
    midpoint = len(calibration_values) // 2
    adaptive_threshold = (
        calibration_values[midpoint - 1] + calibration_values[midpoint]
    ) / 2
    lo = {q for q in qids if qscore[q] < adaptive_threshold}
    acc = bud = verifier = 0
    for q in qids:
        ss = list(range(8)) if q in lo else [0, 1]
        acc += vote(q, ss) == gold(q)
        bud += sum(ntok[(q, s)] for s in ss)
        verifier += sum(verifier_tokens[(q, s)][0.25] for s in (0, 1))
    summary["judge_adaptive"] = {
        "accuracy": acc / nq,
        "generator_tokens_per_problem": bud / nq,
        "verifier_prompt_tokens_per_problem": verifier / nq,
        "threshold": adaptive_threshold,
        "threshold_calibration_scores": calibration_source,
        "expanded_to_8_problems": len(lo),
    }
    print(
        f"judge-adaptive: acc={acc}/{nq} tok/q={bud/nq:.0f} "
        f"verifier_prompt_tok/q={verifier/nq:.0f} expanded={len(lo)}/{nq} "
        f"threshold={adaptive_threshold:.6f}"
    )

    adaptive_rng = random.Random(1)
    adaptive_accs, adaptive_buds, adaptive_verifiers, adaptive_expanded = (
        [],
        [],
        [],
        [],
    )
    for _ in range(args.trials):
        acc = bud = verifier = expanded = 0
        for q in qids:
            order = adaptive_rng.sample(range(8), 8)
            first = order[:2]
            score = sum(scores[(q, s)].get(0.25, 0.5) for s in first) / 2
            use = order if score < adaptive_threshold else first
            expanded += len(use) == 8
            acc += vote(q, use) == gold(q)
            bud += sum(ntok[(q, s)] for s in use)
            verifier += sum(verifier_tokens[(q, s)][0.25] for s in first)
        adaptive_accs.append(acc)
        adaptive_buds.append(bud)
        adaptive_verifiers.append(verifier)
        adaptive_expanded.append(expanded)
    summary["judge_adaptive_randomized_order"] = {
        "accuracy": sum(adaptive_accs) / args.trials / nq,
        "generator_tokens_per_problem": sum(adaptive_buds) / args.trials / nq,
        "verifier_prompt_tokens_per_problem": (
            sum(adaptive_verifiers) / args.trials / nq
        ),
        "mean_expanded_to_8_problems": sum(adaptive_expanded) / args.trials,
        "threshold": adaptive_threshold,
        "threshold_calibration_scores": calibration_source,
    }
    print(
        "judge-adaptive randomized order: "
        f"acc={sum(adaptive_accs) / args.trials:.1f}/{nq} "
        f"tok/q={sum(adaptive_buds) / args.trials / nq:.0f} "
        f"verifier_prompt_tok/q={sum(adaptive_verifiers) / args.trials / nq:.0f} "
        f"expanded={sum(adaptive_expanded) / args.trials:.1f}/{nq}"
    )

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
    agreement_accuracy = sum(accs) / args.trials / nq
    agreement_tokens = sum(buds) / args.trials / nq
    summary["agreement_adaptive"] = {
        "accuracy": agreement_accuracy,
        "generator_tokens_per_problem": agreement_tokens,
        "verifier_prompt_tokens_per_problem": 0,
    }
    print(
        f"agreement-adaptive: acc={agreement_accuracy * nq:.1f}/{nq} "
        f"tok/q={agreement_tokens:.0f}"
    )

    if args.generator_full_inference_wall_time is not None:
        full_tokens = summary["majority"]["8"]["generator_tokens_per_problem"]
        full_time = args.generator_full_inference_wall_time

        def generator_time(policy):
            return full_time * policy["generator_tokens_per_problem"] / full_tokens

        verifier_reference = None
        if args.adaptive_verifier_summary:
            with open(args.adaptive_verifier_summary, encoding="utf-8") as fh:
                verifier_summary = json.load(fh)
            verifier_reference = {
                "summary": args.adaptive_verifier_summary,
                "inference_gpu_seconds": float(
                    verifier_summary["cost"]["inference_wall_time_s"]
                ),
                "prompt_tokens": int(
                    verifier_summary["cost"]["verifier_prompt_tokens"]
                ),
            }

        def compute_policy(policy, include_verifier):
            gen_time = generator_time(policy)
            verifier_time = 0.0
            if include_verifier:
                if verifier_reference is None:
                    verifier_time = None
                else:
                    policy_verifier_tokens = (
                        policy["verifier_prompt_tokens_per_problem"] * nq
                    )
                    verifier_time = (
                        verifier_reference["inference_gpu_seconds"]
                        * policy_verifier_tokens
                        / verifier_reference["prompt_tokens"]
                    )
            total_time = None if verifier_time is None else gen_time + verifier_time
            return {
                "accuracy": policy["accuracy"],
                "generator_gpu_seconds": gen_time,
                "verifier_gpu_seconds": verifier_time,
                "total_gpu_seconds": total_time,
                "savings_vs_full_self_consistency": (
                    None if total_time is None else 1 - total_time / full_time
                ),
            }

        summary["steady_state_compute_estimate"] = {
            "method": "measured_component_time_with_linear_token_scaling",
            "assumptions": [
                "One generator GPU and one verifier GPU are counted in GPU-seconds.",
                "Model initialization is excluded.",
                "Component inference time is scaled linearly by observed token count.",
                "This is an offline estimate, not a directly timed dynamic policy run.",
            ],
            "full_self_consistency": {
                "accuracy": summary["majority"]["8"]["accuracy"],
                "generator_gpu_seconds": full_time,
                "verifier_gpu_seconds": 0.0,
                "total_gpu_seconds": full_time,
                "savings_vs_full_self_consistency": 0.0,
            },
            "judge_adaptive": compute_policy(summary["judge_adaptive"], True),
            "judge_adaptive_randomized_order": compute_policy(
                summary["judge_adaptive_randomized_order"], True
            ),
            "agreement_adaptive": compute_policy(
                summary["agreement_adaptive"], False
            ),
            "generator_full_inference_wall_time_s": full_time,
            "verifier_reference": verifier_reference,
        }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()

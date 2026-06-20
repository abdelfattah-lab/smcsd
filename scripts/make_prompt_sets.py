"""Build prompt JSONL sets for proposal-finetuning collection.

Emits ``{"prompt": <instruction>, "domain": <name>}`` lines whose instruction
text is IDENTICAL to what ``eval_tasks.py`` feeds the engine, so the rollout
collection distribution matches eval.  Feed the output to
``collect_proposal_data.py --dataset <file> --raw-prompts --disable-thinking``.

Splits are chosen disjoint from the eval splits:
  * math  — GSM8K train + Hendrycks MATH train  (eval: GSM8K test, MATH-500)
  * code  — MBPP train+validation+prompt          (eval: MBPP test, HumanEval test)

Usage:
  python scripts/make_prompt_sets.py --out-dir /data/proposal_prompts \
      --n-math 1500 --n-code 1000 --seed 0
"""

import argparse
import json
import os
import random

from datasets import load_dataset

# import the exact instruction templates used at eval time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tasks import instr_gsm8k, instr_math, instr_mbpp  # noqa: E402


def math_prompts(n, rng):
    """GSM8K train (#### format) + Hendrycks MATH train (boxed format)."""
    out = []
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for s in gsm:
        out.append({"prompt": instr_gsm8k(s["question"]), "domain": "math_gsm8k"})
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    for subj in subjects:
        try:
            math = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
        except Exception as e:
            print(f"  (skipping Hendrycks MATH/{subj}: {str(e)[:80]})")
            continue
        for s in math:
            out.append({"prompt": instr_math(s["problem"]), "domain": "math_hendrycks"})
    rng.shuffle(out)
    return out[:n]


def code_prompts(n, rng):
    """MBPP train + validation + prompt splits (test is held out for eval)."""
    out = []
    for split in ("train", "validation", "prompt"):
        try:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        except Exception as e:
            print(f"  (skipping MBPP {split}: {str(e)[:80]})")
            continue
        for s in ds:
            out.append(
                {"prompt": instr_mbpp(s["text"], s["test_list"]), "domain": "code_mbpp"}
            )
    rng.shuffle(out)
    return out[:n]


# --- general-purpose mix (open-perfectblend non-math sources + MBPP + GSM8K) --
# We only use each example's FIRST user turn as a prompt (responses come from
# SMC). open-perfectblend is math-heavy, so we draw only its chat/code/IF
# sources and add the eval-aligned MBPP-train (code) + a small GSM8K-train
# (math-with-headroom) slice.  Domain labels feed train_proposal --balance-domains.
PB_SOURCE_DOMAIN = [
    ("evol-codealpaca", "code"),
    ("ultrachat", "chat"),
    ("lmsys", "chat"),
    ("ultrafeedback", "chat"),
    ("autoif", "if"),
    # skipped (math-contaminated): metamath, orca-math, ultrainteract
]


def _first_user_turn(conv):
    for t in conv:
        if t.get("from") in ("human", "user"):
            return t.get("value", "").strip()
    return ""


def general_prompts(quotas, rng, max_scan=800_000):
    """Stream open-perfectblend; collect `quotas[domain]` prompts per domain
    from the allowed sources."""
    from datasets import load_dataset
    got = {d: [] for d in set(d for _, d in PB_SOURCE_DOMAIN)}
    ds = load_dataset("mlabonne/open-perfectblend", split="train", streaming=True)
    seen = 0
    for row in ds:
        seen += 1
        if seen > max_scan:
            break
        src = (row.get("source") or "").lower()
        dom = next((d for key, d in PB_SOURCE_DOMAIN if key in src), None)
        if dom is None or len(got[dom]) >= quotas.get(dom, 0):
            continue
        prompt = _first_user_turn(row.get("conversations", []))
        if 16 <= len(prompt) <= 4000:  # drop empties / pathological lengths
            got[dom].append({"prompt": prompt, "domain": dom})
        if all(len(got[d]) >= quotas.get(d, 0) for d in quotas):
            break
    out = []
    for d in quotas:
        print(f"  perfectblend {d:6s}: {len(got[d])}/{quotas[d]}")
        out.extend(got[d])
    return out


def build_general(args, rng):
    quotas = {"code": args.n_code_pb, "chat": args.n_chat, "if": args.n_if}
    rows = general_prompts(quotas, rng)
    # eval-aligned code (MBPP train/val/prompt) + small GSM8K math slice
    rows += [{"prompt": r["prompt"], "domain": "code"}
             for r in code_prompts(args.n_mbpp, rng)]
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    rng2 = random.Random(args.seed + 1)
    idx = list(range(len(gsm))); rng2.shuffle(idx)
    rows += [{"prompt": instr_gsm8k(gsm[i]["question"]), "domain": "math"}
             for i in idx[:args.n_gsm8k]]
    rng.shuffle(rows)
    return rows


def main(args):
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.recipe == "general":
        from collections import Counter
        rows = build_general(args, rng)
        path = os.path.join(args.out_dir, "train_general.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(rows)} -> {path}  {dict(Counter(r['domain'] for r in rows))}")
        return

    math = math_prompts(args.n_math, rng)
    code = code_prompts(args.n_code, rng)
    mix = math + code
    rng.shuffle(mix)

    def dump(name, rows):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        c = Counter(r["domain"] for r in rows)
        print(f"  wrote {len(rows):5d} -> {path}  {dict(c)}")

    print("Building prompt sets:")
    dump("train_mix.jsonl", mix)
    dump("train_math.jsonl", math)
    dump("train_code.jsonl", code)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--recipe", choices=["mathcode", "general"], default="mathcode")
    p.add_argument("--n-math", type=int, default=1500)
    p.add_argument("--n-code", type=int, default=1000)
    # general recipe (token-balanced via train_proposal --balance-domains)
    p.add_argument("--n-code-pb", type=int, default=500, help="code prompts from perfectblend (evol-codealpaca)")
    p.add_argument("--n-chat", type=int, default=750, help="chat prompts from perfectblend (ultrachat/lmsys/ultrafeedback)")
    p.add_argument("--n-if", type=int, default=500, help="instruction-following prompts (AutoIF)")
    p.add_argument("--n-mbpp", type=int, default=474, help="eval-aligned MBPP code prompts")
    p.add_argument("--n-gsm8k", type=int, default=400, help="GSM8K math-with-headroom prompts")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())

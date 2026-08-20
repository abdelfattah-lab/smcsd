"""MATH-500 benchmark for SMC speculative decoding and TTS baselines.

Modes share identical preprocessing (chat template, thinking disabled by
flag) so quality/cost comparisons are like-for-like:

  smc_engine : SMC via SMCEngine (posterior-sample output)
  ar         : one stock-engine sample per problem
  majority   : n stock-engine samples per problem, majority vote on the
               normalized boxed answer

Answers are extracted as the last \\boxed{...} group and compared after a
pragmatic LaTeX normalization (consistent across all modes, which is what
fair comparison requires; absolute numbers are conservative).
"""

import argparse
import os
import re
import time
from collections import Counter
from typing import List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

INSTRUCTION = (
    "Solve the following math problem. Reason step by step, and put your "
    "final answer within \\boxed{}.\n\n"
)


def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the last \\boxed{...} with balanced braces."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace(" ", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").rstrip("%")
    s = s.rstrip(".")
    # 0.50 -> 0.5, 3.0 -> 3
    if re.fullmatch(r"-?\d+\.\d*0+", s):
        s = s.rstrip("0").rstrip(".")
    return s or None


def load_math500(tokenizer, num_questions: int, *, disable_thinking: bool):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts, labels = [], []
    kw = {"enable_thinking": False} if disable_thinking else {}
    for row in ds.select(range(num_questions)):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        prompts.append(prompt)
        labels.append(normalize_answer(row["answer"]))
    assert all(l is not None for l in labels), "unparseable gold answer"
    return prompts, labels


def summarize(name, preds: List[Optional[str]], labels, total_tokens, wall):
    n = len(labels)
    invalid = sum(p is None for p in preds)
    acc = sum(p == g for p, g in zip(preds, labels))
    print("=" * 55)
    print(f"  {name}")
    print(f"  Accuracy:          {acc}/{n} ({100 * acc / n:.1f}%)")
    print(f"  Invalid:           {invalid}/{n} ({100 * invalid / n:.1f}%)")
    print(f"  Output throughput: {total_tokens / wall:.1f} tok/s")
    print(f"  Total tokens:      {total_tokens}")
    print(f"  Wall time:         {wall:.1f}s")
    print("=" * 55)


def run_smc(args, prompts, labels):
    from smcsd.engine import SMCEngine

    eng = SMCEngine(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        random_seed=args.seed,
        page_size=1,
        attention_backend=args.attention_backend,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        resample_threshold=args.resample_threshold,
        power_alpha=args.power_alpha,
        tp_size=args.tp,
        disable_custom_all_reduce=args.tp > 1,
        enforce_disable_flashinfer_allreduce_fusion=args.tp > 1,
    )
    try:
        tic = time.perf_counter()
        outs = eng.generate(
            prompts,
            {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        )
        wall = time.perf_counter() - tic
    finally:
        eng.shutdown()
    preds = [normalize_answer(extract_boxed(o["text"])) for o in outs]
    toks = sum(len(o["output_ids"]) for o in outs)
    summarize(
        f"SMCEngine N={args.particles} gamma={args.gamma} seed={args.seed}",
        preds, labels, toks, wall,
    )
    # Particle-level selection rules from the returned collection.
    import math as _math

    maj, wmaj, passk = [], [], []
    for o in outs:
        texts = o.get("smc_particle_texts")
        if not texts:
            maj.append(None); wmaj.append(None); passk.append(None)
            continue
        answers = [normalize_answer(extract_boxed(t)) for t in texts]
        votes = Counter(a for a in answers if a is not None)
        maj.append(votes.most_common(1)[0][0] if votes else None)
        lw = o.get("smc_log_w_tilde") or [0.0] * len(answers)
        m = max(lw)
        wv = {}
        for a, w in zip(answers, lw):
            if a is not None:
                wv[a] = wv.get(a, 0.0) + _math.exp(w - m)
        wmaj.append(max(wv, key=wv.get) if wv else None)
        passk.append(set(a for a in answers if a is not None))
    n = len(labels)
    acc_maj = sum(p == g for p, g in zip(maj, labels))
    acc_wmaj = sum(p == g for p, g in zip(wmaj, labels))
    acc_pass = sum((s is not None and g in s) for s, g in zip(passk, labels))
    print(f"  Particle-majority:  {acc_maj}/{n} ({100 * acc_maj / n:.1f}%)")
    print(f"  Weighted-majority:  {acc_wmaj}/{n} ({100 * acc_wmaj / n:.1f}%)")
    print(f"  Pass@particles:     {acc_pass}/{n} ({100 * acc_pass / n:.1f}%)")


def run_stock(args, prompts, labels, n_samples: int):
    import sglang as sgl

    eng = sgl.Engine(
        model_path=args.model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=args.seed,
        tp_size=args.tp,
        disable_custom_all_reduce=args.tp > 1,
        enforce_disable_flashinfer_allreduce_fusion=args.tp > 1,
    )
    try:
        expanded = [p for p in prompts for _ in range(n_samples)]
        tic = time.perf_counter()
        outs = eng.generate(
            expanded,
            {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature},
        )
        wall = time.perf_counter() - tic
    finally:
        eng.shutdown()
    toks = sum(len(o["output_ids"]) for o in outs)
    if getattr(args, "dump_trajectories", None):
        import json
        with open(args.dump_trajectories, "w") as fh:
            for i in range(len(prompts)):
                for j in range(n_samples):
                    o = outs[i * n_samples + j]
                    a = normalize_answer(extract_boxed(o["text"]))
                    fh.write(json.dumps({
                        "qid": i, "sample": j, "text": o["text"],
                        "answer": a, "gold": labels[i],
                        "correct": bool(a == labels[i]),
                    }) + "\n")
    preds = []
    for i in range(len(prompts)):
        votes = [
            normalize_answer(extract_boxed(outs[i * n_samples + j]["text"]))
            for j in range(n_samples)
        ]
        votes = [v for v in votes if v is not None]
        preds.append(Counter(votes).most_common(1)[0][0] if votes else None)
    name = "AR" if n_samples == 1 else f"majority@{n_samples}"
    summarize(f"{name} seed={args.seed}", preds, labels, toks, wall)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smc_engine", "ar", "majority"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", default=None)
    p.add_argument("--particles", "-N", type=int, default=8)
    p.add_argument("--gamma", "-g", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=8, help="majority mode votes")
    p.add_argument("--num-questions", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--attention-backend", default="triton")
    p.add_argument("--mem-fraction-static", type=float, default=0.75)
    p.add_argument("--max-running-requests", type=int, default=4)
    p.add_argument("--cuda-graph-max-bs", type=int, default=None)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--resample-threshold", type=float, default=0.5)
    p.add_argument("--power-alpha", type=float, default=1.0)
    p.add_argument("--dump-trajectories", default=None,
                   help="stock modes: dump per-sample JSONL with correctness")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, labels = load_math500(
        tok, args.num_questions, disable_thinking=args.disable_thinking
    )
    if args.mode == "smc_engine":
        run_smc(args, prompts, labels)
    elif args.mode == "ar":
        run_stock(args, prompts, labels, 1)
    else:
        run_stock(args, prompts, labels, args.n_samples)


if __name__ == "__main__":
    main()

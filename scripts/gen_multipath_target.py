"""Generate many DIVERSE correct solutions per GSM8k problem from the target.

For multi-path sequence-level distillation: sampling the (accurate) target many
times at high temperature yields several *distinct* correct reasoning paths per
problem. SFT-ing the draft on all of them teaches it varied correct reasoning,
which decorrelates its N samples at inference -> higher coverage@N -> higher SMC
accuracy, with NO change to the SMCSD engine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accuracy_test_gsm8k import extract_answer, normalize_numeric_answer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def safe_norm(x):
    try:
        return normalize_numeric_answer(str(x))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True, help="jsonl with messages[user] + gold_answer")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--target-model", default="meta-llama/Llama-3.1-70B-Instruct"
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start", type=int, default=0, help="skip the first N prompts")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-per-problem", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Problems per target generate call; each expands to --n requests.",
    )
    ap.add_argument("--mem-fraction-static", type=float, default=0.93)
    ap.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Optional SGLang KV-token-pool cap for 70B TP runs.",
    )
    ap.add_argument(
        "--attention-backend",
        choices=("torch_native", "triton", "fa3"),
        default="torch_native",
    )
    ap.add_argument("--tp-size", type=int, default=2,
                    help="Tensor-parallel size for large targets (e.g. 2 for 70B).")
    ap.add_argument("--disable-custom-all-reduce", action="store_true", default=False,
                    help="Use NCCL all-reduce (needed for some multi-GPU TP setups).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file instead of refusing.",
    )
    args = ap.parse_args()
    if args.n < 1 or args.max_per_problem < 1 or args.batch_size < 1:
        raise ValueError("--n, --max-per-problem, and --batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    rows = []
    with open(args.prompts) as f:
        for i, line in enumerate(f):
            if i < args.start:
                continue
            rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break

    tok = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    def user_content(r):
        msgs = r.get("messages") or []
        for m in msgs:
            if m.get("role") == "user":
                return m["content"]
        return None

    prompts = []
    meta = []
    for r in rows:
        uc = user_content(r)
        gold = extract_answer(str(r.get("gold_answer"))) or safe_norm(r.get("gold_answer"))
        if uc is None or gold is None:
            continue
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            ptext = tok.apply_chat_template(
                [{"role": "user", "content": uc}],
                **template_kwargs,
                enable_thinking=False,
            )
        except TypeError:
            ptext = tok.apply_chat_template(
                [{"role": "user", "content": uc}], **template_kwargs
            )
        prompts.append(ptext)
        meta.append({"user": uc, "gold": gold})

    if not prompts:
        raise RuntimeError("No valid teacher prompts were loaded.")
    print(f"problems={len(prompts)} total_gens={len(prompts) * args.n}", flush=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass --overwrite to replace it"
        )

    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.target_model,
        trust_remote_code=True,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=args.seed,
    )
    if args.tp_size > 1:
        engine_kwargs["tp_size"] = args.tp_size
    if args.disable_custom_all_reduce:
        engine_kwargs["disable_custom_all_reduce"] = True
    if args.max_total_tokens is not None:
        engine_kwargs["max_total_tokens"] = args.max_total_tokens
    if args.attention_backend == "torch_native":
        # This backend was the coherent Llama-70B reference in validation and
        # requires eager execution in the supported SGLang snapshot.
        engine_kwargs["disable_cuda_graph"] = True
        engine_kwargs["disable_piecewise_cuda_graph"] = True
    engine = sgl.Engine(**engine_kwargs)
    sp = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    n_written = 0
    n_problems_with = 0
    try:
        with out_path.open("w", encoding="utf-8") as fo:
            for start in range(0, len(prompts), args.batch_size):
                stop = min(start + args.batch_size, len(prompts))
                expanded = [
                    prompt
                    for prompt in prompts[start:stop]
                    for _ in range(args.n)
                ]
                outputs = engine.generate(expanded, sp)
                texts = [output["text"] for output in outputs]
                for local_index, item in enumerate(meta[start:stop]):
                    group = texts[
                        local_index * args.n : (local_index + 1) * args.n
                    ]
                    gold = item["gold"]
                    seen = set()
                    kept = 0
                    for text in group:
                        is_correct = extract_answer(text) == gold
                        if not is_correct:
                            continue
                        key = text.strip()
                        if key in seen:
                            continue
                        seen.add(key)
                        fo.write(
                            json.dumps(
                                {
                                    "sft_messages": [
                                        {"role": "user", "content": item["user"]},
                                        {"role": "assistant", "content": text.strip()},
                                    ],
                                    "source": "gsm8k",
                                    "correct": bool(is_correct),
                                    "teacher_model": args.target_model,
                                    "teacher_sampling": {
                                        "temperature": args.temperature,
                                        "top_p": args.top_p,
                                        "max_new_tokens": args.max_new_tokens,
                                        "seed": args.seed,
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_written += 1
                        kept += 1
                        if kept >= args.max_per_problem:
                            break
                    n_problems_with += int(kept > 0)
                fo.flush()
                os.fsync(fo.fileno())
                print(
                    f"[{stop}/{len(prompts)}] kept={n_written} "
                    f"covered={n_problems_with}",
                    flush=True,
                )
    finally:
        engine.shutdown()
    print(
        f"wrote {n_written} correct solutions across {n_problems_with}/{len(meta)} problems "
        f"(avg {n_written/max(n_problems_with,1):.2f}/problem) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

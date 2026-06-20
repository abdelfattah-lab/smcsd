"""Unified accuracy benchmark for SMC speculative decoding across domains.

Mirrors ``accuracy_test_gsm8k.py`` (same SMCEngine / baseline call pattern and
throughput reporting) but generalizes scoring across tasks so the proposal
finetuning work can measure cross-domain generalization with one harness:

  --task math       MATH-500          (boxed answer, sympy equivalence)
  --task humaneval  OpenAI HumanEval  (pass@1, sandboxed execution)
  --task mbpp       MBPP              (pass@1, sandboxed execution)
  --task gsm8k      GSM8K test        (#### / boxed numeric, same as the
                                       dedicated script — for parity checks)

SMC does not support stop strings, so we generate ``--max-new-tokens`` and
extract the answer post-hoc: the last ``\\boxed{...}`` for math, the first
fenced ``code block`` (or the raw text) for code.

Usage:
  python scripts/eval_tasks.py --task math --mode smc_engine \
      --model Qwen/Qwen3-8B --draft-model Qwen/Qwen3-0.6B \
      -N 8 -g 8 --num-questions 200 --disable-thinking
"""

import argparse
import re
import time
from typing import Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def last_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST ``\\boxed{...}`` with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def _normalize_math(s: str) -> str:
    """Light textual normalization before sympy parsing."""
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\$", "").replace("$", "").replace("%", "")
    s = s.replace("\\%", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace(",", "")  # thousands separators
    s = s.rstrip(".")
    return s.strip()


def math_equal(pred: Optional[str], gold: str) -> bool:
    """MATH equivalence: exact-normalized string match, else sympy equality."""
    if pred is None:
        return False
    p, g = _normalize_math(pred), _normalize_math(gold)
    if p == g:
        return True
    # Try numeric / symbolic equality via sympy's latex parser.
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        ep, eg = parse_latex(p), parse_latex(g)
        diff = simplify(ep - eg)
        if diff == 0:
            return True
        # numeric fallback
        try:
            return abs(float(ep.evalf()) - float(eg.evalf())) < 1e-6
        except Exception:
            return False
    except Exception:
        return False


def extract_gsm8k(text: str) -> Optional[str]:
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    b = last_boxed(text)
    if b is not None:
        nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", b)
        if nums:
            return nums[-1].replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last = lines[-1] if lines else text.strip()
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last)
    return nums[-1].replace(",", "") if nums else None


def extract_code(text: str) -> str:
    """Pull the first fenced code block; fall back to the whole text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # no fence — strip a leading language hint line if present
    return text


# ---------------------------------------------------------------------------
# Sandboxed code execution (pass@1)
# ---------------------------------------------------------------------------

def run_program(program: str, timeout: float = 10.0) -> bool:
    """Run a self-contained program in a subprocess; True iff exit code 0."""
    import subprocess
    import sys
    import tempfile
    import os

    harness = (
        "import signal, resource, sys, math\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))\n"
        "except Exception:\n    pass\n"
        f"signal.alarm(int({timeout}))\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False
    ) as f:
        f.write(harness + program)
        path = f.name
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout + 2,
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Instruction templates (shared with scripts/make_prompt_sets.py so the
# training prompt distribution matches eval exactly).
# ---------------------------------------------------------------------------

def instr_gsm8k(question: str) -> str:
    return (
        "Solve this math problem step by step.\nAt the very end, output ONLY "
        "the final numeric answer on a new line in the exact format:\n"
        f"#### <number>\n\nProblem:\n{question}\n"
    )


def instr_math(problem: str) -> str:
    return (
        "Solve the following math problem step by step. Put your final answer "
        f"inside \\boxed{{}}.\n\nProblem:\n{problem}\n"
    )


def instr_humaneval(prompt: str) -> str:
    return (
        "Complete the following Python function. Return the COMPLETE "
        "function (signature + body) in a single ```python code block, "
        "with no example usage.\n\n```python\n" + prompt + "\n```\n"
    )


def instr_mbpp(text: str, test_list) -> str:
    asserts = "\n".join(test_list)
    return (
        "Write a single Python function that solves the task below. "
        "Return only the function in a ```python code block. It must pass "
        f"these tests:\n{asserts}\n\nTask: {text}\n"
    )


# ---------------------------------------------------------------------------
# Task definitions: load() -> (instructions, refs), score(text, ref) -> bool
# ---------------------------------------------------------------------------

def task_gsm8k(num_questions):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(
        range(num_questions)
    )
    instr = [instr_gsm8k(s["question"]) for s in ds]
    refs = [extract_gsm8k(s["answer"]) for s in ds]
    return instr, refs, lambda t, r: extract_gsm8k(t) == r


def task_math(num_questions):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(
        range(num_questions)
    )
    instr = [instr_math(s["problem"]) for s in ds]
    refs = [s["answer"] for s in ds]
    return instr, refs, lambda t, r: math_equal(last_boxed(t), r)


def task_humaneval(num_questions):
    ds = load_dataset("openai/openai_humaneval", split="test")
    n = min(num_questions, len(ds))
    ds = ds.select(range(n))
    instr, refs = [], []
    for s in ds:
        instr.append(instr_humaneval(s["prompt"]))
        refs.append(
            {"test": s["test"], "entry_point": s["entry_point"],
             "prompt": s["prompt"]}
        )

    def score(text, ref):
        code = extract_code(text)
        # ensure the entry point is defined; if the model omitted the
        # signature, prepend the provided prompt stub.
        if f"def {ref['entry_point']}" not in code:
            code = ref["prompt"] + "\n" + code
        program = code + "\n" + ref["test"] + f"\ncheck({ref['entry_point']})\n"
        return run_program(program)

    return instr, refs, score


def task_mbpp(num_questions):
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    n = min(num_questions, len(ds))
    ds = ds.select(range(n))
    instr, refs = [], []
    for s in ds:
        instr.append(instr_mbpp(s["text"], s["test_list"]))
        refs.append(
            {"setup": s.get("test_setup_code", ""), "tests": s["test_list"]}
        )

    def score(text, ref):
        code = extract_code(text)
        program = (
            (ref["setup"] + "\n" if ref["setup"] else "")
            + code + "\n" + "\n".join(ref["tests"]) + "\n"
        )
        return run_program(program)

    return instr, refs, score


TASKS = {
    "gsm8k": task_gsm8k,
    "math": task_math,
    "humaneval": task_humaneval,
    "mbpp": task_mbpp,
}


# ---------------------------------------------------------------------------
# Engine runners (mirror accuracy_test_gsm8k.py)
# ---------------------------------------------------------------------------

def build_prompts(tokenizer, instructions, disable_thinking):
    kw = {}
    if disable_thinking:
        kw["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ins}],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        for ins in instructions
    ]


def run_smc_engine(args, prompts):
    from smcsd.engine import SMCEngine

    engine_kwargs = dict(
        model_path=args.model,
        draft_model_path=args.draft_model,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        trust_remote_code=True,
        page_size=1,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests or max(args.particles + 4, 16),
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    if args.resample_threshold is not None:
        engine_kwargs["resample_threshold"] = args.resample_threshold
    sp = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    texts, total_tok = [], 0
    with SMCEngine(**engine_kwargs) as engine:
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            outs = engine.generate(prompts[start : start + args.batch_size], sp)
            if not isinstance(outs, list):
                outs = [outs]
            for o in outs:
                texts.append(o["text"])
                total_tok += o["completion_tokens"]
            print(f"\r[gen {len(texts)}/{len(prompts)}] "
                  f"tps={total_tok / (time.perf_counter() - tic):.0f}", end="", flush=True)
        latency = time.perf_counter() - tic
    print()
    return texts, total_tok, latency


def run_baseline(args, prompts):
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model, trust_remote_code=True,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
    )
    if args.seed is not None:
        engine_kwargs["random_seed"] = args.seed
    sp = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    texts, total_tok = [], 0
    with sgl.Engine(**engine_kwargs) as engine:
        tic = time.perf_counter()
        for start in range(0, len(prompts), args.batch_size):
            outs = engine.generate(prompts[start : start + args.batch_size], sp)
            for o in outs:
                texts.append(o["text"])
                total_tok += o["meta_info"]["completion_tokens"]
            print(f"\r[gen {len(texts)}/{len(prompts)}] "
                  f"tps={total_tok / (time.perf_counter() - tic):.0f}", end="", flush=True)
        latency = time.perf_counter() - tic
    print()
    return texts, total_tok, latency


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    print(f"Task: {args.task} | Mode: {args.mode} | Model: {args.model}")
    if args.mode == "smc_engine":
        print(f"  draft={args.draft_model} N={args.particles} g={args.gamma} "
              f"temp={args.temperature}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    instructions, refs, score = TASKS[args.task](args.num_questions)
    prompts = build_prompts(tokenizer, instructions, args.disable_thinking)
    print(f"  {len(prompts)} prompts loaded")

    if args.mode == "smc_engine":
        texts, total_tok, latency = run_smc_engine(args, prompts)
    else:
        texts, total_tok, latency = run_baseline(args, prompts)

    correct = sum(score(t, r) for t, r in zip(texts, refs))
    n = len(texts)
    print(f"\n{'=' * 55}")
    print(f"  {args.task} | {args.mode}"
          + (f" | N={args.particles} g={args.gamma}" if args.mode == "smc_engine" else ""))
    print(f"{'=' * 55}")
    print(f"  Accuracy:          {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"  Output throughput: {total_tok / latency:.1f} tok/s")
    print(f"  Wall time:         {latency:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--mode", choices=["smc_engine", "baseline"], default="smc_engine")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    p.add_argument("--particles", "-N", type=int, default=8)
    p.add_argument("--gamma", "-g", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--resample-threshold", type=float, default=None)
    p.add_argument("--num-questions", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--disable-thinking", action="store_true", default=False)
    p.add_argument("--attention-backend", default="triton", choices=["triton", "fa3"])
    p.add_argument("--mem-fraction-static", type=float, default=0.4)
    p.add_argument("--max-running-requests", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    main(p.parse_args())

#!/usr/bin/env python3
"""Heldout multi-domain diagnostic for SMC-SD EAGLE drafts.

Runs SMC-SD on a held-out prompt set (default: last N rows of
PerfectBlend-Regenerated, never seen during warmstart training) and aggregates
target-only chain diagnostics. The output single-number summary is the
"generalization gauge" — a domain-agnostic measure of how well EAGLE matches
target's per-step distribution outside its training distribution.

Usage:
  python scripts/heldout_diag.py \
    --target meta-llama/Llama-3.1-8B-Instruct \
    --draft <path-or-name> \
    --num-prompts 200 \
    --gamma 4 \
    --output outputs/heldout_diag/<run-name>.jsonl

Reports: top-20 overlap mean, sample-in-target-top-20 mean, target-rank median,
target_minus_draft_lp mean+median, ESS/N mean+median, logw_var mean+median.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--draft", required=True, help="EAGLE3 ckpt path or HF name")
    p.add_argument("--draft-mode", default="eagle3_chain")
    p.add_argument("--data", default=None,
                   help="JSONL of prompts (SpecForge schema). If None, uses "
                        "the *last* --num-prompts of the 200k warmstart data.")
    p.add_argument("--source-jsonl", default=None,
                   help="Pull prompts from this JSONL. If None, defaults to "
                        "the 200k warmstart data and slices the last "
                        "--num-prompts (so they're never seen during warmstart "
                        "or KL-onpolicy training).")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--particles", "-N", type=int, default=12)
    p.add_argument("--gamma", "-g", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-running-requests", type=int, default=24)
    p.add_argument("--cuda-graph-max-bs", type=int, default=24)
    p.add_argument("--eagle-topk", type=int, default=20)
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument("--device", default="0")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write chain_diag.jsonl + smc_metrics.jsonl + summary.json")
    p.add_argument("--smcsd-root", default="/home/yahya/smcsd")
    p.add_argument("--source-skip-last", type=int, default=0,
                   help="When using default --source-jsonl, skip the last N rows after slicing "
                        "for sharding eval across multiple runs.")
    p.add_argument("--python", default="/home/yahya/miniconda3/envs/smcsd/bin/python")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. build the heldout prompt JSONL ----
    if args.source_jsonl is None:
        args.source_jsonl = "/home/yahya/SpecForge/cache/dataset/llama31_8b_smc_warmstart_200k.jsonl"

    src = Path(args.source_jsonl)
    assert src.exists(), f"source jsonl not found: {src}"

    prompt_path = out_dir / "heldout_prompts.jsonl"
    if not prompt_path.exists():
        # Read all rows, take the last --num-prompts (after optional skip)
        # to ensure they're well past the warmstart training set's typical
        # access pattern.
        with open(src) as f:
            rows = f.readlines()
        end = len(rows) - args.source_skip_last
        start = max(end - args.num_prompts, 0)
        slice_ = rows[start:end]
        with open(prompt_path, "w") as f:
            f.writelines(slice_)
        print(f"[heldout] wrote {len(slice_)} prompts to {prompt_path} "
              f"(slice [{start}:{end}] of {len(rows)})", flush=True)

    # ---- 2. run accuracy_test_gsm8k.py with --custom-prompts ----
    # the accuracy script supports running on arbitrary prompt JSONL only via
    # --num-questions on GSM8K. We run it against GSM8K but since the goal is
    # JUST diagnostics, we run a small num-questions, ignore the gsm8k accuracy,
    # and aggregate the chain_diag JSONL only. Simpler: spawn a custom mini-loop
    # that uses smcsd Engine directly to run on these prompts.
    # For simplicity, copy the heldout prompts into a temporary "fake gsm8k"
    # path and run accuracy_test_gsm8k with a custom dataset arg if available.
    # If not, we use a small inline runner below.

    runner_path = out_dir / "_runner.py"
    runner_path.write_text("""
import argparse, json, sys, os, time
# rely on editable installs of smcsd + sglang in the smcsd conda env;
# do NOT prepend '/home/yahya/smcsd/python' (it shadows the vendored sglang).

def main():
    from smcsd.engine import SMCEngine
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--draft', required=True)
    ap.add_argument('--draft-mode', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--particles', type=int, required=True)
    ap.add_argument('--gamma', type=int, required=True)
    ap.add_argument('--temperature', type=float, required=True)
    ap.add_argument('--max-new-tokens', type=int, required=True)
    ap.add_argument('--max-running-requests', type=int, required=True)
    ap.add_argument('--cuda-graph-max-bs', type=int, required=True)
    ap.add_argument('--eagle-topk', type=int, required=True)
    ap.add_argument('--attention-backend', required=True)
    ap.add_argument('--diag-path', required=True)
    ap.add_argument('--metrics-path', required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = []
    for line in open(args.prompts):
        row = json.loads(line)
        convs = row.get('conversations', [])
        if convs and convs[-1].get('role') == 'assistant':
            msgs = convs[:-1]
        else:
            msgs = convs
        if not msgs:
            continue
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f'[runner] loaded {len(prompts)} prompts', flush=True)

    eng = SMCEngine(
        model_path=args.target,
        draft_model_path=args.draft,
        draft_mode=args.draft_mode,
        n_particles=args.particles,
        gamma=args.gamma,
        draft_temperature=args.temperature,
        target_temperature=args.temperature,
        eagle_topk=args.eagle_topk,
        eagle3_collect_path=args.diag_path,
        smc_metrics=True,
        smc_metrics_jsonl=args.metrics_path,
        smc_metrics_log_interval=999999,
        attention_backend=args.attention_backend,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=1,
        log_level='error',
    )

    t0 = time.perf_counter()
    sampling_params = dict(temperature=args.temperature, top_p=1.0, top_k=-1, max_new_tokens=args.max_new_tokens, ignore_eos=False)
    out = eng.generate(prompts, sampling_params=sampling_params)
    elapsed = time.perf_counter() - t0
    out_list = out if isinstance(out, list) else [out]
    total_tokens = sum(o.get('completion_tokens', len(o.get('output_ids', []))) for o in out_list)
    print(f'[runner] {len(prompts)} prompts, {total_tokens} tokens, {elapsed:.1f}s, {total_tokens/elapsed:.1f} tps', flush=True)
    print(f'[runner] diag -> {args.diag_path}', flush=True)
    print(f'[runner] metrics -> {args.metrics_path}', flush=True)

if __name__ == '__main__':
    main()
""")

    diag_path = out_dir / "chain_diag.jsonl"
    metrics_path = out_dir / "smc_metrics.jsonl"
    log_path = out_dir / "runner.log"

    cmd = [
        args.python, str(runner_path),
        "--target", args.target,
        "--draft", args.draft,
        "--draft-mode", args.draft_mode,
        "--prompts", str(prompt_path),
        "--particles", str(args.particles),
        "--gamma", str(args.gamma),
        "--temperature", str(args.temperature),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-running-requests", str(args.max_running_requests),
        "--cuda-graph-max-bs", str(args.cuda_graph_max_bs),
        "--eagle-topk", str(args.eagle_topk),
        "--attention-backend", args.attention_backend,
        "--diag-path", str(diag_path),
        "--metrics-path", str(metrics_path),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"[heldout] launching runner CUDA_VISIBLE_DEVICES={args.device}", flush=True)
    with open(log_path, "w") as logf:
        rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

    print(f"[heldout] runner exit code: {rc.returncode}", flush=True)
    if rc.returncode != 0:
        print(open(log_path).read()[-2000:])
        sys.exit(rc.returncode)

    # ---- 3. aggregate chain diagnostics ----
    if not diag_path.exists():
        print(f"ERROR: diag jsonl not produced at {diag_path}", flush=True)
        sys.exit(1)

    diag_rows = [json.loads(l) for l in open(diag_path)]
    metrics_rows = []
    if metrics_path.exists():
        metrics_rows = [json.loads(l) for l in open(metrics_path)]

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        vals.sort()
        return dict(
            mean=statistics.mean(vals),
            median=statistics.median(vals),
            p10=vals[len(vals)//10],
            p90=vals[9*len(vals)//10],
            n=len(vals),
        )

    summary = dict(
        n_diag_rows=len(diag_rows),
        n_metric_rows=len(metrics_rows),
        gamma=args.gamma,
        particles=args.particles,
        chain_diag={
            k: stats([r[k] for r in diag_rows if k in r])
            for k in [
                "draft_target_topk_overlap_mean",
                "sample_in_target_topk_mean",
                "sample_target_rank_median",
                "target_minus_draft_lp_mean",
            ]
        },
    )
    if metrics_rows:
        ess = [r["aggregate"]["ess_frac_mean"] for r in metrics_rows]
        lvar = [r["aggregate"]["logw_var_mean"] for r in metrics_rows]
        mw = [r["aggregate"]["max_weight_mean"] for r in metrics_rows]
        summary["smc_metrics"] = dict(
            ess_frac=stats(ess),
            logw_var=stats(lvar),
            max_weight=stats(mw),
        )

    summary_path = out_dir / "summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[heldout] summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

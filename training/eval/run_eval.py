"""CLI: evaluate a proposal checkpoint against the target on GSM8K test prefixes.

Runs the block-chi^2 / ESS harness and prints a table. Optionally dumps JSON.

Usage:
    # warm-start-only baseline (no checkpoint)
    CUDA_VISIBLE_DEVICES=7 python -m training.eval.run_eval --num-prefixes 256

    # evaluate a trained checkpoint
    CUDA_VISIBLE_DEVICES=7 python -m training.eval.run_eval \
        --checkpoint path/to/proposal.pt --num-prefixes 1319 \
        --output-json training/eval_results/ckpt_100M.json
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.eval.harness import compute_metrics, sample_from_proposal, score_with_target
from training.model.proposal import ProposalConfig, build_proposal


SYSTEM = "You are a helpful assistant that solves math problems step by step."


def load_gsm8k_test_prefixes(tokenizer, n: int | None) -> list[list[int]]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for ex in ds:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    return out


def load_proposal(checkpoint: str | None, target_model, device):
    proposal = build_proposal(ProposalConfig(), dtype=torch.bfloat16).to(device)
    # Always copy target's embeddings first — cheap, keeps behavior consistent
    # whether the checkpoint saved them or not.
    with torch.no_grad():
        proposal.get_input_embeddings().weight.data.copy_(
            target_model.get_input_embeddings().weight.data
        )
    if checkpoint is not None:
        # weights_only=False: trusted checkpoints from our own trainer; full dict
        # carries optimizer + numpy RNG state which the safe loader rejects.
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = proposal.load_state_dict(state, strict=False)
        print(f"loaded checkpoint: {checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("no checkpoint — warm-start-only baseline")
    proposal.eval()
    return proposal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--target", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--num-prefixes", type=int, default=256)
    ap.add_argument("--n-particles", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--k-values", default="1,2,4,8")
    ap.add_argument("--prefix-batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    k_values = sorted(int(x) for x in args.k_values.split(","))
    assert max(k_values) <= args.k_max

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prefixes = load_gsm8k_test_prefixes(tokenizer, n=args.num_prefixes)
    print(f"prefixes: {len(prefixes)} | prefix-batch: {args.prefix_batch_size} | N={args.n_particles} | K_max={args.k_max} | T={args.temperature}")

    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16).to(device).eval()
    proposal = load_proposal(args.checkpoint, target, device)

    M = len(prefixes)
    N = args.n_particles
    log_p_all = torch.zeros((M, N, args.k_max))
    log_q_all = torch.zeros((M, N, args.k_max))

    for s in tqdm(range(0, M, args.prefix_batch_size), desc="eval"):
        p_batch = prefixes[s : s + args.prefix_batch_size]
        expanded = [p for p in p_batch for _ in range(N)]

        gen, log_q = sample_from_proposal(
            proposal, expanded, k_max=args.k_max,
            pad_id=pad_id, device=device, temperature=args.temperature,
        )
        log_p = score_with_target(
            target, expanded, gen,
            pad_id=pad_id, device=device, temperature=args.temperature,
        )
        # Reshape [B*N, K] into [B, N, K] slice of the aggregate buffer
        B = len(p_batch)
        log_q_all[s : s + B] = log_q.view(B, N, args.k_max).cpu()
        log_p_all[s : s + B] = log_p.view(B, N, args.k_max).cpu()

    log_w_per_step = log_p_all - log_q_all
    summary, per_prefix = compute_metrics(log_w_per_step, k_values)

    print()
    header = f"{'K':>3}  {'log(1+chi2)':>11}  {'chi2':>12}  {'ESS/N mean':>11}  {'ESS/N med':>10}  {'regime':>11}  {'mean log_w':>11}  {'log_w [5% 50% 95%]':>25}"
    print(header)
    print("-" * len(header))
    for k in k_values:
        m = summary[k]
        chi2_str = f"{m['chi2']:.3e}" if abs(m["chi2"]) >= 100 else f"{m['chi2']:.3f}"
        q_str = f"[{m['log_w_q05']:.2f} {m['median_log_w']:.2f} {m['log_w_q95']:.2f}]"
        print(f"{k:>3}  {m['log_1p_chi2']:>11.3f}  {chi2_str:>12}  {m['ess_over_n_mean']:>11.4f}  {m['ess_over_n_median']:>10.4f}  {m['ess_regime']:>11}  {m['mean_log_w']:>11.3f}  {q_str:>25}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "config": vars(args),
                "summary": summary,
                "per_prefix": per_prefix,
            }, f, indent=2)
        print(f"\nsaved: {args.output_json}")


if __name__ == "__main__":
    main()

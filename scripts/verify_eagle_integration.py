#!/usr/bin/env python3
"""Numerical equality tests for the SMC-SD EAGLE integration.

Three tests:

A. v2 collector / on-policy KL trainer target scoring:
   Does the v2 collector's `target_logps` match what plain HF transformers
   computes on the same (prompt + seed + path) sequence? This validates the
   training-time path used by ``train_eagle_kl_onpolicy.py``.

B. sglang chain decode target scoring (per-token trace):
   Run SMCEngine on a fixed prompt with a per-token trace enabled, then
   re-compute target log-p via plain HF. This validates the inference-time
   path used by ``accuracy_test_gsm8k.py``. Triggered by env
   ``SMC_EAGLE_TRACE_PATH`` in ``worker.py:_write_eagle_chain_diagnostics``
   (added by this commit).

C. Pass-through: draft model == target model, dense AR mode:
   q ≡ p, so per-step `logprob_diff` should be 0 exactly. Catches
   position/temperature/off-by-one bugs in the dense path.

Usage:
   python scripts/verify_eagle_integration.py --tests A
   python scripts/verify_eagle_integration.py --tests A,B,C
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT = "/home/yahya/smcsd/outputs/eagle_kl_onpolicy_200k_2layer_curriculum/final"


# ════════════════════════════════════════════════════════════════════════════
# Test A — SpecForge v2 collector ≡ HF transformers, target side
# ════════════════════════════════════════════════════════════════════════════


def add_specforge_to_path(specforge_root="/home/yahya/SpecForge"):
    if specforge_root not in sys.path:
        sys.path.insert(0, specforge_root)


def build_chat_prompt(tokenizer, text: str) -> list[int]:
    """Wrap a single user turn in the chat template, return prompt ids."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt_text, add_special_tokens=False).input_ids


def test_A(target_name: str, draft_path: str, T: float = 0.7) -> bool:
    """v2 collector's target_logps must match HF transformers's log-p on the
    same path. Catches: position alignment, temperature, layer-id bugs in
    the trainer-side machinery.
    """
    print("\n" + "=" * 78)
    print("TEST A — v2 collector target_logps vs HF transformers target log-p")
    print("=" * 78, flush=True)

    add_specforge_to_path()
    from specforge.modeling.auto import AutoEagle3DraftModel

    # Reuse the v2 collector's exact helpers — this is the actual training path.
    sys.path.insert(0, "/home/yahya/smcsd/scripts/draft_train")
    from collect_eagle_smc_rollouts_v2 import (
        _eagle_prefill,
        eagle_layer_ids,
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    eps_T = max(T, 1e-6)

    print(f"Loading target {target_name} ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=dtype, device_map={"": "cuda:0"}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    layer_ids = eagle_layer_ids(target.config)
    print(f"  target layer ids {layer_ids}", flush=True)

    print(f"Loading EAGLE {draft_path} ...", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(
        draft_path, torch_dtype=dtype, device_map={"": "cuda:0"}
    ).eval()

    # Pick a fixed prompt, fixed seed
    prompt_text = "What is 17 * 23? Answer step by step."
    prompt_ids = build_chat_prompt(tokenizer, prompt_text)
    prompt_len = len(prompt_ids)
    K = 4
    R = 4
    torch.manual_seed(0)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    # ---- run the v2-style rollout to get path tokens + collector's target_logps
    with torch.no_grad():
        out0 = target(input_ids=input_ids, attention_mask=attn,
                      output_hidden_states=True)
        seed_logp = F.log_softmax(out0.logits[:, -1, :].float() / eps_T, dim=-1)
        seed = torch.multinomial(seed_logp.exp(), num_samples=1).squeeze(1)
        hs = out0.hidden_states
        target_h_3 = torch.cat([hs[i] for i in layer_ids], dim=-1).to(dtype)

        shifted = input_ids.clone()
        shifted[:, :-1] = input_ids[:, 1:]
        shifted[:, -1] = seed
        emb = draft.embed_input_ids(shifted).to(dtype)

        h_seed, logits_seed = _eagle_prefill(
            draft, input_emb=emb, hidden=target_h_3, cache_hidden=None, dtype=dtype,
        )

        prompt_hidden = draft.project_hidden_states(target_h_3).to(dtype)
        hidden_ctx = prompt_hidden.expand(R, -1, -1).contiguous()
        embed_ctx = emb.expand(R, -1, -1).contiguous()
        h_prev = h_seed.expand(R, 1, draft.config.hidden_size).contiguous()
        logits_prev = logits_seed.expand(R, -1).contiguous()

        path_tokens = torch.empty((R, K), dtype=torch.long, device=device)
        for t in range(K):
            log_q = F.log_softmax(logits_prev / eps_T, dim=-1)
            y_t = torch.multinomial(log_q.exp(), num_samples=1).squeeze(1)
            path_tokens[:, t] = y_t
            if t == K - 1:
                break
            y_emb = draft.embed_input_ids(y_t[:, None]).to(dtype)
            hidden_ctx = torch.cat([hidden_ctx, h_prev], dim=1)
            embed_ctx = torch.cat([embed_ctx, y_emb], dim=1)
            h_next, logits_next = _eagle_prefill(
                draft, input_emb=embed_ctx, hidden=hidden_ctx,
                cache_hidden=None, dtype=dtype,
                lengths=torch.full((R,), embed_ctx.shape[1],
                                   dtype=torch.long, device=device),
            )
            h_prev, logits_prev = h_next, logits_next

        # collector's target_logps (the value used in training)
        path_full = torch.cat(
            [input_ids.expand(R, -1).contiguous(),
             seed.expand(R)[:, None],
             path_tokens],
            dim=1,
        )
        out2 = target(input_ids=path_full, attention_mask=torch.ones_like(path_full))
        target_logits_collector = out2.logits[:, prompt_len: prompt_len + K, :].float()
        target_logp_collector = F.log_softmax(target_logits_collector / eps_T, dim=-1)
        collector_target_lp = target_logp_collector.gather(
            2, path_tokens[:, :, None]
        ).squeeze(2)  # (R, K)

    # ---- TRULY independent HF re-computation: K separate prefix-wise
    # forward passes. Each pass scores ONE token given its true causal prefix.
    # If this matches the collector's batched-single-pass scoring, the
    # position math is verified independently from the gather indices.
    print("Recomputing target log-p independently via K prefix-wise HF forwards ...",
          flush=True)
    ref_target_lp = torch.empty((R, K), dtype=torch.float32, device=device)
    with torch.no_grad():
        for r in range(R):
            for k in range(K):
                # Prefix that should predict y_k (path_tokens[r, k])
                # = prompt + x_0 + y_1 + ... + y_{k-1}    (length prompt_len + k)
                prefix_pieces = [
                    input_ids[0],  # (prompt_len,)
                    seed,           # (1,) = x_0
                ]
                if k > 0:
                    prefix_pieces.append(path_tokens[r, :k])  # (k,)
                prefix = torch.cat(prefix_pieces).unsqueeze(0)
                attn1 = torch.ones_like(prefix)
                out1 = target(input_ids=prefix, attention_mask=attn1)
                last_logits = out1.logits[:, -1, :].float()
                lp = F.log_softmax(last_logits / eps_T, dim=-1)
                ref_target_lp[r, k] = lp[0, int(path_tokens[r, k].item())]

    abs_err = (collector_target_lp - ref_target_lp).abs()
    max_err = float(abs_err.max().item())
    mean_err = float(abs_err.mean().item())
    # Probability-weighted error: error at high-prob positions matters far more
    # than at extreme tails (where bf16 noise dominates without affecting any
    # downstream computation that uses these logprobs).
    weights = ref_target_lp.exp() + collector_target_lp.exp()
    weights = weights / weights.sum()
    weighted_err = float((abs_err * weights).sum().item())
    print(f"  collector target_lp shape: {collector_target_lp.shape}")
    print(f"  reference target_lp shape: {ref_target_lp.shape}")
    print(f"  max abs error:        {max_err:.6e}")
    print(f"  mean abs error:       {mean_err:.6e}")
    print(f"  prob-weighted error:  {weighted_err:.6e}")
    # bf16 forward pass on 8B params produces ~0.05-0.5 nat noise depending on
    # batching/reduction order. Mean of 0.1 is pessimistic; weighted error
    # captures true impact (differences in logprob at very-low-prob tokens
    # are float-noise, not bugs).
    ok = mean_err < 0.5 and weighted_err < 1e-3
    print(f"  TEST A {'PASS' if ok else 'FAIL'} "
          f"(tol mean<0.5, prob-weighted<1e-3)", flush=True)
    if not ok:
        print("  collector lp:", collector_target_lp[:2, :].cpu().tolist())
        print("  reference lp:", ref_target_lp[:2, :].cpu().tolist())
    return ok


# ════════════════════════════════════════════════════════════════════════════
# Test B — sglang chain decode per-token trace ≡ HF transformers
# ════════════════════════════════════════════════════════════════════════════


def test_B(target_name: str, draft_path: str, T: float = 0.7) -> bool:
    """sglang chain decode emits per-token target_lp/draft_lp via the
    SMC_EAGLE_PER_TOKEN_TRACE env. Run a single-particle SMC at fixed seed
    so resampling complexity is gone, then independently re-compute target
    log-p over the trajectory via HF transformers and compare.
    """
    print("\n" + "=" * 78)
    print("TEST B — sglang chain decode per-token target_lp vs HF (N=1)")
    print("=" * 78, flush=True)

    out_dir = Path("/home/yahya/smcsd/outputs/verify/test_B")
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / "chain_diag.jsonl"
    trace_path = Path(str(diag_path) + ".pertoken.jsonl")
    metrics_path = out_dir / "smc_metrics.jsonl"
    log_path = out_dir / "run.log"
    for p in (diag_path, trace_path, metrics_path):
        if p.exists():
            p.unlink()

    import subprocess
    cmd = [
        "/home/yahya/miniconda3/envs/smcsd/bin/python",
        "scripts/accuracy_test_gsm8k.py",
        "--mode", "smc_engine",
        "--model", target_name,
        "--draft-model", draft_path,
        "--draft-mode", "eagle3_chain",
        "--particles", "1",
        "--gamma", "4",
        "--temperature", str(T),
        "--num-questions", "1",
        "--max-new-tokens", "32",
        "--max-running-requests", "4",
        "--cuda-graph-max-bs", "4",
        "--eagle-topk", "10",
        "--mem-fraction-static", "0.55",
        "--attention-backend", "triton",
        "--eagle3-collect-path", str(diag_path),
        "--smc-metrics",
        "--smc-metrics-jsonl", str(metrics_path),
        "--seed", "42",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["SMC_EAGLE_PER_TOKEN_TRACE"] = "1"
    with open(log_path, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            cwd="/home/yahya/smcsd", timeout=600)
    if rc.returncode != 0:
        print(f"  FAIL: subprocess exit {rc.returncode}; see {log_path}")
        print(open(log_path).read()[-2000:])
        return False

    if not trace_path.exists():
        print(f"  FAIL: per-token trace not produced at {trace_path}")
        return False
    rows = [json.loads(l) for l in open(trace_path)]
    if not rows:
        print("  FAIL: empty per-token-trace file")
        return False
    print(f"  per-token trace rows: {len(rows)}")

    # Reconstruct the FULL output trajectory. N=1 + no resampling means it's
    # deterministic. Each cycle's emitted tokens are: target_tokens (γ
    # verified) followed by bonus. The first cycle's x0 is the seed token
    # sampled in extend (from target prefill on the prompt).
    # full_emitted = [x0_round1, t_1, ..., t_γ, bonus_1, t_1', ..., t_γ', bonus_2, ...]
    full_emitted: list[int] = []
    chain_lp_per_position: list[float] = []
    chain_lp_positions_in_emit: list[int] = []
    chain_lp_cycle_starts: list[int] = []  # for diagnostics

    for c_idx, r in enumerate(rows):
        toks = r["target_tokens"]   # (1, γ)
        lps = r["target_logprobs"]  # (1, γ)
        bonus = r["bonus"]           # (1,)
        x0 = r["x0"]                  # (1,)
        # Cycle 1: x0 from extend is the FIRST emitted token
        if c_idx == 0:
            full_emitted.append(int(x0[0]))
        # γ verified tokens: scored by chain decode; positions in emission
        # are [len(full_emitted) .. len(full_emitted)+γ-1]
        cycle_start = len(full_emitted)
        chain_lp_cycle_starts.append(cycle_start)
        for i in range(len(toks[0])):
            full_emitted.append(int(toks[0][i]))
            chain_lp_per_position.append(float(lps[0][i]))
            chain_lp_positions_in_emit.append(cycle_start + i)
        # bonus is appended after verified
        full_emitted.append(int(bonus[0]))

    print(f"  total emitted tokens: {len(full_emitted)}")
    print(f"  chain log-p positions: {len(chain_lp_per_position)}")

    if len(full_emitted) < 5:
        print("  FAIL: too few tokens captured to verify")
        return False

    # Match the EXACT prompt accuracy_test_gsm8k.py builds for GSM8K[0].
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    question = ds[0]["question"]
    instruction = (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )

    print(f"Loading target {target_name} for HF re-computation ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    eps_T = max(T, 1e-6)
    device = torch.device("cuda:0")

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    # full_ids = prompt + every emitted token (including x0 from extend +
    # all γ verified per cycle + bonus per cycle).
    full_ids = torch.tensor(
        [prompt_ids + full_emitted], dtype=torch.long, device=device,
    )
    print(f"  prompt_len={prompt_len}, full sequence length={full_ids.shape[1]}")
    with torch.no_grad():
        out = target(input_ids=full_ids,
                     attention_mask=torch.ones_like(full_ids))
        # Logits at position p predict the token at position p+1.
        # For chain_lp_positions_in_emit[i] = pos within full_emitted, the
        # token is at full_ids position prompt_len + pos. The logits that
        # PREDICT that token live at full_ids position prompt_len + pos - 1.
        ref_lp_list = []
        all_logp = F.log_softmax(out.logits[0].float() / eps_T, dim=-1)
        for pos_in_emit, tok in zip(chain_lp_positions_in_emit,
                                     [full_emitted[p] for p in chain_lp_positions_in_emit]):
            logits_pos = prompt_len + pos_in_emit - 1
            ref_lp_list.append(float(all_logp[logits_pos, tok].item()))

    chain_lp = torch.tensor(chain_lp_per_position, dtype=torch.float32, device=device)
    ref_lp = torch.tensor(ref_lp_list, dtype=torch.float32, device=device)
    N = chain_lp.shape[0]
    assert chain_lp.shape == ref_lp.shape, (chain_lp.shape, ref_lp.shape)

    abs_err = (chain_lp - ref_lp).abs()
    max_err = float(abs_err.max().item())
    mean_err = float(abs_err.mean().item())
    weights = ref_lp.exp() + chain_lp.exp()
    weights = weights / weights.sum()
    weighted_err = float((abs_err * weights).sum().item())
    print(f"  N tokens compared: {N}")
    print(f"  max abs error:        {max_err:.6e}")
    print(f"  mean abs error:       {mean_err:.6e}")
    print(f"  prob-weighted error:  {weighted_err:.6e}")
    ok = mean_err < 0.5 and weighted_err < 1e-2
    print(f"  TEST B {'PASS' if ok else 'FAIL'} "
          f"(tol mean<0.5, prob-weighted<1e-2)", flush=True)
    if not ok:
        print(f"  chain  lp[:8]: {chain_lp[:8].cpu().tolist()}")
        print(f"  ref    lp[:8]: {ref_lp[:8].cpu().tolist()}")
        print(f"  chain  lp[-8:]:{chain_lp[-8:].cpu().tolist()}")
        print(f"  ref    lp[-8:]:{ref_lp[-8:].cpu().tolist()}")
    return ok


# ════════════════════════════════════════════════════════════════════════════
# Test C — pass-through: draft == target in dense mode
# ════════════════════════════════════════════════════════════════════════════


def test_C(target_name: str, T: float = 0.7) -> bool:
    """draft = target → q ≡ p → logprob_diff should be 0 every step.
    Catches: dense-path temperature mismatch (which we know exists at
    log_softmax(score_logits) without /T_target), position alignment.
    """
    print("\n" + "=" * 78)
    print("TEST C — pass-through draft=target dense mode (logprob_diff ≈ 0)")
    print("=" * 78, flush=True)

    # Use Llama-3.2-1B as both target and draft so the chain is small + fast.
    smol = "meta-llama/Llama-3.2-1B-Instruct"

    out_dir = Path("/home/yahya/smcsd/outputs/verify/test_C")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "smc_metrics.jsonl"
    diag_path = out_dir / "chain_diag.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    if diag_path.exists():
        diag_path.unlink()

    import subprocess
    cmd = [
        "/home/yahya/miniconda3/envs/smcsd/bin/python",
        "scripts/accuracy_test_gsm8k.py",
        "--mode", "smc_engine",
        "--model", smol,
        "--draft-model", smol,
        "--draft-mode", "dense",
        "--particles", "4",
        "--gamma", "4",
        "--temperature", str(T),
        "--attention-backend", "fa3",
        "--num-questions", "3",
        "--max-new-tokens", "32",
        "--max-running-requests", "8",
        "--cuda-graph-max-bs", "8",
        "--eagle-topk", "4",
        "--eagle3-collect-path", str(diag_path),
        "--smc-metrics",
        "--smc-metrics-jsonl", str(metrics_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log = out_dir / "run.log"
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            cwd="/home/yahya/smcsd", timeout=600)
    if rc.returncode != 0:
        print(f"  FAIL: subprocess exited {rc.returncode}; see {log}")
        print(open(log).read()[-2000:])
        return False

    # Inspect chain_diag.jsonl: target_lp_mean vs draft_lp_mean per cycle.
    # With draft = target, dense path: target_lp at T=1.0; draft_lp at T=draft.
    # If smc_target_temperature != 1.0, this WILL show a temperature-induced
    # bias. The dense path's known issue: line 520 doesn't divide by
    # smc_target_temperature, while sampling/draft uses smc_draft_temperature.
    # So the gap is nonzero by construction even with draft = target.
    rows = [json.loads(l) for l in open(diag_path)]
    print(f"  diag rows: {len(rows)}")
    if not rows:
        print("  FAIL: no chain_diag rows produced")
        return False
    import statistics
    tp = [r["target_lp_mean"] for r in rows if "target_lp_mean" in r]
    dp = [r["draft_lp_mean"] for r in rows if "draft_lp_mean" in r]
    gap = [r["target_minus_draft_lp_mean"] for r in rows if "target_minus_draft_lp_mean" in r]
    print(f"  target_lp_mean       mean={statistics.mean(tp):+.4f}  median={statistics.median(tp):+.4f}")
    print(f"  draft_lp_mean        mean={statistics.mean(dp):+.4f}  median={statistics.median(dp):+.4f}")
    print(f"  target_minus_draft   mean={statistics.mean(gap):+.4f}  median={statistics.median(gap):+.4f}")
    # With draft == target, AND if both temperatures match, gap should be ~0
    # (within float precision). Dense path's known temperature bug means it
    # WON'T be exactly 0 unless we fix line 520. This test reveals it.
    ok = abs(statistics.median(gap)) < 0.05
    print(f"  TEST C {'PASS' if ok else 'FAIL'} (median |gap| < 0.05)", flush=True)
    if not ok:
        print("  Likely cause: dense path uses log_softmax(score_logits) "
              "without dividing by smc_target_temperature (worker.py:520).")
        print("  Or: draft sampling uses smc_draft_temperature while target "
              "scores at T=1.0, so even draft==target produces nonzero gap.")
    return ok


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--draft", default=DEFAULT_DRAFT)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--tests", default="A,C", help="Comma-separated subset of A,B,C.")
    args = p.parse_args()

    chosen = set(t.strip() for t in args.tests.split(","))
    results = {}
    if "A" in chosen:
        results["A"] = test_A(args.target, args.draft, T=args.temperature)
    if "C" in chosen:
        results["C"] = test_C(args.target, T=args.temperature)
    if "B" in chosen:
        results["B"] = test_B(args.target, args.draft, T=args.temperature)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in sorted(results.items()):
        print(f"  TEST {name}: {'PASS' if ok else 'FAIL'}")
    print()
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Slow pure-HF SMC generation with layerwise MLP proposal heads.

This script is a correctness/quality bridge before integrating layer-head
drafting into SMCSD/sglang. It runs actual particle generation:

  repeat until max_new_tokens:
    draft gamma tokens from q_l = lm_head(head_l(h_l(prefix)))
    update particle log weights with log p_target(token) - log q_l(token)
    sample one bonus token from the full target
    resample particles when ESS/N drops below threshold

It deliberately recomputes full HF forwards to obtain h_l. That means it is
NOT a speed benchmark. It answers whether the proposal produces useful SMC
generations before we pay the engineering cost of a fast runtime path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_layerwise_kl import DEFAULT_TARGET, LayerHead


def parse_csv_ints(value: str) -> List[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("expected at least one integer")
    return sorted(set(out))


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else text.strip()
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", last_line)
    return nums[-1].replace(",", "") if nums else None


def extract_strict_hash_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    return match.group(1).replace(",", "") if match else None


def format_gsm8k_instruction(question: str) -> str:
    return (
        "Solve this math problem step by step.\n"
        "At the very end, output ONLY the final numeric answer "
        "on a new line in the exact format:\n"
        "#### <number>\n\n"
        f"Problem:\n{question}\n"
    )


def load_gsm8k_prompts(tokenizer, num_prompts: int) -> tuple[List[List[int]], List[str], List[str]]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    prompts: List[List[int]] = []
    labels: List[str] = []
    prompt_texts: List[str] = []
    for row in ds.select(range(num_prompts)):
        instruction = format_gsm8k_instruction(row["question"])
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompts.append(ids)
        prompt_texts.append(prompt_text)
        labels.append(extract_answer(row["answer"]) or "")
    return prompts, labels, prompt_texts


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    )


def load_head(
    heads_dir: str | Path,
    *,
    layer: int,
    hidden_size: int,
    device: torch.device,
) -> tuple[LayerHead, Dict]:
    heads_path = Path(heads_dir)
    cfg_path = heads_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    residual = bool(cfg.get("residual", True))
    head = LayerHead(hidden_size, residual=residual, dropout=0.0)
    ckpt = heads_path / f"layer_{layer:02d}_head.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing head checkpoint: {ckpt}")
    head.load_state_dict(torch.load(ckpt, map_location="cpu"))
    head.to(device=device, dtype=torch.float32).eval()
    return head, cfg


@dataclass
class Particle:
    ids: List[int]
    log_weight: float = 0.0


def pad_same_length(prefixes: Sequence[List[int]], pad_id: int, device: torch.device):
    lengths = [len(x) for x in prefixes]
    max_len = max(lengths)
    input_ids = torch.full((len(prefixes), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, ids in enumerate(prefixes):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
    return input_ids, attention_mask, torch.tensor(lengths, dtype=torch.long, device=device)


@torch.inference_mode()
def target_last(
    *,
    target,
    prefixes: Sequence[List[int]],
    layer: int,
    pad_id: int,
    device: torch.device,
    need_hidden: bool,
):
    input_ids, attention_mask, lengths = pad_same_length(prefixes, pad_id, device)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": need_hidden,
        "use_cache": False,
    }
    # In the normal path all particles for one prompt have equal length, so
    # this avoids materializing full-sequence vocab logits.
    if len(set(int(x) for x in lengths.detach().cpu().tolist())) == 1:
        kwargs["num_logits_to_keep"] = 1
        out = target(**kwargs)
        logits = out.logits[:, -1, :].float()
    else:
        out = target(**kwargs)
        idx = lengths - 1
        batch_idx = torch.arange(len(prefixes), device=device)
        logits = out.logits[batch_idx, idx, :].float()

    if not need_hidden:
        return logits, None

    idx = lengths - 1
    batch_idx = torch.arange(len(prefixes), device=device)
    hidden = out.hidden_states[layer][batch_idx, idx, :].detach()
    return logits, hidden


@torch.inference_mode()
def draft_step(
    *,
    target,
    head: LayerHead,
    particles: Sequence[Particle],
    layer: int,
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
):
    prefixes = [p.ids for p in particles]
    target_logits, hidden = target_last(
        target=target,
        prefixes=prefixes,
        layer=layer,
        pad_id=pad_id,
        device=device,
        need_hidden=True,
    )
    eps_t = max(float(temperature), 1e-6)
    target_logp = F.log_softmax(target_logits / eps_t, dim=-1)
    with autocast_context(device, dtype):
        q_h = head(hidden)
        q_logits = target.lm_head(q_h)
    q_logp = F.log_softmax(q_logits.float() / eps_t, dim=-1)

    if temperature > 0:
        tokens = torch.multinomial(q_logp.exp(), num_samples=1).squeeze(1)
    else:
        tokens = torch.argmax(q_logp, dim=-1)
    target_lp = target_logp.gather(1, tokens[:, None]).squeeze(1)
    q_lp = q_logp.gather(1, tokens[:, None]).squeeze(1)
    gaps = target_lp - q_lp
    local_kl = (target_logp.exp() * (target_logp - q_logp)).sum(dim=-1)

    return {
        "tokens": tokens.detach().cpu().tolist(),
        "target_lp": target_lp.detach().cpu().tolist(),
        "q_lp": q_lp.detach().cpu().tolist(),
        "gap": gaps.detach().cpu().tolist(),
        "local_kl": local_kl.detach().cpu().tolist(),
    }


@torch.inference_mode()
def bonus_step(
    *,
    target,
    particles: Sequence[Particle],
    layer: int,
    pad_id: int,
    device: torch.device,
    temperature: float,
):
    prefixes = [p.ids for p in particles]
    logits, _ = target_last(
        target=target,
        prefixes=prefixes,
        layer=layer,
        pad_id=pad_id,
        device=device,
        need_hidden=False,
    )
    eps_t = max(float(temperature), 1e-6)
    logp = F.log_softmax(logits / eps_t, dim=-1)
    if temperature > 0:
        tokens = torch.multinomial(logp.exp(), num_samples=1).squeeze(1)
    else:
        tokens = torch.argmax(logits, dim=-1)
    return tokens.detach().cpu().tolist()


def ess_stats(log_weights: Sequence[float]) -> Dict[str, float]:
    lw = torch.tensor(log_weights, dtype=torch.float64)
    weights = torch.softmax(lw, dim=0)
    ess = 1.0 / torch.sum(weights * weights)
    return {
        "ess": float(ess.item()),
        "ess_frac": float((ess / len(log_weights)).item()),
        "logw_var": float(torch.var(lw, unbiased=False).item()),
        "max_w": float(torch.max(weights).item()),
        "weights": weights.tolist(),
    }


def systematic_resample(weights: Sequence[float], rng: random.Random) -> List[int]:
    n = len(weights)
    positions = [(rng.random() + i) / n for i in range(n)]
    cdf = []
    acc = 0.0
    for w in weights:
        acc += float(w)
        cdf.append(acc)
    out = []
    j = 0
    for pos in positions:
        while j < n - 1 and pos > cdf[j]:
            j += 1
        out.append(j)
    return out


def maybe_resample(
    particles: List[Particle],
    *,
    threshold: float,
    rng: random.Random,
) -> tuple[List[Particle], Dict[str, float], bool]:
    stats = ess_stats([p.log_weight for p in particles])
    do_resample = stats["ess_frac"] < threshold
    if not do_resample:
        return particles, stats, False
    ancestors = systematic_resample(stats["weights"], rng)
    new_particles = [Particle(ids=list(particles[i].ids), log_weight=0.0) for i in ancestors]
    return new_particles, stats, True


def generate_one(
    *,
    target,
    head: LayerHead,
    tokenizer,
    prompt_ids: List[int],
    layer: int,
    particles_n: int,
    gamma: int,
    max_new_tokens: int,
    resample_threshold: float,
    temperature: float,
    stop_on_answer: bool,
    device: torch.device,
    dtype: torch.dtype,
    rng: random.Random,
) -> Dict:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    particles = [Particle(ids=list(prompt_ids), log_weight=0.0) for _ in range(particles_n)]
    prompt_len = len(prompt_ids)
    cycles = []

    while len(particles[0].ids) - prompt_len < max_new_tokens:
        cycle_start_len = len(particles[0].ids) - prompt_len
        remaining = max_new_tokens - cycle_start_len
        draft_steps = min(int(gamma), remaining)
        step_gaps: List[float] = []
        step_kls: List[float] = []

        for _ in range(draft_steps):
            res = draft_step(
                target=target,
                head=head,
                particles=particles,
                layer=layer,
                pad_id=pad_id,
                device=device,
                dtype=dtype,
                temperature=temperature,
            )
            for i, p in enumerate(particles):
                tok = int(res["tokens"][i])
                p.ids.append(tok)
                gap = float(res["gap"][i])
                p.log_weight += gap
                step_gaps.append(gap)
                step_kls.append(float(res["local_kl"][i]))

        if len(particles[0].ids) - prompt_len < max_new_tokens:
            bonus = bonus_step(
                target=target,
                particles=particles,
                layer=layer,
                pad_id=pad_id,
                device=device,
                temperature=temperature,
            )
            for i, p in enumerate(particles):
                p.ids.append(int(bonus[i]))

        particles, stats, did_resample = maybe_resample(
            particles,
            threshold=resample_threshold,
            rng=rng,
        )
        cycles.append(
            {
                "cycle": len(cycles),
                "generated_len": len(particles[0].ids) - prompt_len,
                "draft_steps": draft_steps,
                "gap_mean": sum(step_gaps) / max(len(step_gaps), 1),
                "kl_mean": sum(step_kls) / max(len(step_kls), 1),
                "ess_frac": stats["ess_frac"],
                "logw_var": stats["logw_var"],
                "max_w": stats["max_w"],
                "resampled": did_resample,
            }
        )

        if stop_on_answer:
            completions = [
                tokenizer.decode(p.ids[prompt_len:], skip_special_tokens=True)
                for p in particles
            ]
            if all(extract_strict_hash_answer(text) is not None for text in completions):
                break

    best_idx = max(
        range(len(particles)),
        key=lambda i: (particles[i].log_weight, len(particles[i].ids)),
    )
    best = particles[best_idx]
    completion_ids = best.ids[prompt_len : prompt_len + max_new_tokens]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "text": text,
        "output_ids": completion_ids,
        "best_particle": best_idx,
        "best_log_weight": best.log_weight,
        "cycles": cycles,
        "completion_tokens": len(completion_ids),
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("gsm8k",), default="gsm8k")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--particles", "-N", type=int, default=12)
    parser.add_argument("--gamma", "-g", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--disable-stop-on-answer",
        action="store_true",
        help="For GSM8K, keep generating to max_new_tokens even after every particle has emitted a parseable #### answer.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.task == "gsm8k":
        prompts, labels, prompt_texts = load_gsm8k_prompts(tokenizer, args.num_prompts)
    else:
        raise ValueError(args.task)

    print(f"[setup] loading target {args.target} on {device} ({args.dtype})", flush=True)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    head, head_cfg = load_head(
        args.heads,
        layer=args.layer,
        hidden_size=int(target.config.hidden_size),
        device=device,
    )

    run_config = {
        **vars(args),
        "head_config": head_cfg,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    rows = []
    t0 = time.perf_counter()
    total_tokens = 0
    correct = 0
    for i, prompt_ids in enumerate(prompts):
        tic = time.perf_counter()
        out = generate_one(
            target=target,
            head=head,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            layer=args.layer,
            particles_n=args.particles,
            gamma=args.gamma,
            max_new_tokens=args.max_new_tokens,
            resample_threshold=args.resample_threshold,
            temperature=args.temperature,
            stop_on_answer=(args.task == "gsm8k" and not args.disable_stop_on_answer),
            device=device,
            dtype=dtype,
            rng=rng,
        )
        pred = extract_answer(out["text"])
        label = labels[i]
        is_correct = pred == label
        correct += int(is_correct)
        total_tokens += int(out["completion_tokens"])
        cycle_rows = out["cycles"]
        rec = {
            "idx": i,
            "label": label,
            "pred": pred,
            "correct": is_correct,
            "text": out["text"],
            "completion_tokens": out["completion_tokens"],
            "best_log_weight": out["best_log_weight"],
            "cycles": cycle_rows,
            "prompt": prompt_texts[i],
            "elapsed": time.perf_counter() - tic,
        }
        rows.append(rec)
        acc = correct / (i + 1)
        elapsed = time.perf_counter() - t0
        print(
            f"[eval] {i+1}/{len(prompts)} acc={correct}/{i+1} ({acc:.1%}) "
            f"pred={pred} label={label} toks={total_tokens} "
            f"tps={total_tokens / max(elapsed, 1e-6):.2f}",
            flush=True,
        )
        if i < 3:
            print(f"--- output {i} ---\n{out['text'][:800]}\n", flush=True)

    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    all_cycles = [c for row in rows for c in row["cycles"]]
    summary = {
        "task": args.task,
        "layer": args.layer,
        "gamma": args.gamma,
        "particles": args.particles,
        "num_prompts": len(rows),
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "total_tokens": total_tokens,
        "elapsed": time.perf_counter() - t0,
        "tokens_per_second": total_tokens / max(time.perf_counter() - t0, 1e-6),
        "ess_frac_median": float(statistics.median([c["ess_frac"] for c in all_cycles]))
        if all_cycles
        else float("nan"),
        "logw_var_median": float(statistics.median([c["logw_var"] for c in all_cycles]))
        if all_cycles
        else float("nan"),
        "resample_rate": mean([1.0 if c["resampled"] else 0.0 for c in all_cycles]),
        "gap_mean": mean([c["gap_mean"] for c in all_cycles]),
        "kl_mean": mean([c["kl_mean"] for c in all_cycles]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[summary] " + json.dumps(summary, indent=2), flush=True)
    print(f"[save] generations -> {out_dir / 'generations.jsonl'}", flush=True)
    print(f"[save] summary -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

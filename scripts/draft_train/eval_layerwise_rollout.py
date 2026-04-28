#!/usr/bin/env python3
"""Autoregressive rollout diagnostic for trained layerwise MLP heads.

This is the next test after teacher-forced layerwise KL. It samples draft
tokens from a trained layer head for gamma steps, scores those same tokens
under the full frozen target, and reports SMC-style weight diagnostics.

It is intentionally slow and HF-only. It does not measure throughput and does
not use partial-layer KV caching. Its job is to answer: do the trained heads
stay target-aligned on their own sampled draft paths?
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_layerwise_kl import (
    DEFAULT_DATA,
    DEFAULT_TARGET,
    LayerHead,
    normalize_messages,
    resolve_data_path,
)


def parse_csv_ints(value: str) -> List[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("expected at least one integer")
    return sorted(set(out))


def load_rows(path: str | Path, *, hold_out_last: int, num_prompts: int) -> List[Dict]:
    data_path = resolve_data_path(path)
    lines = data_path.read_text(encoding="utf-8").splitlines()
    hold_out_last = min(max(int(hold_out_last), 0), len(lines))
    selected = lines[len(lines) - hold_out_last :] if hold_out_last else lines
    if num_prompts > 0:
        selected = selected[:num_prompts]

    rows = []
    bad = 0
    for line in selected:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"[data] skipped {bad} malformed heldout rows", flush=True)
    if not rows:
        raise ValueError(f"no valid rows found in heldout slice of {data_path}")
    return rows


def row_to_prompt_ids(tokenizer, row: Dict, max_prompt_tokens: int) -> List[int]:
    messages = normalize_messages(row)
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    if messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
    else:
        text = row.get("prompt") or row.get("text") or ""
        ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    if not ids:
        ids = [tokenizer.eos_token_id]
    return ids


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    )


def load_heads(
    heads_dir: str | Path,
    *,
    layers: Sequence[int] | None,
    hidden_size: int,
    device: torch.device,
) -> tuple[torch.nn.ModuleDict, List[int], Dict]:
    heads_path = Path(heads_dir)
    cfg_path = heads_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    if layers is None:
        if "layers" not in cfg:
            raise ValueError("--layers is required when heads/config.json has no layers")
        layers = [int(x) for x in cfg["layers"]]
    layers = sorted(set(int(x) for x in layers))

    residual = bool(cfg.get("residual", True))
    heads = torch.nn.ModuleDict()
    for layer in layers:
        path = heads_path / f"layer_{layer:02d}_head.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing head checkpoint: {path}")
        head = LayerHead(hidden_size, residual=residual, dropout=0.0)
        state = torch.load(path, map_location="cpu")
        head.load_state_dict(state)
        heads[str(layer)] = head
    heads.to(device=device, dtype=torch.float32).eval()
    return heads, list(layers), cfg


def maybe_num_logits_to_keep(model) -> bool:
    try:
        return "num_logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.inference_mode()
def one_step(
    *,
    target,
    head,
    prefix: List[int],
    layer: int,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    top_k: int,
    use_num_logits_to_keep: bool,
) -> Dict:
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
    }
    if use_num_logits_to_keep:
        kwargs["num_logits_to_keep"] = 1
    out = target(**kwargs)

    eps_t = max(float(temperature), 1e-6)
    target_logits = out.logits[:, -1, :].float()
    target_logp = F.log_softmax(target_logits / eps_t, dim=-1)
    target_p = target_logp.exp()
    h = out.hidden_states[layer][:, -1, :].detach()
    with autocast_context(device, dtype):
        q_h = head(h)
        q_logits = target.lm_head(q_h)
    q_logp = F.log_softmax(q_logits.float() / eps_t, dim=-1)

    if temperature > 0:
        token = torch.multinomial(q_logp.exp(), num_samples=1).squeeze(1)
    else:
        token = torch.argmax(q_logp, dim=-1)
    token_id = int(token.item())

    q_lp = q_logp.gather(1, token[:, None]).squeeze(1)
    target_lp = target_logp.gather(1, token[:, None]).squeeze(1)
    gap = target_lp - q_lp
    local_kl = (target_p * (target_logp - q_logp)).sum(dim=-1)

    k = min(int(top_k), target_logp.shape[-1])
    target_top = torch.topk(target_logp, k, dim=-1).indices
    q_top = torch.topk(q_logp, k, dim=-1).indices
    sample_in_target_topk = bool(target_top.eq(token[:, None]).any().item())
    overlap = (
        q_top.unsqueeze(-1)
        .eq(target_top.unsqueeze(-2))
        .any(dim=-1)
        .sum(dim=-1)
        .float()
        / float(k)
    )
    target_rank = int((target_logp > target_lp[:, None]).sum().item() + 1)

    return {
        "token_id": token_id,
        "token_text": tokenizer.decode([token_id], errors="ignore"),
        "target_lp": float(target_lp.item()),
        "q_lp": float(q_lp.item()),
        "gap": float(gap.item()),
        "local_kl": float(local_kl.item()),
        "topk_overlap": float(overlap.item()),
        "sample_in_target_topk": sample_in_target_topk,
        "target_rank": target_rank,
    }


def ess_from_logweights(logweights: Sequence[float]) -> Dict[str, float]:
    lw = torch.tensor(logweights, dtype=torch.float64)
    weights = torch.softmax(lw, dim=0)
    ess = 1.0 / torch.sum(weights * weights)
    return {
        "ess": float(ess.item()),
        "ess_frac": float((ess / len(logweights)).item()),
        "logw_var": float(torch.var(lw, unbiased=False).item()),
        "max_w": float(torch.max(weights).item()),
    }


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def summarize_records(records: List[Dict], gammas: Sequence[int], particles: int) -> List[Dict]:
    rows = []
    layers = sorted({r["layer"] for r in records})
    for layer in layers:
        layer_records = [r for r in records if r["layer"] == layer]
        for gamma in gammas:
            step_records = [r for r in layer_records if r["step"] <= gamma]
            prompt_particle_keys = sorted(
                {(r["prompt_idx"], r["particle_idx"]) for r in step_records}
            )
            path_logw = []
            by_prompt: Dict[int, List[float]] = {}
            for prompt_idx, particle_idx in prompt_particle_keys:
                vals = [
                    r["gap"]
                    for r in step_records
                    if r["prompt_idx"] == prompt_idx
                    and r["particle_idx"] == particle_idx
                ]
                if len(vals) == gamma:
                    lw = float(sum(vals))
                    path_logw.append(lw)
                    by_prompt.setdefault(prompt_idx, []).append(lw)

            ess_rows = [
                ess_from_logweights(v)
                for v in by_prompt.values()
                if len(v) == particles
            ]
            ranks = [int(r["target_rank"]) for r in step_records]
            rows.append(
                {
                    "layer": layer,
                    "gamma": gamma,
                    "tokens": len(step_records),
                    "target_minus_q_lp_mean": mean([r["gap"] for r in step_records]),
                    "path_target_minus_q_lp_mean": mean(path_logw),
                    "local_kl_mean": mean([r["local_kl"] for r in step_records]),
                    "topk_overlap_mean": mean([r["topk_overlap"] for r in step_records]),
                    "sample_in_target_topk_mean": mean(
                        [1.0 if r["sample_in_target_topk"] else 0.0 for r in step_records]
                    ),
                    "target_rank_median": float(statistics.median(ranks))
                    if ranks
                    else float("nan"),
                    "ess_frac_mean": mean([r["ess_frac"] for r in ess_rows]),
                    "ess_frac_median": float(
                        statistics.median([r["ess_frac"] for r in ess_rows])
                    )
                    if ess_rows
                    else float("nan"),
                    "logw_var_mean": mean([r["logw_var"] for r in ess_rows]),
                    "logw_var_median": float(
                        statistics.median([r["logw_var"] for r in ess_rows])
                    )
                    if ess_rows
                    else float("nan"),
                    "max_w_mean": mean([r["max_w"] for r in ess_rows]),
                    "particles": particles,
                }
            )
    return rows


def print_summary(rows: Sequence[Dict]) -> None:
    print(
        "[summary] layer gamma lp_gap path_gap KL topk sample_topk rank_med "
        "ESS/N_med logw_var_med max_w_mean",
        flush=True,
    )
    for row in sorted(rows, key=lambda r: (r["layer"], r["gamma"])):
        print(
            f"[summary] {row['layer']:>5d} {row['gamma']:>5d} "
            f"{row['target_minus_q_lp_mean']:+.3f} "
            f"{row['path_target_minus_q_lp_mean']:+.3f} "
            f"{row['local_kl_mean']:.3f} "
            f"{row['topk_overlap_mean']:.3f} "
            f"{row['sample_in_target_topk_mean']:.3f} "
            f"{row['target_rank_median']:.1f} "
            f"{row['ess_frac_median']:.3f} "
            f"{row['logw_var_median']:.2f} "
            f"{row['max_w_mean']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--heads", required=True, help="Directory containing config.json and layer_XX_head.pt files")
    parser.add_argument("--layers", default=None, help="Comma-separated layer list. Defaults to heads/config.json")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--hold-out-last", type=int, default=200)
    parser.add_argument("--particles", "-N", type=int, default=12)
    parser.add_argument("--gamma", default="2,4,8")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    gammas = parse_csv_ints(args.gamma)
    max_gamma = max(gammas)
    requested_layers = parse_csv_ints(args.layers) if args.layers else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[setup] loading target {args.target} on {device} ({args.dtype})", flush=True)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    use_keep = maybe_num_logits_to_keep(target)

    heads, layers, head_cfg = load_heads(
        args.heads,
        layers=requested_layers,
        hidden_size=int(target.config.hidden_size),
        device=device,
    )

    rows = load_rows(args.data, hold_out_last=args.hold_out_last, num_prompts=args.num_prompts)
    prompts = [
        row_to_prompt_ids(tokenizer, row, args.max_prompt_tokens)
        for row in rows
    ]

    run_config = {
        **vars(args),
        "layers": layers,
        "gammas": gammas,
        "num_prompts_loaded": len(prompts),
        "head_config": head_cfg,
        "num_logits_to_keep": use_keep,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    records: List[Dict] = []
    t0 = time.perf_counter()
    total = len(layers) * len(prompts) * args.particles
    done = 0
    for layer in layers:
        head = heads[str(layer)]
        for prompt_idx, prompt in enumerate(prompts):
            for particle_idx in range(args.particles):
                prefix = list(prompt)
                cumulative_gap = 0.0
                for step in range(1, max_gamma + 1):
                    rec = one_step(
                        target=target,
                        head=head,
                        prefix=prefix,
                        layer=layer,
                        tokenizer=tokenizer,
                        device=device,
                        dtype=dtype,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        use_num_logits_to_keep=use_keep,
                    )
                    prefix.append(int(rec["token_id"]))
                    cumulative_gap += float(rec["gap"])
                    records.append(
                        {
                            "layer": layer,
                            "prompt_idx": prompt_idx,
                            "particle_idx": particle_idx,
                            "step": step,
                            "cum_gap": cumulative_gap,
                            **rec,
                        }
                    )
                done += 1
                if done % max(args.particles, 1) == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[rollout] {done}/{total} paths "
                        f"({elapsed / max(done, 1):.2f}s/path)",
                        flush=True,
                    )

    records_path = output_dir / "rollout_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary = summarize_records(records, gammas, args.particles)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_summary(summary)
    print(f"[save] records -> {records_path}", flush=True)
    print(f"[save] summary -> {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

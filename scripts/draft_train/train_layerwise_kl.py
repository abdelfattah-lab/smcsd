#!/usr/bin/env python3
"""Train early-exit MLP proposal heads by layerwise KL distillation.

This is the small first experiment for the layerwise-MLP idea:

  frozen target LM hidden state at layer L_i
      -> trainable MLP head
      -> frozen target lm_head
      -> proposal q_i(next token)

The target distribution is the frozen model's final logits on the same
positions. The loss for each selected layer is forward KL:

  KL(softmax(target_logits / T) || softmax(layer_head_logits / T))

The script is intentionally single-process and HF-only. It is meant to answer
the first question before SMC integration: can an intermediate hidden state plus
a small head approximate the full target distribution?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATA = "/home/yahya/SpecForge/cache/dataset/llama31_8b_smc_warmstart_200k.jsonl"


def resolve_data_path(path: str | Path) -> Path:
    """Resolve common local dataset locations.

    The SMCSD repo docs often refer to cache/dataset/... relative to the
    project, but on this box the large warmstart files live in SpecForge's
    raid-backed cache. Accept both forms so the smoke-test command is less
    brittle.
    """
    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw,
                Path("/home/yahya/SpecForge") / raw,
                Path("/home/yahya/SpecForge/cache/dataset") / raw.name,
                Path("/mnt/raid0/yahya/SpecForge_cache/dataset") / raw.name,
            ]
        )
    else:
        candidates.append(Path("/home/yahya/SpecForge/cache/dataset") / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"data file not found. Tried:\n  - {tried}")


def parse_layers(value: str) -> List[int]:
    layers = []
    for item in value.split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise ValueError("--layers must contain at least one integer layer index")
    return sorted(set(layers))


def normalize_messages(row: Dict) -> List[Dict[str, str]]:
    raw = row.get("conversations") or row.get("messages") or []
    messages: List[Dict[str, str]] = []
    for msg in raw:
        role = msg.get("role", msg.get("from", "user"))
        content = msg.get("content", msg.get("value", ""))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "model"):
            role = "assistant"
        elif role != "system":
            role = "user"
        if content:
            messages.append({"role": role, "content": content})
    return messages


def row_to_token_ids(tokenizer, row: Dict, max_length: int) -> List[int]:
    messages = normalize_messages(row)
    if not messages:
        text = row.get("text") or row.get("prompt") or ""
        ids = tokenizer(text, add_special_tokens=False).input_ids
    else:
        add_generation_prompt = messages[-1]["role"] != "assistant"
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


class JsonlChatDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_length: int,
        hold_out_last: int,
        split: str,
        limit_rows: int = 0,
        eval_rows: int = 0,
    ):
        self.path = resolve_data_path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

        lines = self.path.read_text(encoding="utf-8").splitlines()
        if hold_out_last < 0:
            raise ValueError("--hold-out-last must be >= 0")
        hold_out_last = min(hold_out_last, len(lines))

        if split == "train":
            selected = lines[: len(lines) - hold_out_last] if hold_out_last else lines
            if limit_rows > 0:
                selected = selected[:limit_rows]
        elif split == "eval":
            selected = lines[len(lines) - hold_out_last :] if hold_out_last else lines
            if eval_rows > 0:
                selected = selected[:eval_rows]
        else:
            raise ValueError(f"unknown split: {split}")

        if not selected:
            raise ValueError(f"no rows selected for split={split}")
        self.rows = []
        bad_rows = []
        for offset, line in enumerate(selected, start=1):
            if not line.strip():
                continue
            try:
                self.rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                bad_rows.append((offset, exc))
        if bad_rows:
            preview = ", ".join(
                f"{line_no}: {exc.msg}" for line_no, exc in bad_rows[:5]
            )
            print(
                f"[data] skipped {len(bad_rows)} malformed JSONL rows "
                f"from {self.path} split={split}: {preview}",
                file=sys.stderr,
                flush=True,
            )
        if not self.rows:
            raise ValueError(f"no valid JSON rows selected for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> List[int]:
        ids = row_to_token_ids(self.tokenizer, self.rows[idx], self.max_length)
        if len(ids) < 2:
            eos = self.tokenizer.eos_token_id
            ids = [eos, eos]
        return ids


def collate_token_ids(batch: Sequence[List[int]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class LayerHead(nn.Module):
    def __init__(self, hidden_size: int, *, residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.residual = residual
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        if self.residual:
            y = x + y
        return self.norm2(y)


def autocast_context(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scheduled_lr(base_lr: float, step: int, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if max_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_heads(
    output_dir: Path,
    heads: nn.ModuleDict,
    *,
    layers: Sequence[int],
    args: argparse.Namespace,
    target_config,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "step": step,
        "target": args.target,
        "layers": list(layers),
        "hidden_size": int(target_config.hidden_size),
        "num_hidden_layers": int(target_config.num_hidden_layers),
        "temperature": float(args.temperature),
        "residual": not args.no_residual,
        "dropout": float(args.dropout),
        "architecture": "LayerNorm -> Linear(H,H) -> SiLU -> Linear(H,H) -> residual -> LayerNorm -> frozen target lm_head",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for layer in layers:
        torch.save(heads[str(layer)].state_dict(), output_dir / f"layer_{layer:02d}_head.pt")


@torch.no_grad()
def evaluate(
    *,
    target,
    heads: nn.ModuleDict,
    loader: DataLoader,
    layers: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    top_k: int,
    eval_sample_positions: int,
) -> Dict[str, Dict[str, float]]:
    target.eval()
    heads.eval()
    eps_t = max(float(temperature), 1e-6)
    stats = {
        layer: {
            "kl_sum": 0.0,
            "kl_count": 0,
            "overlap_sum": 0.0,
            "overlap_count": 0,
            "gold_gap_sum": 0.0,
            "gold_gap_count": 0,
            "sample_gap_sum": 0.0,
            "sample_gap_count": 0,
            "sample_in_target_topk_sum": 0.0,
            "sample_rank_values": [],
        }
        for layer in layers
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valid = attention_mask.bool()

        out = target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states
        target_logp = F.log_softmax(out.logits.float() / eps_t, dim=-1)
        target_p = target_logp.exp()
        k = min(top_k, target_logp.shape[-1])
        target_top_ids = torch.topk(target_logp, k, dim=-1).indices

        next_valid = valid[:, :-1] & valid[:, 1:]
        gold_next = input_ids[:, 1:]
        target_gold_lp = target_logp[:, :-1, :].gather(
            2, gold_next.unsqueeze(-1)
        ).squeeze(-1)

        valid_positions = valid.nonzero(as_tuple=False)
        if eval_sample_positions > 0 and valid_positions.shape[0] > eval_sample_positions:
            perm = torch.randperm(valid_positions.shape[0], device=device)[
                :eval_sample_positions
            ]
            sample_positions = valid_positions[perm]
        else:
            sample_positions = valid_positions

        for layer in layers:
            h = hidden_states[layer].detach()
            with autocast_context(device, dtype):
                proposal_h = heads[str(layer)](h)
                q_logits = target.lm_head(proposal_h)
            q_logp = F.log_softmax(q_logits.float() / eps_t, dim=-1)

            kl_per_pos = (target_p * (target_logp - q_logp)).sum(dim=-1)
            layer_stats = stats[layer]
            layer_stats["kl_sum"] += float((kl_per_pos * valid).sum().item())
            layer_stats["kl_count"] += int(valid.sum().item())

            q_top_ids = torch.topk(q_logp, k, dim=-1).indices
            overlap = (
                q_top_ids.unsqueeze(-1)
                .eq(target_top_ids.unsqueeze(-2))
                .any(dim=-1)
                .sum(dim=-1)
                .float()
                / float(k)
            )
            layer_stats["overlap_sum"] += float((overlap * valid).sum().item())
            layer_stats["overlap_count"] += int(valid.sum().item())

            q_gold_lp = q_logp[:, :-1, :].gather(2, gold_next.unsqueeze(-1)).squeeze(-1)
            gold_gap = target_gold_lp - q_gold_lp
            layer_stats["gold_gap_sum"] += float((gold_gap * next_valid).sum().item())
            layer_stats["gold_gap_count"] += int(next_valid.sum().item())

            if sample_positions.numel() > 0:
                b_idx = sample_positions[:, 0]
                t_idx = sample_positions[:, 1]
                q_rows = q_logp[b_idx, t_idx, :]
                target_rows = target_logp[b_idx, t_idx, :]
                target_top_rows = target_top_ids[b_idx, t_idx, :]
                sampled = torch.multinomial(q_rows.exp(), num_samples=1).squeeze(1)
                q_sample_lp = q_rows.gather(1, sampled[:, None]).squeeze(1)
                target_sample_lp = target_rows.gather(1, sampled[:, None]).squeeze(1)
                sample_gap = target_sample_lp - q_sample_lp
                sample_in_topk = target_top_rows.eq(sampled[:, None]).any(dim=1).float()
                rank = (target_rows > target_sample_lp[:, None]).sum(dim=1) + 1
                layer_stats["sample_gap_sum"] += float(sample_gap.sum().item())
                layer_stats["sample_gap_count"] += int(sample_gap.numel())
                layer_stats["sample_in_target_topk_sum"] += float(
                    sample_in_topk.sum().item()
                )
                layer_stats["sample_rank_values"].extend(
                    int(x) for x in rank.detach().cpu().tolist()
                )

            del q_logp, q_logits, proposal_h

        del out, hidden_states, target_logp, target_p

    summary: Dict[str, Dict[str, float]] = {}
    for layer in layers:
        s = stats[layer]
        ranks = s["sample_rank_values"]
        summary[str(layer)] = {
            "mean_kl": s["kl_sum"] / max(s["kl_count"], 1),
            f"top_{top_k}_overlap": s["overlap_sum"] / max(s["overlap_count"], 1),
            "gold_target_minus_q_lp": s["gold_gap_sum"] / max(s["gold_gap_count"], 1),
            "sample_target_minus_q_lp": s["sample_gap_sum"]
            / max(s["sample_gap_count"], 1),
            f"sample_in_target_top_{top_k}": s["sample_in_target_topk_sum"]
            / max(s["sample_gap_count"], 1),
            "sample_target_rank_median": float(statistics.median(ranks))
            if ranks
            else float("nan"),
            "positions": float(s["kl_count"]),
        }
    heads.train()
    return summary


def print_eval_summary(summary: Dict[str, Dict[str, float]], top_k: int) -> None:
    print("[eval] layer  mean_kl  topk_overlap  gold_gap  sample_gap  sample_topk  rank_med")
    for layer, row in sorted(summary.items(), key=lambda kv: int(kv[0])):
        print(
            f"[eval] {int(layer):>5d}  "
            f"{row['mean_kl']:.4f}  "
            f"{row[f'top_{top_k}_overlap']:.3f}  "
            f"{row['gold_target_minus_q_lp']:+.3f}  "
            f"{row['sample_target_minus_q_lp']:+.3f}  "
            f"{row[f'sample_in_target_top_{top_k}']:.3f}  "
            f"{row['sample_target_rank_median']:.1f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default="outputs/layerwise_kl_smoke")
    parser.add_argument("--layers", default="24", help="Comma-separated 1-indexed HF hidden-state layers, e.g. 16,24,28")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--hold-out-last", type=int, default=200)
    parser.add_argument("--limit-train-rows", type=int, default=0)
    parser.add_argument("--eval-rows", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-sample-positions", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    layers = parse_layers(args.layers)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = JsonlChatDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        hold_out_last=args.hold_out_last,
        split="train",
        limit_rows=args.limit_train_rows,
    )
    eval_ds = JsonlChatDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        hold_out_last=args.hold_out_last,
        split="eval",
        eval_rows=args.eval_rows,
    )
    collate = lambda batch: collate_token_ids(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    print(f"[setup] loading target {args.target} on {device} ({args.dtype})", flush=True)
    # Avoid device_map here so the smoke trainer does not require accelerate.
    # The 8B target fits on one H100 in bf16; load normally, then move it.
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=dtype,
    ).to(device).eval()
    for param in target.parameters():
        param.requires_grad_(False)

    num_layers = int(target.config.num_hidden_layers)
    hidden_size = int(target.config.hidden_size)
    bad_layers = [layer for layer in layers if layer < 1 or layer >= num_layers]
    if bad_layers:
        raise ValueError(
            f"layers must be in [1, {num_layers - 1}] for early exits; got {bad_layers}"
        )

    heads = nn.ModuleDict(
        {
            str(layer): LayerHead(
                hidden_size,
                residual=not args.no_residual,
                dropout=args.dropout,
            )
            for layer in layers
        }
    ).to(device=device, dtype=torch.float32)
    heads.train()

    optimizer = torch.optim.AdamW(
        heads.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    run_config = {
        **vars(args),
        "layers": layers,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    log_path = out_dir / "train_log.jsonl"

    print(
        f"[setup] train_rows={len(train_ds)} eval_rows={len(eval_ds)} "
        f"layers={layers} max_steps={args.max_steps}",
        flush=True,
    )

    step = 0
    start_time = time.perf_counter()
    train_iter: Iterable[Dict[str, torch.Tensor]]
    while step < args.max_steps:
        train_iter = iter(train_loader)
        for batch in train_iter:
            step += 1
            lr = scheduled_lr(args.lr, step, args.max_steps, args.warmup_steps)
            set_lr(optimizer, lr)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            valid = attention_mask.bool()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                out = target(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = out.hidden_states
                target_logp = F.log_softmax(
                    out.logits.float() / max(args.temperature, 1e-6), dim=-1
                )
                target_p = target_logp.exp()

            layer_losses: Dict[str, float] = {}
            total_loss_value = 0.0
            for layer in layers:
                h = hidden_states[layer].detach()
                with autocast_context(device, dtype):
                    proposal_h = heads[str(layer)](h)
                    q_logits = target.lm_head(proposal_h)
                q_logp = F.log_softmax(
                    q_logits.float() / max(args.temperature, 1e-6), dim=-1
                )
                kl_per_pos = (target_p * (target_logp - q_logp)).sum(dim=-1)
                loss = masked_mean(kl_per_pos, valid)
                loss.backward()
                loss_value = float(loss.detach().item())
                layer_losses[str(layer)] = loss_value
                total_loss_value += loss_value
                del proposal_h, q_logits, q_logp, kl_per_pos, loss

            grad_norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.perf_counter() - start_time
                rec = {
                    "step": step,
                    "lr": lr,
                    "loss_sum": total_loss_value,
                    "loss_by_layer": layer_losses,
                    "grad_norm": float(grad_norm.item())
                    if isinstance(grad_norm, torch.Tensor)
                    else float(grad_norm),
                    "seconds_per_step": elapsed / step,
                }
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

            if args.save_interval > 0 and step % args.save_interval == 0:
                save_heads(
                    out_dir / f"step_{step}",
                    heads,
                    layers=layers,
                    args=args,
                    target_config=target.config,
                    step=step,
                )
                print(f"[save] wrote checkpoint step_{step}", flush=True)

            if args.eval_every > 0 and step % args.eval_every == 0:
                summary = evaluate(
                    target=target,
                    heads=heads,
                    loader=eval_loader,
                    layers=layers,
                    device=device,
                    dtype=dtype,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    eval_sample_positions=args.eval_sample_positions,
                )
                (out_dir / f"eval_step_{step}.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                print_eval_summary(summary, args.top_k)

            del hidden_states, target_logp, target_p, out
            if step >= args.max_steps:
                break

    final_dir = out_dir / "final"
    save_heads(
        final_dir,
        heads,
        layers=layers,
        args=args,
        target_config=target.config,
        step=step,
    )
    print(f"[save] final heads -> {final_dir}", flush=True)

    print("[eval] running final heldout diagnostic", flush=True)
    summary = evaluate(
        target=target,
        heads=heads,
        loader=eval_loader,
        layers=layers,
        device=device,
        dtype=dtype,
        temperature=args.temperature,
        top_k=args.top_k,
        eval_sample_positions=args.eval_sample_positions,
    )
    (out_dir / "eval_final.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_eval_summary(summary, args.top_k)


if __name__ == "__main__":
    main()

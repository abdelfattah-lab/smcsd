"""Stage-1 KL distillation: train the proposal to match cached target top-K distributions.

Single-GPU or DDP. DDP is auto-detected from the LOCAL_RANK env var set by torchrun.
Launch single-GPU with `python -m training.training.distill ...`, DDP with
`torchrun --nproc_per_node=N -m training.training.distill ...`.

Checkpoints carry full state (model, optimizer, RNG, step, tokens, epoch, config) so
runs can be preempted and resumed, or loaded into a fresh schedule for stage-2.
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from training.data.dataset import TargetTraceDataset, make_collate
from training.model.proposal import (
    ProposalConfig,
    build_proposal,
    param_counts,
    warm_start_and_freeze_embeddings,
)
from training.training.loss import top_k_soft_ce


def setup_distributed() -> tuple[bool, int, int, int, torch.device | None]:
    """Return (ddp_active, rank, world_size, local_rank, device).

    Device is None when single-GPU; the caller uses --device instead.
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0, None
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}")


def cosine_lr(step: int, total_steps: int, peak: float, warmup_steps: int, min_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    epoch: int,
    config: dict,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "epoch": epoch,
        "config": config,
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def load_resume(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "rng" in state:
        torch.set_rng_state(state["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(state["rng"]["torch_cuda"].cpu())
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
    return state["step"], state["tokens_seen"], state.get("epoch", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/configs/mvp.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--shard-dir", default="training/data_cache/target_traces")
    ap.add_argument("--device", default="cuda:0", help="single-GPU only; ignored under torchrun")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--find-unused-parameters", action="store_true",
                    help="DDP: safe mode, ~5%% slower; use for validation runs")
    args = ap.parse_args()

    ddp_active, rank, world_size, local_rank, ddp_device = setup_distributed()
    device = ddp_device if ddp_active else torch.device(args.device)
    is_main = (rank == 0)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_active:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    ds = TargetTraceDataset(args.shard_dir)
    train_cfg = cfg["training_stage1"]
    per_rank_batch = train_cfg["batch_size"]
    gen_len = train_cfg["seq_len"]
    tokens_per_step = per_rank_batch * gen_len * world_size   # global tokens/step
    total_steps = int(train_cfg["total_tokens"]) // tokens_per_step
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(int(total_steps * train_cfg["warmup_frac"]), 1)
    ckpt_every_tokens = int(train_cfg["checkpoint_every_tokens"])

    sampler = None
    if ddp_active:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(
        ds, batch_size=per_rank_batch,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=make_collate(pad_id),
        drop_last=True, num_workers=0,
    )

    prop_cfg_d = cfg["proposal"]
    prop_cfg = ProposalConfig(
        hidden_size=prop_cfg_d["hidden_size"],
        num_hidden_layers=prop_cfg_d["num_hidden_layers"],
        num_attention_heads=prop_cfg_d["num_attention_heads"],
        num_key_value_heads=prop_cfg_d["num_key_value_heads"],
        intermediate_size=prop_cfg_d["intermediate_size"],
        vocab_size=prop_cfg_d["vocab_size"],
        max_position_embeddings=prop_cfg_d["max_position_embeddings"],
        tie_word_embeddings=prop_cfg_d["tie_word_embeddings"],
        rope_theta=prop_cfg_d["rope_theta"],
    )
    proposal = build_proposal(prop_cfg, dtype=torch.bfloat16).to(device)
    if prop_cfg_d.get("warm_start_embeddings", True):
        warm_start_and_freeze_embeddings(
            proposal, args.target, freeze=prop_cfg_d.get("freeze_embeddings", True),
        )

    pc = param_counts(proposal)
    if is_main:
        print(f"[ddp={ddp_active} world={world_size}] dataset={len(ds)} | per-rank batch={per_rank_batch} | gen_len={gen_len}")
        print(f"tokens/step (global) = {tokens_per_step}  total_steps = {total_steps}  warmup = {warmup_steps}")
        print(f"ckpt every {ckpt_every_tokens/1e6:.0f}M tokens")
        print(f"params: total={pc['total']/1e6:.1f}M trainable={pc['trainable']/1e6:.1f}M frozen={pc['frozen']/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in proposal.parameters() if p.requires_grad],
        lr=train_cfg["lr_peak"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )

    global_step = 0
    tokens_seen = 0
    epoch = 0
    if args.resume:
        global_step, tokens_seen, epoch = load_resume(args.resume, proposal, optimizer, device)
        if is_main:
            print(f"resumed: step={global_step} tokens={tokens_seen/1e6:.1f}M epoch={epoch}")
    next_ckpt_at = ((tokens_seen // ckpt_every_tokens) + 1) * ckpt_every_tokens

    if ddp_active:
        proposal_train = DDP(
            proposal, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )
    else:
        proposal_train = proposal

    config_snapshot = {
        "proposal": asdict(prop_cfg),
        "training": train_cfg,
        "target": args.target,
        "shard_dir": args.shard_dir,
        "seed": args.seed,
        "world_size": world_size,
    }
    if is_main:
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)
    metrics_path = out_dir / "metrics.jsonl"

    proposal_train.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    loader_iter = iter(loader)
    t_log = time.time()

    while global_step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        topk_idx = batch["topk_idx"].to(device, non_blocking=True)
        topk_lg = batch["topk_lg"].to(device, non_blocking=True)
        pmax = batch["pmax"]

        lr = cosine_lr(global_step, total_steps, train_cfg["lr_peak"], warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        out = proposal_train(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits_pred = out.logits[:, pmax - 1 : pmax - 1 + gen_len, :]
        loss = top_k_soft_ce(logits_pred, topk_idx, topk_lg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in proposal_train.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()

        global_step += 1
        tokens_seen += tokens_per_step

        if global_step % args.log_every == 0 or global_step == total_steps:
            if ddp_active:
                loss_avg = loss.detach().clone()
                gn_avg = gnorm.detach().clone()
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
                dist.all_reduce(gn_avg, op=dist.ReduceOp.AVG)
            else:
                loss_avg = loss.detach()
                gn_avg = gnorm.detach()
            if is_main:
                dt = time.time() - t_log
                sps = args.log_every / max(dt, 1e-6)
                tps = sps * tokens_per_step
                msg = {
                    "step": global_step, "tokens": tokens_seen, "epoch": epoch,
                    "loss": float(loss_avg.item()), "lr": lr,
                    "grad_norm": float(gn_avg.item()),
                    "steps_per_sec": sps, "toks_per_sec": tps,
                }
                print(
                    f"step {global_step:6d} | toks {tokens_seen/1e6:7.2f}M | ep {epoch:2d} | "
                    f"loss {msg['loss']:.4f} | lr {lr:.2e} | gn {msg['grad_norm']:.3f} | "
                    f"{sps:5.1f} step/s | {tps/1e3:6.1f}K tok/s"
                )
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                t_log = time.time()

        if tokens_seen >= next_ckpt_at or global_step == total_steps:
            if is_main:
                ckpt_path = out_dir / f"ckpt_{tokens_seen//1_000_000:05d}M.pt"
                save_checkpoint(ckpt_path, proposal, optimizer,
                                global_step, tokens_seen, epoch, config_snapshot)
                print(f"saved: {ckpt_path}")
            if ddp_active:
                dist.barrier()
            next_ckpt_at += ckpt_every_tokens

    if is_main:
        print("training complete")
    if ddp_active:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

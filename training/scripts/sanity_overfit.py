"""Overfit-one-batch sanity check.

Verifies the proposal + dataset + loss pipeline:
  - Build proposal (4-layer Llama, hidden=2048), warm-start embeddings, freeze them.
  - Take one batch from the target-trace dataset.
  - Run N gradient steps at high LR; loss should drop substantially.

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m training.scripts.sanity_overfit \
        --shard-dir training/data_cache/target_traces
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.data.dataset import TargetTraceDataset, make_collate
from training.model.proposal import (
    ProposalConfig,
    build_proposal,
    param_counts,
    warm_start_and_freeze_embeddings,
)
from training.training.loss import top_k_soft_ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="training/data_cache/target_traces")
    ap.add_argument("--target", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    ds = TargetTraceDataset(args.shard_dir)
    print(f"dataset size: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=make_collate(pad_id), shuffle=True)
    batch = next(iter(loader))

    cfg = ProposalConfig()
    proposal = build_proposal(cfg).to(device)
    warm_start_and_freeze_embeddings(proposal, args.target, freeze=True)
    pc = param_counts(proposal)
    print(f"params: total={pc['total']/1e6:.1f}M  trainable={pc['trainable']/1e6:.1f}M  frozen={pc['frozen']/1e6:.1f}M")

    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    topk_idx = batch["topk_idx"].to(device)
    topk_lg = batch["topk_lg"].to(device)
    pmax = batch["pmax"]
    gen_len = batch["gen_len"]
    print(f"batch: input_ids={tuple(input_ids.shape)}  pmax={pmax}  gen_len={gen_len}")

    optim = torch.optim.AdamW(
        [p for p in proposal.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    proposal.train()
    for step in range(args.steps):
        out = proposal(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits_pred = out.logits[:, pmax - 1 : pmax - 1 + gen_len, :]
        loss = top_k_soft_ce(logits_pred, topk_idx, topk_lg)
        optim.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in proposal.parameters() if p.requires_grad], max_norm=1.0
        )
        optim.step()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}  grad_norm {gnorm.item():.3f}")


if __name__ == "__main__":
    main()

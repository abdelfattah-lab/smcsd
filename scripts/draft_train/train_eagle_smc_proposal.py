#!/usr/bin/env python3
"""Train an EAGLE3 proposal with SMC weighted path-MLE.

This first version supports gamma_train=1 rollouts from
``collect_eagle_smc_rollouts.py``. For each prompt and candidate token y:

    logw = log p_target(y) - log q_old(y)
    w = softmax(logw over candidates)
    loss = - sum_i w_i log q_new(y_i)

This is the direct SMC proposal-learning objective from finetuning_proposal_toy,
adapted to an EAGLE hidden-state-conditioned proposal.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup


def add_specforge_to_path(specforge_root: str):
    if specforge_root and specforge_root not in sys.path:
        sys.path.insert(0, specforge_root)


class RolloutDataset(Dataset):
    def __init__(self, data_dir: str):
        self.rows: List[Dict] = []
        for path in sorted(glob.glob(os.path.join(data_dir, "shard_*.pt"))):
            self.rows.extend(torch.load(path, map_location="cpu", weights_only=False))
        if not self.rows:
            raise ValueError(f"No rollout rows found in {data_dir}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch: List[Dict]) -> Dict:
    # Keep prompt_ids out of tensor batch for now; gamma=1 only needs seed+hidden.
    K = batch[0]["candidates"].shape[0]
    hidden = torch.stack([r["hidden_states"].float() for r in batch], dim=0)
    seed = torch.stack([r["seed_token"].long() for r in batch], dim=0)
    candidates = torch.stack([r["candidates"].long().view(K) for r in batch], dim=0)
    target_logps = torch.stack([r["target_logps"].float().view(K) for r in batch], dim=0)
    draft_logps_old = torch.stack([r["draft_logps_old"].float().view(K) for r in batch], dim=0)
    target_topk_ids = torch.stack([r["target_topk_ids"].long() for r in batch], dim=0)
    target_topk_logps = torch.stack([r["target_topk_logps"].float() for r in batch], dim=0)
    return {
        "hidden_states": hidden,
        "seed_token": seed,
        "candidates": candidates,
        "target_logps": target_logps,
        "draft_logps_old": draft_logps_old,
        "target_topk_ids": target_topk_ids,
        "target_topk_logps": target_topk_logps,
    }


def compute_loss(
    draft,
    batch: Dict,
    device,
    dtype,
    smc_weight: float = 0.1,
    topk_weight: float = 1.0,
    anchor_weight: float = 1.0,
):
    hidden_cat = batch["hidden_states"].to(device=device, dtype=dtype).unsqueeze(1)
    seed = batch["seed_token"].to(device=device).unsqueeze(1)
    candidates = batch["candidates"].to(device=device)
    target_logps = batch["target_logps"].to(device=device)
    draft_logps_old = batch["draft_logps_old"].to(device=device)
    target_topk_ids = batch["target_topk_ids"].to(device=device)
    target_topk_logps = batch["target_topk_logps"].to(device=device)
    B, K = candidates.shape

    hidden_proj = draft.project_hidden_states(hidden_cat)
    embeds = draft.embed_input_ids(seed).to(dtype)
    mask = torch.ones((B, 1), dtype=torch.bool, device=device)
    attn_mask = draft.prepare_decoder_attention_mask(mask, hidden_proj, B, 1, 0)
    pos = torch.zeros((B, 1), dtype=torch.long, device=device)
    h = draft.backbone(
        input_embeds=embeds,
        hidden_states=hidden_proj,
        cache_hidden=[[], []],
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=None,
        use_cache=False,
    )
    logits = draft.compute_logits(h)[:, -1, :].float()
    log_q = F.log_softmax(logits, dim=-1)
    log_q_cand = log_q.gather(1, candidates)
    log_q_topk = log_q.gather(1, target_topk_ids)

    # Stable teacher-forced target KL on the target's top-k support.
    teacher_lp = target_topk_logps - torch.logsumexp(target_topk_logps, dim=1, keepdim=True)
    teacher_p = teacher_lp.exp().detach()
    topk_kl = (teacher_p * (teacher_lp.detach() - log_q_topk)).sum(dim=1).mean()

    # SMC proposal-learning term over sampled candidates.
    logw = target_logps - draft_logps_old
    weights = F.softmax(logw, dim=1).detach()
    smc_loss = -(weights * log_q_cand).sum(dim=1).mean()

    # Conservative anchor: keep q_new close to q_old on sampled candidates.
    anchor = torch.zeros((), device=device)
    if anchor_weight > 0:
        old_probs = F.softmax(draft_logps_old, dim=1).detach()
        old_lp = draft_logps_old - torch.logsumexp(draft_logps_old, dim=1, keepdim=True)
        new_lp = log_q_cand - torch.logsumexp(log_q_cand, dim=1, keepdim=True)
        anchor = (old_probs * (old_lp - new_lp)).sum(dim=1).mean()

    loss = topk_weight * topk_kl + smc_weight * smc_loss + anchor_weight * anchor
    with torch.no_grad():
        ess = 1.0 / (weights * weights).sum(dim=1)
        stats = {
            "loss": float(loss.item()),
            "topk_kl": float(topk_kl.item()),
            "smc_loss": float(smc_loss.item()),
            "anchor": float(anchor.item()),
            "ess": float(ess.mean().item()),
            "ess_frac": float((ess / K).mean().item()),
            "logw_var": float(logw.var(dim=1, unbiased=False).mean().item()),
            "max_w": float(weights.max(dim=1).values.mean().item()),
            "target_lp": float(target_logps.mean().item()),
            "old_q_lp": float(draft_logps_old.mean().item()),
            "new_q_lp": float(log_q_cand.mean().item()),
        }
    return loss, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True, help="Initial EAGLE3 checkpoint")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--specforge-root", default="/home/yahya/SpecForge")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--smc-weight", type=float, default=0.1)
    p.add_argument("--topk-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=200)
    args = p.parse_args()

    add_specforge_to_path(args.specforge_root)
    from specforge.modeling.auto import AutoEagle3DraftModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    ds = RolloutDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    draft = AutoEagle3DraftModel.from_pretrained(args.init, torch_dtype=dtype).to(device)
    draft.train()
    draft.freeze_embedding()

    opt = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, args.warmup_steps, total_steps)

    global_step = 0
    log_path = out_dir / "train_log.jsonl"
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args) | {"n_rows": len(ds), "total_steps": total_steps}, f, indent=2)

    for epoch in range(args.epochs):
        for batch in loader:
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss, stats = compute_loss(
                draft,
                batch,
                device,
                dtype,
                smc_weight=args.smc_weight,
                topk_weight=args.topk_weight,
                anchor_weight=args.anchor_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), 0.5)
            opt.step(); sched.step()

            if global_step % args.log_interval == 0 or global_step == 1:
                entry = {"step": global_step, "epoch": epoch, "lr": sched.get_last_lr()[0], **stats}
                print(json.dumps(entry), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if global_step % args.save_interval == 0:
                ckpt = out_dir / f"step_{global_step}"
                draft.save_pretrained(ckpt)

    final = out_dir / "final"
    draft.save_pretrained(final)
    print(f"Saved final checkpoint to {final}", flush=True)


if __name__ == "__main__":
    main()

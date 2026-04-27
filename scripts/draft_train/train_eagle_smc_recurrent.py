#!/usr/bin/env python3
"""Recurrent path-level SMC-native EAGLE proposal training.

Consumes rollouts produced by ``collect_eagle_smc_rollouts_v2.py``. Each
rollout encodes a single prompt with R candidate paths of length K =
``gamma_train`` together with stored old draft log-probs, target log-probs
under the teacher, and per-step target top-k.

For every path the trainer teacher-forces the path tokens through the
EAGLE recurrent dynamics with the current trainable parameters, exactly
matching SMC-SD inference:

    h_seed = midlayer(input_emb=embed(x0), hidden_states=fc(target_h_3),
                      cache_hidden=[[], []], position_ids=[[0]])
    logits_1 = compute_logits(h_seed)
    for t = 1..K:
        log q_new(y_t)  = log_softmax(logits_t / T)(y_t)
        if t < K:
            h_t      = midlayer(input_emb=embed(y_t), hidden_states=h_{t-1},
                                cache_hidden=growing, position_ids=[[0]])
            logits_{t+1} = compute_logits(h_t)

The objective combines three pieces:

    L = alpha * sum_t KL(target_topk_t || q_new_topk_t)        # warm-start anchor
      + beta  * (- sum_i stopgrad(w_i) * sum_t log q_new(y_i,t)) # SMC weighted MLE
      + gamma * sum_t KL(q_old || q_new)                        # conservative anchor

with SMC weights computed from STORED logps:

    logw_i = sum_t log p_target(y_i,t) - sum_t log q_old(y_i,t)
    w_i    = softmax(logw_i)

Logged stats include ESS/N over candidate paths, log-weight variance, max
weight, mean target top-k KL, and the mean of old/new q log-probs.
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
        # Match any *.pt under data_dir to allow multiple shard prefixes
        # (e.g. when collection was sharded across GPUs with --shard-prefix
        # gpu0, gpu1, ...).
        for path in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
            self.rows.extend(torch.load(path, map_location="cpu", weights_only=False))
        if not self.rows:
            raise ValueError(f"No rollout rows found in {data_dir}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch: List[Dict]) -> Dict:
    """Stack rollouts into per-batch tensors.

    Asserts every row in the batch has the same R (num candidates), K
    (gamma_train), and target_topk dim, since we mix them in flat batched
    tensors. This matches the v2 collector, which uses a fixed R and K per
    run.
    """
    R = batch[0]["candidates"].shape[0]
    K = batch[0]["candidates"].shape[1]
    TK = batch[0]["target_topk_ids"].shape[-1]
    for r in batch:
        assert r["candidates"].shape == (R, K)
        assert r["draft_logps_old"].shape == (R, K)
        assert r["target_logps"].shape == (R, K)
        assert r["target_topk_ids"].shape == (R, K, TK)
        assert r["target_topk_logps"].shape == (R, K, TK)
    target_h_3 = torch.stack([r["target_h_3"].float() for r in batch], dim=0)  # (B, 3H)
    if "target_h_3_seq" in batch[0]:
        lengths = torch.tensor([r["target_h_3_seq"].shape[0] for r in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        hidden_dim = batch[0]["target_h_3_seq"].shape[-1]
        target_h_3_seq = torch.zeros(len(batch), max_len, hidden_dim, dtype=torch.float32)
        shifted_input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        prefill_attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, r in enumerate(batch):
            L = int(lengths[i].item())
            target_h_3_seq[i, :L] = r["target_h_3_seq"].float()
            shifted_input_ids[i, :L] = r["shifted_input_ids"].long()
            prefill_attention_mask[i, :L] = True
    else:
        lengths = None
        target_h_3_seq = None
        shifted_input_ids = None
        prefill_attention_mask = None
    seed = torch.stack([r["seed_token"].long() for r in batch], dim=0)  # (B,)
    candidates = torch.stack([r["candidates"].long() for r in batch], dim=0)  # (B, R, K)
    target_logps = torch.stack(
        [r["target_logps"].float() for r in batch], dim=0
    )  # (B, R, K)
    draft_logps_old = torch.stack(
        [r["draft_logps_old"].float() for r in batch], dim=0
    )  # (B, R, K)
    target_topk_ids = torch.stack(
        [r["target_topk_ids"].long() for r in batch], dim=0
    )  # (B, R, K, TK)
    target_topk_logps = torch.stack(
        [r["target_topk_logps"].float() for r in batch], dim=0
    )  # (B, R, K, TK)
    return {
        "target_h_3": target_h_3,
        "target_h_3_seq": target_h_3_seq,
        "shifted_input_ids": shifted_input_ids,
        "prefill_attention_mask": prefill_attention_mask,
        "prefill_lengths": lengths,
        "seed_token": seed,
        "candidates": candidates,
        "target_logps": target_logps,
        "draft_logps_old": draft_logps_old,
        "target_topk_ids": target_topk_ids,
        "target_topk_logps": target_topk_logps,
        "R": R,
        "K": K,
    }


def _eagle_step(
    draft,
    *,
    input_emb: torch.Tensor,
    hidden: torch.Tensor,
    cache_hidden,
    dtype: torch.dtype,
):
    """Single recurrent EAGLE step on (B, 1, ...) tensors. See collector."""
    B = input_emb.shape[0]
    H = draft.config.hidden_size
    if hidden.shape[-1] == 3 * H:
        h_in = draft.project_hidden_states(hidden).to(dtype)
    else:
        assert hidden.shape[-1] == H, (
            f"hidden last dim {hidden.shape[-1]} != H ({H}) or 3H ({3*H})"
        )
        h_in = hidden.to(dtype)
    mask = torch.ones((B, 1), dtype=torch.bool, device=input_emb.device)
    attn_mask = draft.prepare_decoder_attention_mask(mask, h_in, B, 1, 0)
    pos = torch.zeros((B, 1), dtype=torch.long, device=input_emb.device)
    h_out = draft.backbone(
        input_embeds=input_emb,
        hidden_states=h_in,
        cache_hidden=cache_hidden,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=None,
        use_cache=False,
    )
    logits = draft.compute_logits(h_out)[:, -1, :].float()
    return h_out, logits


def _eagle_prefill(
    draft,
    *,
    input_emb: torch.Tensor,
    hidden: torch.Tensor,
    cache_hidden,
    dtype: torch.dtype,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
):
    """Runtime-like EAGLE prefill over a padded prompt batch."""
    B, S, _ = input_emb.shape
    H = draft.config.hidden_size
    if hidden.shape[-1] == 3 * H:
        h_in = draft.project_hidden_states(hidden).to(dtype)
    else:
        assert hidden.shape[-1] == H, (
            f"hidden last dim {hidden.shape[-1]} != H ({H}) or 3H ({3*H})"
        )
        h_in = hidden.to(dtype)
    mask = attention_mask.to(device=input_emb.device, dtype=torch.bool)
    attn_mask = draft.prepare_decoder_attention_mask(mask, h_in, B, S, 0)
    pos = torch.arange(S, dtype=torch.long, device=input_emb.device)[None, :].expand(B, S)
    h_out = draft.backbone(
        input_embeds=input_emb,
        hidden_states=h_in,
        cache_hidden=cache_hidden,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=None,
        use_cache=False,
    )
    logits_all = draft.compute_logits(h_out).float()
    last_idx = lengths.to(device=input_emb.device, dtype=torch.long) - 1
    batch_idx = torch.arange(B, device=input_emb.device)
    h_last = h_out[batch_idx, last_idx, :].unsqueeze(1)
    logits = logits_all[batch_idx, last_idx, :]
    return h_last, logits


def _expand_cache_for_paths(cache_hidden, R: int):
    expanded = [[], []]
    for side in (0, 1):
        for t in cache_hidden[side]:
            # SpecForge attention cache tensors are batch-major. Add a path
            # axis, tile each prompt's prefix cache across R candidate paths,
            # then flatten (B, R) -> (B*R).
            B = t.shape[0]
            expanded[side].append(
                t[:, None, ...]
                .expand(B, R, *t.shape[1:])
                .reshape(B * R, *t.shape[1:])
                .contiguous()
            )
    return expanded


def compute_loss(
    draft,
    batch: Dict,
    device,
    dtype,
    *,
    alpha_topk: float,
    beta_smc: float,
    gamma_anchor: float,
    temperature: float,
):
    target_h_3 = batch["target_h_3"].to(device=device, dtype=dtype)  # (B, 3H_t)
    seed = batch["seed_token"].to(device=device)  # (B,)
    candidates = batch["candidates"].to(device=device)  # (B, R, K)
    target_logps = batch["target_logps"].to(device=device)  # (B, R, K)
    draft_logps_old = batch["draft_logps_old"].to(device=device)  # (B, R, K)
    target_topk_ids = batch["target_topk_ids"].to(device=device)  # (B, R, K, TK)
    target_topk_logps = batch["target_topk_logps"].to(device=device)  # (B, R, K, TK)
    B, R, K = candidates.shape
    TK = target_topk_ids.shape[-1]

    eps_T = max(temperature, 1e-6)

    # ---- 0. Score stored paths under q_new ----
    BR = B * R
    candidates_BR = candidates.view(BR, K)  # (BR, K)
    new_logp_path = torch.empty(BR, K, dtype=torch.float32, device=device)
    new_logp_topk = torch.empty(BR, K, TK, dtype=torch.float32, device=device)

    if batch.get("target_h_3_seq") is not None:
        # Runtime-matched path: replay the full growing EAGLE sequence without
        # SpecForge's manual cache. This matches SGLang's full-prompt EAGLE
        # prefill semantics while avoiding SpecForge's q_len=1 cache assumption.
        target_h_3_seq = batch["target_h_3_seq"].to(device=device, dtype=dtype)
        shifted_input_ids = batch["shifted_input_ids"].to(device=device)
        prefill_lengths = batch["prefill_lengths"].to(device=device)

        for b in range(B):
            L = int(prefill_lengths[b].item())
            row_start = b * R
            row_end = row_start + R
            prompt_hidden = draft.project_hidden_states(
                target_h_3_seq[b : b + 1, :L, :]
            ).to(dtype)
            shifted_emb = draft.embed_input_ids(
                shifted_input_ids[b : b + 1, :L]
            ).to(dtype)
            hidden_context = prompt_hidden.expand(R, -1, -1).contiguous()
            embed_context = shifted_emb.expand(R, -1, -1).contiguous()
            h_prev, logits_t = _eagle_prefill(
                draft,
                input_emb=embed_context,
                hidden=hidden_context,
                cache_hidden=None,
                dtype=dtype,
                attention_mask=torch.ones((R, L), dtype=torch.bool, device=device),
                lengths=torch.full((R,), L, dtype=torch.long, device=device),
            )
            row_candidates = candidates[b]  # (R, K)
            row_topk_ids = target_topk_ids[b]  # (R, K, TK)

            for t in range(K):
                log_q = F.log_softmax(logits_t / eps_T, dim=-1)
                new_logp_path[row_start:row_end, t] = log_q.gather(
                    1, row_candidates[:, t : t + 1]
                ).squeeze(1)
                new_logp_topk[row_start:row_end, t, :] = log_q.gather(
                    1, row_topk_ids[:, t, :]
                )
                if t == K - 1:
                    break
                y_emb = draft.embed_input_ids(row_candidates[:, t : t + 1]).to(dtype)
                hidden_context = torch.cat([hidden_context, h_prev], dim=1)
                embed_context = torch.cat([embed_context, y_emb], dim=1)
                seq_len = embed_context.shape[1]
                h_prev, logits_t = _eagle_prefill(
                    draft,
                    input_emb=embed_context,
                    hidden=hidden_context,
                    cache_hidden=None,
                    dtype=dtype,
                    attention_mask=torch.ones((R, seq_len), dtype=torch.bool, device=device),
                    lengths=torch.full((R,), seq_len, dtype=torch.long, device=device),
                )
    else:
        # Backward-compatible path for older rollouts. This is not runtime
        # equivalent because it lacks the full prompt draft prefill cache.
        seed_BR = seed[:, None].expand(B, R).contiguous().view(BR)
        target_h_3_BR = (
            target_h_3[:, None, :]
            .expand(B, R, target_h_3.shape[-1])
            .contiguous()
            .view(BR, target_h_3.shape[-1])
        )
        target_h_3_BR = target_h_3_BR.unsqueeze(1)
        cache_hidden = [[], []]
        seed_emb = draft.embed_input_ids(seed_BR[:, None]).to(dtype)
        h_prev, logits_1 = _eagle_step(
            draft,
            input_emb=seed_emb,
            hidden=target_h_3_BR,
            cache_hidden=cache_hidden,
            dtype=dtype,
        )
        for t in range(K):
            log_q = F.log_softmax(logits_1 / eps_T, dim=-1)  # (BR, vocab)
            new_logp_path[:, t] = log_q.gather(1, candidates_BR[:, t : t + 1]).squeeze(1)
            topk_ids_t = target_topk_ids[:, :, t, :].reshape(BR, TK)
            new_logp_topk[:, t, :] = log_q.gather(1, topk_ids_t)
            if t == K - 1:
                break
            # Advance recurrence with the STORED y_t (teacher-forced path).
            y_emb = draft.embed_input_ids(candidates_BR[:, t : t + 1]).to(dtype)
            h_next, logits_next = _eagle_step(
                draft,
                input_emb=y_emb,
                hidden=h_prev,
                cache_hidden=cache_hidden,
                dtype=dtype,
            )
            h_prev = h_next
            logits_1 = logits_next

    # ---- 3. SMC weights (from STORED stale rollout logps) ----
    target_logp_BR = target_logps.view(BR, K)
    old_logp_BR = draft_logps_old.view(BR, K)
    # Per-prompt path weights.
    logw = (target_logp_BR.sum(dim=1) - old_logp_BR.sum(dim=1)).view(B, R)  # (B, R)
    weights = F.softmax(logw, dim=1).detach()  # (B, R)

    # ---- 4. losses ----
    # 4a. recurrent target top-k KL
    teacher_topk_logp = target_topk_logps.view(BR, K, TK)
    teacher_lp = teacher_topk_logp - torch.logsumexp(teacher_topk_logp, dim=-1, keepdim=True)
    teacher_p = teacher_lp.exp().detach()
    topk_kl_per_pos = (teacher_p * (teacher_lp.detach() - new_logp_topk)).sum(dim=-1)
    # average over (BR, K)
    topk_kl = topk_kl_per_pos.mean()

    # 4b. SMC weighted path MLE
    new_logp_path_BR_K = new_logp_path  # (BR, K)
    new_path_logq_sum = new_logp_path_BR_K.sum(dim=1).view(B, R)  # (B, R)
    smc_loss = -(weights * new_path_logq_sum).sum(dim=1).mean()

    # 4c. anchor KL(q_old || q_new) on sampled path tokens, per-position.
    # Treat (q_old, q_new) as Categoricals supported on the R sampled tokens
    # at each step t. This approximates the anchor on the empirical sampled
    # distribution and is cheap to compute.
    if gamma_anchor > 0:
        old_lp_per_step = old_logp_BR.view(B, R, K)
        new_lp_per_step = new_logp_path.view(B, R, K)
        # softmax over R candidates per (B, t)
        old_lp_norm = old_lp_per_step - torch.logsumexp(old_lp_per_step, dim=1, keepdim=True)
        old_p_norm = old_lp_norm.exp().detach()
        new_lp_norm = new_lp_per_step - torch.logsumexp(new_lp_per_step, dim=1, keepdim=True)
        anchor = (old_p_norm * (old_lp_norm.detach() - new_lp_norm)).sum(dim=1).mean()
    else:
        anchor = torch.zeros((), device=device)

    loss = alpha_topk * topk_kl + beta_smc * smc_loss + gamma_anchor * anchor

    with torch.no_grad():
        ess = 1.0 / (weights * weights).sum(dim=1)  # (B,)
        stats = {
            "loss": float(loss.item()),
            "topk_kl": float(topk_kl.item()),
            "smc_loss": float(smc_loss.item()),
            "anchor": float(anchor.item()),
            "ess": float(ess.mean().item()),
            "ess_frac": float((ess / R).mean().item()),
            "logw_var": float(logw.var(dim=1, unbiased=False).mean().item()),
            "logw_var_max": float(logw.var(dim=1, unbiased=False).max().item()),
            "max_w": float(weights.max(dim=1).values.mean().item()),
            "target_lp_mean": float(target_logp_BR.mean().item()),
            "old_q_lp_mean": float(old_logp_BR.mean().item()),
            "new_q_lp_mean": float(new_logp_path.mean().item()),
            "path_logp_diff_mean": float(
                (new_logp_path - old_logp_BR).mean().item()
            ),
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--alpha-topk", type=float, default=1.0)
    p.add_argument("--beta-smc", type=float, default=0.05)
    p.add_argument("--gamma-anchor", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on optimizer steps for short conservative probes.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    add_specforge_to_path(args.specforge_root)
    from specforge.modeling.auto import AutoEagle3DraftModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    print(f"[recurrent] Loading rollouts from {args.data_dir}", flush=True)
    ds = RolloutDataset(args.data_dir)
    print(f"[recurrent] {len(ds)} rollout prompts loaded", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[recurrent] Loading EAGLE init {args.init}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(args.init, torch_dtype=dtype).to(device)
    draft.train()
    if hasattr(draft, "freeze_embedding"):
        draft.freeze_embedding()

    opt = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    full_steps = max(1, args.epochs * len(loader))
    total_steps = min(full_steps, args.max_steps) if args.max_steps else full_steps
    total_steps = max(1, total_steps)
    sched = get_linear_schedule_with_warmup(
        opt, max(0, args.warmup_steps), total_steps
    )

    global_step = 0
    log_path = out_dir / "train_log.jsonl"
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            vars(args)
            | {"n_rows": len(ds), "total_steps": total_steps, "full_steps": full_steps},
            f,
            indent=2,
        )

    print(
        f"[recurrent] training: {total_steps} steps, "
        f"alpha={args.alpha_topk}, beta={args.beta_smc}, gamma={args.gamma_anchor}, "
        f"lr={args.lr}, T={args.temperature}",
        flush=True,
    )

    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss, stats = compute_loss(
                draft,
                batch,
                device,
                dtype,
                alpha_topk=args.alpha_topk,
                beta_smc=args.beta_smc,
                gamma_anchor=args.gamma_anchor,
                temperature=args.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if global_step % args.log_interval == 0 or global_step == 1:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "lr": sched.get_last_lr()[0],
                    **stats,
                }
                print(json.dumps(entry), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            if global_step % args.save_interval == 0:
                ckpt = out_dir / f"step_{global_step}"
                draft.save_pretrained(ckpt)
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final = out_dir / "final"
    draft.save_pretrained(final)
    print(f"[recurrent] saved final checkpoint to {final}", flush=True)


if __name__ == "__main__":
    main()

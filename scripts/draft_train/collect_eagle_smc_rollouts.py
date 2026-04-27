#!/usr/bin/env python3
"""Collect SMC-native one-step EAGLE proposal rollouts.

This is the first, deliberately small, proposal-learning collector for SMC-SD.
It targets gamma_train=1 to make correctness easy to audit:

  prompt -> target samples seed x0 -> EAGLE proposes R candidates y1
  store log q_old(y1 | target_hidden(prompt+x0), x0)
  store log p_target(y1 | prompt+x0)

The resulting shards can be consumed by ``train_eagle_smc_proposal.py`` for
weighted path-MLE with weights proportional to exp(log p - log q_old).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_specforge_to_path(specforge_root: str):
    if specforge_root and specforge_root not in sys.path:
        sys.path.insert(0, specforge_root)


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_prompt(tokenizer, row: Dict, max_prompt_tokens: int) -> List[int]:
    conversations = row["conversations"]
    # Use all messages except final assistant as context when possible.
    if conversations and conversations[-1].get("role") == "assistant":
        prompt_messages = conversations[:-1]
    else:
        prompt_messages = conversations
    if not prompt_messages:
        prompt_messages = conversations[:1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    return ids


def eagle_layer_ids(target_config, explicit: str | None = None) -> List[int]:
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip()]
    n = int(getattr(target_config, "num_hidden_layers"))
    # SpecForge captures outputs of transformer layers [1, N//2 - 1, N - 4].
    # HF output_hidden_states has embedding output at index 0, so add one.
    # This also matches SMC-SD/SGLang EAGLE's default effective captures.
    return [2, n // 2, max(n - 3, 1)]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--draft", required=True, help="EAGLE3 checkpoint path")
    p.add_argument("--data", required=True, help="SpecForge-style conversations jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--specforge-root", default="/home/yahya/SpecForge")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--num-prompts", type=int, default=2000)
    p.add_argument("--num-candidates", type=int, default=8)
    p.add_argument("--target-topk", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-prompt-tokens", type=int, default=768)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--eagle-layer-ids", default=None, help="comma-separated target hidden-state layer ids")
    args = p.parse_args()

    add_specforge_to_path(args.specforge_root)
    from specforge.modeling.auto import AutoEagle3DraftModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading target {args.target} on {device} ({dtype})", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype, device_map={"": args.device}
    ).eval()
    layer_ids = eagle_layer_ids(target.config, args.eagle_layer_ids)
    print(f"Using EAGLE target hidden layers: {layer_ids}", flush=True)

    print(f"Loading EAGLE draft {args.draft}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(
        args.draft, torch_dtype=dtype, device_map={"": args.device}
    ).eval()

    rows = []
    shard_idx = 0
    n_done = 0
    t0 = time.perf_counter()
    metadata = vars(args) | {"target_hidden_layers": layer_ids}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for ex_idx, row in enumerate(iter_jsonl(args.data)):
        if n_done >= args.num_prompts:
            break
        prompt_ids = build_prompt(tokenizer, row, args.max_prompt_tokens)
        if len(prompt_ids) < 2:
            continue
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        # Target seed x0 from p(. | prompt). This mirrors the SMC EAGLE path,
        # where EAGLE drafts after a target-produced anchor token.
        out = target(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
        seed_logits = out.logits[:, -1, :].float()
        seed_logp = F.log_softmax(seed_logits / max(args.temperature, 1e-6), dim=-1)
        seed = torch.multinomial(seed_logp.exp(), num_samples=1).squeeze(1)

        prefix2 = torch.cat([input_ids, seed[:, None]], dim=1)
        attn2 = torch.ones_like(prefix2)
        out2 = target(input_ids=prefix2, attention_mask=attn2, output_hidden_states=True)
        hs = out2.hidden_states
        hidden_cat = torch.cat([hs[i][:, -1:, :] for i in layer_ids], dim=-1).to(dtype)
        target_next_logits = out2.logits[:, -1, :].float()
        target_logp = F.log_softmax(target_next_logits, dim=-1)
        topk_logps, topk_ids = torch.topk(target_logp, k=args.target_topk, dim=-1)

        # EAGLE q(. | hidden(prefix+x0), x0)
        hidden_proj = draft.project_hidden_states(hidden_cat)
        embeds = draft.embed_input_ids(seed[:, None]).to(dtype)
        mask = torch.ones((1, 1), dtype=torch.bool, device=device)
        attn_mask = draft.prepare_decoder_attention_mask(mask, hidden_proj, 1, 1, 0)
        # SpecForge EAGLE3 training uses local one-token draft positions here;
        # absolute target positions are encoded in the target hidden state.
        pos = torch.zeros((1, 1), dtype=torch.long, device=device)
        h1 = draft.backbone(
            input_embeds=embeds,
            hidden_states=hidden_proj,
            cache_hidden=[[], []],
            attention_mask=attn_mask,
            position_ids=pos,
            past_key_values=None,
            use_cache=False,
        )
        logits = draft.compute_logits(h1)[:, -1, :].float()
        q_logp = F.log_softmax(logits / max(args.temperature, 1e-6), dim=-1)
        candidates = torch.multinomial(q_logp.exp(), num_samples=args.num_candidates, replacement=True)[0]
        draft_lp = q_logp[0, candidates]
        target_lp = target_logp[0, candidates]

        rows.append(
            {
                "prompt_id": row.get("id", str(ex_idx)),
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
                "seed_token": seed[0].to(torch.int32).cpu(),
                "hidden_states": hidden_cat[0, 0].to(torch.float16).cpu(),
                "candidates": candidates.to(torch.int32).cpu().view(args.num_candidates, 1),
                "draft_logps_old": draft_lp.to(torch.float16).cpu().view(args.num_candidates, 1),
                "target_logps": target_lp.to(torch.float16).cpu().view(args.num_candidates, 1),
                "target_topk_ids": topk_ids[0].to(torch.int32).cpu(),
                "target_topk_logps": topk_logps[0].to(torch.float16).cpu(),
                "gamma_train": 1,
                "temperature": args.temperature,
            }
        )
        n_done += 1

        if len(rows) >= args.shard_size:
            path = out_dir / f"shard_{shard_idx:05d}.pt"
            torch.save(rows, path)
            rows = []
            shard_idx += 1
        if n_done % 50 == 0:
            rate = n_done / max(time.perf_counter() - t0, 1e-6)
            print(f"[{n_done}/{args.num_prompts}] shards={shard_idx} rate={rate:.2f} prompts/s", flush=True)

    if rows:
        path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(rows, path)
        shard_idx += 1
    print(f"Done: {n_done} prompts, {shard_idx} shards -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

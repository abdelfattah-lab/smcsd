"""Generate target rollouts + cached top-K logits for KL distillation.

One-shot. Expect ~12h on one H100 for 7.5K GSM8K prompts x 4 samples x 256 tokens.
Shard schema (safetensors):
    prompt_tokens : [N, max_prompt_len] int32  (left-padded with pad_id)
    prompt_lens   : [N] int32
    gen_tokens    : [N, gen_len] int32
    topk_indices  : [N, gen_len, K] int32
    topk_logits   : [N, gen_len, K] bfloat16
"""
import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = "You are a helpful assistant that solves math problems step by step."


def format_prompt(tokenizer, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    # Render to string first; transformers 5.x returns tokenizers.Encoding when tokenize=True.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def generate_batch(model, tokenizer, prompt_ids_list, continuation_len, temperature, top_k, device):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    B = len(prompt_ids_list)
    plens = torch.tensor([len(p) for p in prompt_ids_list], dtype=torch.int32)
    max_plen = int(plens.max().item())

    # Left-pad so all prompts end at the same position; attention_mask lets
    # HF's Llama infer correct rotary positions per row.
    input_ids = torch.full((B, max_plen), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_plen), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids_list):
        input_ids[i, max_plen - len(p):] = torch.tensor(p, device=device)
        attn_mask[i, max_plen - len(p):] = 1

    gen_tokens = torch.zeros((B, continuation_len), dtype=torch.int32, device=device)
    topk_idx = torch.zeros((B, continuation_len, top_k), dtype=torch.int32, device=device)
    topk_val = torch.zeros((B, continuation_len, top_k), dtype=torch.bfloat16, device=device)

    past = None
    cur_ids = input_ids
    cur_mask = attn_mask
    for step in range(continuation_len):
        out = model(input_ids=cur_ids, attention_mask=cur_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]  # [B, V]
        past = out.past_key_values

        vals, idx = torch.topk(logits, top_k, dim=-1)
        topk_idx[:, step] = idx.to(torch.int32)
        topk_val[:, step] = vals.to(torch.bfloat16)

        probs = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(-1)
        gen_tokens[:, step] = nxt.to(torch.int32)

        cur_ids = nxt.unsqueeze(-1)
        cur_mask = torch.cat([cur_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=-1)

    return {
        "prompt_tokens": input_ids.to(torch.int32).cpu(),
        "prompt_lens": plens,
        "gen_tokens": gen_tokens.cpu(),
        "topk_indices": topk_idx.cpu(),
        "topk_logits": topk_val.cpu(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--continuation-len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=512, help="rollouts per shard file")
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sort-by-length", action="store_true", help="sort prompts to reduce padding waste")
    ap.add_argument("--rank", type=int, default=0, help="data-parallel rank; this process handles its slice of rollouts")
    ap.add_argument("--world-size", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    if args.max_prompts:
        gsm8k = gsm8k.select(range(args.max_prompts))

    prompts = [format_prompt(tokenizer, ex["question"]) for ex in gsm8k]
    rollouts = [p for p in prompts for _ in range(args.samples_per_prompt)]
    if args.sort_by_length:
        rollouts.sort(key=len)
    total = len(rollouts)
    if args.world_size > 1:
        lo = args.rank * total // args.world_size
        hi = (args.rank + 1) * total // args.world_size
        rollouts = rollouts[lo:hi]
    print(f"[rank {args.rank}/{args.world_size}] rollouts: {len(rollouts)}/{total} | batch: {args.batch_size} | shard: {args.shard_size}")

    t0 = time.time()
    for shard_idx, s in enumerate(range(0, len(rollouts), args.shard_size)):
        shard = rollouts[s:s + args.shard_size]
        batches = []
        for b in tqdm(range(0, len(shard), args.batch_size), desc=f"shard {shard_idx}"):
            batches.append(generate_batch(
                model, tokenizer, shard[b:b + args.batch_size],
                args.continuation_len, args.temperature, args.top_k, args.device,
            ))
        # Different batches may have different max_plen; pad prompt_tokens to shard max.
        shard_max_plen = max(b["prompt_tokens"].shape[1] for b in batches)
        pad_id = tokenizer.pad_token_id
        for b in batches:
            if b["prompt_tokens"].shape[1] < shard_max_plen:
                pad = torch.full(
                    (b["prompt_tokens"].shape[0], shard_max_plen - b["prompt_tokens"].shape[1]),
                    pad_id, dtype=torch.int32,
                )
                b["prompt_tokens"] = torch.cat([pad, b["prompt_tokens"]], dim=1)
        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        fname = f"shard_r{args.rank:02d}_{shard_idx:05d}.safetensors"
        save_file(merged, str(out_dir / fname))
        done = s + len(shard)
        rate = done / max(time.time() - t0, 1e-6)
        eta_h = (len(rollouts) - done) / max(rate, 1e-6) / 3600
        print(f"[{shard_idx}] saved | {done}/{len(rollouts)} | {rate:.1f} rollouts/s | ETA {eta_h:.1f}h")


if __name__ == "__main__":
    main()

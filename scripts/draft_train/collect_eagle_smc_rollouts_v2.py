#!/usr/bin/env python3
"""Collect SMC-native, *recurrent* EAGLE proposal rollouts (gamma_train > 1).

This is the v2 collector for SMC-SD proposal training. It matches the
recurrent EAGLE dynamics used at SMC-SD inference time:

  - Target prefill on ``prompt`` -> sample seed token x0 ~ p(.|prompt) at T.
  - Target prefill on ``prompt + [x0]`` -> capture multi-layer aux target
    hidden state ``target_h_3`` (concat of low/mid/high layers) at the LAST
    position. This is exactly the seed signal sglang's SMC EAGLE chain mode
    feeds into the draft after a verified token.
  - Draft seed pass: feed ``(input_emb=embed(x0), hidden_states=target_h_3)``
    of shape (1, 1, ...) into the EAGLE midlayer with ``cache_hidden=[[], []]``
    and local ``position_ids=[[0]]``. The FC inside ``project_hidden_states``
    handles the 3*hidden_size -> hidden_size projection. The midlayer output
    is the seed recurrent state ``h_seed`` (pre-norm post-residual), and
    ``compute_logits`` produces the proposal distribution over y_1.
  - Recurrent unroll for t = 1..gamma_train:
      logits_t = compute_logits(h_{t-1})       # h_0 = h_seed
      y_t     ~ q_old(. | logits_t / T)
      h_t     = midlayer(input_emb=embed(y_t), hidden_states=h_{t-1},
                         cache_hidden=growing, position_ids=[[0]])
    R candidate paths are drawn in parallel (batched over R; same h_seed,
    different y_t draws diverge them after step 1).
  - Score the entire path under the target model in one teacher-forced pass:
      log p_t = log_softmax(target.logits[:, prompt_len + t - 1, :])(y_t)
      target_topk_t = topk(log_softmax(target.logits[:, prompt_len + t - 1, :]))
  - Store per-prompt: prompt ids, seed x0, target_h_3, R candidate paths,
    per-step old draft log-probs, per-step target log-probs, and per-step
    target top-k for KL anchoring.

The resulting shards are consumed by ``train_eagle_smc_recurrent.py``, which
teacher-forces the saved paths through the EAGLE recurrence with the
trainable draft to compute log q_new and the SMC weighted path-MLE loss.
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
    """Layer indices used for EAGLE3 multi-layer aux hidden capture.

    SpecForge chooses transformer layer IDs ``[1, N//2 - 1, N - 4]`` and
    captures each selected layer's output. HF ``output_hidden_states`` includes
    the embedding output at index 0, so the equivalent indices are each layer
    ID plus one. For Llama-3.1-8B this is ``[2, 16, 29]``.

    SMC-SD/SGLang's default EAGLE capture uses ``[2, N//2, N-3]`` internally
    because it captures the state before those layers, i.e. the same previous
    layer outputs. Keep this collector aligned with that runtime convention.
    """
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip()]
    n = int(getattr(target_config, "num_hidden_layers"))
    return [2, n // 2, max(n - 3, 1)]


def _eagle_prefill(
    draft,
    *,
    input_emb: torch.Tensor,
    hidden: torch.Tensor,
    cache_hidden,
    dtype: torch.dtype,
    attention_mask: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
):
    """EAGLE prefill pass on (B, S, ...) tensors.

    Mirrors sglang's SMC EAGLE3 chain prefill: feed full-prompt
    ``input_emb`` of shape (B, S, H_d) and full-prompt ``hidden`` of shape
    (B, S, 3*H_t) (raw target aux features) or (B, S, H_d) (already
    projected). Uses the streaming attention path with
    ``cache_hidden=[[], []]`` so the prompt's K/V is appended as ONE entry
    (a (B, kv_heads, S, head_dim) tensor) and subsequent 1-token decode
    steps will append 1-token K/V on top of it. Returns (h_last, logits)
    where h_last is (B, 1, H_d) at each row's last non-pad position and logits
    is (B, vocab) for that same position.
    """
    B, S, _ = input_emb.shape
    H = draft.config.hidden_size
    if hidden.shape[-1] == 3 * H:
        h_in = draft.project_hidden_states(hidden).to(dtype)
    else:
        assert hidden.shape[-1] == H, (
            f"hidden last dim {hidden.shape[-1]} != H ({H}) or 3H ({3*H})"
        )
        h_in = hidden.to(dtype)
    mask = (
        torch.ones((B, S), dtype=torch.bool, device=input_emb.device)
        if attention_mask is None
        else attention_mask.to(device=input_emb.device, dtype=torch.bool)
    )
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
    if lengths is None:
        last_idx = torch.full((B,), S - 1, dtype=torch.long, device=input_emb.device)
    else:
        last_idx = lengths.to(device=input_emb.device, dtype=torch.long) - 1
    batch_idx = torch.arange(B, device=input_emb.device)
    h_last = h_out[batch_idx, last_idx, :].unsqueeze(1)
    logits = logits_all[batch_idx, last_idx, :]
    return h_last, logits


def _eagle_step(
    draft,
    *,
    input_emb: torch.Tensor,
    hidden: torch.Tensor,
    cache_hidden,
    dtype: torch.dtype,
):
    """Single recurrent EAGLE step on (B, 1, ...) tensors.

    Use after :func:`_eagle_prefill` to advance one decode step. ``hidden``
    is the recurrent draft hidden state from the previous step (B, 1, H_d).
    The position id passed is local 0 because SpecForge's streaming
    attention adds ``lck`` (= len(cache_hidden[0])) to it, giving
    cumulative positions across appended cache entries.
    """
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
    p.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Skip the first N prompts in --data (for sharding across processes).",
    )
    p.add_argument(
        "--shard-prefix",
        default="shard",
        help="Filename prefix for output shards (use a unique prefix per process when sharding).",
    )
    p.add_argument("--num-candidates", "-R", type=int, default=8)
    p.add_argument(
        "--gamma-train",
        type=int,
        default=2,
        help="Number of recurrent draft steps per path (>= 1).",
    )
    p.add_argument("--target-topk", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument(
        "--eagle-layer-ids",
        default=None,
        help="comma-separated target hidden-state layer ids (default: low/mid/high)",
    )
    p.add_argument(
        "--save-prompt-ids",
        action="store_true",
        help="Persist prompt_ids per row (slightly larger shards).",
    )
    args = p.parse_args()

    if args.gamma_train < 1:
        raise ValueError("--gamma-train must be >= 1")

    add_specforge_to_path(args.specforge_root)
    from specforge.modeling.auto import AutoEagle3DraftModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[v2] Loading target {args.target} on {device} ({dtype})", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype, device_map={"": args.device}
    ).eval()
    layer_ids = eagle_layer_ids(target.config, args.eagle_layer_ids)
    H_target = int(target.config.hidden_size)
    print(
        f"[v2] EAGLE target hidden layer ids: {layer_ids} (3H={3*H_target})",
        flush=True,
    )

    print(f"[v2] Loading EAGLE draft {args.draft}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(
        args.draft, torch_dtype=dtype, device_map={"": args.device}
    ).eval()
    H_draft = int(draft.config.hidden_size)

    R = int(args.num_candidates)
    K = int(args.gamma_train)
    T = float(args.temperature)
    eps_T = max(T, 1e-6)

    rows: List[Dict] = []
    shard_idx = 0
    n_done = 0
    t0 = time.perf_counter()
    metadata = vars(args) | {
        "target_hidden_layers": layer_ids,
        "target_hidden_size": H_target,
        "draft_hidden_size": H_draft,
        "vocab_size": int(target.config.vocab_size),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for ex_idx, row in enumerate(iter_jsonl(args.data)):
        if ex_idx < args.start_offset:
            continue
        if n_done >= args.num_prompts:
            break
        prompt_ids = build_prompt(tokenizer, row, args.max_prompt_tokens)
        if len(prompt_ids) < 2:
            continue
        prompt_len = len(prompt_ids)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        # ---- 1. target prefill on prompt; sample seed x0 ~ p(.|prompt) at T ----
        # Runtime SMC-SD EAGLE chain feeds the draft the target aux hidden
        # states for the prompt positions, with shifted draft input IDs ending
        # in x0. Do not use target_hidden(prompt+x0) for the first proposal.
        out0 = target(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        seed_logits = out0.logits[:, -1, :].float()
        seed_logp = F.log_softmax(seed_logits / eps_T, dim=-1)
        seed = torch.multinomial(seed_logp.exp(), num_samples=1).squeeze(1)  # (1,)
        hs = out0.hidden_states  # tuple of (1, prompt_len, H_target)
        target_h_3_seq = torch.cat([hs[i] for i in layer_ids], dim=-1).to(dtype)
        shifted_ids = input_ids.clone()
        shifted_ids[:, :-1] = input_ids[:, 1:]
        shifted_ids[:, -1] = seed

        # ---- 2. runtime-like draft prefill over the full prompt ----
        # SpecForge's manual cache assumes the first cached segment has q_len=1,
        # while SGLang runtime uses paged KV for the whole prompt. To mirror the
        # runtime distribution offline, replay the whole growing EAGLE sequence
        # without cache; this is slower but equivalent for these short K rollouts.
        cache_hidden = None
        shifted_emb = draft.embed_input_ids(shifted_ids).to(dtype)
        h_seed, logits_seed = _eagle_prefill(
            draft,
            input_emb=shifted_emb,
            hidden=target_h_3_seq,
            cache_hidden=cache_hidden,
            dtype=dtype,
        )
        # h_seed: (1, 1, H_draft); logits_seed: (1, vocab)

        # ---- 4. recurrent unroll over R candidate paths, length K = gamma_train ----
        prompt_hidden = draft.project_hidden_states(target_h_3_seq).to(dtype)
        hidden_context = prompt_hidden.expand(R, -1, -1).contiguous()
        embed_context = shifted_emb.expand(R, -1, -1).contiguous()
        h_prev = h_seed.expand(R, 1, H_draft).contiguous()
        logits_prev = logits_seed.expand(R, -1).contiguous()

        path_tokens = torch.empty((R, K), dtype=torch.long, device=device)
        path_old_logp = torch.empty((R, K), dtype=torch.float32, device=device)

        for t in range(K):
            log_q = F.log_softmax(logits_prev / eps_T, dim=-1)  # (R, vocab)
            sample_probs = log_q.exp()
            y_t = torch.multinomial(sample_probs, num_samples=1).squeeze(1)  # (R,)
            path_tokens[:, t] = y_t
            path_old_logp[:, t] = log_q.gather(1, y_t[:, None]).squeeze(1)

            if t == K - 1:
                break

            y_emb = draft.embed_input_ids(y_t[:, None]).to(dtype)  # (R, 1, H_draft)
            hidden_context = torch.cat([hidden_context, h_prev], dim=1)
            embed_context = torch.cat([embed_context, y_emb], dim=1)
            h_next, logits_next = _eagle_prefill(
                draft,
                input_emb=embed_context,
                hidden=hidden_context,
                cache_hidden=None,
                dtype=dtype,
                lengths=torch.full((R,), embed_context.shape[1], dtype=torch.long, device=device),
            )
            h_prev = h_next
            logits_prev = logits_next

        # ---- 5. score path under target with one teacher-forced pass ----
        # Build R sequences of length prompt_len + 1 + K with tokens
        # prompt_ids + [x0] + y_1..y_K. Run target once.
        path_full = torch.cat(
            [
                input_ids.expand(R, -1).contiguous(),
                seed.expand(R)[:, None],
                path_tokens,
            ],
            dim=1,
        )  # (R, prompt_len + 1 + K)
        attn_full = torch.ones_like(path_full)
        out2 = target(
            input_ids=path_full, attention_mask=attn_full, output_hidden_states=False
        )
        # logits at positions prompt_len + t - 1 (0-indexed) predict y_t for
        # t=1..K. position prompt_len holds x0; logits[:, prompt_len, :]
        # predicts y_1 (next token after x0). We score with the SAME
        # temperature used in SMC inference (both target and draft are tempered
        # in scheduler/worker.py), so logw matches sglang's logprob_diff.
        target_logits_path = (
            out2.logits[:, prompt_len : prompt_len + K, :].float()
        )  # (R, K, vocab)
        target_logp_full = F.log_softmax(target_logits_path / eps_T, dim=-1)
        target_path_logp = target_logp_full.gather(
            2, path_tokens[:, :, None]
        ).squeeze(2)  # (R, K)
        topk_logps, topk_ids = torch.topk(target_logp_full, k=args.target_topk, dim=-1)

        rec = {
            "prompt_id": row.get("id", str(ex_idx)),
            "prompt_len": prompt_len,
            "seed_token": seed[0].to(torch.int32).cpu(),
            "target_h_3": target_h_3_seq[0, -1].to(torch.float16).cpu(),  # legacy/debug
            "target_h_3_seq": target_h_3_seq[0].to(torch.float16).cpu(),  # (prompt_len, 3H)
            "shifted_input_ids": shifted_ids[0].to(torch.int32).cpu(),  # prompt shifted left, ending x0
            "candidates": path_tokens.to(torch.int32).cpu(),  # (R, K)
            "draft_logps_old": path_old_logp.to(torch.float32).cpu(),  # (R, K)
            "target_logps": target_path_logp.to(torch.float32).cpu(),  # (R, K)
            "target_topk_ids": topk_ids.to(torch.int32).cpu(),  # (R, K, topk)
            "target_topk_logps": topk_logps.to(torch.float32).cpu(),  # (R, K, topk)
            "gamma_train": K,
            "num_candidates": R,
            "temperature": T,
        }
        if args.save_prompt_ids:
            rec["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.int32)
        rows.append(rec)
        n_done += 1

        if len(rows) >= args.shard_size:
            path = out_dir / f"{args.shard_prefix}_{shard_idx:05d}.pt"
            torch.save(rows, path)
            rows = []
            shard_idx += 1
        if n_done % 25 == 0 or n_done == 1:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            rate = n_done / elapsed
            with torch.no_grad():
                logw = path_old_logp.new_empty((R,))
                logw = (
                    target_path_logp.sum(dim=1) - path_old_logp.sum(dim=1)
                ).detach()
                logw_var = float(logw.var(unbiased=False).item())
                ess = float(
                    (
                        F.softmax(logw, dim=0).pow(2).sum().reciprocal()
                    ).item()
                )
            print(
                f"[v2 {n_done}/{args.num_prompts}] rate={rate:.2f}/s shards={shard_idx} "
                f"K={K} R={R} prompt_len={prompt_len} "
                f"path_logw_var={logw_var:.2f} ess={ess:.2f}/{R}",
                flush=True,
            )

    if rows:
        path = out_dir / f"{args.shard_prefix}_{shard_idx:05d}.pt"
        torch.save(rows, path)
        shard_idx += 1
    print(
        f"[v2] Done: {n_done} prompts, {shard_idx} shards -> {out_dir}", flush=True
    )


if __name__ == "__main__":
    main()

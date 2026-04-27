#!/usr/bin/env python3
"""On-policy Rao-Blackwellized KL distillation for SMC-SD EAGLE drafts.

Target objective:  minimize  E_traj~q_old [ Sum_n KL( target_p(.|prefix)
                                                 || q_theta(.|prefix) ) ]

For each batch we:
  1. Run target prefill on the prompt and sample x0 ~ p(.|prompt) at temp T.
  2. Roll out EAGLE recurrently for K steps using the *snapshotted* old
     weights q_old (no_grad) to produce tokens y_1..y_{K-1}. This is the
     on-policy trajectory the model actually sees at inference.
  3. Run target ONCE on (prompt + x0 + y_1..y_{K-1}) to get target's full
     next-token distribution at every position from prompt_len to
     prompt_len + K - 1. These are the K teacher distributions p_t.
  4. Teacher-force the SAME trajectory through the trainable EAGLE
     recurrence to get q_theta_t at every position t = 0..K-1.
  5. Loss = (1/(B*K)) Sum_{b,t} sum_v p_t(v) [log p_t(v) - log q_theta_t(v)]
     This is the Rao-Blackwellized KL estimator (Amini et al., 2025): an
     exact KL on full per-step distributions, provably variance-bounded
     vs any path-level Monte Carlo estimator.
  6. Backprop into EAGLE only.

q_old is snapshotted at the start and refreshed every --refresh-old-every
steps. We omit the off-policy importance weight (Eq. 21) since with small
LR and frequent snapshots q_theta ~= q_old, so IS = 1 + O(lr * dist).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def eagle_layer_ids(target_config, explicit: Optional[str] = None) -> List[int]:
    if explicit:
        return [int(x) for x in explicit.split(",")]
    n = int(getattr(target_config, "num_hidden_layers"))
    return [2, n // 2, max(n - 3, 1)]


def _eagle_prefill(
    draft,
    *,
    input_emb: torch.Tensor,
    hidden: torch.Tensor,
    cache_hidden,
    dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
):
    """Run EAGLE recurrence over (B, S, ...) inputs and return (h_last, logits)."""
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


def eagle_step_logits(
    draft,
    *,
    target_h_3_seq: torch.Tensor,  # (1, ctx_len, 3H)
    shifted_input_ids: torch.Tensor,  # (1, ctx_len) ending in seed
    rolled_tokens: torch.Tensor,  # (1, K) tokens y_1..y_K
    dtype: torch.dtype,
):
    """Run the EAGLE recurrence for K+1 positions and return logits at each.

    The recurrence is identical to v3 collector: prefill on (target_h_3_seq,
    shifted_input_ids) gives h_seed and logits at position 0 (predicting
    y_1). For each subsequent step t we extend (hidden_context, embed_context)
    by (h_prev, embed(y_t)) and run prefill on the full sequence; the last-
    position logits predict y_{t+1}.

    Returns logits of shape (1, K, vocab) — logits[:, t, :] predicts the
    (t+1)th rolled token y_{t+1}, given prefix that includes y_1..y_t.
    Wait — semantics: logits[:, 0, :] predicts y_1 from prefix ending at
    seed; logits[:, 1, :] predicts y_2 from prefix ending at y_1; ...
    logits[:, K, :] predicts y_{K+1} from prefix ending at y_K.
    Total K+1 logit rows are returned.
    """
    B = shifted_input_ids.shape[0]
    H_d = draft.config.hidden_size
    device = shifted_input_ids.device
    K = int(rolled_tokens.shape[-1])

    # prompt prefill
    shifted_emb = draft.embed_input_ids(shifted_input_ids).to(dtype)
    h_seed, logits_seed = _eagle_prefill(
        draft, input_emb=shifted_emb, hidden=target_h_3_seq,
        cache_hidden=None, dtype=dtype,
    )  # h_seed: (B, 1, H_d); logits_seed: (B, V)

    out_logits = [logits_seed]

    prompt_hidden = draft.project_hidden_states(target_h_3_seq).to(dtype)
    hidden_context = prompt_hidden  # (B, ctx_len, H_d)
    embed_context = shifted_emb       # (B, ctx_len, H_d)
    h_prev = h_seed                    # (B, 1, H_d)

    for t in range(K):
        y_t = rolled_tokens[:, t]
        y_emb = draft.embed_input_ids(y_t[:, None]).to(dtype)  # (B, 1, H_d)
        hidden_context = torch.cat([hidden_context, h_prev], dim=1)
        embed_context = torch.cat([embed_context, y_emb], dim=1)
        h_next, logits_next = _eagle_prefill(
            draft,
            input_emb=embed_context,
            hidden=hidden_context,
            cache_hidden=None,
            dtype=dtype,
            lengths=torch.full(
                (B,), embed_context.shape[1], dtype=torch.long, device=device
            ),
        )
        out_logits.append(logits_next)
        h_prev = h_next

    # logits stacked: (B, K+1, V)
    return torch.stack(out_logits, dim=1)


@torch.no_grad()
def rollout_old(
    draft,
    *,
    target_h_3_seq: torch.Tensor,  # (1, ctx_len, 3H)
    shifted_input_ids: torch.Tensor,  # (1, ctx_len) ending in seed
    K: int,
    temperature: float,
    dtype: torch.dtype,
):
    """Sample y_1..y_K from q_old (this draft's current weights, no grad)."""
    B = shifted_input_ids.shape[0]
    device = shifted_input_ids.device

    shifted_emb = draft.embed_input_ids(shifted_input_ids).to(dtype)
    h_seed, logits_seed = _eagle_prefill(
        draft, input_emb=shifted_emb, hidden=target_h_3_seq,
        cache_hidden=None, dtype=dtype,
    )

    rolled = torch.empty((B, K), dtype=torch.long, device=device)
    prompt_hidden = draft.project_hidden_states(target_h_3_seq).to(dtype)
    hidden_context = prompt_hidden
    embed_context = shifted_emb
    h_prev = h_seed
    logits_prev = logits_seed

    for t in range(K):
        log_q = F.log_softmax(logits_prev / max(temperature, 1e-6), dim=-1)
        y_t = torch.multinomial(log_q.exp(), num_samples=1).squeeze(1)
        rolled[:, t] = y_t
        if t == K - 1:
            break
        y_emb = draft.embed_input_ids(y_t[:, None]).to(dtype)
        hidden_context = torch.cat([hidden_context, h_prev], dim=1)
        embed_context = torch.cat([embed_context, y_emb], dim=1)
        h_next, logits_next = _eagle_prefill(
            draft, input_emb=embed_context, hidden=hidden_context,
            cache_hidden=None, dtype=dtype,
            lengths=torch.full(
                (B,), embed_context.shape[1], dtype=torch.long, device=device
            ),
        )
        h_prev, logits_prev = h_next, logits_next

    return rolled


def kl_full_dist(target_logp: torch.Tensor, q_logp: torch.Tensor) -> torch.Tensor:
    """KL(p || q) = sum_v p(v) * (log p(v) - log q(v)) per row.

    target_logp, q_logp: (B*K, V) log-softmax distributions in fp32.
    Returns (B*K,) per-row KL.
    """
    p = target_logp.exp()
    return (p * (target_logp - q_logp)).sum(dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--draft", required=True, help="EAGLE3 init checkpoint")
    p.add_argument("--data", required=True, help="SpecForge-style conversations jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--specforge-root", default="/home/yahya/SpecForge")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gamma-train", "-K", type=int, default=8,
                   help="Length of EAGLE rollout per training example. "
                        "Ignored if --k-schedule is given.")
    p.add_argument("--k-schedule", type=str, default=None,
                   help="Curriculum on K: comma-separated 'K:start_step' pairs, "
                        "e.g. '2:0,4:2500,8:5000,12:7500'. K is set to the "
                        "value of the highest K whose start_step <= current step.")
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--refresh-old-every", type=int, default=200,
                   help="Refresh q_old snapshot every N steps. With small lr "
                        "this can be skipped (snapshot once at start).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--eagle-layer-ids", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    add_specforge_to_path(args.specforge_root)
    from specforge.modeling.auto import AutoEagle3DraftModel

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[kl] Loading target {args.target} on {device} ({dtype})", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype, device_map={"": args.device}
    ).eval()
    for p_ in target.parameters():
        p_.requires_grad_(False)

    layer_ids = eagle_layer_ids(target.config, args.eagle_layer_ids)
    H_target = int(target.config.hidden_size)
    vocab = int(target.config.vocab_size)
    print(f"[kl] target layer ids {layer_ids} (3H={3*H_target}) vocab={vocab}",
          flush=True)

    print(f"[kl] Loading EAGLE init {args.draft}", flush=True)
    draft = AutoEagle3DraftModel.from_pretrained(
        args.draft, torch_dtype=dtype, device_map={"": args.device}
    ).train()

    # snapshot for q_old (no grad). Refreshed periodically.
    print("[kl] Building q_old snapshot ...", flush=True)
    draft_old = copy.deepcopy(draft).eval()
    for p_ in draft_old.parameters():
        p_.requires_grad_(False)

    optim = torch.optim.AdamW(draft.parameters(), lr=args.lr, betas=(0.9, 0.95))

    K_default = int(args.gamma_train)
    T = float(args.temperature)
    B = int(args.batch_size)
    eps_T = max(T, 1e-6)

    # parse curriculum schedule
    k_schedule = None
    if args.k_schedule:
        k_schedule = []
        for tok in args.k_schedule.split(","):
            k_str, s_str = tok.split(":")
            k_schedule.append((int(s_str), int(k_str)))
        k_schedule.sort()
        print(f"[kl] K curriculum: {k_schedule}", flush=True)

    def current_K(step: int) -> int:
        if not k_schedule:
            return K_default
        active = K_default
        for s, k in k_schedule:
            if step >= s:
                active = k
        return active

    # iterator over data, recycled if we run out
    data_path = Path(args.data)
    rows = []  # buffer of next prompts (each as a list[int])
    data_iter = iter_jsonl(args.data)

    def pop_prompt():
        nonlocal data_iter
        for _ in range(20):
            try:
                row = next(data_iter)
            except StopIteration:
                data_iter = iter_jsonl(args.data)
                row = next(data_iter)
            ids = build_prompt(tokenizer, row, args.max_prompt_tokens)
            if len(ids) >= 4:
                return ids
        raise RuntimeError("Could not find any prompt with >=4 tokens.")

    log_path = out_dir / "train.log.jsonl"
    log_fh = open(log_path, "a", buffering=1)

    t0 = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        # warmup + linear decay
        if step <= args.warmup_steps:
            lr_scale = step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            lr_scale = max(1.0 - progress, 0.0)
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_scale

        # ---- 1. fetch B prompts (variable length, no padding within K steps) ----
        # process one example at a time; accumulate gradients to effective batch B
        K = current_K(step)
        optim.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_kl_per_pos_sum = 0.0
        step_kl_per_pos_count = 0
        step_target_lp_mean = 0.0
        step_q_lp_mean = 0.0

        for b in range(B):
            prompt_ids = pop_prompt()
            prompt_len = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)

            # ---- 2. target prefill on prompt; sample x0 ~ p(.|prompt) ----
            with torch.no_grad():
                out0 = target(
                    input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True,
                )
                seed_logits = out0.logits[:, -1, :].float()
                seed_logp = F.log_softmax(seed_logits / eps_T, dim=-1)
                seed = torch.multinomial(seed_logp.exp(), num_samples=1).squeeze(1)
                hs = out0.hidden_states
                target_h_3_seq = torch.cat(
                    [hs[i] for i in layer_ids], dim=-1
                ).to(dtype)
                shifted_ids = input_ids.clone()
                shifted_ids[:, :-1] = input_ids[:, 1:]
                shifted_ids[:, -1] = seed

            # ---- 3. roll out EAGLE for K steps using q_old ----
            with torch.no_grad():
                rolled = rollout_old(
                    draft_old,
                    target_h_3_seq=target_h_3_seq,
                    shifted_input_ids=shifted_ids,
                    K=K,
                    temperature=T,
                    dtype=dtype,
                )  # (1, K) tokens y_1..y_K

            # ---- 4. target forward on (prompt + x0 + y_1..y_{K-1}) ----
            # We need target's distribution at positions prompt_len..prompt_len+K-1
            # which predict y_1..y_K. The input at position prompt_len-1 is the
            # last prompt token; position prompt_len = x0; position
            # prompt_len + t = y_t for t=1..K-1 (no need to include y_K).
            with torch.no_grad():
                full_input = torch.cat(
                    [input_ids, seed[:, None], rolled[:, :K-1]], dim=1
                )  # (1, prompt_len + K)
                full_attn = torch.ones_like(full_input)
                out_full = target(
                    input_ids=full_input, attention_mask=full_attn,
                    output_hidden_states=False,
                )
                target_logits_traj = out_full.logits[
                    :, prompt_len - 1 : prompt_len - 1 + K, :
                ].float()  # (1, K, vocab) — predicts y_1..y_K? wait
                # Actually: logits[:, n, :] predicts position n+1.
                # We want predictions of y_1..y_K, which are at positions
                # prompt_len..prompt_len+K-1 in full_input. So we read logits
                # at positions prompt_len-1..prompt_len+K-2.
                target_logits_traj = out_full.logits[
                    :, prompt_len - 1 : prompt_len - 1 + K, :
                ].float()
                target_logp_traj = F.log_softmax(target_logits_traj / eps_T, dim=-1)
                # (1, K, vocab) — target_logp_traj[:, t, :] predicts y_{t+1}
                # for t = 0..K-1, i.e. distribution over y_1, y_2, ..., y_K.

            # ---- 5. teacher-force trajectory through trainable EAGLE ----
            # eagle_step_logits returns logits at K+1 positions: predicts
            # y_1, y_2, ..., y_{K+1}. We only need the first K (predicting
            # y_1..y_K), aligned with target.
            q_logits_full = eagle_step_logits(
                draft,
                target_h_3_seq=target_h_3_seq,
                shifted_input_ids=shifted_ids,
                rolled_tokens=rolled,  # (1, K)
                dtype=dtype,
            )  # (1, K+1, vocab)
            q_logits_traj = q_logits_full[:, :K, :]  # (1, K, vocab)
            q_logp_traj = F.log_softmax(q_logits_traj / eps_T, dim=-1)

            # ---- 6. RB-KL loss ----
            # KL(p || q) per position, mean over K positions
            p_traj = target_logp_traj.exp()
            kl_per_pos = (
                p_traj * (target_logp_traj - q_logp_traj)
            ).sum(dim=-1)  # (1, K)
            loss_b = kl_per_pos.mean()
            (loss_b / B).backward()

            step_loss += float(loss_b.item())
            step_kl_per_pos_sum += float(kl_per_pos.sum().item())
            step_kl_per_pos_count += int(kl_per_pos.numel())
            with torch.no_grad():
                # diagnostics: log-probs of *sampled* tokens under target and q
                idx = rolled[:, :K]  # (1, K)
                target_lp_sampled = target_logp_traj.gather(
                    2, idx[:, :, None]
                ).squeeze(2)
                q_lp_sampled = q_logp_traj.gather(
                    2, idx[:, :, None]
                ).squeeze(2)
                step_target_lp_mean += float(target_lp_sampled.mean().item())
                step_q_lp_mean += float(q_lp_sampled.mean().item())

        # gradient clip + step
        gn = torch.nn.utils.clip_grad_norm_(draft.parameters(), args.grad_clip)
        optim.step()

        # refresh q_old periodically
        if args.refresh_old_every > 0 and step % args.refresh_old_every == 0:
            with torch.no_grad():
                draft_old.load_state_dict(draft.state_dict())

        loss = step_loss / B
        kl_mean = step_kl_per_pos_sum / max(step_kl_per_pos_count, 1)
        target_lp = step_target_lp_mean / B
        q_lp = step_q_lp_mean / B

        rec = {
            "step": step,
            "K": K,
            "lr": args.lr * lr_scale,
            "loss": loss,
            "kl_per_pos_mean": kl_mean,
            "target_lp_sampled": target_lp,
            "q_lp_sampled": q_lp,
            "target_minus_q_sampled": target_lp - q_lp,
            "grad_norm": float(gn.item()) if isinstance(gn, torch.Tensor) else float(gn),
        }
        log_fh.write(json.dumps(rec) + "\n")

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"[kl step {step}/{args.max_steps} K={K}] "
                f"lr={args.lr*lr_scale:.2e} loss={loss:.4f} "
                f"kl/pos={kl_mean:.4f} t_lp={target_lp:.3f} q_lp={q_lp:.3f} "
                f"gap={target_lp-q_lp:+.3f} gn={rec['grad_norm']:.3f} "
                f"({elapsed/step:.2f}s/step)",
                flush=True,
            )

        if step % args.save_interval == 0:
            ckpt_dir = out_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            draft.save_pretrained(ckpt_dir)
            print(f"[kl] saved checkpoint to {ckpt_dir}", flush=True)

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(final)
    log_fh.close()
    print(f"[kl] done. final checkpoint -> {final}", flush=True)


if __name__ == "__main__":
    main()

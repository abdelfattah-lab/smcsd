"""Finetune the SMC proposal (draft) model on collected rollouts.

Consumes the JSONL dumps produced by ``scripts/collect_proposal_data.py``
and trains the draft model so its proposal distribution better matches the
tempered target distribution the SMC importance weights are computed
against (``worker.py``: ``log w += alpha * log p_T(x) - log q_Td(x)``).
Lower KL(p || q) means lower weight variance, higher ESS, fewer resamples —
quality at lower N / higher gamma.

Two losses:

* ``--loss kl`` (default, recommended) — token-level distillation on the
  particle trajectories.  Teacher distribution is EXACTLY the engine's
  per-token target, ``softmax(alpha * z_target / target_temperature)``;
  student distribution is the engine's per-token proposal,
  ``softmax(z_draft / draft_temperature)``.  Trajectories were sampled by
  the proposal (plus resampling toward the target), so this is on-policy
  distillation: it shrinks the KL where the proposal actually goes.
  Temperatures and alpha default to the values recorded in the dump's meta
  line.

* ``--loss wsft`` — posterior-weighted SFT baseline: plain cross-entropy on
  each particle's trajectory, weighted by the particle's normalized final
  weight ``softmax(log_w_tilde)``.  No teacher forward needed (much
  cheaper), but a strictly weaker, sequence-level signal.

The finetuned checkpoint is saved in merged HF format (bf16 safetensors) and
drops straight into the engine via ``--draft-model <output_dir>/final``.

Usage:
  python scripts/train_proposal.py \
      --data /data/proposal_data/gsm8k_train_N8g8.jsonl \
      --output-dir /data/proposal_ckpts/llama1b-kl \
      --loss kl --epochs 1 --batch-size 4 --grad-accum 8 --lr 1e-5
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dumps(paths):
    """Read one or more collection dumps.  Returns (meta, records); meta is
    taken from the first file and checked for consistency on the rest."""
    meta = None
    records = []
    for path in paths:
        with open(path) as f:
            first = json.loads(f.readline())
            file_meta = first["meta"]
            if meta is None:
                meta = file_meta
            else:
                for key in (
                    "model", "draft_model", "draft_temperature",
                    "target_temperature", "power_alpha",
                ):
                    assert file_meta[key] == meta[key], (
                        f"{path}: meta[{key}]={file_meta[key]!r} conflicts "
                        f"with {meta[key]!r} from the first dump"
                    )
            for line in f:
                records.append(json.loads(line))
    return meta, records


def build_examples(records, *, max_seq_len, dedup, weighting):
    """Flatten records into per-particle training examples.

    Each example: (input_ids, prompt_len, weight).  ``weight`` is the
    particle's posterior weight normalized to mean 1 within its group
    (``N * softmax(log_w_tilde)``) for ``weighting='posterior'``, or 1 for
    ``'uniform'``.  Resampling produces duplicate particles; with
    ``dedup``, identical (prompt, output) pairs are merged by summing
    weights — without it duplicates simply appear multiple times, which
    represents the same empirical distribution.
    """
    examples = []
    n_skipped = 0
    for rec in records:
        prompt_ids = rec["prompt_ids"]
        if len(prompt_ids) >= max_seq_len - 1:
            n_skipped += len(rec["particle_output_ids"])
            continue
        log_w = torch.tensor(rec["log_w_tilde"], dtype=torch.float64)
        n = len(rec["particle_output_ids"])
        if weighting == "posterior":
            w = (torch.softmax(log_w, dim=0) * n).tolist()
        else:
            w = [1.0] * n
        seen = {}
        for out_ids, wi in zip(rec["particle_output_ids"], w):
            ids = (prompt_ids + out_ids)[:max_seq_len]
            if dedup:
                key = (id(rec), tuple(ids))
                if key in seen:
                    examples[seen[key]][2] += wi
                    continue
                seen[key] = len(examples)
            examples.append([ids, len(prompt_ids), wi])
    if n_skipped:
        print(f"Skipped {n_skipped} particles (prompt >= max_seq_len)")
    return examples


def make_batches(examples, batch_size, *, rng):
    """Length-bucketed batches: shuffle, sort by length, slice, shuffle
    batch order.  Keeps padding waste low without fixed buckets."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(examples[i][0]))
    batches = [
        order[i : i + batch_size] for i in range(0, len(order), batch_size)
    ]
    rng.shuffle(batches)
    return batches


def collate(examples, idxs, pad_id, device):
    """Pad a batch and build the shifted completion mask.

    Position ``t`` of the (length L-1) logits predicts token ``t+1``;
    completion positions are those with ``t+1 >= prompt_len`` and
    ``t+1 < seq_len``.
    """
    batch = [examples[i] for i in idxs]
    max_len = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    comp_mask = torch.zeros((len(batch), max_len - 1), dtype=torch.float32)
    weights = torch.tensor([w for _, _, w in batch], dtype=torch.float32)
    for r, (ids, prompt_len, _) in enumerate(batch):
        input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[r, : len(ids)] = 1
        comp_mask[r, prompt_len - 1 : len(ids) - 1] = 1.0
    return (
        input_ids.to(device),
        attn.to(device),
        comp_mask.to(device),
        weights.to(device),
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def kl_loss(student_logits, teacher_logits, comp_mask, weights, cfg):
    """Per-token KL between p_T^alpha and q_Td over completion positions.

    Matches the engine's weight equation exactly: teacher logprobs are
    ``log_softmax(alpha * z_t / T_target)`` (the normalized tempered-power
    target the bonus token is sampled from), student logprobs are
    ``log_softmax(z_s / T_draft)`` (what the draft AR loop samples from).

    ``cfg.kl_direction`` picks the divergence:

    * ``forward``  — KL(p || q): coverage.  Penalizes q underweighting
      tokens p likes (the "sibling found a token I missed" side of SMC
      weight variance).
    * ``reverse``  — KL(q || p): precision.  Penalizes q proposing tokens
      p rejects — the per-token weight increment IS log p - log q at
      q's own samples, so this directly targets particle death.
    * ``both``     — mean of the two; both tails drive ESS.

    Returns (loss, mean_token_divergence).
    """
    p_log = F.log_softmax(
        cfg.power_alpha * teacher_logits.float() / cfg.target_temperature,
        dim=-1,
    )
    q_log = F.log_softmax(student_logits.float() / cfg.draft_temperature, dim=-1)
    if cfg.kl_direction == "forward":
        kl = (p_log.exp() * (p_log - q_log)).sum(-1)  # (B, L-1)
    elif cfg.kl_direction == "reverse":
        kl = (q_log.exp() * (q_log - p_log)).sum(-1)
    else:  # both
        kl = 0.5 * (
            (p_log.exp() * (p_log - q_log)).sum(-1)
            + (q_log.exp() * (q_log - p_log)).sum(-1)
        )
    w = comp_mask * weights.unsqueeze(1)
    denom = w.sum().clamp_min(1.0)
    loss = (kl * w).sum() / denom
    return loss, loss.detach()


def wsft_loss(student_logits, input_ids, comp_mask, weights):
    """Posterior-weighted cross-entropy on completion tokens."""
    targets = input_ids[:, 1:]
    ce = F.cross_entropy(
        student_logits.float().transpose(1, 2), targets, reduction="none"
    )  # (B, L-1)
    w = comp_mask * weights.unsqueeze(1)
    denom = w.sum().clamp_min(1.0)
    loss = (ce * w).sum() / denom
    return loss, loss.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_eval(model, teacher, examples, batches, pad_id, device, args, cfg):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for idxs in batches:
            input_ids, attn, comp_mask, weights = collate(
                examples, idxs, pad_id, device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = model(input_ids, attention_mask=attn).logits[:, :-1]
                if args.loss == "kl":
                    t_logits = teacher(input_ids, attention_mask=attn).logits[
                        :, :-1
                    ]
                    loss, _ = kl_loss(s_logits, t_logits, comp_mask, weights, cfg)
                else:
                    loss, _ = wsft_loss(s_logits, input_ids, comp_mask, weights)
            ntok = comp_mask.sum().item()
            total += loss.item() * ntok
            count += ntok
    model.train()
    return total / max(count, 1)


def main(args):
    meta, records = load_dumps(args.data)
    cfg = argparse.Namespace(
        draft_temperature=args.draft_temperature
        if args.draft_temperature is not None
        else meta["draft_temperature"],
        target_temperature=args.target_temperature
        if args.target_temperature is not None
        else meta["target_temperature"],
        power_alpha=args.power_alpha
        if args.power_alpha is not None
        else meta["power_alpha"],
        kl_direction=args.kl_direction,
    )
    init_from = args.init_from or meta["draft_model"]
    teacher_path = args.teacher or meta["model"]
    print(f"{len(records)} records | loss={args.loss} | init={init_from}")
    print(f"  T_draft={cfg.draft_temperature} T_target={cfg.target_temperature} "
          f"alpha={cfg.power_alpha}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac)) if args.val_frac > 0 else 0
    val_records, train_records = records[:n_val], records[n_val:]

    train_examples = build_examples(
        train_records, max_seq_len=args.max_seq_len, dedup=args.dedup,
        weighting=args.weighting,
    )
    val_examples = build_examples(
        val_records, max_seq_len=args.max_seq_len, dedup=args.dedup,
        weighting=args.weighting,
    )
    print(f"  {len(train_examples)} train / {len(val_examples)} val examples")

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(init_from)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Student in fp32 master weights, bf16 autocast forward; teacher bf16.
    model = AutoModelForCausalLM.from_pretrained(
        init_from, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    teacher = None
    if args.loss == "kl":
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        teacher.eval()
        teacher.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(
        len(train_examples) / (args.batch_size * args.grad_accum)
    )
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "meta": meta, **vars(cfg)}, f, indent=2, default=str)

    val_batches = (
        make_batches(val_examples, args.batch_size, rng=random.Random(0))
        if val_examples
        else []
    )

    def save(tag):
        out = os.path.join(args.output_dir, tag)
        # bf16 state dict — half the disk, and the dtype the engine runs.
        sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        model.save_pretrained(out, state_dict=sd, safe_serialization=True)
        # The config dtype must say bfloat16 too: it's what sglang loads the
        # draft as, and a float32 draft crashes CUDA-graph buffer sharing
        # against the bf16 target ("input_embeds has different dtype").
        # save_pretrained re-stamps the config dtype from the live (fp32)
        # master weights, so rewrite the file rather than the config object.
        cfg_path = os.path.join(out, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        for key in ("dtype", "torch_dtype"):
            if key in cfg:
                cfg[key] = "bfloat16"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        tokenizer.save_pretrained(out)
        print(f"  saved {out}")

    step = 0
    tokens_seen = 0
    tic = time.perf_counter()
    for epoch in range(args.epochs):
        batches = make_batches(train_examples, args.batch_size, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        running = []
        for bi, idxs in enumerate(batches):
            input_ids, attn, comp_mask, weights = collate(
                train_examples, idxs, pad_id, device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = model(input_ids, attention_mask=attn).logits[:, :-1]
                if args.loss == "kl":
                    with torch.no_grad():
                        t_logits = teacher(
                            input_ids, attention_mask=attn
                        ).logits[:, :-1]
                    loss, metric = kl_loss(
                        s_logits, t_logits, comp_mask, weights, cfg
                    )
                else:
                    loss, metric = wsft_loss(
                        s_logits, input_ids, comp_mask, weights
                    )
            (loss / args.grad_accum).backward()
            running.append(metric.item())
            tokens_seen += int(comp_mask.sum().item())

            if (bi + 1) % args.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    elapsed = time.perf_counter() - tic
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"{args.loss}={sum(running) / len(running):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"tok/s={tokens_seen / elapsed:.0f}",
                        flush=True,
                    )
                    running = []

        if val_batches:
            val = run_eval(
                model, teacher, val_examples, val_batches, pad_id, device,
                args, cfg,
            )
            print(f"epoch {epoch} done: val_{args.loss}={val:.4f}")
        save(f"epoch_{epoch}")

    save("final")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", nargs="+", required=True,
                        help="collection dump(s) from collect_proposal_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loss", choices=["kl", "wsft"], default="kl")
    parser.add_argument("--kl-direction", choices=["forward", "reverse", "both"],
                        default="forward",
                        help="divergence for --loss kl: forward=KL(p||q) "
                             "(coverage), reverse=KL(q||p) (precision, targets "
                             "particle death), both=mean of the two")
    parser.add_argument("--init-from", type=str, default=None,
                        help="student init (default: dump meta draft_model)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="teacher for kl loss (default: dump meta model)")
    parser.add_argument("--weighting", choices=["uniform", "posterior"],
                        default=None,
                        help="particle weighting (default: uniform for kl, "
                             "posterior for wsft)")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="merge duplicate post-resample particles")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")

    # Loss config — default to the dump's engine config.
    parser.add_argument("--draft-temperature", type=float, default=None)
    parser.add_argument("--target-temperature", type=float, default=None)
    parser.add_argument("--power-alpha", type=float, default=None)

    # Optimization
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--val-frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    if args.weighting is None:
        args.weighting = "uniform" if args.loss == "kl" else "posterior"
    main(args)

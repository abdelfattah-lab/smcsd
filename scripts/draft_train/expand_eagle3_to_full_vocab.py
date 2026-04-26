#!/usr/bin/env python3
"""Expand a hot-vocab EAGLE3 checkpoint to full target vocabulary.

This is an Option-A utility for SMC-SD experiments. Standard EAGLE3 checkpoints
are often trained with a reduced hot vocabulary, e.g. ``draft_vocab_size=32000``
plus ``t2d``/``d2t`` mapping buffers. For SMC importance weights it is cleaner
to use a full-vocabulary proposal, so this script creates a full-vocab EAGLE3
checkpoint by:

1. Loading the hot-vocab EAGLE3 checkpoint.
2. Creating a new lm_head of shape [target_vocab_size, hidden_size].
3. Initializing the full head from the target model lm_head when possible.
4. Overwriting rows for hot tokens with the trained hot-vocab EAGLE rows.
5. Writing config.json with ``architectures=["LlamaForCausalLMEagle3"]`` and
   ``draft_vocab_size == vocab_size``.

The resulting checkpoint is not expected to be a good proposal by itself; it is
a support-correct warm start for SMC-native EAGLE finetuning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def _repo_or_path(path_or_repo: str) -> str:
    if os.path.exists(path_or_repo):
        return path_or_repo
    return snapshot_download(path_or_repo)


def _load_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=False)
    # Minimal safetensors support if future checkpoints switch format.
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        return load_file(st_path, device="cpu")
    raise FileNotFoundError(f"No pytorch_model.bin or model.safetensors in {model_dir}")


def _load_target_lm_head(target_dir: str, key: str) -> torch.Tensor:
    # Common unsharded path.
    st_path = os.path.join(target_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors import safe_open

        with safe_open(st_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    bin_path = os.path.join(target_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        return sd[key]

    # Sharded safetensors/bin index path.
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(target_dir, index_name)
        if not os.path.exists(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        shard = index.get("weight_map", {}).get(key)
        if shard is None:
            continue
        shard_path = os.path.join(target_dir, shard)
        if shard_path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                return f.get_tensor(key)
        sd = torch.load(shard_path, map_location="cpu", weights_only=False)
        return sd[key]

    raise FileNotFoundError(f"Could not find {key!r} in target checkpoint {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eagle", required=True, help="Hot-vocab EAGLE3 checkpoint path or HF repo id")
    parser.add_argument("--target", required=True, help="Target model path or HF repo id for full lm_head init")
    parser.add_argument("--output", required=True, help="Output full-vocab EAGLE3 checkpoint directory")
    parser.add_argument("--lm-head-key", default="lm_head.weight")
    parser.add_argument(
        "--no-target-init",
        action="store_true",
        help="Initialize non-hot rows randomly instead of from target lm_head.",
    )
    args = parser.parse_args()

    eagle_dir = _repo_or_path(args.eagle)
    target_dir = _repo_or_path(args.target)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(eagle_dir, "config.json")) as f:
        cfg = json.load(f)
    state = _load_state_dict(eagle_dir)

    old_head = state.get("lm_head.weight")
    if old_head is None:
        raise KeyError("EAGLE checkpoint has no lm_head.weight")
    old_vocab, hidden = old_head.shape
    target_vocab = int(cfg.get("vocab_size"))
    old_draft_vocab = int(cfg.get("draft_vocab_size", old_vocab))
    if old_vocab != old_draft_vocab:
        raise ValueError(f"lm_head rows ({old_vocab}) != draft_vocab_size ({old_draft_vocab})")

    d2t = state.get("d2t")
    if d2t is not None:
        d2t = d2t.to(torch.long) + torch.arange(d2t.numel(), dtype=torch.long)
    else:
        d2t = torch.arange(old_vocab, dtype=torch.long)

    if d2t.numel() != old_vocab:
        raise ValueError(f"d2t length {d2t.numel()} != old vocab {old_vocab}")
    if int(d2t.max()) >= target_vocab or int(d2t.min()) < 0:
        raise ValueError("d2t maps outside target vocabulary")

    if args.no_target_init:
        std = float(cfg.get("initializer_range", 0.02))
        full_head = torch.empty((target_vocab, hidden), dtype=old_head.dtype)
        torch.nn.init.normal_(full_head, mean=0.0, std=std)
    else:
        full_head = _load_target_lm_head(target_dir, args.lm_head_key).to(dtype=old_head.dtype)
        if full_head.shape != (target_vocab, hidden):
            raise ValueError(
                f"target lm_head shape {tuple(full_head.shape)} != {(target_vocab, hidden)}"
            )
        full_head = full_head.clone()

    full_head[d2t] = old_head
    state["lm_head.weight"] = full_head
    state.pop("d2t", None)
    state.pop("t2d", None)

    cfg["architectures"] = ["LlamaForCausalLMEagle3"]
    cfg["draft_vocab_size"] = target_vocab
    cfg["vocab_size"] = target_vocab

    torch.save(state, out_dir / "pytorch_model.bin")
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")

    # Copy tokenizer/metadata side files if present. Missing files are fine
    # because SMC uses the target tokenizer, but this makes the directory more
    # HF-compatible.
    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "README.md",
        "LICENSE",
    ]:
        src = os.path.join(eagle_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, out_dir / name)

    print(f"Wrote full-vocab EAGLE3 checkpoint to {out_dir}")
    print(f"old draft vocab: {old_vocab}; target vocab: {target_vocab}; hidden: {hidden}")
    print(f"hot rows copied: {d2t.numel()}; non-hot init: {'random' if args.no_target_init else 'target lm_head'}")


if __name__ == "__main__":
    main()

"""Torch Dataset over target-trace shards.

Each example yields the unpadded prompt, the target-sampled continuation, and
the top-K logits/indices of the target at each continuation position. The
collator left-pads prompts within a batch so all gen tokens occupy the same
absolute positions; training code can then slice logits at one fixed range.
"""
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class TargetTraceDataset(Dataset):
    def __init__(self, shard_dir: str):
        shards = sorted(Path(shard_dir).glob("shard_*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no shards found under {shard_dir}")

        self.prompts: list[torch.Tensor] = []
        gen_list, ki_list, kv_list = [], [], []
        for s in shards:
            d = load_file(str(s))
            pt = d["prompt_tokens"]
            pl = d["prompt_lens"]
            for i in range(pt.shape[0]):
                plen = int(pl[i])
                self.prompts.append(pt[i, -plen:].clone().contiguous())
            gen_list.append(d["gen_tokens"])
            ki_list.append(d["topk_indices"])
            kv_list.append(d["topk_logits"])
        self.gen_tokens = torch.cat(gen_list, dim=0)       # [N, gen_len] int32
        self.topk_indices = torch.cat(ki_list, dim=0)      # [N, gen_len, K] int32
        self.topk_logits = torch.cat(kv_list, dim=0)       # [N, gen_len, K] bf16
        assert len(self.prompts) == self.gen_tokens.shape[0]

    def __len__(self) -> int:
        return self.gen_tokens.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "prompt": self.prompts[i],
            "gen": self.gen_tokens[i],
            "topk_idx": self.topk_indices[i],
            "topk_lg": self.topk_logits[i],
        }


def make_collate(pad_token_id: int):
    """Left-pad prompts to batch max so gen positions are column-aligned."""
    def collate(batch: list[dict]) -> dict:
        B = len(batch)
        plens = torch.tensor([b["prompt"].numel() for b in batch], dtype=torch.long)
        pmax = int(plens.max())
        gen_len = batch[0]["gen"].numel()
        L = pmax + gen_len

        input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((B, L), dtype=torch.long)
        for i, b in enumerate(batch):
            plen = int(plens[i])
            start = pmax - plen
            input_ids[i, start:pmax] = b["prompt"].long()
            input_ids[i, pmax:pmax + gen_len] = b["gen"].long()
            attn_mask[i, start:] = 1

        topk_idx = torch.stack([b["topk_idx"] for b in batch]).long()   # [B, gen_len, K]
        topk_lg = torch.stack([b["topk_lg"] for b in batch]).float()    # [B, gen_len, K]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "pmax": pmax,
            "gen_len": gen_len,
            "topk_idx": topk_idx,
            "topk_lg": topk_lg,
        }
    return collate

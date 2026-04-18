"""Small Llama-arch proposal with warm-started, frozen token embeddings."""
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


@dataclass
class ProposalConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 4
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 4096
    vocab_size: int = 128256
    max_position_embeddings: int = 2048
    tie_word_embeddings: bool = True
    rope_theta: float = 500000.0


def build_proposal(cfg: ProposalConfig, dtype: torch.dtype = torch.bfloat16) -> LlamaForCausalLM:
    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_word_embeddings,
        torch_dtype=dtype,
    )
    model = LlamaForCausalLM(llama_cfg)
    return model.to(dtype)


def warm_start_and_freeze_embeddings(
    proposal: LlamaForCausalLM,
    target_model_path: str,
    freeze: bool = True,
) -> None:
    """Copy target's embedding table into proposal. With tied weights this also pins the LM head."""
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.bfloat16)
    tgt = target.get_input_embeddings().weight.data
    prop = proposal.get_input_embeddings().weight
    if tgt.shape != prop.shape:
        raise ValueError(
            f"embedding shape mismatch: target {tuple(tgt.shape)} vs "
            f"proposal {tuple(prop.shape)} — set proposal.hidden_size = target.hidden_size"
        )
    with torch.no_grad():
        prop.data.copy_(tgt.to(prop.dtype))
    if freeze:
        prop.requires_grad = False
    del target


def param_counts(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}

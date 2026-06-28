import os as _os
import torch
from transformers import AutoModelForCausalLM
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.bfloat16).cuda().eval()
m = model.model
layer = m.layers[0]
print("layer forward params:", list(layer.forward.__code__.co_varnames[:layer.forward.__code__.co_argcount]))
S = 24; hidden = model.config.hidden_size
h = (torch.randn(1, S, hidden, device="cuda", dtype=torch.bfloat16) * 0.1)
pos_ids = torch.arange(S, device="cuda").unsqueeze(0)
cos, sin = m.rotary_emb(h, pos_ids)                       # HF exact RoPE (theta+llama3 scaling)
# causal mask additive [1,1,S,S]
mask = torch.full((S,S), float("-inf"), device="cuda", dtype=torch.bfloat16).triu(1)[None,None]
with torch.no_grad():
    out = layer(h, attention_mask=mask, position_ids=pos_ids, position_embeddings=(cos,sin))
ref = out[0] if isinstance(out, tuple) else out
L0 = layer
torch.save({
  "h": h.cpu(), "cos": cos.cpu(), "sin": sin.cpu(), "ref": ref.detach().cpu(),
  "Wq": L0.self_attn.q_proj.weight.detach().cpu(), "Wk": L0.self_attn.k_proj.weight.detach().cpu(),
  "Wv": L0.self_attn.v_proj.weight.detach().cpu(), "Wo": L0.self_attn.o_proj.weight.detach().cpu(),
  "Wg": L0.mlp.gate_proj.weight.detach().cpu(), "Wu": L0.mlp.up_proj.weight.detach().cpu(),
  "Wd": L0.mlp.down_proj.weight.detach().cpu(),
  "g1": L0.input_layernorm.weight.detach().cpu(), "g2": L0.post_attention_layernorm.weight.detach().cpu(),
  "eps": model.config.rms_norm_eps,
  "cfg": dict(H=model.config.num_attention_heads, Hkv=model.config.num_key_value_heads,
              D=getattr(model.config,'head_dim',hidden//model.config.num_attention_heads), hidden=hidden, S=S),
}, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"ref_block.pt"))
print("ref captured, S=%d hidden=%d, ref norm=%.4f" % (S, hidden, ref.float().norm().item()))

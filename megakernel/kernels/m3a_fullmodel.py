import sys, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import m_kernels as mk
from transformers import AutoModelForCausalLM, AutoTokenizer

name="meta-llama/Llama-3.2-1B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
model.requires_grad_(False)
m=model.model
cfg=dict(H=model.config.num_attention_heads, Hkv=model.config.num_key_value_heads,
         D=getattr(model.config,'head_dim',model.config.hidden_size//model.config.num_attention_heads),
         hidden=model.config.hidden_size)
eps=model.config.rms_norm_eps

ids=tok("The capital of France is", return_tensors="pt").input_ids.cuda()
S=ids.shape[1]
with torch.no_grad():
    ref_logits=model(ids).logits[0].float()              # [S, vocab]

# my full-model forward through CuTe kernels
with torch.no_grad():
    h=m.embed_tokens(ids)[0].contiguous()                # [S,hidden] bf16
    pos=torch.arange(S,device="cuda").unsqueeze(0)
    cos,sin=m.rotary_emb(h.unsqueeze(0),pos)
    cos=cos[0].contiguous(); sin=sin[0].contiguous()
    t0=time.time()
    for L in range(model.config.num_hidden_layers):
        ly=m.layers[L]
        w=dict(g1=ly.input_layernorm.weight, g2=ly.post_attention_layernorm.weight,
               Wq=ly.self_attn.q_proj.weight, Wk=ly.self_attn.k_proj.weight,
               Wv=ly.self_attn.v_proj.weight, Wo=ly.self_attn.o_proj.weight,
               Wg=ly.mlp.gate_proj.weight, Wu=ly.mlp.up_proj.weight, Wd=ly.mlp.down_proj.weight)
        h=mk.block(h, w, cos, sin, cfg, eps)
    # final norm + lm_head
    g=m.norm.weight
    hf=h.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+eps)
    hn=(hf*inv*g.float()).to(torch.bfloat16)
    my_logits=mk.f32(S, model.config.vocab_size)
    mk.gemv(mk.from_dlpack(hn), mk.from_dlpack(model.lm_head.weight), mk.from_dlpack(my_logits), S, cfg["hidden"])
    torch.cuda.synchronize(); dt=time.time()-t0

rel=(my_logits-ref_logits).abs().max().item()/ref_logits.abs().max().item()
# token-level agreement (what actually matters for decode)
my_top1=my_logits.argmax(-1); ref_top1=ref_logits.argmax(-1)
agree=(my_top1==ref_top1).float().mean().item()
# last-position next-token (the decode token)
print(f"full 16-layer model through CuTe kernels: {dt*1000:.0f} ms")
print(f"logits rel err {rel:.3e}   top-1 token agreement {agree*100:.1f}%")
print(f"next-token mine={tok.decode(my_top1[-1])!r}  ref={tok.decode(ref_top1[-1])!r}")
print("RESULT:", "PASS" if agree>0.99 else "FAIL")

"""M6 step 1: validate the TARGET (8B) VERIFY FORWARD standalone vs HF, using the validated per-op
CuTe kernels (m_kernels.block + gemv) — the 8B analogue of M3a. The verify phase runs one causal
forward over the N x (gamma+1) drafted block and extracts the TEMPERED TARGET LOGPROB per drafted
position (no sampling): logp[s] = (logits[s]/T)[tok_{s+1}] - logsumexp(logits[s]/T).
This proves the 8B forward + logprob extraction is correct before fusing it into the persistent kernel.
"""
import sys, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import m_kernels as mk
from transformers import AutoModelForCausalLM, AutoTokenizer

name="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
model.requires_grad_(False)
m=model.model; cf=model.config
cfg=dict(H=cf.num_attention_heads, Hkv=cf.num_key_value_heads,
         D=getattr(cf,'head_dim',cf.hidden_size//cf.num_attention_heads), hidden=cf.hidden_size)
eps=cf.rms_norm_eps; V=cf.vocab_size; NL=cf.num_hidden_layers
print(f"[8B cfg] hidden={cfg['hidden']} NL={NL} H={cfg['H']} Hkv={cfg['Hkv']} D={cfg['D']} I={cf.intermediate_size} V={V}")

# A representative drafted block: a prompt followed by greedily-drafted continuation tokens.
# (Provenance is irrelevant for forward-correctness; what matters is a causal forward over S tokens.)
ids=tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    gen=model.generate(ids, max_new_tokens=8, do_sample=False)
ids=gen  # [1, S]  (prompt + 8 drafted tokens)
S=ids.shape[1]
print(f"[block] S={S} tokens: {tok.decode(ids[0])!r}")

with torch.no_grad():
    ref_logits=model(ids).logits[0].float()              # [S, V]

# ---- my full 32-layer 8B forward through the CuTe kernels ----
with torch.no_grad():
    h=m.embed_tokens(ids)[0].contiguous()                # [S,hidden] bf16
    pos=torch.arange(S,device="cuda").unsqueeze(0)
    cos,sin=m.rotary_emb(h.unsqueeze(0),pos)             # llama3 RoPE scaling handled internally
    cos=cos[0].contiguous(); sin=sin[0].contiguous()
    t0=time.time()
    for L in range(NL):
        ly=m.layers[L]
        w=dict(g1=ly.input_layernorm.weight, g2=ly.post_attention_layernorm.weight,
               Wq=ly.self_attn.q_proj.weight, Wk=ly.self_attn.k_proj.weight,
               Wv=ly.self_attn.v_proj.weight, Wo=ly.self_attn.o_proj.weight,
               Wg=ly.mlp.gate_proj.weight, Wu=ly.mlp.up_proj.weight, Wd=ly.mlp.down_proj.weight)
        h=mk.block(h, w, cos, sin, cfg, eps)
    g=m.norm.weight
    hf=h.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+eps)
    hn=(hf*inv*g.float()).to(torch.bfloat16)
    my_logits=mk.f32(S, V)
    mk.gemv(mk.from_dlpack(hn), mk.from_dlpack(model.lm_head.weight), mk.from_dlpack(my_logits), S, cfg["hidden"])
    torch.cuda.synchronize(); dt=time.time()-t0

# ---- raw-logits agreement (like M3a) ----
rel=(my_logits-ref_logits).abs().max().item()/ref_logits.abs().max().item()
my_top1=my_logits.argmax(-1); ref_top1=ref_logits.argmax(-1)
agree=(my_top1==ref_top1).float().mean().item()
print(f"\nfull {NL}-layer 8B forward through CuTe kernels: {dt*1000:.0f} ms")
print(f"logits rel err {rel:.3e}   top-1 token agreement {agree*100:.1f}%")

# ---- the actual verify output: TEMPERED TARGET LOGPROB per drafted position ----
def tempered_logp(logits, toks_next, T):
    # logits: [S,V]; score token toks_next[s] at position s (predicting s+1). Last position has no target.
    lg=logits[:-1]/T                                     # [S-1, V]
    lse=torch.logsumexp(lg, dim=-1)                      # [S-1]
    idx=toks_next.view(-1,1)                             # [S-1,1]
    chosen=lg.gather(-1, idx).squeeze(-1)                # [S-1]
    return chosen-lse                                    # [S-1]

toks_next=ids[0,1:].long()                               # teacher-forced next tokens
for T in (1.0, 0.7):
    my_lp =tempered_logp(my_logits,  toks_next, T)
    ref_lp=tempered_logp(ref_logits, toks_next, T)
    max_abs=(my_lp-ref_lp).abs().max().item()
    mean_abs=(my_lp-ref_lp).abs().mean().item()
    print(f"[tempered logp T={T}] max|Δ|={max_abs:.4e}  mean|Δ|={mean_abs:.4e}   (ref logp range [{ref_lp.min():.2f},{ref_lp.max():.2f}])")

# PASS if top-1 agreement is high AND the verify logprobs match HF tightly (bf16-accum level).
my_lp1=tempered_logp(my_logits,toks_next,1.0); ref_lp1=tempered_logp(ref_logits,toks_next,1.0)
logp_ok=(my_lp1-ref_lp1).abs().max().item()<5e-2
print("RESULT:", "PASS" if (agree>0.95 and logp_ok) else "FAIL",
      f"(top1 {agree*100:.0f}%, logp max|Δ| {(my_lp1-ref_lp1).abs().max().item():.3e})")

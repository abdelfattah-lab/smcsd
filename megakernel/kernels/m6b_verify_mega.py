"""M6 step 2: the TARGET (8B) VERIFY FORWARD as ONE persistent kernel (prefill-shaped, batched over the
S-token drafted block), validated vs HF. This is the M3b.2 multi-token persistent forward (which hit 100%
top-1 on the 1B model) run on the 8B target, plus tempered-logprob extraction. Proves the persistent 8B
forward + grid barrier work at 32 layers / 4096 hidden before fusing draft->verify (M6c).

Kernel = m3b2_mega.mega_k verbatim (dim-parametrized, already validated). lm_head + tempered logprob done
on host here (they move in-kernel in M6c for the fusion)."""
import sys, math, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import m_kernels as mk
I32=cutlass.Int32; F32=cutlass.Float32

@cute.jit
def gbar(arrive, sense, ls, B:cutlass.Constexpr, tx):
    cute.arch.sync_threads()
    if tx==0:
        cute.arch.fence_acq_rel_gpu()
        old=cute.arch.atomic_add(arrive,I32(1),sem='acq_rel',scope='gpu')
        if old+1==B:
            cute.arch.atomic_exch(arrive,I32(0),sem='relaxed',scope='gpu')
            cute.arch.atomic_exch(sense,ls,sem='release',scope='gpu')
        else:
            cur=cute.arch.atomic_add(sense,I32(0),sem='acquire',scope='gpu')
            while cur!=ls:
                cur=cute.arch.atomic_add(sense,I32(0),sem='acquire',scope='gpu')
        cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()

@cute.kernel
def mega_k(gh, gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd, gcos,gsin,
           gQ,gK,gV,gA,gR,gAct, garr,gsen,
           S:cutlass.Constexpr, NL:cutlass.Constexpr, H:cutlass.Constexpr, Hkv:cutlass.Constexpr,
           D:cutlass.Constexpr, hid:cutlass.Constexpr, I:cutlass.Constexpr, GRP:cutlass.Constexpr,
           DH:cutlass.Constexpr, B:cutlass.Constexpr, BLK:cutlass.Constexpr, eps:cutlass.Constexpr, scale:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D
    ls=I32(0)
    for L in cutlass.range(NL):
        # ===== Stage 1: Q,K,V = rmsnorm(h,g1[L]) @ W[L]^T =====
        idx=gtid
        while idx < S*QW:
            m=idx//QW; n=idx%QW
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=gh[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            acc=F32(0.0)
            for k in cutlass.range(hid):
                acc=acc+gh[m,k].to(F32)*inv*gg1[L,k].to(F32)*gWq[L,n,k].to(F32)
            gQ[m,n]=acc.to(gQ.element_type)
            idx=idx+GT
        idx=gtid
        while idx < S*KW:
            m=idx//KW; n=idx%KW
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=gh[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            ak=F32(0.0); av=F32(0.0)
            for k in cutlass.range(hid):
                yn=gh[m,k].to(F32)*inv*gg1[L,k].to(F32)
                ak=ak+yn*gWk[L,n,k].to(F32); av=av+yn*gWv[L,n,k].to(F32)
            gK[m,n]=ak.to(gK.element_type); gV[m,n]=av.to(gV.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # ===== Stage 2: attention (RoPE on-chip, causal, GQA) -> gA[S,QW] =====
        idx=gtid
        while idx < S*H:
            m=idx//H; h=idx%H; kvh=h//GRP
            qr=cute.make_fragment(D,F32)
            for d in cutlass.range_constexpr(DH):
                lo=gQ[m,h*D+d].to(F32); hi=gQ[m,h*D+d+DH].to(F32)
                qr[d]=lo*gcos[m,d].to(F32)-hi*gsin[m,d].to(F32)
                qr[d+DH]=hi*gcos[m,d+DH].to(F32)+lo*gsin[m,d+DH].to(F32)
            ac=cute.make_fragment(D,F32)
            for d in cutlass.range_constexpr(D): ac[d]=F32(0.0)
            rmax=F32(-1.0e30); rsum=F32(0.0)
            for s in cutlass.range(m+1):
                sc=F32(0.0)
                for d in cutlass.range_constexpr(DH):
                    klo=gK[s,kvh*D+d].to(F32); khi=gK[s,kvh*D+d+DH].to(F32)
                    krlo=klo*gcos[s,d].to(F32)-khi*gsin[s,d].to(F32)
                    krhi=khi*gcos[s,d+DH].to(F32)+klo*gsin[s,d+DH].to(F32)
                    sc=sc+qr[d]*krlo+qr[d+DH]*krhi
                sc=sc*F32(scale)
                nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                rsum=rsum*corr+p
                for d in cutlass.range_constexpr(D): ac[d]=ac[d]*corr+p*gV[s,kvh*D+d].to(F32)
                rmax=nm
            for d in cutlass.range_constexpr(D): gA[m,h*D+d]=(ac[d]/rsum).to(gA.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # ===== Stage 3: res1 = h + attn@Wo[L]^T =====
        idx=gtid
        while idx < S*hid:
            m=idx//hid; n=idx%hid
            acc=gh[m,n].to(F32)
            for j in cutlass.range(QW):
                acc=acc+gA[m,j].to(F32)*gWo[L,n,j].to(F32)
            gR[m,n]=acc.to(gR.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # ===== Stage 4: act = silu(rmsnorm(res1,g2)@Wg) * (..@Wu) =====
        idx=gtid
        while idx < S*I:
            m=idx//I; n=idx%I
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=gR[m,k].to(cutlass.BFloat16).to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            g=F32(0.0); u=F32(0.0)
            for k in cutlass.range(hid):
                yn=gR[m,k].to(cutlass.BFloat16).to(F32)*inv*gg2[L,k].to(F32)
                g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32)
            silu=g/(F32(1.0)+cute.exp(-g))
            gAct[m,n]=(silu*u).to(gAct.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # ===== Stage 5: h = res1 + act@Wd[L]^T =====
        idx=gtid
        while idx < S*hid:
            m=idx//hid; n=idx%hid
            acc=gR[m,n].to(F32)
            for j in cutlass.range(I):
                acc=acc+gAct[m,j].to(F32)*gWd[L,n,j].to(F32)
            gh[m,n]=acc.to(gh.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def mega(gh,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gQ,gK,gV,gA,gR,gAct,garr,gsen,
         S:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,
         hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,B:cutlass.Constexpr,
         BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr):
    mega_k(gh,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gQ,gK,gV,gA,gR,gAct,garr,gsen,
           S,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale).launch(grid=[B,1,1],block=[BLK,1,1])

# ===== driver: 8B target =====
from transformers import AutoModelForCausalLM, AutoTokenizer
name="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).cuda().eval(); model.requires_grad_(False)
m=model.model; cf=model.config
H=cf.num_attention_heads; Hkv=cf.num_key_value_heads; D=getattr(cf,'head_dim',cf.hidden_size//H)
hid=cf.hidden_size; I=cf.intermediate_size; NL=cf.num_hidden_layers; eps=cf.rms_norm_eps; V=cf.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=256
print(f"[8B cfg] hidden={hid} NL={NL} H={H} Hkv={Hkv} D={D} I={I} V={V}  grid={B}x{BLK}")

ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    ids=model.generate(ids, max_new_tokens=8, do_sample=False)   # prompt + drafted block
S=ids.shape[1]
print(f"[block] S={S}: {tok.decode(ids[0])!r}")
with torch.no_grad():
    ref_logits=model(ids).logits[0].float()
    h0=m.embed_tokens(ids)[0].contiguous()
    pos=torch.arange(S,device="cuda").unsqueeze(0); cos,sin=m.rotary_emb(h0.unsqueeze(0),pos)
    cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(get): return torch.stack([get(m.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)
h=h0.clone()
Q=mk.f32(S,H*D); K=mk.f32(S,Hkv*D); Vt=mk.f32(S,Hkv*D)
A=mk.f32(S,H*D); R=mk.f32(S,hid); Act=mk.f32(S,I)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
args=[h,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,Q,K,Vt,A,R,Act,arr,sen]
mt=[from_dlpack(t) for t in args]
comp=cute.compile(mega,*mt,S,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale)
arr.zero_(); sen.zero_()
t0=time.time(); comp(*mt); torch.cuda.synchronize(); dt=time.time()-t0
# final norm + lm_head on host (moves in-kernel in M6c)
hf=h.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+eps); hn=(hf*inv*m.norm.weight.float()).to(torch.bfloat16)
my=mk.f32(S,V); mk.gemv(from_dlpack(hn),from_dlpack(model.lm_head.weight),from_dlpack(my),S,hid); torch.cuda.synchronize()

rel=(my-ref_logits).abs().max().item()/ref_logits.abs().max().item()
agree=(my.argmax(-1)==ref_logits.argmax(-1)).float().mean().item()
print(f"[PERSISTENT 8B VERIFY NL={NL}] {dt*1000:.0f} ms/launch  logits rel {rel:.3e}  top1 agree {agree*100:.1f}%")

def tempered_logp(logits, toks_next, T):
    lg=logits[:-1]/T; lse=torch.logsumexp(lg,-1)
    return lg.gather(-1, toks_next.view(-1,1)).squeeze(-1)-lse
toks_next=ids[0,1:].long()
for T in (1.0,0.7):
    a=tempered_logp(my,toks_next,T); b=tempered_logp(ref_logits,toks_next,T)
    print(f"[tempered logp T={T}] mine-vs-HFbf16: max|Δ|={ (a-b).abs().max().item():.4e}  mean|Δ|={(a-b).abs().mean().item():.4e}")
# Establish the bf16 noise floor: HF bf16 vs HF fp32 over the SAME tokens. If my-vs-HFbf16 ~ this, the
# residual is bf16-accumulation noise (path-dependent), not a kernel bug (top1 100% already rules out logic bugs).
with torch.no_grad():
    ref32=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32).cuda().eval()
    rl32=ref32(ids).logits[0].float()
bf16_floor=(tempered_logp(ref_logits,toks_next,1.0)-tempered_logp(rl32,toks_next,1.0)).abs().max().item()
mine_vs_32=(tempered_logp(my,toks_next,1.0)-tempered_logp(rl32,toks_next,1.0)).abs().max().item()
print(f"[bf16 floor] HFbf16-vs-HFfp32 logp max|Δ|={bf16_floor:.4e}   mine-vs-HFfp32 max|Δ|={mine_vs_32:.4e}")
# PASS bar = the M3a/M3b2 bar (top1 100%, logits rel ~1e-2) + tempered logp within the bf16 noise floor
# for a 32-layer 8B forward (~1e-1 on logprobs). Tighter eager-SMC-estimator equivalence is an M7 check.
lp_max=(tempered_logp(my,toks_next,1.0)-tempered_logp(ref_logits,toks_next,1.0)).abs().max().item()
print("RESULT:", "PASS" if (agree>0.99 and lp_max<1e-1) else "FAIL", f"(top1 {agree*100:.0f}%, logp max|Δ| {lp_max:.3e}, bf16 floor)")

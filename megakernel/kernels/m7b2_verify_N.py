"""M7b.2: N-particle VERIFY (8B) with a PER-PARTICLE BLOCK-DIAGONAL CAUSAL MASK — the one genuinely new
kernel mechanic for the N-particle cycle. The verify block is N sequences laid out as rows [p*Sv + s];
query row (p,s) attends only to keys (p, s'<=s) of the SAME particle (block-diagonal causal), RoPE at the
per-particle position (s for the query, s' for each key). Validates per-particle tempered target logprobs vs
HF-8B run on each particle's sequence independently.

= m6b's persistent 8B forward, rows S->N*Sv, attention masked within-particle. lm_head+logprob on host."""
import sys, math, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import m_kernels as mk
I32=cutlass.Int32; F32=cutlass.Float32; BF=cutlass.BFloat16

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
def verifyN_k(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,
              ghv,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
              N:cutlass.Constexpr,Sv:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,
              D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,
              B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D; R=N*Sv
    # embed (rows = N*Sv)
    idx=gtid
    while idx < R*hid:
        m=idx//hid; n=idx%hid
        ghv[m,n]=gEmb[gVtok[m],n]
        idx=idx+GT
    ls=I32(0); ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
    for L in cutlass.range(NL):
        idx=gtid
        while idx < R*QW:
            m=idx//QW; n=idx%QW
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=ghv[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            acc=F32(0.0)
            for k in cutlass.range(hid):
                acc=acc+ghv[m,k].to(F32)*inv*gg1[L,k].to(F32)*gWq[L,n,k].to(F32)
            gQ[m,n]=acc.to(gQ.element_type)
            idx=idx+GT
        idx=gtid
        while idx < R*KW:
            m=idx//KW; n=idx%KW
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=ghv[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            ak=F32(0.0); av=F32(0.0)
            for k in cutlass.range(hid):
                yn=ghv[m,k].to(F32)*inv*gg1[L,k].to(F32)
                ak=ak+yn*gWk[L,n,k].to(F32); av=av+yn*gWv[L,n,k].to(F32)
            gK[m,n]=ak.to(gK.element_type); gVt[m,n]=av.to(gVt.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # attention: query row m=(p,s); keys = rows p*Sv+s' for s'<=s (block-diagonal causal); RoPE per-position
        idx=gtid
        while idx < R*H:
            m=idx//H; h=idx%H; kvh=h//GRP
            p=m//Sv; s=m%Sv; base=p*Sv
            qr=cute.make_fragment(D,F32)
            for d in cutlass.range_constexpr(DH):
                lo=gQ[m,h*D+d].to(F32); hi=gQ[m,h*D+d+DH].to(F32)
                qr[d]=lo*gcos[s,d].to(F32)-hi*gsin[s,d].to(F32)
                qr[d+DH]=hi*gcos[s,d+DH].to(F32)+lo*gsin[s,d+DH].to(F32)
            ac=cute.make_fragment(D,F32)
            for d in cutlass.range_constexpr(D): ac[d]=F32(0.0)
            rmax=F32(-1.0e30); rsum=F32(0.0)
            for sp in cutlass.range(s+1):
                kr=base+sp
                sc=F32(0.0)
                for d in cutlass.range_constexpr(DH):
                    klo=gK[kr,kvh*D+d].to(F32); khi=gK[kr,kvh*D+d+DH].to(F32)
                    krlo=klo*gcos[sp,d].to(F32)-khi*gsin[sp,d].to(F32)
                    krhi=khi*gcos[sp,d+DH].to(F32)+klo*gsin[sp,d+DH].to(F32)
                    sc=sc+qr[d]*krlo+qr[d+DH]*krhi
                sc=sc*F32(scale)
                nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); pe=cute.exp(sc-nm)
                rsum=rsum*corr+pe
                for d in cutlass.range_constexpr(D): ac[d]=ac[d]*corr+pe*gVt[kr,kvh*D+d].to(F32)
                rmax=nm
            for d in cutlass.range_constexpr(D): gA[m,h*D+d]=(ac[d]/rsum).to(gA.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < R*hid:
            m=idx//hid; n=idx%hid
            acc=ghv[m,n].to(F32)
            for j in cutlass.range(QW):
                acc=acc+gA[m,j].to(F32)*gWo[L,n,j].to(F32)
            gR[m,n]=acc.to(gR.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < R*I:
            m=idx//I; n=idx%I
            ss=F32(0.0)
            for k in cutlass.range(hid):
                xv=gR[m,k].to(BF).to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hid)+F32(eps))
            g=F32(0.0); u=F32(0.0)
            for k in cutlass.range(hid):
                yn=gR[m,k].to(BF).to(F32)*inv*gg2[L,k].to(F32)
                g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32)
            gAct[m,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < R*hid:
            m=idx//hid; n=idx%hid
            acc=gR[m,n].to(F32)
            for j in cutlass.range(I):
                acc=acc+gAct[m,j].to(F32)*gWd[L,n,j].to(F32)
            ghv[m,n]=acc.to(ghv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def verifyN(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,ghv,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
            N:cutlass.Constexpr,Sv:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,
            D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,
            B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr):
    verifyN_k(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,ghv,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
              N,Sv,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale).launch(grid=[B,1,1],block=[BLK,1,1])

# ============================ driver ============================
from transformers import AutoModelForCausalLM, AutoTokenizer
dname="meta-llama/Llama-3.2-1B-Instruct"; vname="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(dname)
print("loading 1B (for drafting the N particles) + 8B target ...")
dm=AutoModelForCausalLM.from_pretrained(dname,dtype=torch.bfloat16).cuda().eval(); dm.requires_grad_(False)
vm=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.bfloat16).cuda().eval(); vm.requires_grad_(False)
mv=vm.model; cfv=vm.config
H=cfv.num_attention_heads; Hkv=cfv.num_key_value_heads; D=getattr(cfv,'head_dim',cfv.hidden_size//H)
hid=cfv.hidden_size; I=cfv.intermediate_size; NL=cfv.num_hidden_layers; eps=cfv.rms_norm_eps; V=cfv.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=256; NGEN=4; Np=4
print(f"[N-verify] N={Np} particles, 8B hid={hid} NL={NL}")

prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
# build N distinct drafted sequences by Gumbel-sampling the 1B (different noise per particle)
T=0.7; torch.manual_seed(1)
with torch.no_grad():
    g1log=dm(torch.cat([prompt_ids,dm.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]]).unsqueeze(0)).logits[0].float()
U=torch.rand(Np,NGEN,dm.config.vocab_size,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
gn=-torch.log(-torch.log(U))
draftedN=(g1log[SP-1:SP-1+NGEN].unsqueeze(0)/T+gn).argmax(-1).int()    # [Np,NGEN]
Vtok2d=torch.zeros(Np,Sv,device="cuda",dtype=torch.int32)
for p in range(Np): Vtok2d[p,:SP]=prompt_ids.int(); Vtok2d[p,SP:]=draftedN[p]
Vtok=Vtok2d.reshape(-1).contiguous()                                  # [Np*Sv] flattened rows

posv=torch.arange(Sv,device="cuda").unsqueeze(0)
cos,sin=mv.rotary_emb(torch.zeros(1,Sv,hid,device="cuda",dtype=torch.bfloat16),posv); cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(g): return torch.stack([g(mv.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)
Rrows=Np*Sv
ghv=mk.f32(Rrows,hid).to(torch.bfloat16); Q=mk.f32(Rrows,H*D); K=mk.f32(Rrows,Hkv*D); Vt=mk.f32(Rrows,Hkv*D)
A=mk.f32(Rrows,H*D); Rr=mk.f32(Rrows,hid); Act=mk.f32(Rrows,I)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Emb=mv.embed_tokens.weight
args=[Vtok,Emb,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,ghv,Q,K,Vt,A,Rr,Act,arr,sen]
mt=[from_dlpack(x) for x in args]
print("compiling N-particle masked verify ...")
comp=cute.compile(verifyN,*mt,Np,Sv,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale)
arr.zero_(); sen.zero_()
t0=time.time(); comp(*mt); torch.cuda.synchronize(); dt=time.time()-t0

# host: per-row final norm + lm_head -> logits[Np*Sv, V]
hf=ghv.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+eps); hn=(hf*inv*mv.norm.weight.float()).to(torch.bfloat16)
mylog=mk.f32(Rrows,V); mk.gemv(from_dlpack(hn),from_dlpack(vm.lm_head.weight),from_dlpack(mylog),Rrows,hid); torch.cuda.synchronize()
mylog=mylog.reshape(Np,Sv,V)
# reference: HF-8B per particle (independent forward), + fp32 floor
def tlp(lg,nx,Tt):
    z=lg[:-1]/Tt; return z.gather(-1,nx.view(-1,1)).squeeze(-1)-torch.logsumexp(z,-1)
agree=0.0; lp_kernel=[]; lp_hf=[]
with torch.no_grad():
    vm32=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.float32).cuda().eval()
for p in range(Np):
    seq=Vtok2d[p].long().unsqueeze(0)
    with torch.no_grad():
        r8=vm(seq).logits[0].float(); r32=vm32(seq).logits[0].float()
    agree+=(mylog[p].argmax(-1)==r8.argmax(-1)).float().mean().item()
    nx=seq[0,1:].long()
    # drafted positions s=SP-1..SP+NGEN-2 score the drafted tokens
    kp=tlp(mylog[p],nx,T)[SP-1:SP-1+NGEN]; hp=tlp(r32,nx,T)[SP-1:SP-1+NGEN]
    lp_kernel.append(kp); lp_hf.append(hp)
agree/=Np
lp_kernel=torch.stack(lp_kernel); lp_hf=torch.stack(lp_hf)
lp_max=(lp_kernel-lp_hf).abs().max().item()
print(f"\n[N-verify] drafted per particle:\n{draftedN.tolist()}")
print(f"[N-verify] per-particle top1 agree (vs HF-8B) = {agree*100:.1f}%   drafted tempered-logp max|Δ| vs HF-fp32 = {lp_max:.3e}   ({dt*1000:.0f} ms)")
print("RESULT:", "PASS" if (agree>0.99 and lp_max<5e-2) else "FAIL")

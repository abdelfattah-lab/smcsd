"""M7b.1: N-particle BATCHED draft megakernel (the prerequisite for the cross-particle SMC stages).
The draft now processes N particles per step (N rows through each layer = a batched decode; the GEMVs go
"fat" over N, better arithmetic intensity as the paper argues). Each particle has its own KV-cache slice and
its own Gumbel noise, so the N draws diverge. Teacher-forced (shared prompt prefix, per-particle noise) for a
clean deterministic validation of the batched machinery — free-running feed-back is wired in M7b.4.

Per-row layer math = m3b2 (idx=row*W+out), AR loop + per-particle KV + Gumbel = m4/m5. lm_head reads a
pre-staged normed-hidden gHN[N,hid] (avoids the per-vocab-output rmsnorm recompute). Validates each particle's
NGEN tokens vs torch Gumbel-max on the 1B logits with that particle's noise."""
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
def draftN_k(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,
             gh,gKc,gVc,gQ,gA,gR,gAct,gHN,gLog,gWBV,gWBI,garr,gsen,
             N:cutlass.Constexpr,SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,
             H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,
             GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,
             eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D
    ls=I32(0)
    for t in cutlass.range(SP+NGEN-1):
        idx=gtid
        while idx < N*hid:
            p=idx//hid; k=idx%hid
            gh[p,k]=gEmb[gTok[p,t],k]
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NL):
            # Stage1 Q,K,V (N rows; per-output rmsnorm recompute, m3b2-style)
            idx=gtid
            while idx < N*QW:
                p=idx//QW; n=idx%QW
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gh[p,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                acc=F32(0.0)
                for k in cutlass.range(hid):
                    acc=acc+gh[p,k].to(F32)*inv*gg1[L,k].to(F32)*gWq[L,n,k].to(F32)
                gQ[p,n]=acc.to(gQ.element_type)
                idx=idx+GT
            idx=gtid
            while idx < N*KW:
                p=idx//KW; n=idx%KW; kvh=n//D; d=n%D; part=n+DH if d<DH else n-DH
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gh[p,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0)
                for k in cutlass.range(hid):
                    yn=gh[p,k].to(F32)*inv*gg1[L,k].to(F32)
                    km=km+yn*gWk[L,n,k].to(F32); kp=kp+yn*gWk[L,part,k].to(F32); vv=vv+yn*gWv[L,n,k].to(F32)
                cl=gcos[t,d].to(F32); sl=gsin[t,d].to(F32); kr=F32(0.0)
                if d<DH: kr=km*cl-kp*sl
                else: kr=km*cl+kp*sl
                gKc[L,p,t,n]=kr.to(gKc.element_type); gVc[L,p,t,n]=vv.to(gVc.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage2 attention (N rows x H heads; per-particle causal over its own KV up to t)
            idx=gtid
            while idx < N*H:
                p=idx//H; h=idx%H; kvh=h//GRP
                qr=cute.make_fragment(D,F32)
                for d in cutlass.range_constexpr(DH):
                    lo=gQ[p,h*D+d].to(F32); hi=gQ[p,h*D+d+DH].to(F32)
                    qr[d]=lo*gcos[t,d].to(F32)-hi*gsin[t,d].to(F32)
                    qr[d+DH]=hi*gcos[t,d+DH].to(F32)+lo*gsin[t,d+DH].to(F32)
                ac=cute.make_fragment(D,F32)
                for d in cutlass.range_constexpr(D): ac[d]=F32(0.0)
                rmax=F32(-1.0e30); rsum=F32(0.0)
                for s in cutlass.range(t+1):
                    sc=F32(0.0)
                    for d in cutlass.range_constexpr(D): sc=sc+qr[d]*gKc[L,p,s,kvh*D+d].to(F32)
                    sc=sc*F32(scale)
                    nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); pe=cute.exp(sc-nm)
                    rsum=rsum*corr+pe
                    for d in cutlass.range_constexpr(D): ac[d]=ac[d]*corr+pe*gVc[L,p,s,kvh*D+d].to(F32)
                    rmax=nm
                for d in cutlass.range_constexpr(D): gA[p,h*D+d]=(ac[d]/rsum).to(gA.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage3 O + residual
            idx=gtid
            while idx < N*hid:
                p=idx//hid; n=idx%hid
                acc=gh[p,n].to(F32)
                for j in cutlass.range(QW):
                    acc=acc+gA[p,j].to(F32)*gWo[L,n,j].to(F32)
                gR[p,n]=acc.to(gR.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage4 gate/up + silu
            idx=gtid
            while idx < N*I:
                p=idx//I; n=idx%I
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gR[p,k].to(BF).to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                g=F32(0.0); u=F32(0.0)
                for k in cutlass.range(hid):
                    yn=gR[p,k].to(BF).to(F32)*inv*gg2[L,k].to(F32)
                    g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32)
                gAct[p,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage5 down + residual
            idx=gtid
            while idx < N*hid:
                p=idx//hid; n=idx%hid
                acc=gR[p,n].to(F32)
                for j in cutlass.range(I):
                    acc=acc+gAct[p,j].to(F32)*gWd[L,n,j].to(F32)
                gh[p,n]=acc.to(gh.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        if t+1>=SP:
            ii=t+1-SP
            # stage normed-hidden gHN[p,k] = norm(gh[p])*gnorm  (one norm per row, reused by lm_head)
            idx=gtid
            while idx < N*hid:
                p=idx//hid; k=idx%hid
                ss=F32(0.0)
                for kk in cutlass.range(hid):
                    xv=gh[p,kk].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                gHN[p,k]=(gh[p,k].to(F32)*inv*gnorm[k].to(F32)).to(gHN.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # lm_head: gLog[p,ii,n]
            idx=gtid
            while idx < N*V:
                p=idx//V; n=idx%V
                acc=F32(0.0)
                for k in cutlass.range(hid):
                    acc=acc+gHN[p,k].to(F32)*gLM[n,k].to(F32)
                gLog[p,ii,n]=acc
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # per-particle Gumbel-argmax (sequential over N; reuse gWBV/gWBI per particle)
            for p in cutlass.range(N):
                bval=F32(-1.0e30); bidx=I32(0); v=gtid
                while v<V:
                    gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gU[p,ii,v].to(F32)))
                    val=gLog[p,ii,v]*F32(invT)+gg
                    if val>bval: bval=val; bidx=I32(v)
                    v=v+GT
                o=16
                while o>0:
                    oval=cute.arch.shuffle_sync_bfly(bval,o); oidx=cute.arch.shuffle_sync_bfly(bidx,o)
                    if oval>bval: bval=oval; bidx=oidx
                    o=o//2
                if lane==0: gWBV[wid]=bval; gWBI[wid]=bidx
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                if gtid==0:
                    best=F32(-1.0e30); bi=I32(0); w=0
                    while w<NW:
                        if gWBV[w]>best: best=gWBV[w]; bi=gWBI[w]
                        w=w+1
                    gPred[p,ii]=bi
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def draftN(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,
           gh,gKc,gVc,gQ,gA,gR,gAct,gHN,gLog,gWBV,gWBI,garr,gsen,
           N:cutlass.Constexpr,SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,
           H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,
           GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,
           eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr):
    draftN_k(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,
             gh,gKc,gVc,gQ,gA,gR,gAct,gHN,gLog,gWBV,gWBI,garr,gsen,
             N,SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,invT,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])

# ============================ driver ============================
from transformers import AutoModelForCausalLM, AutoTokenizer
name="meta-llama/Llama-3.2-1B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).cuda().eval(); model.requires_grad_(False)
m=model.model; cf=model.config
H=cf.num_attention_heads; Hkv=cf.num_key_value_heads; D=getattr(cf,'head_dim',cf.hidden_size//H)
hid=cf.hidden_size; I=cf.intermediate_size; NL=cf.num_hidden_layers; eps=cf.rms_norm_eps; V=cf.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=256; NGEN=4; Np=4
print(f"[N-draft] N={Np} particles, hid={hid} NL={NL} NGEN={NGEN}")

prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
with torch.no_grad():
    ref=model.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]  # shared teacher-forced suffix
posv=torch.arange(Sv,device="cuda").unsqueeze(0)
cos,sin=m.rotary_emb(torch.zeros(1,Sv,hid,device="cuda",dtype=torch.bfloat16),posv); cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(g): return torch.stack([g(m.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)

# all particles share the same teacher-forced sequence [prompt, ref]; per-particle Gumbel noise differs.
Tok=torch.zeros(Np,Sv,device="cuda",dtype=torch.int32)
for p in range(Np): Tok[p,:SP]=prompt_ids.int(); Tok[p,SP:]=ref.int()
T=0.7; torch.manual_seed(1)
U=torch.rand(Np,NGEN,V,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
Pred=torch.full((Np,NGEN),-1,device="cuda",dtype=torch.int32)
gh=mk.f32(Np,hid).to(torch.bfloat16)
Kc=torch.zeros(NL,Np,Sv,Hkv*D,device="cuda",dtype=torch.bfloat16); Vc=torch.zeros_like(Kc)
Q=mk.f32(Np,H*D); A=mk.f32(Np,H*D); R=mk.f32(Np,hid); Act=mk.f32(Np,I); HN=mk.f32(Np,hid).to(torch.bfloat16); Log=mk.f32(Np*NGEN,V).reshape(Np,NGEN,V).contiguous()
NWv=B*BLK//32; WBV=mk.f32(NWv,1).reshape(NWv).contiguous(); WBI=torch.zeros(NWv,device='cuda',dtype=torch.int32)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Emb=m.embed_tokens.weight; LM=model.lm_head.weight; norm=m.norm.weight
args=[Tok,Pred,U,Emb,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,norm,LM,gh,Kc,Vc,Q,A,R,Act,HN,Log,WBV,WBI,arr,sen]
mt=[from_dlpack(x) for x in args]
print("compiling N-particle draft ...")
comp=cute.compile(draftN,*mt,Np,SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,1.0/T)
arr.zero_(); sen.zero_()
t0=time.time(); comp(*mt); torch.cuda.synchronize(); dt=time.time()-t0

# reference: torch Gumbel-max on the 1B logits of [prompt, ref], per-particle noise
with torch.no_grad():
    full=torch.cat([prompt_ids,ref]).unsqueeze(0); hflog=model(full).logits[0].float()  # [Sv, V]
gnoise=-torch.log(-torch.log(U))                                                          # [Np,NGEN,V]
ref_pred=(hflog[SP-1:SP-1+NGEN].unsqueeze(0)/T+gnoise).argmax(-1).int()                   # [Np,NGEN]
mine=Pred
match=(mine==ref_pred).sum().item(); tot=Np*NGEN
print(f"\n[N-draft] per-particle preds (mine):\n{mine.tolist()}")
print(f"[N-draft] reference (torch gumbel):\n{ref_pred.tolist()}")
print(f"[N-draft] match {match}/{tot}  ({dt*1000:.0f} ms)")
print("RESULT:", "PASS" if match>=tot-1 else "FAIL")

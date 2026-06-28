"""Interleaved A/B: M5 (re-read activation per output) vs M5d (smem-staged activation).
One model load, one set of tensors, both kernels compiled once, timed alternately so
GPU-contention drift cancels in the ratio. Reports both us/token and the speedup ratio."""
import sys, math, torch, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))+"/kernels")
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
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

@cute.jit
def wreduce(x):
    o=16
    while o>0:
        x=x+cute.arch.shuffle_sync_bfly(x,o); o=o//2
    return x

# ---------- M5: original (re-read gh*inv*g1 from global per output) ----------
@cute.kernel
def k_m5(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
         SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D; ls=I32(0)
    for t in cutlass.range(SP+NGEN-1):
        tok=gTok[t]; i=gtid
        while i<hid: gh[0,i]=gEmb[tok,i]; i=i+B*BLK
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NL):
            ss=F32(0.0); k=lane
            while k<hid: xv=gh[0,k].to(F32); ss=ss+xv*xv; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps))
            n=wid
            while n<QW:
                acc=F32(0.0); k=lane
                while k<hid: acc=acc+gh[0,k].to(F32)*inv*gg1[L,k].to(F32)*gWq[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gQ[0,n]=acc.to(gQ.element_type)
                n=n+NW
            n=wid
            while n<KW:
                kvh=n//D; d=n%D; part=n+DH if d<DH else n-DH
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0); k=lane
                while k<hid:
                    yn=gh[0,k].to(F32)*inv*gg1[L,k].to(F32)
                    km=km+yn*gWk[L,n,k].to(F32); kp=kp+yn*gWk[L,part,k].to(F32); vv=vv+yn*gWv[L,n,k].to(F32); k=k+32
                km=wreduce(km); kp=wreduce(kp); vv=wreduce(vv)
                if lane==0:
                    cl=gcos[t,d].to(F32); sl=gsin[t,d].to(F32); kr=F32(0.0)
                    if d<DH: kr=km*cl-kp*sl
                    else: kr=km*cl+kp*sl
                    gKc[L,t,n]=kr.to(gKc.element_type); gVc[L,t,n]=vv.to(gVc.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=gtid
            while i<H:
                h=i; kvh=h//GRP
                qr=cute.make_fragment(D,F32)
                for d in cutlass.range_constexpr(DH):
                    lo=gQ[0,h*D+d].to(F32); hi=gQ[0,h*D+d+DH].to(F32)
                    qr[d]=lo*gcos[t,d].to(F32)-hi*gsin[t,d].to(F32)
                    qr[d+DH]=hi*gcos[t,d+DH].to(F32)+lo*gsin[t,d+DH].to(F32)
                ac=cute.make_fragment(D,F32)
                for d in cutlass.range_constexpr(D): ac[d]=F32(0.0)
                rmax=F32(-1.0e30); rsum=F32(0.0)
                for s in cutlass.range(t+1):
                    sc=F32(0.0)
                    for d in cutlass.range_constexpr(D): sc=sc+qr[d]*gKc[L,s,kvh*D+d].to(F32)
                    sc=sc*F32(scale); nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                    rsum=rsum*corr+p
                    for d in cutlass.range_constexpr(D): ac[d]=ac[d]*corr+p*gVc[L,s,kvh*D+d].to(F32)
                    rmax=nm
                for d in cutlass.range_constexpr(D): gA[0,h*D+d]=(ac[d]/rsum).to(gA.element_type)
                i=i+B*BLK
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            n=wid
            while n<hid:
                acc=F32(0.0); k=lane
                while k<QW: acc=acc+gA[0,k].to(F32)*gWo[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gR[0,n]=(gh[0,n].to(F32)+acc).to(gR.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            ss=F32(0.0); k=lane
            while k<hid: xv=gR[0,k].to(BF).to(F32); ss=ss+xv*xv; k=k+32
            inv2=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps))
            n=wid
            while n<I:
                g=F32(0.0); u=F32(0.0); k=lane
                while k<hid:
                    yn=gR[0,k].to(BF).to(F32)*inv2*gg2[L,k].to(F32)
                    g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32); k=k+32
                g=wreduce(g); u=wreduce(u)
                if lane==0: gAct[0,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            n=wid
            while n<hid:
                acc=F32(0.0); k=lane
                while k<I: acc=acc+gAct[0,k].to(F32)*gWd[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gh[0,n]=(gR[0,n].to(F32)+acc).to(gh.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        if t+1>=SP:
            ss=F32(0.0); k=lane
            while k<hid: xv=gh[0,k].to(F32); ss=ss+xv*xv; k=k+32
            invf=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps)); ii=t+1-SP
            n=wid
            while n<V:
                acc=F32(0.0); k=lane
                while k<hid: acc=acc+gh[0,k].to(F32)*invf*gnorm[k].to(F32)*gLM[n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gLog[ii,n]=acc
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            bval=F32(-1.0e30); bidx=I32(0); v=gtid
            while v<V:
                gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gU[ii,v].to(F32)))
                val=gLog[ii,v]*F32(invT)+gg
                if val>bval: bval=val; bidx=I32(v)
                v=v+B*BLK
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
                gPred[ii]=bi
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def m5(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
       SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr):
    k_m5(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
         SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,invT,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])

# Staged kernel inline (kept in sync with kernels/m5d_draft_mega.py).
@cute.kernel
def k_m5d(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
          SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D
    smem=utils.SmemAllocator(); sY=smem.allocate_tensor(F32, cute.make_layout(I), byte_alignment=16)
    ls=I32(0)
    for t in cutlass.range(SP+NGEN-1):
        tok=gTok[t]; i=gtid
        while i<hid: gh[0,i]=gEmb[tok,i]; i=i+B*BLK
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NL):
            ss=F32(0.0); k=lane
            while k<hid: xv=gh[0,k].to(F32); ss=ss+xv*xv; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps))
            i=tx
            while i<hid: sY[i]=gh[0,i].to(F32)*inv*gg1[L,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<QW:
                acc=F32(0.0); k=lane
                while k<hid: acc=acc+sY[k]*gWq[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gQ[0,n]=acc.to(gQ.element_type)
                n=n+NW
            n=wid
            while n<KW:
                kvh=n//D; d=n%D; part=n+DH if d<DH else n-DH
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0); k=lane
                while k<hid:
                    yn=sY[k]
                    km=km+yn*gWk[L,n,k].to(F32); kp=kp+yn*gWk[L,part,k].to(F32); vv=vv+yn*gWv[L,n,k].to(F32); k=k+32
                km=wreduce(km); kp=wreduce(kp); vv=wreduce(vv)
                if lane==0:
                    cl=gcos[t,d].to(F32); sl=gsin[t,d].to(F32); kr=F32(0.0)
                    if d<DH: kr=km*cl-kp*sl
                    else: kr=km*cl+kp*sl
                    gKc[L,t,n]=kr.to(gKc.element_type); gVc[L,t,n]=vv.to(gVc.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            h=wid
            while h<H:
                kvh=h//GRP; d=lane
                lo=gQ[0,h*D+d].to(F32); hi=gQ[0,h*D+d+DH].to(F32)
                qlo=lo*gcos[t,d].to(F32)-hi*gsin[t,d].to(F32)
                qhi=hi*gcos[t,d+DH].to(F32)+lo*gsin[t,d+DH].to(F32)
                alo=F32(0.0); ahi=F32(0.0); rmax=F32(-1.0e30); rsum=F32(0.0)
                for s in cutlass.range(t+1):
                    klo=gKc[L,s,kvh*D+d].to(F32); khi=gKc[L,s,kvh*D+d+DH].to(F32)
                    sc=wreduce(qlo*klo+qhi*khi)*F32(scale)
                    nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                    rsum=rsum*corr+p
                    alo=alo*corr+p*gVc[L,s,kvh*D+d].to(F32); ahi=ahi*corr+p*gVc[L,s,kvh*D+d+DH].to(F32)
                    rmax=nm
                gA[0,h*D+d]=(alo/rsum).to(gA.element_type); gA[0,h*D+d+DH]=(ahi/rsum).to(gA.element_type)
                h=h+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=tx
            while i<QW: sY[i]=gA[0,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<hid:
                acc=F32(0.0); k=lane
                while k<QW: acc=acc+sY[k]*gWo[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gR[0,n]=(gh[0,n].to(F32)+acc).to(gR.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            ss=F32(0.0); k=lane
            while k<hid: xv=gR[0,k].to(BF).to(F32); ss=ss+xv*xv; k=k+32
            inv2=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps))
            i=tx
            while i<hid: sY[i]=gR[0,i].to(BF).to(F32)*inv2*gg2[L,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<I:
                g=F32(0.0); u=F32(0.0); k=lane
                while k<hid:
                    yn=sY[k]
                    g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32); k=k+32
                g=wreduce(g); u=wreduce(u)
                if lane==0: gAct[0,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=tx
            while i<I: sY[i]=gAct[0,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<hid:
                acc=F32(0.0); k=lane
                while k<I: acc=acc+sY[k]*gWd[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gh[0,n]=(gR[0,n].to(F32)+acc).to(gh.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        if t+1>=SP:
            ss=F32(0.0); k=lane
            while k<hid: xv=gh[0,k].to(F32); ss=ss+xv*xv; k=k+32
            invf=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps)); ii=t+1-SP
            i=tx
            while i<hid: sY[i]=gh[0,i].to(F32)*invf*gnorm[i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<V:
                acc=F32(0.0); k=lane
                while k<hid: acc=acc+sY[k]*gLM[n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gLog[ii,n]=acc
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            bval=F32(-1.0e30); bidx=I32(0); v=gtid
            while v<V:
                gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gU[ii,v].to(F32)))
                val=gLog[ii,v]*F32(invT)+gg
                if val>bval: bval=val; bidx=I32(v)
                v=v+B*BLK
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
                gPred[ii]=bi
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def m5d(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
        SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr):
    k_m5d(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,gWBV,gWBI,garr,gsen,
          SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,invT,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1],smem=I*4+256)

# ---------- driver ----------
from transformers import AutoModelForCausalLM, AutoTokenizer
name="meta-llama/Llama-3.2-1B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).cuda().eval(); model.requires_grad_(False)
m=model.model; cf=model.config
H=cf.num_attention_heads; Hkv=cf.num_key_value_heads; D=getattr(cf,'head_dim',cf.hidden_size//H)
hid=cf.hidden_size; I=cf.intermediate_size; NL=cf.num_hidden_layers; eps=cf.rms_norm_eps; V=cf.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=512; NGEN=4
prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; maxT=SP+NGEN
with torch.no_grad():
    ref=model.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]
posv=torch.arange(maxT,device="cuda").unsqueeze(0)
cos,sin=m.rotary_emb(torch.zeros(1,maxT,hid,device="cuda",dtype=torch.bfloat16),posv); cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(g): return torch.stack([g(m.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)
Tok=torch.zeros(maxT,device="cuda",dtype=torch.int32); Tok[:SP]=prompt_ids.int(); Tok[SP:SP+NGEN]=ref.int()
T=0.7; torch.manual_seed(1); U=torch.rand(NGEN,V,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
Pred=torch.full((NGEN,),-1,device="cuda",dtype=torch.int32)
h=mk.f32(1,hid).to(torch.bfloat16); Kc=torch.zeros(NL,maxT,Hkv*D,device="cuda",dtype=torch.bfloat16); Vc=torch.zeros_like(Kc)
Q=mk.f32(1,H*D); A=mk.f32(1,H*D); R=mk.f32(1,hid); Act=mk.f32(1,I); Log=mk.f32(NGEN,V)
NWv=B*512//32; WBV=mk.f32(NWv,1).reshape(NWv).contiguous(); WBI=torch.zeros(NWv,device='cuda',dtype=torch.int32)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Emb=m.embed_tokens.weight; LM=model.lm_head.weight; norm=m.norm.weight
targs=[Tok,Pred,U,Emb,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,norm,LM,h,Kc,Vc,Q,A,R,Act,Log,WBV,WBI,arr,sen]
mt=[from_dlpack(x) for x in targs]
cargs=(SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,1.0/T)
comp5=cute.compile(m5,*mt,*cargs)
comp5d=cute.compile(m5d,*mt,*cargs)

def run5():  arr.zero_(); sen.zero_(); comp5(*mt)
def run5d(): arr.zero_(); sen.zero_(); comp5d(*mt)

# validate both produce identical preds
run5();  torch.cuda.synchronize(); p5=Pred.tolist()
run5d(); torch.cuda.synchronize(); p5d=Pred.tolist()
print("m5 pred ", p5)
print("m5d pred", p5d)
print("preds identical:", p5==p5d)

# interleaved timing
NT=SP+NGEN-1
def timeit(fn,iters):
    for _ in range(3): fn()
    torch.cuda.synchronize(); e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/iters
ROUNDS=8; ITERS=10
t5=[]; t5d=[]
for r in range(ROUNDS):
    t5.append(timeit(run5,ITERS)); t5d.append(timeit(run5d,ITERS))
import statistics as st
m5_us=st.median(t5)*1000; m5d_us=st.median(t5d)*1000
print(f"[median over {ROUNDS} rounds]  M5={m5_us:.0f} us/launch ({m5_us/NT:.0f} us/tok)   M5d={m5d_us:.0f} us/launch ({m5d_us/NT:.0f} us/tok)")
print(f"[speedup] M5/M5d = {m5_us/m5d_us:.3f}x   (per-round ratios: "+", ".join(f"{a/b:.2f}" for a,b in zip(t5,t5d))+")")

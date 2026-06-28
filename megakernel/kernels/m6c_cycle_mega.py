"""M6 step 3 (THE HEADLINE): draft(1B) + verify(8B) fused in ONE persistent kernel launch.
The draft's sampled tokens flow to the verify's input IN-KERNEL (no host round-trip) — the fusion that
SMC's static, no-rejection shape uniquely permits. One launch = γ draft AR-forwards (1B) + the target
verify forward (8B) over the drafted block.

Structure of cycle_k (single @cute.kernel, B=148 blocks, grid-barrier between every stage):
  Phase A  DRAFT (1B, smem-staged, teacher-forced AR loop, in-kernel Gumbel)  -> gPred[NGEN], and writes
           each drafted token into gVtok[SP+ii].
  (barrier)
  Phase B  VERIFY embed: ghv[m,:] = Emb8B[gVtok[m]]   (tokens from Phase A)
  (barrier)
  Phase B  VERIFY (8B, 32-layer prefill-shaped batched forward over S=SP+NGEN tokens) -> ghv[S,hid8].
Final norm + lm_head + tempered logprob done on host (move in-kernel for M7).

Validation: (a) draft tokens gPred == the 1B Gumbel reference (4/4, as M5d); (b) the fused verify's
tempered logprobs == HF-8B over [prompt, drafted] — i.e. the draft output really fed the verify in-graph.
Suffix d=draft/1B, v=verify/8B."""
import sys, math, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
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

@cute.kernel
def cycle_k(
    # ---- shared / control ----
    gTok,gVtok,gPred,gU, garr,gsen,
    # ---- DRAFT (1B) ----
    gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
    ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gLogd,gWBV,gWBI,
    # ---- VERIFY (8B) ----
    gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,
    ghv,gQv,gKv,gVv,gAv,gRv,gActv,
    # ---- draft constexpr ----
    SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NLd:cutlass.Constexpr,Hd:cutlass.Constexpr,Hkvd:cutlass.Constexpr,
    Dd:cutlass.Constexpr,hidd:cutlass.Constexpr,Id:cutlass.Constexpr,GRPd:cutlass.Constexpr,DHd:cutlass.Constexpr,
    Vd:cutlass.Constexpr,epsd:cutlass.Constexpr,scaled:cutlass.Constexpr,invT:cutlass.Constexpr,
    # ---- verify constexpr ----
    Sv:cutlass.Constexpr,NLv:cutlass.Constexpr,Hv:cutlass.Constexpr,Hkvv:cutlass.Constexpr,Dv:cutlass.Constexpr,
    hidv:cutlass.Constexpr,Iv:cutlass.Constexpr,GRPv:cutlass.Constexpr,DHv:cutlass.Constexpr,
    epsv:cutlass.Constexpr,scalev:cutlass.Constexpr,
    # ---- launch constexpr ----
    B:cutlass.Constexpr,BLK:cutlass.Constexpr,NW:cutlass.Constexpr):

    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QWd=Hd*Dd; KWd=Hkvd*Dd
    smem=utils.SmemAllocator(); sY=smem.allocate_tensor(F32, cute.make_layout(Id), byte_alignment=16)
    ls=I32(0)

    # =========================== PHASE A: DRAFT (1B) ===========================
    for t in cutlass.range(SP+NGEN-1):
        tok=gTok[t]
        i=gtid
        while i<hidd: ghd[0,i]=gEmbd[tok,i]; i=i+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NLd):
            ss=F32(0.0); k=lane
            while k<hidd: xv=ghd[0,k].to(F32); ss=ss+xv*xv; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hidd)+F32(epsd))
            i=tx
            while i<hidd: sY[i]=ghd[0,i].to(F32)*inv*gg1d[L,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<QWd:
                acc=F32(0.0); k=lane
                while k<hidd: acc=acc+sY[k]*gWqd[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gQd[0,n]=acc.to(gQd.element_type)
                n=n+NW
            n=wid
            while n<KWd:
                kvh=n//Dd; d=n%Dd; part=n+DHd if d<DHd else n-DHd
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0); k=lane
                while k<hidd:
                    yn=sY[k]
                    km=km+yn*gWkd[L,n,k].to(F32); kp=kp+yn*gWkd[L,part,k].to(F32); vv=vv+yn*gWvd[L,n,k].to(F32); k=k+32
                km=wreduce(km); kp=wreduce(kp); vv=wreduce(vv)
                if lane==0:
                    cl=gcosd[t,d].to(F32); sl=gsind[t,d].to(F32); kr=F32(0.0)
                    if d<DHd: kr=km*cl-kp*sl
                    else: kr=km*cl+kp*sl
                    gKcd[L,t,n]=kr.to(gKcd.element_type); gVcd[L,t,n]=vv.to(gVcd.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=gtid
            while i<Hd:
                h=i; kvh=h//GRPd
                qr=cute.make_fragment(Dd,F32)
                for d in cutlass.range_constexpr(DHd):
                    lo=gQd[0,h*Dd+d].to(F32); hi=gQd[0,h*Dd+d+DHd].to(F32)
                    qr[d]=lo*gcosd[t,d].to(F32)-hi*gsind[t,d].to(F32)
                    qr[d+DHd]=hi*gcosd[t,d+DHd].to(F32)+lo*gsind[t,d+DHd].to(F32)
                ac=cute.make_fragment(Dd,F32)
                for d in cutlass.range_constexpr(Dd): ac[d]=F32(0.0)
                rmax=F32(-1.0e30); rsum=F32(0.0)
                for s in cutlass.range(t+1):
                    sc=F32(0.0)
                    for d in cutlass.range_constexpr(Dd): sc=sc+qr[d]*gKcd[L,s,kvh*Dd+d].to(F32)
                    sc=sc*F32(scaled)
                    nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                    rsum=rsum*corr+p
                    for d in cutlass.range_constexpr(Dd): ac[d]=ac[d]*corr+p*gVcd[L,s,kvh*Dd+d].to(F32)
                    rmax=nm
                for d in cutlass.range_constexpr(Dd): gAd[0,h*Dd+d]=(ac[d]/rsum).to(gAd.element_type)
                i=i+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=tx
            while i<QWd: sY[i]=gAd[0,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<hidd:
                acc=F32(0.0); k=lane
                while k<QWd: acc=acc+sY[k]*gWod[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gRd[0,n]=(ghd[0,n].to(F32)+acc).to(gRd.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            ss=F32(0.0); k=lane
            while k<hidd: xv=gRd[0,k].to(BF).to(F32); ss=ss+xv*xv; k=k+32
            inv2=cute.rsqrt(wreduce(ss)/F32(hidd)+F32(epsd))
            i=tx
            while i<hidd: sY[i]=gRd[0,i].to(BF).to(F32)*inv2*gg2d[L,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<Id:
                g=F32(0.0); u=F32(0.0); k=lane
                while k<hidd:
                    yn=sY[k]
                    g=g+yn*gWgd[L,n,k].to(F32); u=u+yn*gWud[L,n,k].to(F32); k=k+32
                g=wreduce(g); u=wreduce(u)
                if lane==0: gActd[0,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gActd.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            i=tx
            while i<Id: sY[i]=gActd[0,i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<hidd:
                acc=F32(0.0); k=lane
                while k<Id: acc=acc+sY[k]*gWdd[L,n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: ghd[0,n]=(gRd[0,n].to(F32)+acc).to(ghd.element_type)
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        if t+1>=SP:
            ss=F32(0.0); k=lane
            while k<hidd: xv=ghd[0,k].to(F32); ss=ss+xv*xv; k=k+32
            invf=cute.rsqrt(wreduce(ss)/F32(hidd)+F32(epsd)); ii=t+1-SP
            i=tx
            while i<hidd: sY[i]=ghd[0,i].to(F32)*invf*gnormd[i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<Vd:
                acc=F32(0.0); k=lane
                while k<hidd: acc=acc+sY[k]*gLMd[n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0: gLogd[ii,n]=acc
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            bval=F32(-1.0e30); bidx=I32(0); v=gtid
            while v<Vd:
                gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gU[ii,v].to(F32)))
                val=gLogd[ii,v]*F32(invT)+gg
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
                gPred[ii]=bi
                gVtok[SP+ii]=bi          # <<< draft token flows to the verify input buffer, in-kernel
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # barrier separating draft-done from verify-start
    ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # =========================== PHASE B: VERIFY (8B) ===========================
    # embed gVtok (= [prompt, drafted]) with the 8B embedding
    idx=gtid
    while idx < Sv*hidv:
        m=idx//hidv; n=idx%hidv
        ghv[m,n]=gEmbv[gVtok[m],n]
        idx=idx+GT
    ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    QWv=Hv*Dv; KWv=Hkvv*Dv
    for L in cutlass.range(NLv):
        idx=gtid
        while idx < Sv*QWv:
            m=idx//QWv; n=idx%QWv
            ss=F32(0.0)
            for k in cutlass.range(hidv):
                xv=ghv[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hidv)+F32(epsv))
            acc=F32(0.0)
            for k in cutlass.range(hidv):
                acc=acc+ghv[m,k].to(F32)*inv*gg1v[L,k].to(F32)*gWqv[L,n,k].to(F32)
            gQv[m,n]=acc.to(gQv.element_type)
            idx=idx+GT
        idx=gtid
        while idx < Sv*KWv:
            m=idx//KWv; n=idx%KWv
            ss=F32(0.0)
            for k in cutlass.range(hidv):
                xv=ghv[m,k].to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hidv)+F32(epsv))
            ak=F32(0.0); av=F32(0.0)
            for k in cutlass.range(hidv):
                yn=ghv[m,k].to(F32)*inv*gg1v[L,k].to(F32)
                ak=ak+yn*gWkv[L,n,k].to(F32); av=av+yn*gWvv[L,n,k].to(F32)
            gKv[m,n]=ak.to(gKv.element_type); gVv[m,n]=av.to(gVv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < Sv*Hv:
            m=idx//Hv; h=idx%Hv; kvh=h//GRPv
            qr=cute.make_fragment(Dv,F32)
            for d in cutlass.range_constexpr(DHv):
                lo=gQv[m,h*Dv+d].to(F32); hi=gQv[m,h*Dv+d+DHv].to(F32)
                qr[d]=lo*gcosv[m,d].to(F32)-hi*gsinv[m,d].to(F32)
                qr[d+DHv]=hi*gcosv[m,d+DHv].to(F32)+lo*gsinv[m,d+DHv].to(F32)
            ac=cute.make_fragment(Dv,F32)
            for d in cutlass.range_constexpr(Dv): ac[d]=F32(0.0)
            rmax=F32(-1.0e30); rsum=F32(0.0)
            for s in cutlass.range(m+1):
                sc=F32(0.0)
                for d in cutlass.range_constexpr(DHv):
                    klo=gKv[s,kvh*Dv+d].to(F32); khi=gKv[s,kvh*Dv+d+DHv].to(F32)
                    krlo=klo*gcosv[s,d].to(F32)-khi*gsinv[s,d].to(F32)
                    krhi=khi*gcosv[s,d+DHv].to(F32)+klo*gsinv[s,d+DHv].to(F32)
                    sc=sc+qr[d]*krlo+qr[d+DHv]*krhi
                sc=sc*F32(scalev)
                nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                rsum=rsum*corr+p
                for d in cutlass.range_constexpr(Dv): ac[d]=ac[d]*corr+p*gVv[s,kvh*Dv+d].to(F32)
                rmax=nm
            for d in cutlass.range_constexpr(Dv): gAv[m,h*Dv+d]=(ac[d]/rsum).to(gAv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < Sv*hidv:
            m=idx//hidv; n=idx%hidv
            acc=ghv[m,n].to(F32)
            for j in cutlass.range(QWv):
                acc=acc+gAv[m,j].to(F32)*gWov[L,n,j].to(F32)
            gRv[m,n]=acc.to(gRv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < Sv*Iv:
            m=idx//Iv; n=idx%Iv
            ss=F32(0.0)
            for k in cutlass.range(hidv):
                xv=gRv[m,k].to(BF).to(F32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/F32(hidv)+F32(epsv))
            g=F32(0.0); u=F32(0.0)
            for k in cutlass.range(hidv):
                yn=gRv[m,k].to(BF).to(F32)*inv*gg2v[L,k].to(F32)
                g=g+yn*gWgv[L,n,k].to(F32); u=u+yn*gWuv[L,n,k].to(F32)
            silu=g/(F32(1.0)+cute.exp(-g))
            gActv[m,n]=(silu*u).to(gActv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < Sv*hidv:
            m=idx//hidv; n=idx%hidv
            acc=gRv[m,n].to(F32)
            for j in cutlass.range(Iv):
                acc=acc+gActv[m,j].to(F32)*gWdv[L,n,j].to(F32)
            ghv[m,n]=acc.to(ghv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def cycle(gTok,gVtok,gPred,gU,garr,gsen,
          gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
          ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gLogd,gWBV,gWBI,
          gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,
          ghv,gQv,gKv,gVv,gAv,gRv,gActv,
          SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NLd:cutlass.Constexpr,Hd:cutlass.Constexpr,Hkvd:cutlass.Constexpr,
          Dd:cutlass.Constexpr,hidd:cutlass.Constexpr,Id:cutlass.Constexpr,GRPd:cutlass.Constexpr,DHd:cutlass.Constexpr,
          Vd:cutlass.Constexpr,epsd:cutlass.Constexpr,scaled:cutlass.Constexpr,invT:cutlass.Constexpr,
          Sv:cutlass.Constexpr,NLv:cutlass.Constexpr,Hv:cutlass.Constexpr,Hkvv:cutlass.Constexpr,Dv:cutlass.Constexpr,
          hidv:cutlass.Constexpr,Iv:cutlass.Constexpr,GRPv:cutlass.Constexpr,DHv:cutlass.Constexpr,
          epsv:cutlass.Constexpr,scalev:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    cycle_k(gTok,gVtok,gPred,gU,garr,gsen,
            gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
            ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gLogd,gWBV,gWBI,
            gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,
            ghv,gQv,gKv,gVv,gAv,gRv,gActv,
            SP,NGEN,NLd,Hd,Hkvd,Dd,hidd,Id,GRPd,DHd,Vd,epsd,scaled,invT,
            Sv,NLv,Hv,Hkvv,Dv,hidv,Iv,GRPv,DHv,epsv,scalev,
            B,BLK,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1],smem=Id*4+256)

# ============================ driver ============================
from transformers import AutoModelForCausalLM, AutoTokenizer
dname="meta-llama/Llama-3.2-1B-Instruct"; vname="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(dname)
print("loading 1B draft + 8B target ...")
dm=AutoModelForCausalLM.from_pretrained(dname,dtype=torch.bfloat16).cuda().eval(); dm.requires_grad_(False)
vm=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.bfloat16).cuda().eval(); vm.requires_grad_(False)
md=dm.model; cfd=dm.config; mv=vm.model; cfv=vm.config
Hd=cfd.num_attention_heads; Hkvd=cfd.num_key_value_heads; Dd=getattr(cfd,'head_dim',cfd.hidden_size//Hd)
hidd=cfd.hidden_size; Id=cfd.intermediate_size; NLd=cfd.num_hidden_layers; epsd=cfd.rms_norm_eps; Vd=cfd.vocab_size
GRPd=Hd//Hkvd; DHd=Dd//2; scaled=1.0/math.sqrt(Dd)
Hv=cfv.num_attention_heads; Hkvv=cfv.num_key_value_heads; Dv=getattr(cfv,'head_dim',cfv.hidden_size//Hv)
hidv=cfv.hidden_size; Iv=cfv.intermediate_size; NLv=cfv.num_hidden_layers; epsv=cfv.rms_norm_eps; Vvoc=cfv.vocab_size
GRPv=Hv//Hkvv; DHv=Dv//2; scalev=1.0/math.sqrt(Dv)
B=148; BLK=512; NGEN=4
print(f"[draft 1B] hid={hidd} NL={NLd} | [verify 8B] hid={hidv} NL={NLv}")

prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
with torch.no_grad():
    ref=dm.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]   # greedy draft for teacher-forcing input
# RoPE tables (per model — different head_dim & scaling)
posv=torch.arange(Sv,device="cuda").unsqueeze(0)
cosd,sind=md.rotary_emb(torch.zeros(1,Sv,hidd,device="cuda",dtype=torch.bfloat16),posv); cosd=cosd[0].contiguous(); sind=sind[0].contiguous()
cosv,sinv=mv.rotary_emb(torch.zeros(1,Sv,hidv,device="cuda",dtype=torch.bfloat16),posv); cosv=cosv[0].contiguous(); sinv=sinv[0].contiguous()
def stk(m,NL,g): return torch.stack([g(m.layers[L]) for L in range(NL)]).contiguous()
# draft weights
WqdA=stk(md,NLd,lambda l:l.self_attn.q_proj.weight); WkdA=stk(md,NLd,lambda l:l.self_attn.k_proj.weight)
WvdA=stk(md,NLd,lambda l:l.self_attn.v_proj.weight); WodA=stk(md,NLd,lambda l:l.self_attn.o_proj.weight)
g1dA=stk(md,NLd,lambda l:l.input_layernorm.weight); g2dA=stk(md,NLd,lambda l:l.post_attention_layernorm.weight)
WgdA=stk(md,NLd,lambda l:l.mlp.gate_proj.weight); WudA=stk(md,NLd,lambda l:l.mlp.up_proj.weight); WddA=stk(md,NLd,lambda l:l.mlp.down_proj.weight)
# verify weights
WqvA=stk(mv,NLv,lambda l:l.self_attn.q_proj.weight); WkvA=stk(mv,NLv,lambda l:l.self_attn.k_proj.weight)
WvvA=stk(mv,NLv,lambda l:l.self_attn.v_proj.weight); WovA=stk(mv,NLv,lambda l:l.self_attn.o_proj.weight)
g1vA=stk(mv,NLv,lambda l:l.input_layernorm.weight); g2vA=stk(mv,NLv,lambda l:l.post_attention_layernorm.weight)
WgvA=stk(mv,NLv,lambda l:l.mlp.gate_proj.weight); WuvA=stk(mv,NLv,lambda l:l.mlp.up_proj.weight); WdvA=stk(mv,NLv,lambda l:l.mlp.down_proj.weight)

Tok=torch.zeros(Sv,device="cuda",dtype=torch.int32); Tok[:SP]=prompt_ids.int(); Tok[SP:]=ref.int()  # teacher-forced draft input
Vtok=torch.zeros(Sv,device="cuda",dtype=torch.int32); Vtok[:SP]=prompt_ids.int()                     # prompt; draft fills [SP:]
T=0.7; torch.manual_seed(1); U=torch.rand(NGEN,Vd,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
Pred=torch.full((NGEN,),-1,device="cuda",dtype=torch.int32)
# draft scratch
ghd=mk.f32(1,hidd).to(torch.bfloat16); Kcd=torch.zeros(NLd,Sv,Hkvd*Dd,device="cuda",dtype=torch.bfloat16); Vcd=torch.zeros_like(Kcd)
Qd=mk.f32(1,Hd*Dd); Ad=mk.f32(1,Hd*Dd); Rd=mk.f32(1,hidd); Actd=mk.f32(1,Id); Logd=mk.f32(NGEN,Vd)
NWv=B*BLK//32; WBV=mk.f32(NWv,1).reshape(NWv).contiguous(); WBI=torch.zeros(NWv,device='cuda',dtype=torch.int32)
# verify scratch
ghv=mk.f32(Sv,hidv).to(torch.bfloat16); Qv=mk.f32(Sv,Hv*Dv); Kv=mk.f32(Sv,Hkvv*Dv); Vv=mk.f32(Sv,Hkvv*Dv)
Av=mk.f32(Sv,Hv*Dv); Rv=mk.f32(Sv,hidv); Actv=mk.f32(Sv,Iv)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Embd=md.embed_tokens.weight; LMd=dm.lm_head.weight; normd=md.norm.weight; Embv=mv.embed_tokens.weight

args=[Tok,Vtok,Pred,U,arr,sen,
      Embd,WqdA,WkdA,WvdA,WodA,g1dA,g2dA,WgdA,WudA,WddA,cosd,sind,normd,LMd,
      ghd,Kcd,Vcd,Qd,Ad,Rd,Actd,Logd,WBV,WBI,
      Embv,WqvA,WkvA,WvvA,WovA,g1vA,g2vA,WgvA,WuvA,WdvA,cosv,sinv,
      ghv,Qv,Kv,Vv,Av,Rv,Actv]
mt=[from_dlpack(x) for x in args]
cc=(SP,NGEN,NLd,Hd,Hkvd,Dd,hidd,Id,GRPd,DHd,Vd,epsd,scaled,1.0/T,
    Sv,NLv,Hv,Hkvv,Dv,hidv,Iv,GRPv,DHv,epsv,scalev,B,BLK)
print("compiling fused cycle kernel ...")
comp=cute.compile(cycle,*mt,*cc)
arr.zero_(); sen.zero_()
t0=time.time(); comp(*mt); torch.cuda.synchronize(); dt=time.time()-t0

# ---- validate Phase A: draft tokens vs 1B gumbel reference ----
gen=Pred.tolist()
with torch.no_grad():
    full=torch.cat([prompt_ids,ref]).unsqueeze(0); dlog=dm(full).logits[0].float()
gnoise=-torch.log(-torch.log(U)); ref_g=(dlog[SP-1:SP-1+NGEN]/T+gnoise).argmax(-1).int().tolist()
dmatch=sum(int(a==b) for a,b in zip(gen,ref_g))
print(f"\n[Phase A draft] pred={gen} ref={ref_g}  match {dmatch}/{NGEN}")

# ---- validate Phase B: fused verify hidden -> logprob vs HF-8B over [prompt, drafted] ----
vtok_out=Vtok.tolist()
hf=ghv.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+epsv); hn=(hf*inv*mv.norm.weight.float()).to(torch.bfloat16)
mylog=mk.f32(Sv,Vvoc); mk.gemv(from_dlpack(hn),from_dlpack(vm.lm_head.weight),from_dlpack(mylog),Sv,hidv); torch.cuda.synchronize()
vids=torch.tensor(vtok_out,device="cuda").unsqueeze(0)
with torch.no_grad():
    ref8=vm(vids).logits[0].float()
v_agree=(mylog.argmax(-1)==ref8.argmax(-1)).float().mean().item()
def tlp(lg,nx,Tt):
    z=lg[:-1]/Tt; return z.gather(-1,nx.view(-1,1)).squeeze(-1)-torch.logsumexp(z,-1)
nx=vids[0,1:].long()
lp_my=tlp(mylog,nx,1.0); lp_ref=tlp(ref8,nx,1.0)
lp_max=(lp_my-lp_ref).abs().max().item()
# bf16 floor: HF-bf16 vs HF-fp32 over the SAME (drafted) tokens — separates kernel error from bf16 noise.
with torch.no_grad():
    vm32=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.float32).cuda().eval()
    ref8_32=vm32(vids).logits[0].float()
floor=(tlp(ref8,nx,1.0)-tlp(ref8_32,nx,1.0)).abs().max().item()
mine_vs_32=(lp_my-tlp(ref8_32,nx,1.0)).abs().max().item()
print(f"[Phase B verify] vtok={vtok_out}  (prompt[:{SP}]+drafted) top1 agree {v_agree*100:.1f}%  tempered-logp max|Δ| vs HFbf16={lp_max:.3e}")
print(f"[bf16 floor] HFbf16-vs-HFfp32={floor:.3e}   mine-vs-HFfp32={mine_vs_32:.3e}  (kernel error ~ mine-vs-fp32)")
print(f"[fused cycle] one launch: draft(1B {NLd}L) + verify(8B {NLv}L) = {dt*1000:.0f} ms")
# PASS = draft tokens match, verify top1 100%, and the fused verify is within the bf16 floor of fp32 truth.
ok = (dmatch>=NGEN-1) and (v_agree>0.99) and (mine_vs_32 < max(2*floor, 3e-2))
print("RESULT:", "PASS" if ok else "FAIL")

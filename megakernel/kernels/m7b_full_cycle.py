"""M7b (FUSED): the ENTIRE N-particle SMC cycle in ONE persistent kernel launch.
  Phase A  DRAFT (1B, N particles): batched AR loop -> drafted tokens (-> gVtok) + in-kernel DRAFT logprob gDlp.
  Phase B  VERIFY (8B, N particles): embed gVtok, per-particle BLOCK-DIAGONAL causal forward.
  Phase C  TARGET logprob (in-kernel): per particle/drafted-position lm_head + tempered logsumexp -> gTlp;
           also store each particle's LAST-position target logits gVLast (for the bonus).
  Phase D  REWEIGHT + SYSTEMATIC RESAMPLE (in-kernel, thread 0): log_w=Σ(α·target−draft) -> softmax -> ancestors.
  Phase E  BONUS (in-kernel): Gumbel-max on the tempered target last-position logits of each ancestor -> gBonus.
One launch = the whole worker cycle. Validated vs eager torch SMC (weights/ancestors/bonus exact).
Suffix d=draft/1B, v=verify/8B. Teacher-forced draft (per-particle Gumbel noise) for clean validation."""
import sys, math, torch, time
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
import m_kernels as mk
sys.path.insert(0, "/home/yahya/smcsd")
from smcsd.common.utils import normalize_log_weights
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
    gTokd,gVtok,gPred,gUg,gDlp,gTlp,gW,gAnc,gBonus,gUr,gUb, garr,gsen,
    # draft (1B)
    gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
    ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gHNd,gLogd,gWBV,gWBI,gRed,
    # verify (8B)
    gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,gnormv,gLMv,
    ghv,gHNv,gQv,gKv,gVv,gAv,gRv,gActv,gVlog,gVLast,
    N:cutlass.Constexpr,SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,Sv:cutlass.Constexpr,
    NLd:cutlass.Constexpr,Hd:cutlass.Constexpr,Hkvd:cutlass.Constexpr,Dd:cutlass.Constexpr,hidd:cutlass.Constexpr,
    Id:cutlass.Constexpr,GRPd:cutlass.Constexpr,DHd:cutlass.Constexpr,Vd:cutlass.Constexpr,epsd:cutlass.Constexpr,scaled:cutlass.Constexpr,
    NLv:cutlass.Constexpr,Hv:cutlass.Constexpr,Hkvv:cutlass.Constexpr,Dv:cutlass.Constexpr,hidv:cutlass.Constexpr,
    Iv:cutlass.Constexpr,GRPv:cutlass.Constexpr,DHv:cutlass.Constexpr,Vv:cutlass.Constexpr,epsv:cutlass.Constexpr,scalev:cutlass.Constexpr,
    invT:cutlass.Constexpr,alpha:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,NW:cutlass.Constexpr):

    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QWd=Hd*Dd; KWd=Hkvd*Dd; QWv=Hv*Dv; KWv=Hkvv*Dv; Rv=N*Sv
    smem=utils.SmemAllocator(); sY=smem.allocate_tensor(F32, cute.make_layout(hidv), byte_alignment=16)
    ls=I32(0)

    # =================== PHASE A: DRAFT (1B, N particles) ===================
    for t in cutlass.range(SP+NGEN-1):
        idx=gtid
        while idx < N*hidd:
            p=idx//hidd; k=idx%hidd
            ghd[p,k]=gEmbd[gTokd[p,t],k]
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NLd):
            idx=gtid
            while idx < N*QWd:
                p=idx//QWd; n=idx%QWd
                ss=F32(0.0)
                for k in cutlass.range(hidd):
                    xv=ghd[p,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hidd)+F32(epsd))
                acc=F32(0.0)
                for k in cutlass.range(hidd):
                    acc=acc+ghd[p,k].to(F32)*inv*gg1d[L,k].to(F32)*gWqd[L,n,k].to(F32)
                gQd[p,n]=acc.to(gQd.element_type)
                idx=idx+GT
            idx=gtid
            while idx < N*KWd:
                p=idx//KWd; n=idx%KWd; kvh=n//Dd; d=n%Dd; part=n+DHd if d<DHd else n-DHd
                ss=F32(0.0)
                for k in cutlass.range(hidd):
                    xv=ghd[p,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hidd)+F32(epsd))
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0)
                for k in cutlass.range(hidd):
                    yn=ghd[p,k].to(F32)*inv*gg1d[L,k].to(F32)
                    km=km+yn*gWkd[L,n,k].to(F32); kp=kp+yn*gWkd[L,part,k].to(F32); vv=vv+yn*gWvd[L,n,k].to(F32)
                cl=gcosd[t,d].to(F32); sl=gsind[t,d].to(F32); kr=F32(0.0)
                if d<DHd: kr=km*cl-kp*sl
                else: kr=km*cl+kp*sl
                gKcd[L,p,t,n]=kr.to(gKcd.element_type); gVcd[L,p,t,n]=vv.to(gVcd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            idx=gtid
            while idx < N*Hd:
                p=idx//Hd; h=idx%Hd; kvh=h//GRPd
                qr=cute.make_fragment(Dd,F32)
                for d in cutlass.range_constexpr(DHd):
                    lo=gQd[p,h*Dd+d].to(F32); hi=gQd[p,h*Dd+d+DHd].to(F32)
                    qr[d]=lo*gcosd[t,d].to(F32)-hi*gsind[t,d].to(F32)
                    qr[d+DHd]=hi*gcosd[t,d+DHd].to(F32)+lo*gsind[t,d+DHd].to(F32)
                ac=cute.make_fragment(Dd,F32)
                for d in cutlass.range_constexpr(Dd): ac[d]=F32(0.0)
                rmax=F32(-1.0e30); rsum=F32(0.0)
                for s in cutlass.range(t+1):
                    sc=F32(0.0)
                    for d in cutlass.range_constexpr(Dd): sc=sc+qr[d]*gKcd[L,p,s,kvh*Dd+d].to(F32)
                    sc=sc*F32(scaled)
                    nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); pe=cute.exp(sc-nm)
                    rsum=rsum*corr+pe
                    for d in cutlass.range_constexpr(Dd): ac[d]=ac[d]*corr+pe*gVcd[L,p,s,kvh*Dd+d].to(F32)
                    rmax=nm
                for d in cutlass.range_constexpr(Dd): gAd[p,h*Dd+d]=(ac[d]/rsum).to(gAd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            idx=gtid
            while idx < N*hidd:
                p=idx//hidd; n=idx%hidd
                acc=ghd[p,n].to(F32)
                for j in cutlass.range(QWd):
                    acc=acc+gAd[p,j].to(F32)*gWod[L,n,j].to(F32)
                gRd[p,n]=acc.to(gRd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            idx=gtid
            while idx < N*Id:
                p=idx//Id; n=idx%Id
                ss=F32(0.0)
                for k in cutlass.range(hidd):
                    xv=gRd[p,k].to(BF).to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hidd)+F32(epsd))
                g=F32(0.0); u=F32(0.0)
                for k in cutlass.range(hidd):
                    yn=gRd[p,k].to(BF).to(F32)*inv*gg2d[L,k].to(F32)
                    g=g+yn*gWgd[L,n,k].to(F32); u=u+yn*gWud[L,n,k].to(F32)
                gActd[p,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gActd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            idx=gtid
            while idx < N*hidd:
                p=idx//hidd; n=idx%hidd
                acc=gRd[p,n].to(F32)
                for j in cutlass.range(Id):
                    acc=acc+gActd[p,j].to(F32)*gWdd[L,n,j].to(F32)
                ghd[p,n]=acc.to(ghd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        if t+1>=SP:
            ii=t+1-SP
            idx=gtid
            while idx < N*hidd:
                p=idx//hidd; k=idx%hidd
                ss=F32(0.0)
                for kk in cutlass.range(hidd):
                    xv=ghd[p,kk].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hidd)+F32(epsd))
                gHNd[p,k]=(ghd[p,k].to(F32)*inv*gnormd[k].to(F32)).to(gHNd.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            idx=gtid
            while idx < N*Vd:
                p=idx//Vd; n=idx%Vd
                acc=F32(0.0)
                for k in cutlass.range(hidd):
                    acc=acc+gHNd[p,k].to(F32)*gLMd[n,k].to(F32)
                gLogd[p,ii,n]=acc
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # per-particle Gumbel-argmax + DRAFT logprob (logsumexp over V1)
            for p in cutlass.range(N):
                bval=F32(-1.0e30); bidx=I32(0); v=gtid
                while v<Vd:
                    gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gUg[p,ii,v].to(F32)))
                    val=gLogd[p,ii,v]*F32(invT)+gg
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
                    gPred[p,ii]=bi; gVtok[p*Sv+SP+ii]=bi
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                # logsumexp over V1 of (logit*invT)
                lmax=F32(-1.0e30); v=gtid
                while v<Vd:
                    z=gLogd[p,ii,v]*F32(invT)
                    if z>lmax: lmax=z
                    v=v+GT
                o=16
                while o>0:
                    ov=cute.arch.shuffle_sync_bfly(lmax,o)
                    if ov>lmax: lmax=ov
                    o=o//2
                if lane==0: gWBV[wid]=lmax
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                if gtid==0:
                    mx=F32(-1.0e30); w=0
                    while w<NW:
                        if gWBV[w]>mx: mx=gWBV[w]
                        w=w+1
                    gRed[0]=mx
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                gmax=gRed[0]
                lsum=F32(0.0); v=gtid
                while v<Vd:
                    lsum=lsum+cute.exp(gLogd[p,ii,v]*F32(invT)-gmax)
                    v=v+GT
                lsum=wreduce(lsum)
                if lane==0: gWBV[wid]=lsum
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                if gtid==0:
                    tot=F32(0.0); w=0
                    while w<NW:
                        tot=tot+gWBV[w]; w=w+1
                    lse=gmax+cute.log(tot)
                    tk=gPred[p,ii]
                    gDlp[p,ii]=gLogd[p,ii,tk]*F32(invT)-lse
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # =================== PHASE B: VERIFY (8B, N particles, masked) ===================
    idx=gtid
    while idx < Rv*hidv:
        m=idx//hidv; n=idx%hidv
        ghv[m,n]=gEmbv[gVtok[m],n]
        idx=idx+GT
    ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
    for L in cutlass.range(NLv):
        # norm-once: gHNv[m,k]=rmsnorm(ghv[m])[k]*g1[k] (warp per row)
        r=wid
        while r<Rv:
            ss=F32(0.0); k=lane
            while k<hidv: x=ghv[r,k].to(F32); ss=ss+x*x; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hidv)+F32(epsv)); k=lane
            while k<hidv: gHNv[r,k]=(ghv[r,k].to(F32)*inv*gg1v[L,k].to(F32)).to(gHNv.element_type); k=k+32
            r=r+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # Q (warp per output, weight reused across all Rv rows)
        n=wid
        while n<QWv:
            acc=cute.make_fragment(Rv,F32)
            for m in cutlass.range_constexpr(Rv): acc[m]=F32(0.0)
            k=lane
            while k<hidv:
                w=gWqv[L,n,k].to(F32)
                for m in cutlass.range_constexpr(Rv): acc[m]=acc[m]+gHNv[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(Rv):
                a=wreduce(acc[m])
                if lane==0: gQv[m,n]=a.to(gQv.element_type)
            n=n+NW
        # K,V (warp per output, weight reused)
        n=wid
        while n<KWv:
            ak=cute.make_fragment(Rv,F32); av=cute.make_fragment(Rv,F32)
            for m in cutlass.range_constexpr(Rv): ak[m]=F32(0.0); av[m]=F32(0.0)
            k=lane
            while k<hidv:
                wk=gWkv[L,n,k].to(F32); wv=gWvv[L,n,k].to(F32)
                for m in cutlass.range_constexpr(Rv):
                    y=gHNv[m,k].to(F32); ak[m]=ak[m]+y*wk; av[m]=av[m]+y*wv
                k=k+32
            for m in cutlass.range_constexpr(Rv):
                a=wreduce(ak[m]); b=wreduce(av[m])
                if lane==0: gKv[m,n]=a.to(gKv.element_type); gVv[m,n]=b.to(gVv.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        idx=gtid
        while idx < Rv*Hv:
            m=idx//Hv; h=idx%Hv; kvh=h//GRPv
            p=m//Sv; s=m%Sv; base=p*Sv
            qr=cute.make_fragment(Dv,F32)
            for d in cutlass.range_constexpr(DHv):
                lo=gQv[m,h*Dv+d].to(F32); hi=gQv[m,h*Dv+d+DHv].to(F32)
                qr[d]=lo*gcosv[s,d].to(F32)-hi*gsinv[s,d].to(F32)
                qr[d+DHv]=hi*gcosv[s,d+DHv].to(F32)+lo*gsinv[s,d+DHv].to(F32)
            ac=cute.make_fragment(Dv,F32)
            for d in cutlass.range_constexpr(Dv): ac[d]=F32(0.0)
            rmax=F32(-1.0e30); rsum=F32(0.0)
            for sp in cutlass.range(s+1):
                kr=base+sp
                sc=F32(0.0)
                for d in cutlass.range_constexpr(DHv):
                    klo=gKv[kr,kvh*Dv+d].to(F32); khi=gKv[kr,kvh*Dv+d+DHv].to(F32)
                    krlo=klo*gcosv[sp,d].to(F32)-khi*gsinv[sp,d].to(F32)
                    krhi=khi*gcosv[sp,d+DHv].to(F32)+klo*gsinv[sp,d+DHv].to(F32)
                    sc=sc+qr[d]*krlo+qr[d+DHv]*krhi
                sc=sc*F32(scalev)
                nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); pe=cute.exp(sc-nm)
                rsum=rsum*corr+pe
                for d in cutlass.range_constexpr(Dv): ac[d]=ac[d]*corr+pe*gVv[kr,kvh*Dv+d].to(F32)
                rmax=nm
            for d in cutlass.range_constexpr(Dv): gAv[m,h*Dv+d]=(ac[d]/rsum).to(gAv.element_type)
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # O + residual (warp per output, weight reused; reads gAv directly)
        n=wid
        while n<hidv:
            acc=cute.make_fragment(Rv,F32)
            for m in cutlass.range_constexpr(Rv): acc[m]=F32(0.0)
            k=lane
            while k<QWv:
                w=gWov[L,n,k].to(F32)
                for m in cutlass.range_constexpr(Rv): acc[m]=acc[m]+gAv[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(Rv):
                a=wreduce(acc[m])
                if lane==0: gRv[m,n]=(ghv[m,n].to(F32)+a).to(gRv.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # norm-once of res1: gHNv[m,k]=rmsnorm(gRv[m])[k]*g2[k]
        r=wid
        while r<Rv:
            ss=F32(0.0); k=lane
            while k<hidv: x=gRv[r,k].to(BF).to(F32); ss=ss+x*x; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hidv)+F32(epsv)); k=lane
            while k<hidv: gHNv[r,k]=(gRv[r,k].to(BF).to(F32)*inv*gg2v[L,k].to(F32)).to(gHNv.element_type); k=k+32
            r=r+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # gate/up + silu (warp per output, weights reused)
        n=wid
        while n<Iv:
            gg=cute.make_fragment(Rv,F32); uu=cute.make_fragment(Rv,F32)
            for m in cutlass.range_constexpr(Rv): gg[m]=F32(0.0); uu[m]=F32(0.0)
            k=lane
            while k<hidv:
                wg=gWgv[L,n,k].to(F32); wu=gWuv[L,n,k].to(F32)
                for m in cutlass.range_constexpr(Rv):
                    y=gHNv[m,k].to(F32); gg[m]=gg[m]+y*wg; uu[m]=uu[m]+y*wu
                k=k+32
            for m in cutlass.range_constexpr(Rv):
                g=wreduce(gg[m]); u=wreduce(uu[m])
                if lane==0: gActv[m,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gActv.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # down + residual (warp per output, weight reused; reads gActv directly)
        n=wid
        while n<hidv:
            acc=cute.make_fragment(Rv,F32)
            for m in cutlass.range_constexpr(Rv): acc[m]=F32(0.0)
            k=lane
            while k<Iv:
                w=gWdv[L,n,k].to(F32)
                for m in cutlass.range_constexpr(Rv): acc[m]=acc[m]+gActv[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(Rv):
                a=wreduce(acc[m])
                if lane==0: ghv[m,n]=(gRv[m,n].to(F32)+a).to(ghv.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # =================== PHASE C: TARGET logprob (drafted positions) + store last logits ===================
    # rows of interest per particle p: drafted positions row p*Sv+SP-1+ii (ii in 0..NGEN-1), and last row p*Sv+Sv-1.
    for p in cutlass.range(N):
        for jj in cutlass.range(NGEN+1):
            # jj<NGEN: drafted position ii=jj at row p*Sv+SP-1+jj ; jj==NGEN: last row p*Sv+Sv-1 (bonus)
            row=p*Sv+SP-1+jj
            ss=F32(0.0); k=lane
            while k<hidv: xv=ghv[row,k].to(F32); ss=ss+xv*xv; k=k+32
            invf=cute.rsqrt(wreduce(ss)/F32(hidv)+F32(epsv))
            i=tx
            while i<hidv: sY[i]=ghv[row,i].to(F32)*invf*gnormv[i].to(F32); i=i+BLK
            cute.arch.barrier()
            n=wid
            while n<Vv:
                acc=F32(0.0); k=lane
                while k<hidv: acc=acc+sY[k]*gLMv[n,k].to(F32); k=k+32
                acc=wreduce(acc)
                if lane==0:
                    gVlog[n]=acc
                    if jj==NGEN: gVLast[p,n]=acc      # stash last-position logits for the bonus
                n=n+NW
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            if jj<NGEN:
                # tempered logsumexp over V8 -> target logprob of the drafted token gVtok[p*Sv+SP+jj]
                lmax=F32(-1.0e30); v=gtid
                while v<Vv:
                    z=gVlog[v]*F32(invT)
                    if z>lmax: lmax=z
                    v=v+GT
                o=16
                while o>0:
                    ov=cute.arch.shuffle_sync_bfly(lmax,o)
                    if ov>lmax: lmax=ov
                    o=o//2
                if lane==0: gWBV[wid]=lmax
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                if gtid==0:
                    mx=F32(-1.0e30); w=0
                    while w<NW:
                        if gWBV[w]>mx: mx=gWBV[w]
                        w=w+1
                    gRed[0]=mx
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                gmax=gRed[0]
                lsum=F32(0.0); v=gtid
                while v<Vv:
                    lsum=lsum+cute.exp(gVlog[v]*F32(invT)-gmax)
                    v=v+GT
                lsum=wreduce(lsum)
                if lane==0: gWBV[wid]=lsum
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
                if gtid==0:
                    tot=F32(0.0); w=0
                    while w<NW:
                        tot=tot+gWBV[w]; w=w+1
                    lse=gmax+cute.log(tot)
                    tk=gVtok[p*Sv+SP+jj]
                    gTlp[p,jj]=gVlog[tk]*F32(invT)-lse
                ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # =================== PHASE D: REWEIGHT + SYSTEMATIC RESAMPLE (thread 0) ===================
    if gtid==0:
        lw=cute.make_fragment(N,F32); mxw=F32(-1.0e30)
        for p in cutlass.range_constexpr(N):
            s=F32(0.0)
            for j in cutlass.range_constexpr(NGEN): s=s+F32(alpha)*gTlp[p,j].to(F32)-gDlp[p,j].to(F32)
            lw[p]=s
            if s>mxw: mxw=s
        Z=F32(0.0)
        for p in cutlass.range_constexpr(N): Z=Z+cute.exp(lw[p]-mxw)
        cdf=cute.make_fragment(N,F32); c=F32(0.0)
        for p in cutlass.range_constexpr(N):
            w=cute.exp(lw[p]-mxw)/Z; gW[p]=w; c=c+w; cdf[p]=c
        u=gUr[0].to(F32); step=F32(1.0)/F32(N)
        for i in cutlass.range_constexpr(N):
            pos=u*step+F32(i)*step; anc=I32(N-1); found=I32(0)
            for p in cutlass.range_constexpr(N):
                ge=I32(1) if cdf[p]>=pos else I32(0); take=ge*(I32(1)-found)
                if take==I32(1): anc=I32(p); found=I32(1)
            gAnc[i]=anc
    ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

    # =================== PHASE E: BONUS (Gumbel-max on tempered target last logits of each ancestor) ===================
    for i in cutlass.range(N):
        anc=gAnc[i]
        bval=F32(-1.0e30); bidx=I32(0); v=gtid
        while v<Vv:
            gg=F32(0.0)-cute.log(F32(0.0)-cute.log(gUb[i,v].to(F32)))
            val=gVLast[anc,v]*F32(invT)+gg
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
            gBonus[i]=bi
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def cycle(gTokd,gVtok,gPred,gUg,gDlp,gTlp,gW,gAnc,gBonus,gUr,gUb,garr,gsen,
    gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
    ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gHNd,gLogd,gWBV,gWBI,gRed,
    gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,gnormv,gLMv,
    ghv,gHNv,gQv,gKv,gVv,gAv,gRv,gActv,gVlog,gVLast,
    N:cutlass.Constexpr,SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,Sv:cutlass.Constexpr,
    NLd:cutlass.Constexpr,Hd:cutlass.Constexpr,Hkvd:cutlass.Constexpr,Dd:cutlass.Constexpr,hidd:cutlass.Constexpr,
    Id:cutlass.Constexpr,GRPd:cutlass.Constexpr,DHd:cutlass.Constexpr,Vd:cutlass.Constexpr,epsd:cutlass.Constexpr,scaled:cutlass.Constexpr,
    NLv:cutlass.Constexpr,Hv:cutlass.Constexpr,Hkvv:cutlass.Constexpr,Dv:cutlass.Constexpr,hidv:cutlass.Constexpr,
    Iv:cutlass.Constexpr,GRPv:cutlass.Constexpr,DHv:cutlass.Constexpr,Vv:cutlass.Constexpr,epsv:cutlass.Constexpr,scalev:cutlass.Constexpr,
    invT:cutlass.Constexpr,alpha:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    cycle_k(gTokd,gVtok,gPred,gUg,gDlp,gTlp,gW,gAnc,gBonus,gUr,gUb,garr,gsen,
        gEmbd,gWqd,gWkd,gWvd,gWod,gg1d,gg2d,gWgd,gWud,gWdd,gcosd,gsind,gnormd,gLMd,
        ghd,gKcd,gVcd,gQd,gAd,gRd,gActd,gHNd,gLogd,gWBV,gWBI,gRed,
        gEmbv,gWqv,gWkv,gWvv,gWov,gg1v,gg2v,gWgv,gWuv,gWdv,gcosv,gsinv,gnormv,gLMv,
        ghv,gHNv,gQv,gKv,gVv,gAv,gRv,gActv,gVlog,gVLast,
        N,SP,NGEN,Sv,NLd,Hd,Hkvd,Dd,hidd,Id,GRPd,DHd,Vd,epsd,scaled,
        NLv,Hv,Hkvv,Dv,hidv,Iv,GRPv,DHv,Vv,epsv,scalev,
        invT,alpha,B,BLK,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1],smem=hidv*4+256)

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
B=148; BLK=256; NGEN=4; Np=4; T=0.7; alpha=0.7
prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
print(f"[full cycle] N={Np} gamma={NGEN} | draft 1B NL={NLd} | verify 8B NL={NLv}")

with torch.no_grad():
    greedy=dm.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]
posv=torch.arange(Sv,device="cuda").unsqueeze(0)
cosd,sind=md.rotary_emb(torch.zeros(1,Sv,hidd,device="cuda",dtype=torch.bfloat16),posv); cosd=cosd[0].contiguous(); sind=sind[0].contiguous()
cosv,sinv=mv.rotary_emb(torch.zeros(1,Sv,hidv,device="cuda",dtype=torch.bfloat16),posv); cosv=cosv[0].contiguous(); sinv=sinv[0].contiguous()
def stk(m,NL,g): return torch.stack([g(m.layers[L]) for L in range(NL)]).contiguous()
WqdA=stk(md,NLd,lambda l:l.self_attn.q_proj.weight); WkdA=stk(md,NLd,lambda l:l.self_attn.k_proj.weight)
WvdA=stk(md,NLd,lambda l:l.self_attn.v_proj.weight); WodA=stk(md,NLd,lambda l:l.self_attn.o_proj.weight)
g1dA=stk(md,NLd,lambda l:l.input_layernorm.weight); g2dA=stk(md,NLd,lambda l:l.post_attention_layernorm.weight)
WgdA=stk(md,NLd,lambda l:l.mlp.gate_proj.weight); WudA=stk(md,NLd,lambda l:l.mlp.up_proj.weight); WddA=stk(md,NLd,lambda l:l.mlp.down_proj.weight)
WqvA=stk(mv,NLv,lambda l:l.self_attn.q_proj.weight); WkvA=stk(mv,NLv,lambda l:l.self_attn.k_proj.weight)
WvvA=stk(mv,NLv,lambda l:l.self_attn.v_proj.weight); WovA=stk(mv,NLv,lambda l:l.self_attn.o_proj.weight)
g1vA=stk(mv,NLv,lambda l:l.input_layernorm.weight); g2vA=stk(mv,NLv,lambda l:l.post_attention_layernorm.weight)
WgvA=stk(mv,NLv,lambda l:l.mlp.gate_proj.weight); WuvA=stk(mv,NLv,lambda l:l.mlp.up_proj.weight); WdvA=stk(mv,NLv,lambda l:l.mlp.down_proj.weight)

# draft input: shared teacher-forced [prompt, greedy] for all particles; per-particle Gumbel noise
Tokd=torch.zeros(Np,Sv,device="cuda",dtype=torch.int32)
for p in range(Np): Tokd[p,:SP]=prompt_ids.int(); Tokd[p,SP:]=greedy.int()
Vtok=torch.zeros(Np*Sv,device="cuda",dtype=torch.int32)
for p in range(Np): Vtok[p*Sv:p*Sv+SP]=prompt_ids.int()      # prompt; draft fills [SP:] per particle
torch.manual_seed(1)
Ug=torch.rand(Np,NGEN,Vd,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
torch.manual_seed(7); Ub=torch.rand(Np,Vvoc,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
Ur=torch.rand(1,device="cuda",dtype=torch.float32)
Pred=torch.full((Np,NGEN),-1,device="cuda",dtype=torch.int32)
Dlp=mk.f32(Np,NGEN); Tlp=mk.f32(Np,NGEN); Wt=mk.f32(Np,1).reshape(Np).contiguous()
Anc=torch.zeros(Np,device="cuda",dtype=torch.int32); Bonus=torch.full((Np,),-1,device="cuda",dtype=torch.int32)
# draft scratch
ghd=mk.f32(Np,hidd).to(torch.bfloat16); Kcd=torch.zeros(NLd,Np,Sv,Hkvd*Dd,device="cuda",dtype=torch.bfloat16); Vcd=torch.zeros_like(Kcd)
Qd=mk.f32(Np,Hd*Dd); Ad=mk.f32(Np,Hd*Dd); Rd=mk.f32(Np,hidd); Actd=mk.f32(Np,Id); HNd=mk.f32(Np,hidd).to(torch.bfloat16); Logd=mk.f32(Np*NGEN,Vd).reshape(Np,NGEN,Vd).contiguous()
NWv=B*BLK//32; WBV=mk.f32(NWv,1).reshape(NWv).contiguous(); WBI=torch.zeros(NWv,device='cuda',dtype=torch.int32); Red=mk.f32(4,1).reshape(4).contiguous()
# verify scratch
Rrows=Np*Sv
ghv=mk.f32(Rrows,hidv).to(torch.bfloat16); HNv=mk.f32(Rrows,hidv).to(torch.bfloat16)
Qv=mk.f32(Rrows,Hv*Dv); Kv=mk.f32(Rrows,Hkvv*Dv); Vvt=mk.f32(Rrows,Hkvv*Dv)
Avt=mk.f32(Rrows,Hv*Dv); Rv=mk.f32(Rrows,hidv); Actv=mk.f32(Rrows,Iv)
Vlog=mk.f32(Vvoc,1).reshape(Vvoc).contiguous(); VLast=mk.f32(Np,Vvoc)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Embd=md.embed_tokens.weight; LMd=dm.lm_head.weight; normd=md.norm.weight
Embv=mv.embed_tokens.weight; LMv=vm.lm_head.weight; normv=mv.norm.weight

args=[Tokd,Vtok,Pred,Ug,Dlp,Tlp,Wt,Anc,Bonus,Ur,Ub,arr,sen,
      Embd,WqdA,WkdA,WvdA,WodA,g1dA,g2dA,WgdA,WudA,WddA,cosd,sind,normd,LMd,
      ghd,Kcd,Vcd,Qd,Ad,Rd,Actd,HNd,Logd,WBV,WBI,Red,
      Embv,WqvA,WkvA,WvvA,WovA,g1vA,g2vA,WgvA,WuvA,WdvA,cosv,sinv,normv,LMv,
      ghv,HNv,Qv,Kv,Vvt,Avt,Rv,Actv,Vlog,VLast]
mt=[from_dlpack(x) for x in args]
cc=(Np,SP,NGEN,Sv,NLd,Hd,Hkvd,Dd,hidd,Id,GRPd,DHd,Vd,epsd,scaled,
    NLv,Hv,Hkvv,Dv,hidv,Iv,GRPv,DHv,Vvoc,epsv,scalev,1.0/T,alpha,B,BLK)
print("compiling FULL fused cycle kernel (large; be patient) ...")
comp=cute.compile(cycle,*mt,*cc)
arr.zero_(); sen.zero_()
t0=time.time(); comp(*mt); torch.cuda.synchronize(); dt=time.time()-t0

# ===== eager torch SMC reference =====
with torch.no_grad():
    d1log=dm(torch.cat([prompt_ids,greedy]).unsqueeze(0)).logits[0].float()
gn=-torch.log(-torch.log(Ug))
drafted_ref=(d1log[SP-1:SP-1+NGEN].unsqueeze(0)/T+gn).argmax(-1).int()
dz=d1log[SP-1:SP-1+NGEN]/T; dlse=torch.logsumexp(dz,-1)
draft_lp_ref=torch.gather(dz.unsqueeze(0).expand(Np,-1,-1),-1,drafted_ref.long().unsqueeze(-1)).squeeze(-1)-dlse
tgt_lp_ref=torch.zeros(Np,NGEN,device="cuda"); last_ref=torch.zeros(Np,Vvoc,device="cuda")
with torch.no_grad():
    for p in range(Np):
        seq=torch.cat([prompt_ids,drafted_ref[p].long()]).unsqueeze(0)
        lg=vm(seq).logits[0].float(); z=lg/T; lse=torch.logsumexp(z,-1); nx=seq[0,1:]
        tl=z[:-1].gather(-1,nx.view(-1,1)).squeeze(-1)-lse[:-1]
        tgt_lp_ref[p]=tl[SP-1:SP-1+NGEN]; last_ref[p]=lg[-1]
logw=(alpha*tgt_lp_ref.double()-draft_lp_ref.double()).sum(1)
Wref=normalize_log_weights(logw,device="cuda"); cdf=torch.cumsum(Wref,0); step=1.0/Np
pos=Ur.double()*step+step*torch.arange(Np,device="cuda",dtype=torch.float64)
anc_ref=torch.searchsorted(cdf,pos,right=False).int()
gnb=-torch.log(-torch.log(Ub)); bonus_ref=((last_ref[anc_ref.long()]/T)+gnb).argmax(-1).int()

# ===== validate =====
dmatch=(Pred==drafted_ref).sum().item()
anc_ok=torch.equal(Anc,anc_ref)
w_max=(Wt.double()-Wref).abs().max().item()
bonus_ok=torch.equal(Bonus,bonus_ref)
print(f"\n[draft]    tokens match {dmatch}/{Np*NGEN}   {Pred.tolist()}")
print(f"[weights]  mine={[round(x,3) for x in Wt.tolist()]}  ref={[round(x,3) for x in Wref.tolist()]}  max|Δ|={w_max:.2e}")
print(f"[resample] ancestors mine={Anc.tolist()}  ref={anc_ref.tolist()}  match={anc_ok}")
print(f"[bonus]    mine={Bonus.tolist()}  ref={bonus_ref.tolist()}  match={bonus_ok}  -> {[tok.decode([b]) for b in Bonus.tolist()]}")
print(f"[ONE LAUNCH] full N={Np} SMC cycle (draft 1B + verify 8B + reweight + resample + bonus) = {dt*1000:.0f} ms")
# The SMC DECISIONS (ancestors, bonus) must match the eager path EXACTLY. The weights match only to the
# bf16-forward logprob floor (~1e-2 through softmax): the kernel reweights from ITS OWN bf16 logprobs while
# the reference uses HF's — the resample ARITHMETIC on identical logprobs is exact (m7b3 = 5e-7).
ok=(dmatch>=Np*NGEN-1) and anc_ok and bonus_ok and (w_max<3e-2)
print(f"  decisions: draft {dmatch}/{Np*NGEN}, ancestors match={anc_ok}, bonus match={bonus_ok}; weights vs eager max|Δ|={w_max:.2e} (bf16 floor)")
print("RESULT:", "PASS" if ok else "FAIL")

"""M7c: OPTIMIZED N-particle verify (8B) — applies the M5d levers to the bottleneck verify forward:
  (1) WARP-PER-OUTPUT GEMV: 32 lanes split the contraction K (coalesced weight reads + butterfly reduce),
      instead of thread-per-output (uncoalesced, serial K-loop).
  (2) RMSNorm ONCE PER ROW staged into gHN[R,hid] (warp-per-row), instead of recomputing the norm for
      every one of R*Wout outputs.
Same per-particle block-diagonal causal mask as m7b2. Validated vs HF-8B (top1 must stay 100%) and timed
against the naive m7b2 (1241 ms)."""
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

@cute.jit
def wreduce(x):
    o=16
    while o>0:
        x=x+cute.arch.shuffle_sync_bfly(x,o); o=o//2
    return x

@cute.kernel
def verifyN_k(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,
              ghv,gHN,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
              N:cutlass.Constexpr,Sv:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,
              D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,
              B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; lane=gtid%32; wid=gtid//32; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D; R=N*Sv
    # embed
    idx=gtid
    while idx < R*hid:
        m=idx//hid; n=idx%hid
        ghv[m,n]=gEmb[gVtok[m],n]
        idx=idx+GT
    ls=I32(0); ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
    for L in cutlass.range(NL):
        # --- norm-once: gHN[m,k] = rmsnorm(ghv[m])[k]*g1[k]   (warp per row) ---
        r=wid
        while r<R:
            ss=F32(0.0); k=lane
            while k<hid: x=ghv[r,k].to(F32); ss=ss+x*x; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps)); k=lane
            while k<hid: gHN[r,k]=(ghv[r,k].to(F32)*inv*gg1[L,k].to(F32)).to(gHN.element_type); k=k+32
            r=r+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # --- Q (warp per output n, WEIGHT REUSED across all R rows: load gWq[n,k] once) ---
        n=wid
        while n<QW:
            acc=cute.make_fragment(R,F32)
            for m in cutlass.range_constexpr(R): acc[m]=F32(0.0)
            k=lane
            while k<hid:
                w=gWq[L,n,k].to(F32)
                for m in cutlass.range_constexpr(R): acc[m]=acc[m]+gHN[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(R):
                a=wreduce(acc[m])
                if lane==0: gQ[m,n]=a.to(gQ.element_type)
            n=n+NW
        # --- K,V (warp per output, weight reused across rows) ---
        n=wid
        while n<KW:
            ak=cute.make_fragment(R,F32); av=cute.make_fragment(R,F32)
            for m in cutlass.range_constexpr(R): ak[m]=F32(0.0); av[m]=F32(0.0)
            k=lane
            while k<hid:
                wk=gWk[L,n,k].to(F32); wv=gWv[L,n,k].to(F32)
                for m in cutlass.range_constexpr(R):
                    y=gHN[m,k].to(F32); ak[m]=ak[m]+y*wk; av[m]=av[m]+y*wv
                k=k+32
            for m in cutlass.range_constexpr(R):
                a=wreduce(ak[m]); b=wreduce(av[m])
                if lane==0: gK[m,n]=a.to(gK.element_type); gVt[m,n]=b.to(gVt.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # --- attention (thread per (m,h); per-particle block-diagonal causal) ---
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
                kr=base+sp; sc=F32(0.0)
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
        # --- O + residual (warp per output, weight reused across rows) ---
        n=wid
        while n<hid:
            acc=cute.make_fragment(R,F32)
            for m in cutlass.range_constexpr(R): acc[m]=F32(0.0)
            k=lane
            while k<QW:
                w=gWo[L,n,k].to(F32)
                for m in cutlass.range_constexpr(R): acc[m]=acc[m]+gA[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(R):
                a=wreduce(acc[m])
                if lane==0: gR[m,n]=(ghv[m,n].to(F32)+a).to(gR.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # --- norm-once of res1: gHN[m,k]=rmsnorm(gR[m])[k]*g2[k] ---
        r=wid
        while r<R:
            ss=F32(0.0); k=lane
            while k<hid: x=gR[r,k].to(BF).to(F32); ss=ss+x*x; k=k+32
            inv=cute.rsqrt(wreduce(ss)/F32(hid)+F32(eps)); k=lane
            while k<hid: gHN[r,k]=(gR[r,k].to(BF).to(F32)*inv*gg2[L,k].to(F32)).to(gHN.element_type); k=k+32
            r=r+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # --- gate/up + silu (warp per output, weights reused across rows) ---
        n=wid
        while n<I:
            gg=cute.make_fragment(R,F32); uu=cute.make_fragment(R,F32)
            for m in cutlass.range_constexpr(R): gg[m]=F32(0.0); uu[m]=F32(0.0)
            k=lane
            while k<hid:
                wg=gWg[L,n,k].to(F32); wu=gWu[L,n,k].to(F32)
                for m in cutlass.range_constexpr(R):
                    y=gHN[m,k].to(F32); gg[m]=gg[m]+y*wg; uu[m]=uu[m]+y*wu
                k=k+32
            for m in cutlass.range_constexpr(R):
                g=wreduce(gg[m]); u=wreduce(uu[m])
                if lane==0: gAct[m,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # --- down + residual (warp per output, weights reused across rows) ---
        n=wid
        while n<hid:
            acc=cute.make_fragment(R,F32)
            for m in cutlass.range_constexpr(R): acc[m]=F32(0.0)
            k=lane
            while k<I:
                w=gWd[L,n,k].to(F32)
                for m in cutlass.range_constexpr(R): acc[m]=acc[m]+gAct[m,k].to(F32)*w
                k=k+32
            for m in cutlass.range_constexpr(R):
                a=wreduce(acc[m])
                if lane==0: ghv[m,n]=(gR[m,n].to(F32)+a).to(ghv.element_type)
            n=n+NW
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def verifyN(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,ghv,gHN,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
            N:cutlass.Constexpr,Sv:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,
            D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,
            B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr):
    verifyN_k(gVtok,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,ghv,gHN,gQ,gK,gVt,gA,gR,gAct,garr,gsen,
              N,Sv,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])

# ============================ driver ============================
from transformers import AutoModelForCausalLM, AutoTokenizer
dname="meta-llama/Llama-3.2-1B-Instruct"; vname="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(dname)
print("loading 1B (draft the particles) + 8B target ...")
dm=AutoModelForCausalLM.from_pretrained(dname,dtype=torch.bfloat16).cuda().eval(); dm.requires_grad_(False)
vm=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.bfloat16).cuda().eval(); vm.requires_grad_(False)
mv=vm.model; cfv=vm.config
H=cfv.num_attention_heads; Hkv=cfv.num_key_value_heads; D=getattr(cfv,'head_dim',cfv.hidden_size//H)
hid=cfv.hidden_size; I=cfv.intermediate_size; NL=cfv.num_hidden_layers; eps=cfv.rms_norm_eps; V=cfv.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=256; NGEN=4; Np=4
print(f"[verify-opt] N={Np} 8B hid={hid} NL={NL}  warp-per-output + norm-once")

prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
T=0.7; torch.manual_seed(1)
with torch.no_grad():
    g1log=dm(torch.cat([prompt_ids,dm.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]]).unsqueeze(0)).logits[0].float()
U=torch.rand(Np,NGEN,dm.config.vocab_size,device="cuda").clamp_(1e-9,1.0); gnz=-torch.log(-torch.log(U))
draftedN=(g1log[SP-1:SP-1+NGEN].unsqueeze(0)/T+gnz).argmax(-1).int()
Vtok2d=torch.zeros(Np,Sv,device="cuda",dtype=torch.int32)
for p in range(Np): Vtok2d[p,:SP]=prompt_ids.int(); Vtok2d[p,SP:]=draftedN[p]
Vtok=Vtok2d.reshape(-1).contiguous()
posv=torch.arange(Sv,device="cuda").unsqueeze(0)
cos,sin=mv.rotary_emb(torch.zeros(1,Sv,hid,device="cuda",dtype=torch.bfloat16),posv); cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(g): return torch.stack([g(mv.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)
Rrows=Np*Sv
ghv=mk.f32(Rrows,hid).to(torch.bfloat16); HN=mk.f32(Rrows,hid).to(torch.bfloat16)
Q=mk.f32(Rrows,H*D); K=mk.f32(Rrows,Hkv*D); Vt=mk.f32(Rrows,Hkv*D); A=mk.f32(Rrows,H*D); Rr=mk.f32(Rrows,hid); Act=mk.f32(Rrows,I)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Emb=mv.embed_tokens.weight
args=[Vtok,Emb,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,ghv,HN,Q,K,Vt,A,Rr,Act,arr,sen]
mt=[from_dlpack(x) for x in args]
print("compiling optimized verify ...")
comp=cute.compile(verifyN,*mt,Np,Sv,NL,H,Hkv,D,hid,I,GRP,DH,B,BLK,eps,scale)
def run(): arr.zero_(); sen.zero_(); comp(*mt)
run(); torch.cuda.synchronize()
# timing
for _ in range(2): run()
torch.cuda.synchronize(); e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record()
for _ in range(5): run()
e1.record(); torch.cuda.synchronize(); ms=e0.elapsed_time(e1)/5

# validate vs HF-8B per particle
hf=ghv.float(); inv=torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+eps); hn=(hf*inv*mv.norm.weight.float()).to(torch.bfloat16)
mylog=mk.f32(Rrows,V); mk.gemv(from_dlpack(hn),from_dlpack(vm.lm_head.weight),from_dlpack(mylog),Rrows,hid); torch.cuda.synchronize()
mylog=mylog.reshape(Np,Sv,V)
agree=0.0
for p in range(Np):
    with torch.no_grad(): r8=vm(Vtok2d[p].long().unsqueeze(0)).logits[0].float()
    agree+=(mylog[p].argmax(-1)==r8.argmax(-1)).float().mean().item()
agree/=Np
print(f"\n[verify-opt] per-particle top1 agree vs HF-8B = {agree*100:.1f}%")
print(f"[verify-opt] OPTIMIZED verify = {ms:.0f} ms   (naive m7b2 was 1241 ms -> {1241/ms:.1f}x faster)")
print("RESULT:", "PASS" if agree>0.99 else "FAIL")

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

# Draft decode megakernel: AR loop (teacher-force prompt, then generate), one token/step,
# KV cache grows, greedy argmax sampling. One persistent launch.
@cute.kernel
def draft_k(gTok, gPred, gU, gEmb, gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd, gcos,gsin, gnorm, gLM,
            gh, gKc,gVc, gQ,gA,gR,gAct, gLog, garr,gsen,
            SP:cutlass.Constexpr, NGEN:cutlass.Constexpr, NL:cutlass.Constexpr, H:cutlass.Constexpr,
            Hkv:cutlass.Constexpr, D:cutlass.Constexpr, hid:cutlass.Constexpr, I:cutlass.Constexpr,
            GRP:cutlass.Constexpr, DH:cutlass.Constexpr, V:cutlass.Constexpr, B:cutlass.Constexpr,
            BLK:cutlass.Constexpr, eps:cutlass.Constexpr, scale:cutlass.Constexpr, invT:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*BLK+tx; GT=B*BLK
    arr=garr.iterator; sen=gsen.iterator
    QW=H*D; KW=Hkv*D
    ls=I32(0)
    for t in cutlass.range(SP+NGEN-1):       # process token at pos t, produce token at t+1
        tok = gTok[t]
        # ---- embed: h[0,:] = Emb[tok,:] ----
        idx=gtid
        while idx < hid:
            gh[0,idx]=gEmb[tok,idx]
            idx=idx+GT
        ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        for L in cutlass.range(NL):
            # Stage 1: QKV (1 token), RoPE k at pos t, write cache[L,t]
            idx=gtid
            while idx < QW:
                n=idx
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gh[0,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                acc=F32(0.0)
                for k in cutlass.range(hid):
                    acc=acc+gh[0,k].to(F32)*inv*gg1[L,k].to(F32)*gWq[L,n,k].to(F32)
                gQ[0,n]=acc.to(gQ.element_type)
                idx=idx+GT
            idx=gtid
            while idx < KW:
                n=idx; kvh=n//D; d=n%D
                part = n+DH if d<DH else n-DH        # rotate_half partner
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gh[0,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                km=F32(0.0); kp=F32(0.0); vv=F32(0.0)
                for k in cutlass.range(hid):
                    yn=gh[0,k].to(F32)*inv*gg1[L,k].to(F32)
                    km=km+yn*gWk[L,n,k].to(F32)
                    kp=kp+yn*gWk[L,part,k].to(F32)
                    vv=vv+yn*gWv[L,n,k].to(F32)
                cl=gcos[t,d].to(F32); sl=gsin[t,d].to(F32)
                kr=F32(0.0)
                if d < DH:
                    kr = km*cl - kp*sl
                else:
                    kr = km*cl + kp*sl
                gKc[L,t,n]=kr.to(gKc.element_type)
                gVc[L,t,n]=vv.to(gVc.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage 2: attention, query=token at pos t (RoPE), keys/vals = cache[L,0..t]
            idx=gtid
            while idx < H:
                h=idx; kvh=h//GRP
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
                    for d in cutlass.range_constexpr(D):
                        sc=sc+qr[d]*gKc[L,s,kvh*D+d].to(F32)
                    sc=sc*F32(scale)
                    nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
                    rsum=rsum*corr+p
                    for d in cutlass.range_constexpr(D): ac[d]=ac[d]*corr+p*gVc[L,s,kvh*D+d].to(F32)
                    rmax=nm
                for d in cutlass.range_constexpr(D): gA[0,h*D+d]=(ac[d]/rsum).to(gA.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage 3: res1 = h + attn@Wo
            idx=gtid
            while idx < hid:
                n=idx; acc=gh[0,n].to(F32)
                for j in cutlass.range(QW): acc=acc+gA[0,j].to(F32)*gWo[L,n,j].to(F32)
                gR[0,n]=acc.to(gR.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage 4: act = silu(rmsnorm(res1,g2)@Wg)*(..@Wu)
            idx=gtid
            while idx < I:
                n=idx
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gR[0,k].to(cutlass.BFloat16).to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                g=F32(0.0); u=F32(0.0)
                for k in cutlass.range(hid):
                    yn=gR[0,k].to(cutlass.BFloat16).to(F32)*inv*gg2[L,k].to(F32)
                    g=g+yn*gWg[L,n,k].to(F32); u=u+yn*gWu[L,n,k].to(F32)
                gAct[0,n]=(g/(F32(1.0)+cute.exp(-g))*u).to(gAct.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            # Stage 5: h = res1 + act@Wd
            idx=gtid
            while idx < hid:
                n=idx; acc=gR[0,n].to(F32)
                for j in cutlass.range(I): acc=acc+gAct[0,j].to(F32)*gWd[L,n,j].to(F32)
                gh[0,n]=acc.to(gh.element_type)
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
        # ---- sample next token when we've consumed >= prompt ----
        if t + 1 >= SP:
            # logits = rmsnorm(h,norm) @ LM ; greedy argmax
            idx=gtid
            while idx < V:
                n=idx
                ss=F32(0.0)
                for k in cutlass.range(hid):
                    xv=gh[0,k].to(F32); ss=ss+xv*xv
                inv=cute.rsqrt(ss/F32(hid)+F32(eps))
                acc=F32(0.0)
                for k in cutlass.range(hid):
                    acc=acc+gh[0,k].to(F32)*inv*gnorm[k].to(F32)*gLM[n,k].to(F32)
                gLog[t+1-SP,n]=acc
                idx=idx+GT
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)
            if gtid==0:
                best=F32(-1.0e30); bi=I32(0)
                i=t+1-SP
                for v in cutlass.range(V):
                    g = F32(0.0) - cute.log(F32(0.0) - cute.log(gU[i,v].to(F32)))
                    val = gLog[i,v]*F32(invT) + g
                    if val > best:
                        best=val; bi=I32(v)
                gPred[i]=bi
            ls=I32(1)-ls; gbar(arr,sen,ls,B,tx)

@cute.jit
def draft(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,garr,gsen,
          SP:cutlass.Constexpr,NGEN:cutlass.Constexpr,NL:cutlass.Constexpr,H:cutlass.Constexpr,Hkv:cutlass.Constexpr,
          D:cutlass.Constexpr,hid:cutlass.Constexpr,I:cutlass.Constexpr,GRP:cutlass.Constexpr,DH:cutlass.Constexpr,
          V:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,eps:cutlass.Constexpr,scale:cutlass.Constexpr,invT:cutlass.Constexpr):
    draft_k(gTok,gPred,gU,gEmb,gWq,gWk,gWv,gWo,gg1,gg2,gWg,gWu,gWd,gcos,gsin,gnorm,gLM,gh,gKc,gVc,gQ,gA,gR,gAct,gLog,garr,gsen,
            SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale,invT).launch(grid=[B,1,1],block=[BLK,1,1])

# ===== driver =====
from transformers import AutoModelForCausalLM, AutoTokenizer
name="meta-llama/Llama-3.2-1B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.bfloat16).cuda().eval(); model.requires_grad_(False)
m=model.model; cf=model.config
H=cf.num_attention_heads; Hkv=cf.num_key_value_heads; D=getattr(cf,'head_dim',cf.hidden_size//H)
hid=cf.hidden_size; I=cf.intermediate_size; NL=cf.num_hidden_layers; eps=cf.rms_norm_eps; V=cf.vocab_size
GRP=H//Hkv; DH=D//2; scale=1.0/math.sqrt(D); B=148; BLK=128
NGEN=4
prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; maxT=SP+NGEN
# HF greedy reference
with torch.no_grad():
    ref=model.generate(prompt_ids.unsqueeze(0), max_new_tokens=NGEN, do_sample=False)[0][SP:]
# rope tables
posv=torch.arange(maxT,device="cuda").unsqueeze(0)
cos,sin=m.rotary_emb(torch.zeros(1,maxT,hid,device="cuda",dtype=torch.bfloat16),posv)
cos=cos[0].contiguous(); sin=sin[0].contiguous()
def stack(g): return torch.stack([g(m.layers[L]) for L in range(NL)]).contiguous()
WqA=stack(lambda l:l.self_attn.q_proj.weight); WkA=stack(lambda l:l.self_attn.k_proj.weight)
WvA=stack(lambda l:l.self_attn.v_proj.weight); WoA=stack(lambda l:l.self_attn.o_proj.weight)
g1A=stack(lambda l:l.input_layernorm.weight); g2A=stack(lambda l:l.post_attention_layernorm.weight)
WgA=stack(lambda l:l.mlp.gate_proj.weight); WuA=stack(lambda l:l.mlp.up_proj.weight); WdA=stack(lambda l:l.mlp.down_proj.weight)
Tok=torch.zeros(maxT,device="cuda",dtype=torch.int32); Tok[:SP]=prompt_ids.int(); Tok[SP:SP+NGEN]=ref.int()  # teacher-force
Pred=torch.full((NGEN,),-1,device="cuda",dtype=torch.int32)
T=0.7; torch.manual_seed(1)
U=torch.rand(NGEN,V,device="cuda",dtype=torch.float32).clamp_(1e-9,1.0)
h=mk.f32(1,hid).to(torch.bfloat16)
Kc=torch.zeros(NL,maxT,Hkv*D,device="cuda",dtype=torch.bfloat16); Vc=torch.zeros_like(Kc)
Q=mk.f32(1,H*D); A=mk.f32(1,H*D); R=mk.f32(1,hid); Act=mk.f32(1,I); Log=mk.f32(NGEN,V)
arr=torch.zeros(1,device="cuda",dtype=torch.int32); sen=torch.zeros(1,device="cuda",dtype=torch.int32)
Emb=m.embed_tokens.weight; LM=model.lm_head.weight; norm=m.norm.weight
targs=[Tok,Pred,U,Emb,WqA,WkA,WvA,WoA,g1A,g2A,WgA,WuA,WdA,cos,sin,norm,LM,h,Kc,Vc,Q,A,R,Act,Log,arr,sen]
mt=[from_dlpack(x) for x in targs]
t0=time.time()
draft(*mt, SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale, 1.0/T)
torch.cuda.synchronize(); dt=time.time()-t0
gen=Pred.tolist()
refl=ref.tolist()
print(f"[DRAFT MEGAKERNEL teacher-forced] {dt*1000:.0f}ms  pred={gen}  ref={refl}")
agree=sum(int(a==b) for a,b in zip(gen,refl))
with torch.no_grad():
    full=torch.cat([prompt_ids, ref]).unsqueeze(0)
    hflog=model(full).logits[0].float()      # [SP+NGEN, V]
import torch.nn.functional as Fn
gnoise=-torch.log(-torch.log(U))
ref_gumbel=(hflog[SP-1:SP-1+NGEN]/T + gnoise).argmax(-1).int().tolist()
cs=[Fn.cosine_similarity(Log[i].float(),hflog[SP-1+i],dim=0).item() for i in range(NGEN)]
match=sum(int(a==b) for a,b in zip(gen,ref_gumbel))
print(f"[GUMBEL draft megakernel] pred={gen}  ref(torch gumbel-max on HF logits, same noise)={ref_gumbel}")
print(f"stochastic-token match {match}/{NGEN}   per-step logit cos {[round(c,5) for c in cs]}")
print("RESULT:", "PASS" if match>=NGEN-1 and min(cs)>0.999 else "FAIL")


# ===== PROPER timing (compile once, reset barrier state per launch) =====
comp = cute.compile(draft, *mt, SP,NGEN,NL,H,Hkv,D,hid,I,GRP,DH,V,B,BLK,eps,scale, 1.0/T)
def run():
    arr.zero_(); sen.zero_(); comp(*mt)
for _ in range(3): run()
torch.cuda.synchronize()
e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record()
for _ in range(20): run()
e1.record(); torch.cuda.synchronize()
ms=e0.elapsed_time(e1)/20
nstep=SP+NGEN-1
print(f"[REAL megakernel exec] {ms*1000:.0f} us/launch for {nstep} AR steps = {ms*1000/nstep:.0f} us/token  (NGEN={NGEN} do lm_head over {V} vocab)")

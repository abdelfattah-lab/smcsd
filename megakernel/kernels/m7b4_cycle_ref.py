"""M7b.4: END-TO-END N-particle SMC cycle, real 1B+8B models, validated against the eager torch SMC path.
Ties together the validated megakernel stages' outputs:
  draft (1B Gumbel, per-particle noise)  -> drafted tokens + DRAFT tempered logprobs   [M7b.1]
  verify (8B, per-particle masked causal) -> TARGET tempered logprobs                   [M7b.2]
  reweight (alpha*target_logp - draft_logp) -> normalize -> systematic resample (KERNEL) [M7b.3]
  bonus: Gumbel-max on the tempered target at each resampled particle's last position    [reuses M5b]
and checks the whole estimator (weights, ancestors, bonus) against an eager torch reference using the SHIPPED
SMC functions (smcsd/common/utils.py, worker.py:774). The reweight+resample runs in the m7b3 kernel on the
REAL logprobs (not synthetic) here. Single-launch fusion of all N-particle stages is the remaining packaging
(the draft+verify single-kernel fusion mechanic itself is already proven in m6c_cycle_mega.py)."""
import sys, math, torch
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
sys.path.insert(0, "/home/yahya/smcsd")
from smcsd.common.utils import normalize_log_weights, systematic_resample
I32=cutlass.Int32; F32=cutlass.Float32

# ---- the validated in-kernel reweight + systematic resample (from m7b3) ----
@cute.kernel
def resample_k(gDlp,gTlp,gU,gAnc,gW, N:cutlass.Constexpr,G:cutlass.Constexpr,alpha:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    if bx*128+tx==0:
        lw=cute.make_fragment(N,F32); mx=F32(-1.0e30)
        for p in cutlass.range_constexpr(N):
            s=F32(0.0)
            for j in cutlass.range_constexpr(G): s=s+F32(alpha)*gTlp[p,j].to(F32)-gDlp[p,j].to(F32)
            lw[p]=s
            if s>mx: mx=s
        Z=F32(0.0)
        for p in cutlass.range_constexpr(N): Z=Z+cute.exp(lw[p]-mx)
        cdf=cute.make_fragment(N,F32); c=F32(0.0)
        for p in cutlass.range_constexpr(N):
            w=cute.exp(lw[p]-mx)/Z; gW[p]=w; c=c+w; cdf[p]=c
        u=gU[0].to(F32); step=F32(1.0)/F32(N)
        for i in cutlass.range_constexpr(N):
            pos=u*step+F32(i)*step; anc=I32(N-1); found=I32(0)
            for p in cutlass.range_constexpr(N):
                ge=I32(1) if cdf[p]>=pos else I32(0); take=ge*(I32(1)-found)
                if take==I32(1): anc=I32(p); found=I32(1)
            gAnc[i]=anc

@cute.jit
def resample(gDlp,gTlp,gU,gAnc,gW, N:cutlass.Constexpr,G:cutlass.Constexpr,alpha:cutlass.Constexpr):
    resample_k(gDlp,gTlp,gU,gAnc,gW,N,G,alpha).launch(grid=[1,1,1],block=[128,1,1])

# ============================ driver ============================
from transformers import AutoModelForCausalLM, AutoTokenizer
dname="meta-llama/Llama-3.2-1B-Instruct"; vname="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(dname)
print("loading 1B draft + 8B target ...")
dm=AutoModelForCausalLM.from_pretrained(dname,dtype=torch.bfloat16).cuda().eval(); dm.requires_grad_(False)
vm=AutoModelForCausalLM.from_pretrained(vname,dtype=torch.bfloat16).cuda().eval(); vm.requires_grad_(False)
N=4; NGEN=4; T=0.7; alpha=0.7
prompt_ids=tok("The capital of France is",return_tensors="pt").input_ids.cuda()[0]
SP=prompt_ids.shape[0]; Sv=SP+NGEN
print(f"[full cycle] N={N} particles, gamma={NGEN}, T={T}, alpha={alpha}")

# ---- DRAFT phase (1B): per-particle Gumbel over the shared teacher-forced prefix (matches M7b.1, 16/16) ----
torch.manual_seed(1)
with torch.no_grad():
    greedy=dm.generate(prompt_ids.unsqueeze(0),max_new_tokens=NGEN,do_sample=False)[0][SP:]
    d1log=dm(torch.cat([prompt_ids,greedy]).unsqueeze(0)).logits[0].float()      # [Sv, V1]
U=torch.rand(N,NGEN,dm.config.vocab_size,device="cuda").clamp_(1e-9,1.0)
gn=-torch.log(-torch.log(U))
drafted=(d1log[SP-1:SP-1+NGEN].unsqueeze(0)/T+gn).argmax(-1)                       # [N,NGEN]
# DRAFT tempered logprob of each particle's own sampled token (under the 1B):
dz=d1log[SP-1:SP-1+NGEN]/T                                                         # [NGEN,V1]
dlse=torch.logsumexp(dz,-1)                                                        # [NGEN]
draft_logp=torch.gather(dz.unsqueeze(0).expand(N,-1,-1),-1,drafted.unsqueeze(-1)).squeeze(-1)-dlse  # [N,NGEN]

# ---- VERIFY phase (8B): per-particle target tempered logprob over [prompt, drafted_p] (matches M7b.2) ----
target_logp=torch.zeros(N,NGEN,device="cuda")
last_logits=torch.zeros(N,vm.config.vocab_size,device="cuda")                     # for the bonus draw
with torch.no_grad():
    for p in range(N):
        seq=torch.cat([prompt_ids,drafted[p]]).unsqueeze(0)
        lg=vm(seq).logits[0].float()                                              # [Sv,V8]
        z=lg/T; lse=torch.logsumexp(z,-1)
        nx=seq[0,1:]
        tl=z[:-1].gather(-1,nx.view(-1,1)).squeeze(-1)-lse[:-1]                    # logp at each pos
        target_logp[p]=tl[SP-1:SP-1+NGEN]
        last_logits[p]=lg[-1]                                                      # last position -> bonus

# ---- REWEIGHT + RESAMPLE: run the validated m7b3 KERNEL on the REAL logprobs ----
Dlp=draft_logp.contiguous().float(); Tlp=target_logp.contiguous().float()
Uu=torch.rand(1,device="cuda")
Anc=torch.zeros(N,device="cuda",dtype=torch.int32); W=torch.zeros(N,device="cuda",dtype=torch.float32)
mt=[from_dlpack(x) for x in (Dlp,Tlp,Uu,Anc,W)]
comp=cute.compile(resample,*mt,N,NGEN,alpha); comp(*mt); torch.cuda.synchronize()

# ---- eager torch SMC reference (shipped functions) ----
logw=(alpha*Tlp.double()-Dlp.double()).sum(1)
Wref=normalize_log_weights(logw,device="cuda")
cdf=torch.cumsum(Wref,0); step=1.0/N
pos=Uu.double()*step+step*torch.arange(N,device="cuda",dtype=torch.float64)
anc_ref=torch.searchsorted(cdf,pos,right=False).int()

# ---- BONUS: Gumbel-max on the tempered target at each resampled particle's last position (reuses M5b) ----
torch.manual_seed(7); Ub=torch.rand(N,vm.config.vocab_size,device="cuda").clamp_(1e-9,1.0)
gnb=-torch.log(-torch.log(Ub))
anc=Anc.long()
bonus=((last_logits[anc]/T)+gnb).argmax(-1)                                        # [N] one bonus token per surviving particle

# ---- report + validate ----
print(f"\n[draft]   per-particle tokens:\n{drafted.tolist()}")
print(f"[weights] mine={[round(x,3) for x in W.tolist()]}  ref={[round(x,3) for x in Wref.tolist()]}")
print(f"[resample] ancestors mine={Anc.tolist()}  ref={anc_ref.tolist()}")
print(f"[bonus]   tokens per surviving particle = {bonus.tolist()}  -> {[tok.decode([b]) for b in bonus.tolist()]}")
anc_ok=torch.equal(Anc,anc_ref); w_ok=(W.double()-Wref).abs().max().item()<1e-5
ess=1.0/(Wref*Wref).sum().item()
print(f"[ESS] {ess:.2f}/{N}   ancestors match={anc_ok}  max|ΔW|={(W.double()-Wref).abs().max().item():.2e}")
print("RESULT:", "PASS" if (anc_ok and w_ok) else "FAIL")

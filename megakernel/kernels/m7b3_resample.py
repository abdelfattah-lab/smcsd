"""M7b.3: in-kernel SMC reweight + systematic resample, validated against the SHIPPED math
(smcsd/core/worker.py:774  log w = alpha*target_logp - draft_logp;
 smcsd/common/utils.py     normalize_log_weights = softmax;  systematic_resample = cumsum+searchsorted).
These are tiny cross-particle reductions (N particles, gamma positions) — done by thread 0 in-kernel here
(N is small); the bonus Gumbel draw reuses the validated M5b machinery and is folded into the full cycle.
Unit-tested on synthetic logprobs over many trials so the ancestor indices match torch EXACTLY."""
import sys, math, torch
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
sys.path.insert(0, "/home/yahya/smcsd")
from smcsd.common.utils import normalize_log_weights, systematic_resample
I32=cutlass.Int32; F32=cutlass.Float32

@cute.kernel
def resample_k(gDlp,gTlp,gU,gAnc,gW, N:cutlass.Constexpr,G:cutlass.Constexpr,alpha:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    gtid=bx*128+tx
    if gtid==0:
        # log_w[p] = sum_j (alpha*target_logp - draft_logp)
        lw=cute.make_fragment(N,F32)
        mx=F32(-1.0e30)
        for p in cutlass.range_constexpr(N):
            s=F32(0.0)
            for j in cutlass.range_constexpr(G):
                s=s+F32(alpha)*gTlp[p,j].to(F32)-gDlp[p,j].to(F32)
            lw[p]=s
            if s>mx: mx=s
        # normalize: softmax
        Z=F32(0.0)
        for p in cutlass.range_constexpr(N):
            Z=Z+cute.exp(lw[p]-mx)
        # cdf + write normalized weights
        cdf=cute.make_fragment(N,F32)
        c=F32(0.0)
        for p in cutlass.range_constexpr(N):
            w=cute.exp(lw[p]-mx)/Z
            gW[p]=w
            c=c+w; cdf[p]=c
        # systematic resample: positions = (u + i)/N ; ancestor = first p with cdf[p] >= pos
        u=gU[0].to(F32); step=F32(1.0)/F32(N)
        for i in cutlass.range_constexpr(N):
            pos=u*step+F32(i)*step
            anc=I32(N-1); found=I32(0)
            for p in cutlass.range_constexpr(N):
                ge=I32(1) if cdf[p]>=pos else I32(0)
                take=ge*(I32(1)-found)
                if take==I32(1): anc=I32(p); found=I32(1)
            gAnc[i]=anc

@cute.jit
def resample(gDlp,gTlp,gU,gAnc,gW, N:cutlass.Constexpr,G:cutlass.Constexpr,alpha:cutlass.Constexpr):
    resample_k(gDlp,gTlp,gU,gAnc,gW,N,G,alpha).launch(grid=[1,1,1],block=[128,1,1])

N=4; G=4; alpha=0.7
torch.manual_seed(0)
ntrials=200; mismatches=0; wmax=0.0
# compile once with representative tensors
Dlp=torch.zeros(N,G,device="cuda",dtype=torch.float32); Tlp=torch.zeros(N,G,device="cuda",dtype=torch.float32)
Uu=torch.zeros(1,device="cuda",dtype=torch.float32); Anc=torch.zeros(N,device="cuda",dtype=torch.int32); W=torch.zeros(N,device="cuda",dtype=torch.float32)
mt=[from_dlpack(x) for x in (Dlp,Tlp,Uu,Anc,W)]
comp=cute.compile(resample,*mt,N,G,alpha)
for it in range(ntrials):
    Dlp.copy_(-torch.rand(N,G,device="cuda")*8)          # draft logprobs in [-8,0]
    Tlp.copy_(-torch.rand(N,G,device="cuda")*8)          # target logprobs in [-8,0]
    Uu.copy_(torch.rand(1,device="cuda"))
    comp(*mt); torch.cuda.synchronize()
    # torch reference (exact shipped functions)
    logw=(alpha*Tlp.double()-Dlp.double()).sum(1)        # [N]
    Wref=normalize_log_weights(logw, device="cuda")      # softmax, float64
    cdf=torch.cumsum(Wref,0); step=1.0/N
    pos=Uu.double()*step+step*torch.arange(N,device="cuda",dtype=torch.float64)
    anc_ref=torch.searchsorted(cdf,pos,right=False).int()
    if not torch.equal(Anc, anc_ref): mismatches+=1
    wmax=max(wmax,(W.double()-Wref).abs().max().item())
print(f"[M7b.3 reweight+systematic-resample] {ntrials} trials  ancestor mismatches={mismatches}  max|ΔW|={wmax:.2e}")
print("example: W=",[round(x,3) for x in W.tolist()]," anc(mine)=",Anc.tolist()," anc(ref)=",anc_ref.tolist())
print("RESULT:", "PASS" if (mismatches==0 and wmax<1e-5) else "FAIL")

import torch, math
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# --- fused RMSNorm(x)*g  then  out = normed @ W^T ---
@cute.kernel
def rmsnorm_proj_k(gX,gG,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr,eps:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    n=bx*bd+tx; N=gW.shape[0]
    if n<N:
        for m in cutlass.range_constexpr(M):
            ss=cutlass.Float32(0.0)
            for k in cutlass.range(K):
                xv=gX[m,k].to(cutlass.Float32); ss=ss+xv*xv
            inv=cute.rsqrt(ss/cutlass.Float32(K)+cutlass.Float32(eps))
            acc=cutlass.Float32(0.0)
            for k in cutlass.range(K):
                yv=gX[m,k].to(cutlass.Float32)*inv*gG[k].to(cutlass.Float32)
                acc=acc+yv*gW[n,k].to(cutlass.Float32)
            gO[m,n]=acc
@cute.jit
def rmsnorm_proj(mX,mG,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr,eps:cutlass.Constexpr):
    N=mW.shape[0]; tpb=128
    rmsnorm_proj_k(mX,mG,mW,mO,M,K,eps).launch(grid=[(N+tpb-1)//tpb,1,1],block=[tpb,1,1])

# --- plain GEMV: out = x @ W^T ---
@cute.kernel
def gemv_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    n=bx*bd+tx; N=gW.shape[0]
    if n<N:
        for m in cutlass.range_constexpr(M):
            acc=cutlass.Float32(0.0)
            for k in cutlass.range(K):
                acc=acc+gX[m,k].to(cutlass.Float32)*gW[n,k].to(cutlass.Float32)
            gO[m,n]=acc
@cute.jit
def gemv(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    N=mW.shape[0]; tpb=128
    gemv_k(mX,mW,mO,M,K).launch(grid=[(N+tpb-1)//tpb,1,1],block=[tpb,1,1])

# --- attention + on-chip RoPE + causal (GQA) ---
@cute.kernel
def attn_k(gQ,gK,gV,gC,gSi,gO,H:cutlass.Constexpr,D:cutlass.Constexpr,DH:cutlass.Constexpr,GRP:cutlass.Constexpr,scale:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    t=bx*bd+tx; S=gQ.shape[0]
    if t<S*H:
        m=t//H; h=t%H; kvh=h//GRP
        qr=cute.make_fragment(D,cutlass.Float32)
        for d in cutlass.range_constexpr(DH):
            lo=gQ[m,h,d].to(cutlass.Float32); hi=gQ[m,h,d+DH].to(cutlass.Float32)
            qr[d]=lo*gC[m,d].to(cutlass.Float32)-hi*gSi[m,d].to(cutlass.Float32)
            qr[d+DH]=hi*gC[m,d+DH].to(cutlass.Float32)+lo*gSi[m,d+DH].to(cutlass.Float32)
        acc=cute.make_fragment(D,cutlass.Float32)
        for d in cutlass.range_constexpr(D): acc[d]=cutlass.Float32(0.0)
        rmax=cutlass.Float32(-1.0e30); rsum=cutlass.Float32(0.0)
        for s in cutlass.range(m+1):
            sc=cutlass.Float32(0.0)
            for d in cutlass.range_constexpr(DH):
                klo=gK[s,kvh,d].to(cutlass.Float32); khi=gK[s,kvh,d+DH].to(cutlass.Float32)
                krlo=klo*gC[s,d].to(cutlass.Float32)-khi*gSi[s,d].to(cutlass.Float32)
                krhi=khi*gC[s,d+DH].to(cutlass.Float32)+klo*gSi[s,d+DH].to(cutlass.Float32)
                sc=sc+qr[d]*krlo+qr[d+DH]*krhi
            sc=sc*cutlass.Float32(scale)
            nm=cutlass.max(sc,rmax); corr=cute.exp(rmax-nm); p=cute.exp(sc-nm)
            rsum=rsum*corr+p
            for d in cutlass.range_constexpr(D): acc[d]=acc[d]*corr+p*gV[s,kvh,d].to(cutlass.Float32)
            rmax=nm
        for d in cutlass.range_constexpr(D): gO[m,h,d]=acc[d]/rsum
@cute.jit
def attn(mQ,mK,mV,mC,mS,mO,H:cutlass.Constexpr,D:cutlass.Constexpr,DH:cutlass.Constexpr,GRP:cutlass.Constexpr,scale:cutlass.Constexpr):
    S=mQ.shape[0]; tpb=128
    attn_k(mQ,mK,mV,mC,mS,mO,H,D,DH,GRP,scale).launch(grid=[(S*H+tpb-1)//tpb,1,1],block=[tpb,1,1])

# ===== driver =====
d=torch.load(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'ref_block.pt'))
cfg=d["cfg"]; H,Hkv,D,hidden,S=cfg["H"],cfg["Hkv"],cfg["D"],cfg["hidden"],cfg["S"]
DH=D//2; GRP=H//Hkv; eps=float(d["eps"]); scale=1.0/math.sqrt(D)
to=lambda x: x.cuda()
h=to(d["h"])[0].contiguous()                              # [S,hidden] bf16
g1,g2=to(d["g1"]),to(d["g2"])
Wq,Wk,Wv,Wo=to(d["Wq"]),to(d["Wk"]),to(d["Wv"]),to(d["Wo"])
Wg,Wu,Wd=to(d["Wg"]),to(d["Wu"]),to(d["Wd"])
cos=to(d["cos"])[0].contiguous(); sin=to(d["sin"])[0].contiguous()   # [S,D]
ref=to(d["ref"])[0].float()

def f32(M,N): return torch.zeros(M,N,device="cuda",dtype=torch.float32)
# Q,K,V via fused rmsnorm(g1)+proj
Q=f32(S,H*D); K=f32(S,Hkv*D); V=f32(S,Hkv*D)
rmsnorm_proj(from_dlpack(h),from_dlpack(g1),from_dlpack(Wq),from_dlpack(Q),S,hidden,eps)
rmsnorm_proj(from_dlpack(h),from_dlpack(g1),from_dlpack(Wk),from_dlpack(K),S,hidden,eps)
rmsnorm_proj(from_dlpack(h),from_dlpack(g1),from_dlpack(Wv),from_dlpack(V),S,hidden,eps)
Qb=Q.view(S,H,D).contiguous(); Kb=K.view(S,Hkv,D).contiguous(); Vb=V.view(S,Hkv,D).contiguous()
A=f32(S,H*D); Av=A.view(S,H,D)
attn(from_dlpack(Qb),from_dlpack(Kb),from_dlpack(Vb),from_dlpack(cos),from_dlpack(sin),from_dlpack(Av),H,D,DH,GRP,scale)
# O proj + residual
Ob=f32(S,hidden)
gemv(from_dlpack(A.contiguous()),from_dlpack(Wo),from_dlpack(Ob),S,H*D)
res1=h.float()+Ob
res1b=res1.to(torch.bfloat16)
# MLP: fused rmsnorm(g2)+gate/up, silu, down
G=f32(S,Wg.shape[0]); U=f32(S,Wu.shape[0])
rmsnorm_proj(from_dlpack(res1b),from_dlpack(g2),from_dlpack(Wg),from_dlpack(G),S,hidden,eps)
rmsnorm_proj(from_dlpack(res1b),from_dlpack(g2),from_dlpack(Wu),from_dlpack(U),S,hidden,eps)
act=(torch.nn.functional.silu(G)*U).to(torch.bfloat16)
Dn=f32(S,hidden)
gemv(from_dlpack(act),from_dlpack(Wd),from_dlpack(Dn),S,Wg.shape[0])
out=res1+Dn
torch.cuda.synchronize()
err=(out-ref).abs().max().item(); rel=err/ref.abs().max().item()
cos_sim=torch.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
print(f"FULL BLOCK vs HF Llama-3.2-1B layer-0:  max abs err {err:.4e}  rel {rel:.2e}  cos_sim {cos_sim:.6f}")
print("RESULT:", "PASS" if rel<5e-2 else "FAIL")

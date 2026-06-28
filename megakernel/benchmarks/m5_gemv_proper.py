import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32
@cute.kernel
def naive_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    n=bx*bd+tx; N=gW.shape[0]
    if n<N:
        for m in cutlass.range_constexpr(M):
            acc=F32(0.0)
            for k in cutlass.range(K): acc=acc+gX[m,k].to(F32)*gW[n,k].to(F32)
            gO[m,n]=acc
@cute.jit
def naive(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    N=mW.shape[0]; tpb=128
    naive_k(mX,mW,mO,M,K).launch(grid=[(N+tpb-1)//tpb,1,1],block=[tpb,1,1])
@cute.kernel
def warp_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; lane=gtid%32; warp=gtid//32; N=gW.shape[0]; n=warp
    while n<N:
        for m in cutlass.range_constexpr(M):
            part=F32(0.0); k=lane
            while k<K: part=part+gX[m,k].to(F32)*gW[n,k].to(F32); k=k+32
            o=16
            while o>0: part=part+cute.arch.shuffle_sync_down(part,o); o=o//2
            if lane==0: gO[m,n]=part
        n=n+NW
@cute.jit
def warp(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    warp_k(mX,mW,mO,M,K,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])

M,K,N=4,2048,8192; peak=7.67e12; Wb=N*K*2
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16); W=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.02
o=torch.zeros(M,N,device="cuda",dtype=torch.float32)
mx,mw,mo=from_dlpack(x),from_dlpack(W),from_dlpack(o)
cn=cute.compile(naive,mx,mw,mo,M,K)
cw=cute.compile(warp,mx,mw,mo,M,K,148,256)
def bench(fn,iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): fn()
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
tn=bench(lambda:cn(mx,mw,mo)); tw=bench(lambda:cw(mx,mw,mo)); tt=bench(lambda:torch.nn.functional.linear(x,W))
for nm,t in [("naive scalar",tn),("warp+coalesced",tw),("cuBLAS",tt)]:
    bw=Wb/(t*1e-3); print(f"  {nm:16s} {t:7.1f} us   {bw/1e12:5.2f} TB/s ({bw/peak*100:4.1f}% roofline)")
print(f"gap naive/cuBLAS: {tn/tt:.1f}x   warp/cuBLAS: {tw/tt:.1f}x")

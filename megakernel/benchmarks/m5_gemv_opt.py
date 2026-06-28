import torch, time
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32

# NAIVE: one thread per output, strided (uncoalesced) loads
@cute.kernel
def naive_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    n=bx*bd+tx; N=gW.shape[0]
    if n<N:
        for m in cutlass.range_constexpr(M):
            acc=F32(0.0)
            for k in cutlass.range(K):
                acc=acc+gX[m,k].to(F32)*gW[n,k].to(F32)
            gO[m,n]=acc
@cute.jit
def naive(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr):
    N=mW.shape[0]; tpb=128
    naive_k(mX,mW,mO,M,K).launch(grid=[(N+tpb-1)//tpb,1,1],block=[tpb,1,1])

# OPTIMIZED: warp per output, coalesced K-strided loads, shuffle reduction
@cute.kernel
def opt_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; lane=gtid%32; warp=gtid//32
    N=gW.shape[0]
    n=warp
    while n<N:
        for m in cutlass.range_constexpr(M):
            part=F32(0.0)
            k=lane
            while k<K:
                part=part+gX[m,k].to(F32)*gW[n,k].to(F32)
                k=k+32
            o=16
            while o>0:
                part=part+cute.arch.shuffle_sync_down(part,o)
                o=o//2
            if lane==0:
                gO[m,n]=part
        n=n+NW
@cute.jit
def opt(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    NW=B*BLK//32
    opt_k(mX,mW,mO,M,K,NW).launch(grid=[B,1,1],block=[BLK,1,1])

def bench(fn, *a, iters=30):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize()
    e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record()
    for _ in range(iters): fn(*a)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/iters

torch.manual_seed(0)
M,K,N = 4, 2048, 8192          # MLP-gate decode shape
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
W=(torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.02)
o1=torch.zeros(M,N,device="cuda",dtype=torch.float32)
o2=torch.zeros(M,N,device="cuda",dtype=torch.float32)
B,BLK=148,256
mX,mW=from_dlpack(x),from_dlpack(W)
# correctness
naive(mX,mW,from_dlpack(o1),M,K); opt(mX,mW,from_dlpack(o2),M,K,B,BLK); torch.cuda.synchronize()
ref=torch.nn.functional.linear(x.float(),W.float())
print(f"opt correctness rel {((o2-ref).abs().max()/ref.abs().max()).item():.2e}  naive rel {((o1-ref).abs().max()/ref.abs().max()).item():.2e}")
Wbytes=N*K*2
t_naive=bench(lambda: naive(mX,mW,from_dlpack(o1),M,K))
t_opt=bench(lambda: opt(mX,mW,from_dlpack(o2),M,K,B,BLK))
t_torch=bench(lambda: torch.nn.functional.linear(x,W))
peak=7.67e12
for nm,t in [("naive",t_naive),("opt(warp+coalesced)",t_opt),("torch/cuBLAS",t_torch)]:
    bw=Wbytes/(t*1e-3); print(f"  {nm:22s} {t*1000:7.1f} us   {bw/1e12:5.2f} TB/s  ({bw/peak*100:4.1f}% roofline)")
print(f"speedup opt/naive: {t_naive/t_opt:.1f}x   gap opt/cuBLAS: {t_opt/t_torch:.1f}x")

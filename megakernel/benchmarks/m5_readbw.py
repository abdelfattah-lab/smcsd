import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32
# pure coalesced streaming read of a flat [Ntot] bf16 array; grid-stride; sum -> out[block]
@cute.kernel
def sum_k(gW,gO,Ntot:cutlass.Constexpr,GT:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx
    acc=F32(0.0)
    i=gtid
    while i<Ntot:
        acc=acc+gW[i].to(F32)
        i=i+GT
    gO[gtid]=acc
@cute.jit
def summ(mW,mO,Ntot:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    sum_k(mW,mO,Ntot,B*BLK).launch(grid=[B,1,1],block=[BLK,1,1])
def bench(fn,iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/iters
Ntot=8192*2048
W=torch.randn(Ntot,device="cuda",dtype=torch.bfloat16)
peak=7.67e12; bytes_=Ntot*2
for B,BLK in [(148,256),(148,1024),(1184,256),(2048,512)]:
    O=torch.zeros(B*BLK,device="cuda",dtype=torch.float32)
    t=bench(lambda:summ(from_dlpack(W),from_dlpack(O),Ntot,B,BLK))
    bw=bytes_/(t*1e-3); print(f"  grid={B}x{BLK} ({B*BLK} thr): {t*1000:7.1f} us  {bw/1e12:5.2f} TB/s ({bw/peak*100:4.1f}% roofline)")
# torch reference read BW (sum)
t=bench(lambda: W.float().sum()); print(f"  torch .sum(): {t*1000:.1f} us  {bytes_/(t*1e-3)/1e12:.2f} TB/s")

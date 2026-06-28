import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32
# ILP read: 8 independent accumulators, 8 loads/iter (memory-level parallelism)
@cute.kernel
def sum8_k(gW,gO,Ntot:cutlass.Constexpr,GT:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx
    a0=F32(0.0);a1=F32(0.0);a2=F32(0.0);a3=F32(0.0);a4=F32(0.0);a5=F32(0.0);a6=F32(0.0);a7=F32(0.0)
    i=gtid
    while i+7*GT<Ntot:
        a0=a0+gW[i].to(F32); a1=a1+gW[i+GT].to(F32); a2=a2+gW[i+2*GT].to(F32); a3=a3+gW[i+3*GT].to(F32)
        a4=a4+gW[i+4*GT].to(F32); a5=a5+gW[i+5*GT].to(F32); a6=a6+gW[i+6*GT].to(F32); a7=a7+gW[i+7*GT].to(F32)
        i=i+8*GT
    gO[gtid]=a0+a1+a2+a3+a4+a5+a6+a7
@cute.jit
def sum8(mW,mO,Ntot:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    sum8_k(mW,mO,Ntot,B*BLK).launch(grid=[B,1,1],block=[BLK,1,1])
def bench(fn,iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): fn()
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
Ntot=8192*2048; W=torch.randn(Ntot,device="cuda",dtype=torch.bfloat16); peak=7.67e12; by=Ntot*2
for B,BLK in [(148,512),(296,512),(592,512)]:
    O=torch.zeros(B*BLK,device="cuda",dtype=torch.float32)
    t=bench(lambda:sum8(from_dlpack(W),from_dlpack(O),Ntot,B,BLK))
    bw=by/(t*1e-3); print(f"  ILP8 grid={B}x{BLK}: {t*1000:6.1f} us  {bw/1e12:5.2f} TB/s ({bw/peak*100:4.1f}% roofline)")

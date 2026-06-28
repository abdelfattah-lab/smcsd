import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32
@cute.kernel
def sum_k(gW,gO,Ntot:cutlass.Constexpr,GT:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; acc=F32(0.0); i=gtid
    while i<Ntot:
        acc=acc+gW[i].to(F32); i=i+GT
    gO[gtid]=acc
@cute.jit
def summ(mW,mO,Ntot:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    sum_k(mW,mO,Ntot,B*BLK).launch(grid=[B,1,1],block=[BLK,1,1])

Ntot=8192*2048; B,BLK=592,512; peak=7.67e12
W=torch.randn(Ntot,device="cuda",dtype=torch.bfloat16); O=torch.zeros(B*BLK,device="cuda",dtype=torch.float32)
mW=from_dlpack(W); mO=from_dlpack(O)
# COMPILE ONCE
compiled=cute.compile(summ, mW, mO, Ntot, B, BLK)
def bench(iters=50):
    for _ in range(5): compiled(mW,mO)
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): compiled(mW,mO)
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
t=bench(); bw=Ntot*2/(t*1e-3)
print(f"[PROPER timing, compiled, tensors pre-converted] scalar read: {t*1000:.1f} us  {bw/1e12:.2f} TB/s ({bw/peak*100:.1f}% roofline)")
# compare: same kernel timed the OLD (polluted) way
def bench_old(iters=50):
    for _ in range(5): summ(from_dlpack(W),from_dlpack(O),Ntot,B,BLK)
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): summ(from_dlpack(W),from_dlpack(O),Ntot,B,BLK)
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
t2=bench_old(); print(f"[OLD polluted timing, from_dlpack+jit in loop] {t2*1000:.1f} us  ({Ntot*2/(t2*1e-3)/peak*100:.2f}% roofline)")
print(f"overhead per call was ~{(t2-t)*1000:.0f} us (Python from_dlpack+dispatch)")

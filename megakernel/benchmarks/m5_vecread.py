import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32
# Vectorized read: each thread reads a contiguous 8-bf16 chunk via 128-bit load (recast),
# grid-strided across chunks. Measures whether 128-bit loads hit roofline.
@cute.kernel
def vread_k(gW8, gO, NV:cutlass.Constexpr, GT:cutlass.Constexpr, VEC:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx
    acc=F32(0.0)
    i=gtid
    while i<NV:
        v=gW8[i].load()                 # 128-bit vectorized load of 8 bf16
        for j in cutlass.range_constexpr(VEC):
            acc=acc+v[j].to(F32)
        i=i+GT
    gO[gtid]=acc
@cute.jit
def vread(mW8,mO,NV:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr,VEC:cutlass.Constexpr):
    vread_k(mW8,mO,NV,B*BLK,VEC).launch(grid=[B,1,1],block=[BLK,1,1])
def bench(fn,iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): fn()
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
Ntot=8192*2048; VEC=8
W=torch.randn(Ntot,device="cuda",dtype=torch.bfloat16)
mW=from_dlpack(W)
# recast flat bf16 [Ntot] -> [Ntot/8] vectors of 8xbf16
print("trying recast bf16->vec8 ...")
try:
    mW8=cute.recast_tensor(mW, cute.BFloat16)  # placeholder; will adjust
except Exception as e:
    print("recast err", str(e)[:200])

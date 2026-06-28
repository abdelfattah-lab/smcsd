import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
I32=cutlass.Int32; F32=cutlass.Float32

# one thread per sequence: token = argmax_v( logits[v]/T - log(-log(u[v])) )
@cute.kernel
def gumbel_k(gL, gU, gOut, V:cutlass.Constexpr, invT:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    r = bx*bd+tx                      # sequence id
    R = gL.shape[0]
    if r < R:
        best = F32(-1.0e30); bi = I32(0)
        for v in cutlass.range(V):
            u = gU[r,v].to(F32)
            g = F32(0.0) - cute.log(F32(0.0) - cute.log(u))
            val = gL[r,v].to(F32)*F32(invT) + g
            if val > best:
                best = val; bi = I32(v)
        gOut[r] = bi
@cute.jit
def gumbel(mL,mU,mOut,V:cutlass.Constexpr,invT:cutlass.Constexpr):
    R=mL.shape[0]; tpb=128
    gumbel_k(mL,mU,mOut,V,invT).launch(grid=[(R+tpb-1)//tpb,1,1],block=[tpb,1,1])

torch.manual_seed(0)
R, V, T = 8, 128256, 0.7
logits = torch.randn(R, V, device="cuda", dtype=torch.float32)
u = torch.rand(R, V, device="cuda", dtype=torch.float32).clamp_(1e-9, 1.0)   # uniform noise
out = torch.zeros(R, device="cuda", dtype=torch.int32)
gumbel(from_dlpack(logits), from_dlpack(u), from_dlpack(out), V, 1.0/T)
torch.cuda.synchronize()
# torch reference: identical gumbel-max
g = -torch.log(-torch.log(u))
ref = (logits/T + g).argmax(-1).int()
match = (out == ref).float().mean().item()
print(f"in-kernel Gumbel-max vs torch: {out[:8].tolist()}  ref {ref[:8].tolist()}")
print(f"token match {match*100:.1f}%")
print("RESULT:", "PASS" if match==1.0 else "FAIL")

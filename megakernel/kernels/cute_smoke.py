import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    m, n = gA.shape
    ni = i % n
    mi = i // n
    gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def add(mA, mB, mC):
    m, n = mA.shape
    tpb = 256
    nblk = (m * n) // tpb
    add_kernel(mA, mB, mC).launch(grid=[nblk, 1, 1], block=[tpb, 1, 1])

M, N = 256, 256
a = torch.randn(M, N, device="cuda", dtype=torch.float32)
b = torch.randn(M, N, device="cuda", dtype=torch.float32)
c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
mA, mB, mC = from_dlpack(a), from_dlpack(b), from_dlpack(c)
print("compiling+launching CuTe DSL kernel on", torch.cuda.get_device_name(0))
add(mA, mB, mC)
torch.cuda.synchronize()
err = (c - (a + b)).abs().max().item()
print(f"max abs err vs torch: {err:.2e}")
print("RESULT:", "PASS" if err == 0 else "FAIL")

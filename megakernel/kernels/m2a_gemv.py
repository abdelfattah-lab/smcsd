import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemv_kernel(gX: cute.Tensor, gW: cute.Tensor, gO: cute.Tensor, M: cutlass.Constexpr, K: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = bidx * bdim + tidx
    N = gW.shape[0]
    if n < N:
        for m in cutlass.range_constexpr(M):
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(K):
                acc = acc + gX[m, k].to(cutlass.Float32) * gW[n, k].to(cutlass.Float32)
            gO[m, n] = acc

@cute.jit
def gemv(mX, mW, mO, M: cutlass.Constexpr, K: cutlass.Constexpr):
    N = mW.shape[0]
    tpb = 128
    nblk = (N + tpb - 1) // tpb
    gemv_kernel(mX, mW, mO, M, K).launch(grid=[nblk, 1, 1], block=[tpb, 1, 1])

torch.manual_seed(0)
M, K, N = 4, 2048, 2048
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
o = torch.zeros(M, N, device="cuda", dtype=torch.float32)
mX, mW, mO = from_dlpack(x), from_dlpack(W), from_dlpack(o)
print("running GEMV (M=4,K=2048,N=2048) on", torch.cuda.get_device_name(0))
gemv(mX, mW, mO, M, K)
torch.cuda.synchronize()
ref = torch.nn.functional.linear(x.float(), W.float())
err = (o - ref).abs().max().item(); rel = err / ref.abs().max().item()
print(f"max abs err {err:.4e}  rel {rel:.2e}")
print("RESULT:", "PASS" if rel < 1e-2 else "FAIL")

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Fused: y = rmsnorm(x)*gamma ; out = y @ W^T   (intermediate y never hits HBM)
@cute.kernel
def fused_kernel(gX, gG, gW, gO, M: cutlass.Constexpr, K: cutlass.Constexpr, eps: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    n = bidx * bdim + tidx
    N = gW.shape[0]
    if n < N:
        for m in cutlass.range_constexpr(M):
            # pass 1: sum of squares over K  (RMS)
            ss = cutlass.Float32(0.0)
            for k in cutlass.range(K):
                xv = gX[m, k].to(cutlass.Float32)
                ss = ss + xv * xv
            inv = cute.rsqrt(ss / cutlass.Float32(K) + cutlass.Float32(eps))
            # pass 2: project normalized x  (fused, no HBM write of the normed vector)
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(K):
                yv = gX[m, k].to(cutlass.Float32) * inv * gG[k].to(cutlass.Float32)
                acc = acc + yv * gW[n, k].to(cutlass.Float32)
            gO[m, n] = acc

@cute.jit
def fused(mX, mG, mW, mO, M: cutlass.Constexpr, K: cutlass.Constexpr, eps: cutlass.Constexpr):
    N = mW.shape[0]; tpb = 128; nblk = (N + tpb - 1) // tpb
    fused_kernel(mX, mG, mW, mO, M, K, eps).launch(grid=[nblk,1,1], block=[tpb,1,1])

torch.manual_seed(0)
M, K, N, eps = 4, 2048, 2048, 1e-5
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
g = torch.randn(K, device="cuda", dtype=torch.bfloat16)
W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
o = torch.zeros(M, N, device="cuda", dtype=torch.float32)
fused(from_dlpack(x), from_dlpack(g), from_dlpack(W), from_dlpack(o), M, K, eps)
torch.cuda.synchronize()
# reference
xf = x.float()
y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()
ref = torch.nn.functional.linear(y, W.float())
err = (o - ref).abs().max().item(); rel = err / ref.abs().max().item()
print(f"fused RMSNorm+proj: max abs err {err:.4e}  rel {rel:.2e}")
print("RESULT:", "PASS" if rel < 1e-2 else "FAIL")

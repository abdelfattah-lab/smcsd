import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# act[s,j] = silu(x@Wg^T)[s,j] * (x@Wu^T)[s,j]   (gate/up/silu fused per output j)
@cute.kernel
def act_kernel(gX, gWg, gWu, gA, S: cutlass.Constexpr, K: cutlass.Constexpr):
    tidx,_,_ = cute.arch.thread_idx(); bidx,_,_ = cute.arch.block_idx(); bdim,_,_ = cute.arch.block_dim()
    t = bidx*bdim + tidx
    I = gWg.shape[0]
    if t < S * I:
        s = t // I; j = t % I
        g = cutlass.Float32(0.0); u = cutlass.Float32(0.0)
        for k in cutlass.range(K):
            xv = gX[s,k].to(cutlass.Float32)
            g = g + xv * gWg[j,k].to(cutlass.Float32)
            u = u + xv * gWu[j,k].to(cutlass.Float32)
        silu = g / (cutlass.Float32(1.0) + cute.exp(-g))
        gA[s,j] = silu * u

# out[s,k] = act @ Wd^T
@cute.kernel
def down_kernel(gA, gWd, gO, S: cutlass.Constexpr, I: cutlass.Constexpr):
    tidx,_,_ = cute.arch.thread_idx(); bidx,_,_ = cute.arch.block_idx(); bdim,_,_ = cute.arch.block_dim()
    t = bidx*bdim + tidx
    Kout = gWd.shape[0]
    if t < S * Kout:
        s = t // Kout; k = t % Kout
        acc = cutlass.Float32(0.0)
        for j in cutlass.range(I):
            acc = acc + gA[s,j].to(cutlass.Float32) * gWd[k,j].to(cutlass.Float32)
        gO[s,k] = acc

@cute.jit
def mlp(mX, mWg, mWu, mWd, mA, mO, S: cutlass.Constexpr, K: cutlass.Constexpr, I: cutlass.Constexpr):
    tpb=128
    act_kernel(mX,mWg,mWu,mA,S,K).launch(grid=[(S*I+tpb-1)//tpb,1,1], block=[tpb,1,1])
    down_kernel(mA,mWd,mO,S,I).launch(grid=[(S*K+tpb-1)//tpb,1,1], block=[tpb,1,1])

torch.manual_seed(0)
S, K, I = 8, 2048, 8192
x = torch.randn(S,K,device="cuda",dtype=torch.bfloat16)
Wg = (torch.randn(I,K,device="cuda",dtype=torch.bfloat16)*0.02)
Wu = (torch.randn(I,K,device="cuda",dtype=torch.bfloat16)*0.02)
Wd = (torch.randn(K,I,device="cuda",dtype=torch.bfloat16)*0.02)
A = torch.zeros(S,I,device="cuda",dtype=torch.float32)
O = torch.zeros(S,K,device="cuda",dtype=torch.float32)
mlp(*[from_dlpack(t) for t in (x,Wg,Wu,Wd,A,O)], S, K, I)
torch.cuda.synchronize()
xf=x.float()
ref = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(xf,Wg.float()))*torch.nn.functional.linear(xf,Wu.float()), Wd.float())
err=(O-ref).abs().max().item(); rel=err/ref.abs().max().item()
print(f"SwiGLU MLP: max abs err {err:.4e}  rel {rel:.2e}")
print("RESULT:", "PASS" if rel<2e-2 else "FAIL")

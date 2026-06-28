import torch, math
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Decode attention, GQA, online softmax. One thread per (m, h).
# Q:[M,H,D]  K,V:[S,Hkv,D]  O:[M,H,D]  (RoPE assumed pre-applied; non-causal full-S here)
@cute.kernel
def attn_kernel(gQ, gK, gV, gO, M: cutlass.Constexpr, H: cutlass.Constexpr,
                D: cutlass.Constexpr, GRP: cutlass.Constexpr, scale: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    t = bidx * bdim + tidx           # global thread = m*H + h
    S = gK.shape[0]
    if t < M * H:
        m = t // H
        h = t % H
        kvh = h // GRP
        acc = cute.make_fragment(D, cutlass.Float32)
        for d in cutlass.range_constexpr(D):
            acc[d] = cutlass.Float32(0.0)
        run_max = cutlass.Float32(-1.0e30)
        run_sum = cutlass.Float32(0.0)
        for s in cutlass.range(S):
            score = cutlass.Float32(0.0)
            for d in cutlass.range_constexpr(D):
                score = score + gQ[m, h, d].to(cutlass.Float32) * gK[s, kvh, d].to(cutlass.Float32)
            score = score * cutlass.Float32(scale)
            new_max = cutlass.max(score, run_max)
            corr = cute.exp(run_max - new_max)
            p = cute.exp(score - new_max)
            run_sum = run_sum * corr + p
            for d in cutlass.range_constexpr(D):
                acc[d] = acc[d] * corr + p * gV[s, kvh, d].to(cutlass.Float32)
            run_max = new_max
        for d in cutlass.range_constexpr(D):
            gO[m, h, d] = acc[d] / run_sum

@cute.jit
def attn(mQ, mK, mV, mO, M: cutlass.Constexpr, H: cutlass.Constexpr,
         D: cutlass.Constexpr, GRP: cutlass.Constexpr, scale: cutlass.Constexpr):
    tpb = 128
    nblk = (M * H + tpb - 1) // tpb
    attn_kernel(mQ, mK, mV, mO, M, H, D, GRP, scale).launch(grid=[nblk,1,1], block=[tpb,1,1])

torch.manual_seed(0)
M, H, Hkv, D, S = 4, 32, 8, 64, 200
GRP = H // Hkv
scale = 1.0 / math.sqrt(D)
Q = torch.randn(M, H, D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(S, Hkv, D, device="cuda", dtype=torch.bfloat16)
V = torch.randn(S, Hkv, D, device="cuda", dtype=torch.bfloat16)
O = torch.zeros(M, H, D, device="cuda", dtype=torch.float32)
attn(from_dlpack(Q), from_dlpack(K), from_dlpack(V), from_dlpack(O), M, H, D, GRP, scale)
torch.cuda.synchronize()
# reference: torch SDPA with GQA (expand kv heads), non-causal
Qr = Q.float().permute(1,0,2)                      # [H,M,D]
Kr = K.float().repeat_interleave(GRP,dim=1).permute(1,0,2)  # [H,S,D]
Vr = V.float().repeat_interleave(GRP,dim=1).permute(1,0,2)
ref = torch.nn.functional.scaled_dot_product_attention(Qr, Kr, Vr).permute(1,0,2)  # [M,H,D]
err = (O - ref).abs().max().item(); rel = err / ref.abs().max().item()
print(f"decode attention GQA: max abs err {err:.4e}  rel {rel:.2e}")
print("RESULT:", "PASS" if rel < 2e-2 else "FAIL")

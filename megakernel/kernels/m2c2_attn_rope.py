import torch, math
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Self-attention over S tokens, GQA, RoPE applied on-chip, causal. One thread per (m,h).
# Q:[S,H,D] K,V:[S,Hkv,D] cos,sin:[S,D] O:[S,H,D]
@cute.kernel
def attn_kernel(gQ, gK, gV, gC, gSi, gO, H: cutlass.Constexpr, D: cutlass.Constexpr,
                DH: cutlass.Constexpr, GRP: cutlass.Constexpr, scale: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx(); bdim, _, _ = cute.arch.block_dim()
    t = bidx * bdim + tidx
    S = gQ.shape[0]
    if t < S * H:
        m = t // H; h = t % H; kvh = h // GRP
        # RoPE(Q[m,h]) -> qr[D]
        qr = cute.make_fragment(D, cutlass.Float32)
        for d in cutlass.range_constexpr(DH):
            lo = gQ[m, h, d].to(cutlass.Float32); hi = gQ[m, h, d + DH].to(cutlass.Float32)
            cl = gC[m, d].to(cutlass.Float32); sl = gSi[m, d].to(cutlass.Float32)
            ch = gC[m, d + DH].to(cutlass.Float32); sh = gSi[m, d + DH].to(cutlass.Float32)
            qr[d] = lo * cl - hi * sl
            qr[d + DH] = hi * ch + lo * sh
        acc = cute.make_fragment(D, cutlass.Float32)
        for d in cutlass.range_constexpr(D):
            acc[d] = cutlass.Float32(0.0)
        run_max = cutlass.Float32(-1.0e30); run_sum = cutlass.Float32(0.0)
        for s in cutlass.range(m + 1):                      # causal: s <= m
            score = cutlass.Float32(0.0)
            for d in cutlass.range_constexpr(DH):
                klo = gK[s, kvh, d].to(cutlass.Float32); khi = gK[s, kvh, d + DH].to(cutlass.Float32)
                cl = gC[s, d].to(cutlass.Float32); sl = gSi[s, d].to(cutlass.Float32)
                ch = gC[s, d + DH].to(cutlass.Float32); sh = gSi[s, d + DH].to(cutlass.Float32)
                kr_lo = klo * cl - khi * sl
                kr_hi = khi * ch + klo * sh
                score = score + qr[d] * kr_lo + qr[d + DH] * kr_hi
            score = score * cutlass.Float32(scale)
            new_max = cutlass.max(score, run_max)
            corr = cute.exp(run_max - new_max); p = cute.exp(score - new_max)
            run_sum = run_sum * corr + p
            for d in cutlass.range_constexpr(D):
                acc[d] = acc[d] * corr + p * gV[s, kvh, d].to(cutlass.Float32)
            run_max = new_max
        for d in cutlass.range_constexpr(D):
            gO[m, h, d] = acc[d] / run_sum

@cute.jit
def attn(mQ, mK, mV, mC, mS, mO, H: cutlass.Constexpr, D: cutlass.Constexpr,
         DH: cutlass.Constexpr, GRP: cutlass.Constexpr, scale: cutlass.Constexpr):
    S = mQ.shape[0]; tpb = 128; nblk = (S * H + tpb - 1) // tpb
    attn_kernel(mQ, mK, mV, mC, mS, mO, H, D, DH, GRP, scale).launch(grid=[nblk,1,1], block=[tpb,1,1])

torch.manual_seed(0)
S, H, Hkv, D = 24, 32, 8, 64
DH = D // 2; GRP = H // Hkv; scale = 1.0 / math.sqrt(D)
Q = torch.randn(S,H,D, device="cuda", dtype=torch.bfloat16)
K = torch.randn(S,Hkv,D, device="cuda", dtype=torch.bfloat16)
Vv = torch.randn(S,Hkv,D, device="cuda", dtype=torch.bfloat16)
# RoPE tables (theta=10000 for the unit test; exact Llama3 scaling applied at HF-validation stage)
inv = 1.0 / (10000.0 ** (torch.arange(0,DH,device="cuda").float()/DH))
pos = torch.arange(S,device="cuda").float()
ang = torch.outer(pos, inv)                       # [S,DH]
emb = torch.cat([ang,ang],dim=-1)                 # [S,D]
cos = emb.cos().to(torch.bfloat16); sin = emb.sin().to(torch.bfloat16)
O = torch.zeros(S,H,D, device="cuda", dtype=torch.float32)
attn(*[from_dlpack(t) for t in (Q,K,Vv,cos,sin,O)], H, D, DH, GRP, scale)
torch.cuda.synchronize()
# torch reference: identical RoPE (rotate_half), causal SDPA, GQA
def rope(x, cos, sin):  # x:[S,nh,D]
    c = cos.float()[:,None,:]; s = sin.float()[:,None,:]
    xh = torch.cat([-x[...,DH:], x[...,:DH]], dim=-1)
    return x.float()*c + xh*s
Qr = rope(Q,cos,sin).permute(1,0,2)
Kr = rope(K,cos,sin).repeat_interleave(GRP,dim=1).permute(1,0,2)
Vr = Vv.float().repeat_interleave(GRP,dim=1).permute(1,0,2)
ref = torch.nn.functional.scaled_dot_product_attention(Qr,Kr,Vr,is_causal=True).permute(1,0,2)
err=(O-ref).abs().max().item(); rel=err/ref.abs().max().item()
print(f"attn+RoPE+causal: max abs err {err:.4e}  rel {rel:.2e}")
print("RESULT:", "PASS" if rel<2e-2 else "FAIL")

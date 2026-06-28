import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32; BF=cutlass.BFloat16
VEC=8
# warp per output, each lane loads VEC=8 contiguous bf16 (128-bit) per step
@cute.kernel
def vg_k(gX,gW,gO,M:cutlass.Constexpr,K:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; lane=gtid%32; warp=gtid//32; N=gW.shape[0]
    atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=128)
    n=warp
    while n<N:
        gWrow=gW[n,None]                      # [K] row view
        acc=F32(0.0)
        base=0
        while base < K:
            tile=cute.local_tile(gWrow,(VEC,),(base//VEC+lane,))   # this lane's 8 elems
            fr=cute.make_fragment(VEC,BF)
            cute.copy(atom,tile,fr)
            for j in cutlass.range_constexpr(VEC):
                acc=acc+gX[0,base+lane*VEC+j].to(F32)*fr[j].to(F32)
            base=base+32*VEC
        o=16
        while o>0: acc=acc+cute.arch.shuffle_sync_bfly(acc,o); o=o//2
        if lane==0: gO[0,n]=acc
        n=n+NW
@cute.jit
def vg(mX,mW,mO,M:cutlass.Constexpr,K:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    vg_k(mX,mW,mO,M,K,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])

M,K,N=1,2048,8192; peak=7.67e12; Wb=N*K*2
x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16); W=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.02
o=torch.zeros(M,N,device="cuda",dtype=torch.float32)
mx=from_dlpack(x,assumed_align=16); mw=from_dlpack(W,assumed_align=16); mo=from_dlpack(o)
try:
    c=cute.compile(vg,mx,mw,mo,M,K,148,512)
    c(mx,mw,mo); torch.cuda.synchronize()
    ref=torch.nn.functional.linear(x.float(),W.float())
    print("vec-gemv correctness rel", ((o-ref).abs().max()/ref.abs().max()).item())
    def bench(fn,it=50):
        for _ in range(5): fn()
        torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
        for _ in range(it): fn()
        e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/it
    t=bench(lambda:c(mx,mw,mo)); print(f"vec-gemv: {t*1000:.1f} us  {Wb/(t*1e-3)/peak*100:.1f}% roofline")
except Exception as e:
    import traceback; traceback.print_exc()

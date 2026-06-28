import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32

# Vectorized read via TiledCopy: each CTA processes tiles of (BLK*VEC) bf16 with 128-bit loads.
@cute.kernel
def vr_k(gW, gO, tiled_copy, NTILE:cutlass.Constexpr, TILE:cutlass.Constexpr, NB:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    thr=tiled_copy.get_slice(tx)
    acc=F32(0.0)
    tile=bx
    while tile<NTILE:
        gWt = cute.local_tile(gW, (TILE,), (tile,))         # this CTA's tile [TILE]
        tg = thr.partition_S(gWt)
        tr = cute.make_fragment_like(tg)
        cute.copy(tiled_copy, tg, tr)
        v = tr.load().to(F32)
        acc = acc + v.reduce(cutlass.cute.ReductionOp.ADD, F32(0.0), 0)
        tile=tile+NB
    gO[bx*0 + tx]=acc

@cute.jit
def vr(mW,mO,NTILE:cutlass.Constexpr,TILE:cutlass.Constexpr,BLK:cutlass.Constexpr,VEC:cutlass.Constexpr,NB:cutlass.Constexpr):
    copy_atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
    tv=cute.make_layout((BLK,VEC), stride=(VEC,1))
    tiled_copy=cute.make_tiled_copy(copy_atom, tv, (TILE,))
    vr_k(mW,mO,tiled_copy,NTILE,TILE,NB).launch(grid=[NB,1,1],block=[BLK,1,1])

def bench(fn,iters=30):
    for _ in range(3): fn()
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(iters): fn()
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/iters
Ntot=8192*2048; BLK=256; VEC=8; TILE=BLK*VEC; NTILE=Ntot//TILE; NB=296; peak=7.67e12
W=torch.randn(Ntot,device="cuda",dtype=torch.bfloat16); O=torch.zeros(NB*BLK,device="cuda",dtype=torch.float32)
try:
    t=bench(lambda:vr(from_dlpack(W,assumed_align=16),from_dlpack(O,assumed_align=16),NTILE,TILE,BLK,VEC,NB))
    bw=Ntot*2/(t*1e-3); print(f"vectorized TiledCopy read: {t*1000:.1f} us  {bw/1e12:.2f} TB/s ({bw/peak*100:.1f}% roofline)")
except Exception as e:
    import traceback; traceback.print_exc()

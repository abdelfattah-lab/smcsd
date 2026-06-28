import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/kernels")
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils as utils
F32=cutlass.Float32; I32=cutlass.Int32

@cute.kernel
def k(gIn, gOut, N:cutlass.Constexpr, BLK:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    smem=utils.SmemAllocator()
    sBuf=smem.allocate_tensor(F32, cute.make_layout(N), byte_alignment=16)
    # cooperatively stage: each block stages gIn*2 into its own smem
    i=tx
    while i<N:
        sBuf[i]=gIn[i].to(F32)*F32(2.0); i=i+BLK
    cute.arch.barrier()
    # each thread sums the whole staged buffer (tests broadcast read) and writes one output per block-thread
    if bx==0:
        j=tx
        while j<N:
            s=F32(0.0); kk=0
            while kk<N:
                s=s+sBuf[kk]; kk=kk+1
            gOut[j]=sBuf[j]+s*F32(0.0)  # just echo sBuf[j], but force reading all of sBuf
            j=j+BLK

@cute.jit
def run(gIn, gOut, N:cutlass.Constexpr, BLK:cutlass.Constexpr):
    k(gIn,gOut,N,BLK).launch(grid=[2,1,1], block=[BLK,1,1], smem=N*4+256)

N=2048; BLK=512
x=torch.arange(N,device="cuda",dtype=torch.float32)/N
out=torch.zeros(N,device="cuda",dtype=torch.float32)
mi=from_dlpack(x); mo=from_dlpack(out)
comp=cute.compile(run, mi, mo, N, BLK)
comp(mi,mo); torch.cuda.synchronize()
ref=x*2.0
err=(out-ref).abs().max().item()
print(f"max abs err = {err:.3e}  -> {'PASS' if err<1e-4 else 'FAIL'}")
print("out[:5]", out[:5].tolist(), "ref[:5]", ref[:5].tolist())

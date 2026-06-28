import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
F32=cutlass.Float32; BF=cutlass.BFloat16
# scalar warp GEMV
@cute.kernel
def sw_k(gX,gW,gO,K:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; lane=gtid%32; warp=gtid//32; N=gW.shape[0]; n=warp
    while n<N:
        acc=F32(0.0); k=lane
        while k<K: acc=acc+gX[0,k].to(F32)*gW[n,k].to(F32); k=k+32
        o=16
        while o>0: acc=acc+cute.arch.shuffle_sync_bfly(acc,o); o=o//2
        if lane==0: gO[0,n]=acc
        n=n+NW
@cute.jit
def sw(mX,mW,mO,K:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    sw_k(mX,mW,mO,K,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])
# vec warp GEMV
@cute.kernel
def vw_k(gX,gW,gO,K:cutlass.Constexpr,NW:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx(); bd,_,_=cute.arch.block_dim()
    gtid=bx*bd+tx; lane=gtid%32; warp=gtid//32; N=gW.shape[0]
    atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),BF,num_bits_per_copy=128); n=warp
    while n<N:
        gWr=gW[n,None]; acc=F32(0.0); base=0
        while base<K:
            tile=cute.local_tile(gWr,(8,),(base//8+lane,)); fr=cute.make_fragment(8,BF); cute.copy(atom,tile,fr)
            for j in cutlass.range_constexpr(8): acc=acc+gX[0,base+lane*8+j].to(F32)*fr[j].to(F32)
            base=base+256
        o=16
        while o>0: acc=acc+cute.arch.shuffle_sync_bfly(acc,o); o=o//2
        if lane==0: gO[0,n]=acc
        n=n+NW
@cute.jit
def vw(mX,mW,mO,K:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    vw_k(mX,mW,mO,K,B*BLK//32).launch(grid=[B,1,1],block=[BLK,1,1])
def bench(fn,it=100):
    for _ in range(10): fn()
    torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
    for _ in range(it): fn()
    e1.record();torch.cuda.synchronize();return e0.elapsed_time(e1)/it
peak=7.67e12
for K,N in [(2048,8192),(8192,2048),(2048,2048)]:
    x=torch.randn(1,K,device="cuda",dtype=torch.bfloat16); W=torch.randn(N,K,device="cuda",dtype=torch.bfloat16)*0.02; o=torch.zeros(1,N,device="cuda",dtype=torch.float32)
    mx=from_dlpack(x,assumed_align=16); mw=from_dlpack(W,assumed_align=16); mo=from_dlpack(o)
    cs=cute.compile(sw,mx,mw,mo,K,148,512); cv=cute.compile(vw,mx,mw,mo,K,148,512)
    ts=bench(lambda:cs(mx,mw,mo)); tv=bench(lambda:cv(mx,mw,mo)); Wb=N*K*2
    print(f"  K={K} N={N}: scalar {ts*1000:.1f}us ({Wb/(ts*1e-3)/peak*100:.0f}%)  vec {tv*1000:.1f}us ({Wb/(tv*1e-3)/peak*100:.0f}%)  speedup {ts/tv:.2f}x")

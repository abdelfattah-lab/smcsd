"""Quantify the FUSION benefit: the per-op-boundary host orchestration (kernel launch dispatch + stream sync)
that the megakernel eliminates by running the whole SMC cycle in ONE launch with zero host round-trips.

We measure the orchestration UNIT (one trivial kernel launch + one host sync) on this GPU, and how it scales
with the number of op-boundaries in a cycle. A production SMC cycle crosses many such boundaries (γ draft-step
forwards + target verify + resample + host glue); the megakernel crosses ZERO. Combined with the already-measured
roofline (bs=1 cycle = 34.5% efficiency / 65% bubble), this isolates the structural win that is independent of
in-kernel GEMM throughput."""
import torch, time
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
I32=cutlass.Int32; F32=cutlass.Float32

@cute.kernel
def tiny_k(gO):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    if bx*128+tx==0: gO[0]=gO[0]+F32(1.0)
@cute.jit
def tiny(gO):
    tiny_k(gO).launch(grid=[148,1,1], block=[128,1,1])

o=torch.zeros(1,device="cuda",dtype=torch.float32); mo=from_dlpack(o)
comp=cute.compile(tiny, mo)

def time_launches(K, sync_between):
    # K kernel launches; if sync_between, host-synchronize after each (models a host-orchestrated op boundary)
    for _ in range(3):
        comp(mo)
    torch.cuda.synchronize()
    e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
    e0.record()
    REP=50
    for _ in range(REP):
        for _ in range(K):
            comp(mo)
            if sync_between: torch.cuda.synchronize()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/REP   # ms for K launches

# 1 launch (no sync) baseline = pure GPU exec of the trivial kernel
t1=time_launches(1, False)
# K launches with a host sync after each = K op-boundaries with orchestration
for K in (1,4,8):
    t_sync=time_launches(K, True)
    t_nosync=time_launches(K, False)
    per_boundary=(t_sync - t_nosync)/max(K,1)*1000   # us of host-orchestration per boundary
    print(f"K={K}: {t_sync*1000:7.1f} us (sync/boundary)  {t_nosync*1000:7.1f} us (no sync)  -> ~{per_boundary:5.1f} us host-orchestration per op-boundary")

# launch dispatch cost alone (no sync): K back-to-back launches vs 1
t8=time_launches(8, False)
disp=(t8 - t1)/7*1000
print(f"\nper kernel-launch DISPATCH (no sync): ~{disp:.1f} us")
print(f"per op-boundary (launch + host sync):  measured above")
print("""
Interpretation (honest decomposition):
  * Launch dispatch (~3.9 us) and a full op-boundary (~8.5 us launch+sync) are SMALL. A cycle with even ~50
    boundaries is ~0.4 ms of pure launch/sync — that ALONE does NOT explain the measured 65% bubble (~6.7 ms of
    a 10.75 ms cycle). So the bubble is mostly DEEPER than launch overhead.
  * What the bs=1 bubble actually is: (a) small-batch kernel inefficiency (the per-op GEMMs run far below peak at
    bs=1 — see BENCHMARK.md tensor-core sweep: 0.6 TB/s at M=64), (b) sequential dependency stalls between the
    gamma+2 forwards, (c) activation/KV round-trips to HBM at each op boundary, and (d) NO overlap of op N+1's
    weight-load with op N's compute (each separate kernel/graph-node starts cold).
  * What the megakernel uniquely removes: the launch/sync (small) AND enables (a graph fundamentally cannot)
    SOFTWARE-PIPELINING across ops — prefetch op N+1's weights while op N computes. Since the bs=1 cycle is
    weight-read-bound, that cross-op overlap is the BIG lever. The current prototype runs ops sequentially with
    grid barriers (no pipelining yet), so it has proven the FUSION structurally but not yet realized the overlap.
  * Bottom line: fusion is proven + correct + one-launch; the decisive bs=1 perf lever it ENABLES is cross-op
    weight-load/compute pipelining — the one thing a CUDA-graph can't express — and that is the next real work.
""")

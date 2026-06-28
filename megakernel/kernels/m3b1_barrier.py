import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
I32=cutlass.Int32

@cute.kernel
def bar_test_k(gArr, gSense, gData, gOut, B:cutlass.Constexpr, NPH:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    arrive = gArr.iterator
    sense  = gSense.iterator
    ls = I32(0)
    for ph in cutlass.range_constexpr(NPH):
        if tx==0:
            gData[bx] = I32(ph*1000) + bx
        # ---- barrier (write visibility) ----
        ls = I32(1) - ls
        cute.arch.sync_threads()
        if tx==0:
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(arrive, I32(1), sem='acq_rel', scope='gpu')
            if old + 1 == B:
                cute.arch.atomic_exch(arrive, I32(0), sem='relaxed', scope='gpu')
                cute.arch.atomic_exch(sense, ls, sem='release', scope='gpu')
            else:
                cur = cute.arch.atomic_add(sense, I32(0), sem='acquire', scope='gpu')
                while cur != ls:
                    cur = cute.arch.atomic_add(sense, I32(0), sem='acquire', scope='gpu')
            cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()
        # ---- read all blocks' data ----
        if tx==0:
            s = I32(0)
            for j in cutlass.range(B):
                s = s + gData[j]
            gOut[bx] = s
        # ---- barrier (before next phase overwrites data) ----
        ls = I32(1) - ls
        cute.arch.sync_threads()
        if tx==0:
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(arrive, I32(1), sem='acq_rel', scope='gpu')
            if old + 1 == B:
                cute.arch.atomic_exch(arrive, I32(0), sem='relaxed', scope='gpu')
                cute.arch.atomic_exch(sense, ls, sem='release', scope='gpu')
            else:
                cur = cute.arch.atomic_add(sense, I32(0), sem='acquire', scope='gpu')
                while cur != ls:
                    cur = cute.arch.atomic_add(sense, I32(0), sem='acquire', scope='gpu')
            cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()

@cute.jit
def run(mArr,mSense,mData,mOut,B:cutlass.Constexpr,NPH:cutlass.Constexpr):
    bar_test_k(mArr,mSense,mData,mOut,B,NPH).launch(grid=[B,1,1], block=[128,1,1])

B=148; NPH=5
arr=torch.zeros(1,device="cuda",dtype=torch.int32)
sense=torch.zeros(1,device="cuda",dtype=torch.int32)
data=torch.zeros(B,device="cuda",dtype=torch.int32)
out=torch.full((B,),-1,device="cuda",dtype=torch.int32)
run(*[from_dlpack(t) for t in (arr,sense,data,out)], B, NPH)
torch.cuda.synchronize()
expect = (NPH-1)*1000*B + B*(B-1)//2     # last phase sum seen by every block
ok = bool((out==expect).all().item())
print(f"B={B} phases={NPH}  expected last-phase sum={expect}")
print(f"out[:5]={out[:5].tolist()}  all_equal={ok}  unique={out.unique().tolist()[:5]}")
print("RESULT:", "PASS" if ok else "FAIL")

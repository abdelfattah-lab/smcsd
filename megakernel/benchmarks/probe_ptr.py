import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
t=torch.zeros(4,device="cuda",dtype=torch.int32)
mt=from_dlpack(t)
print("Tensor attrs w/ ptr/iter:", [x for x in dir(mt) if any(k in x.lower() for k in ('ptr','iter','data','engine'))])
print("atomic_exch present:", hasattr(cute.arch,'atomic_exch'))
# in-kernel probe: get pointer to element and atomic_add
@cute.kernel
def k(gT):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    if tx==0:
        p = gT.iterator + bx       # pointer arithmetic?
        cute.arch.atomic_add(p, cutlass.Int32(1), sem='relaxed', scope='gpu')
@cute.jit
def run(mT):
    k(mT).launch(grid=[4,1,1], block=[32,1,1])
try:
    run(mt); torch.cuda.synchronize(); print("atomic via gT.iterator+bx ->", t.tolist())
except Exception as e:
    print("ERR:", str(e)[:300])

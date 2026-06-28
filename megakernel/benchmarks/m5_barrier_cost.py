import torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
I32=cutlass.Int32
@cute.kernel
def bar_k(garr,gsen,gout,NBAR:cutlass.Constexpr,B:cutlass.Constexpr):
    tx,_,_=cute.arch.thread_idx(); bx,_,_=cute.arch.block_idx()
    arr=garr.iterator; sen=gsen.iterator; ls=I32(0)
    i=0
    while i<NBAR:
        ls=I32(1)-ls
        cute.arch.sync_threads()
        if tx==0:
            cute.arch.fence_acq_rel_gpu()
            old=cute.arch.atomic_add(arr,I32(1),sem='acq_rel',scope='gpu')
            if old+1==B:
                cute.arch.atomic_exch(arr,I32(0),sem='relaxed',scope='gpu')
                cute.arch.atomic_exch(sen,ls,sem='release',scope='gpu')
            else:
                cur=cute.arch.atomic_add(sen,I32(0),sem='acquire',scope='gpu')
                while cur!=ls: cur=cute.arch.atomic_add(sen,I32(0),sem='acquire',scope='gpu')
            cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()
        i=i+1
    if tx==0: gout[bx]=ls
@cute.jit
def barr(ga,gs,go,NBAR:cutlass.Constexpr,B:cutlass.Constexpr,BLK:cutlass.Constexpr):
    bar_k(ga,gs,go,NBAR,B).launch(grid=[B,1,1],block=[BLK,1,1])
B,BLK,NBAR=148,512,720
a=torch.zeros(1,device="cuda",dtype=torch.int32);ss=torch.zeros(1,device="cuda",dtype=torch.int32);o=torch.zeros(B,device="cuda",dtype=torch.int32)
ma,ms,mo=from_dlpack(a),from_dlpack(ss),from_dlpack(o)
c=cute.compile(barr,ma,ms,mo,NBAR,B,BLK)
def run(): a.zero_();ss.zero_();c(ma,ms,mo)
for _ in range(3): run()
torch.cuda.synchronize();e0=torch.cuda.Event(True);e1=torch.cuda.Event(True);e0.record()
for _ in range(20): run()
e1.record();torch.cuda.synchronize();ms_=e0.elapsed_time(e1)/20
print(f"{NBAR} grid barriers: {ms_*1000:.0f} us total = {ms_*1000/NBAR:.2f} us/barrier")
print(f"=> ~720 barriers/token cost ~{ms_*1000:.0f} us of the 2760 us/token megakernel")

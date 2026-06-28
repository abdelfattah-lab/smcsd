import inspect, cutlass, cutlass.cute as cute
for nm in ['shuffle_sync_down','shuffle_sync_bfly','shuffle_sync','warp_reduction_max','redux']:
    o=getattr(cute.arch,nm,None)
    if o is None: print(nm,"MISSING"); continue
    try: print(nm,"->",str(inspect.signature(o)))
    except Exception as e: print(nm,"(no sig)",type(o))
print("WARP_SIZE",getattr(cute.arch,'WARP_SIZE',None))

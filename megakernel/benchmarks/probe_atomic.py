import inspect, cutlass, cutlass.cute as cute
for nm in ['atomic_add','atomic_cas','fence','fence_acq_rel_gpu','fence_acq_rel_sys','threadfence','make_ptr']:
    o=getattr(cute.arch, nm, None) or getattr(cute, nm, None)
    if o is None: print(nm, "-> MISSING"); continue
    try: print(nm, "->", str(inspect.signature(o)))
    except: print(nm, "-> (no sig)", type(o))
print("--- all arch fences ---", [x for x in dir(cute.arch) if 'fence' in x.lower()])
print("--- SM count ---")
import cutlass.utils as U
print([x for x in dir(U) if 'hardware' in x.lower() or 'sm' in x.lower()][:10])

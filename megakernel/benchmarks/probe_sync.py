import cutlass, cutlass.cute as cute
arch=[x for x in dir(cute.arch)]
print("grid/coop sync:", [x for x in arch if any(k in x.lower() for k in ('grid','coop','barrier','sync','cluster'))])
print("atomics:", [x for x in arch if 'atomic' in x.lower()])
print("launch has cooperative?:", 'cooperative' in str(cute.Kernel.launch.__doc__ if hasattr(cute,'Kernel') else ''))
# does .launch accept a cooperative/grid flag?
import inspect
try:
    from cutlass.cute.typing import Kernel
except Exception as e:
    pass

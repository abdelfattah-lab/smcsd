import cutlass, cutlass.cute as cute
print("while_loop:", [x for x in dir(cutlass) if 'while' in x.lower()])
print("cute while:", [x for x in dir(cute) if 'while' in x.lower()])
import cutlass.utils as U
hi=U.HardwareInfo()
print("HardwareInfo methods:", [x for x in dir(hi) if not x.startswith('_')])
try: print("SM count:", hi.get_device_multiprocessor_count())
except Exception as e: print("sm count err:", e)

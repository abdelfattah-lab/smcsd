import cutlass, cutlass.cute as cute
try:
    import cutlass.cute.math as cmath
    print("cute.math:", [x for x in dir(cmath) if not x.startswith('_')])
except Exception as e:
    print("no cute.math:", e)
print("cutlass max/maximum:", hasattr(cutlass,'max'), hasattr(cutlass,'maximum'))
print("cute.arch fmax-ish:", [x for x in dir(cute.arch) if 'max' in x.lower() or 'abs' in x.lower()])

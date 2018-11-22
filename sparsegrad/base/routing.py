import numpy as np
from sparsegrad.impl.multipledispatch import dispatch

@dispatch(object)
def implementation_priority(obj):
    return getattr(obj, '__array_priority__', 0)

def find_implementation(arrays, default=np, default_priority=0):
    highest = default
    current = default_priority
    for a in arrays:
        priority = implementation_priority(a)
        if highest is None or priority > current:
            highest, current = a, priority
    return highest

@dispatch(object, object)
def sparsesum(impl, terms, **kwargs):
    return impl.sparsesum(terms, **kwargs)

@dispatch(object, object)
def hstack(impl, arrays):
    return impl.hstack(arrays)

"""
Basic helpers that should be available independent of the api.

This is for module level helpers that should be available without importing
bempp_cl.api first.

"""

import time as _time
import functools as _functools
import collections as _collections


def timeit(fun):
    """Time a method in Bempp."""

    @_functools.wraps(fun)
    def timed_fun(*args, **kwargs):
        """Time an operation."""
        from bempp_cl.api import log

        start_time = _time.time()
        res = fun(*args, **kwargs)
        end_time = _time.time()
        log(fun.__qualname__ + " : {0:.3e}s".format(end_time - start_time), "timing")
        return res

    return timed_fun


IndexList = _collections.namedtuple("IndexList", ["indices", "indexptr"])


def jit_logger(name):
    """Emit a log message whenever Numba jits something."""
    import bempp_cl.api

    def closure(func):
        """Closure that has the name variable."""

        def inner(*args, **kwargs):
            origsigs = set(func.signatures)
            result = func(*args, **kwargs)
            newsigs = set(func.signatures)
            if newsigs != origsigs:
                new = (newsigs ^ origsigs).pop()
                bempp_cl.api.log(f"Compiled {name} for signature {new}", level="timing")
            return result

        return inner

    return closure

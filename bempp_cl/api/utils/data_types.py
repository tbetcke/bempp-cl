"""Data type checks."""

import numpy as _np


def check_type(name, default="float64"):
    """Try to convert input into a numpy.dtype object."""
    value = str(name) if name is not None else default
    if value not in ["float32", "float64", "complex64", "complex128"]:
        raise ValueError("Incorrect type (%s)" % value)

    return _np.dtype(value)


def combined_type(dtype1, dtype2):
    """Return a type that is compatible with dtype1 and dtype2."""
    new_d1 = check_type(dtype1)
    new_d2 = check_type(dtype2)

    return (new_d1.type(1) * new_d2.type(1)).dtype

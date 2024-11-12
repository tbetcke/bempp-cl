"""Basic operator imports and definitions."""

import collections as _collections
import numpy as _np

from . import boundary
from . import potential
from . import far_field


OperatorDescriptor = _collections.namedtuple(
    "OperatorDescriptor",
    "identifier options kernel_type assembly_type precision is_complex singular_part kernel_dimension",
)

MultitraceOperatorDescriptor = _collections.namedtuple(
    "MultitraceOperatorDescriptor",
    "identifier options multitrace_kernel singular_contribution",
)


def _add_wavenumber(options, wavenumber, identifier="WAVENUMBER"):
    """Add a real/complex wavenumber to an options array."""
    if _np.iscomplexobj(wavenumber):
        options[identifier + "_REAL"] = 1.0 * _np.real(wavenumber)
        options[identifier + "_COMPLEX"] = 1.0 * _np.imag(wavenumber)
    else:
        options[identifier + "_REAL"] = 1.0 * _np.real(wavenumber)

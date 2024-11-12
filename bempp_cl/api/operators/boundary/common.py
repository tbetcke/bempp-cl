"""Common helpers uses for all boundary operators."""

from bempp_cl.api.assembly import assembler as _assembler
from bempp_cl.api.assembly import boundary_operator as _boundary_operator
from bempp_cl.api.assembly import blocked_operator as _blocked_operator
import numpy as _np


# pylint: disable=too-many-arguments
def create_operator(
    identifier,
    domain,
    range_,
    dual_to_range,
    parameters,
    assembler,
    operator_options,
    kernel_type,
    assembly_type,
    device_interface,
    precision,
    is_complex,
):
    """Create a generic operator."""
    from bempp_cl.api.operators import OperatorDescriptor
    import bempp_cl.api

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    if assembler != "only_singular_part":
        # Add operator for singular part assembly. Needed for certain fast methods.
        singular_part = create_operator(
            identifier,
            domain,
            range_,
            dual_to_range,
            parameters,
            "only_singular_part",
            operator_options,
            kernel_type,
            assembly_type,
            device_interface,
            precision,
            is_complex,
        )
    else:
        singular_part = None

    assembler = _assembler.AssemblerInterface(domain, dual_to_range, assembler, device_interface, precision, parameters)

    kernel_dimension = 1

    descriptor = OperatorDescriptor(
        identifier,
        operator_options,
        kernel_type,
        assembly_type,
        precision,
        is_complex,
        singular_part,
        kernel_dimension,
    )
    return _boundary_operator.BoundaryOperatorWithAssembler(domain, range_, dual_to_range, assembler, descriptor)


def create_multitrace_operator(
    identifier,
    domain,
    range_,
    dual_to_range,
    parameters,
    assembler,
    operator_options,
    multitrace_kernel,
    singular_contribution,
    device_interface,
    precision,
):
    """Create a generic operator."""
    from bempp_cl.api.operators import MultitraceOperatorDescriptor
    import bempp_cl.api

    if device_interface is None:
        device_interface = bempp_cl.api.DEFAULT_DEVICE

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    assembler = _assembler.AssemblerInterface(domain, dual_to_range, assembler, device_interface, precision, parameters)

    descriptor = MultitraceOperatorDescriptor(identifier, operator_options, multitrace_kernel, singular_contribution)
    return _blocked_operator.MultitraceOperatorFromAssembler(domain, range_, dual_to_range, assembler, descriptor)


def pade_coeffs(order, angle):
    """
    Calculate the coefficients of the Pade series expansion of the square root.

    Parameters
    ----------
    order : int
        The order of the expansion.
    angle : float
        The branch-cut angle of the expansion.

    Returns
    -------
    c_0 : complex
    a_j : numpy.ndarray[complex]
    b_j : numpy.ndarray[complex]
    r_0 : complex
        The coefficients of the Pade expansion of the square root.
    """
    idx = _np.arange(order) + 1
    sin_j = 2 / (2 * order + 1) * _np.sin(idx * _np.pi / (2 * order + 1)) ** 2
    cos_j = _np.cos(idx * _np.pi / (2 * order + 1)) ** 2
    z = _np.exp(-1j * angle) - 1
    z_j = 1 + cos_j * z

    c_0 = _np.exp(0.5j * angle) * (1 + _np.sum(sin_j * z / z_j))
    a_j = _np.exp(-0.5j * angle) * sin_j / z_j**2
    b_j = _np.exp(-1j * angle) * cos_j / z_j
    r_0 = c_0 + _np.sum(a_j / b_j)

    return c_0, a_j, b_j, r_0


def inv_sqrt_pade_coeffs(order, angle=_np.pi / 3):
    """
    Calculate the coefficients of the Pade series expansion of the inverse square root.

    Parameters
    ----------
    order : int
        The order of the expansion.
    angle : float
        The branch-cut angle of the expansion (pi/3 is recommended).

    Returns
    -------
    s_j : numpy.ndarray[complex]
    r_j : numpy.ndarray[complex]
        The coefficients of the Pade expansion of the inverse square root.
    """
    idx = _np.arange(order)
    tan_j = _np.tan(0.5 * _np.pi * (0.5 + idx) / order) ** 2
    d_j = 1 + tan_j
    c_j = d_j / order

    r_j = _np.exp(1j * angle * 0.5) * c_j
    s_j = 1 + _np.exp(1j * angle) * (d_j - 1)

    return r_j, s_j

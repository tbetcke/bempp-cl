"""Helmholtz far-field operators."""

import numpy as _np


def single_layer(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz single-layer far-field potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_far_field_single_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_far_field_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )


def double_layer(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz double-layer far-field potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_far_field_double_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_far_field_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )

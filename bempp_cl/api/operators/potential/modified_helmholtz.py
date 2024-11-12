"""Modified Helmholtz potential operators."""

import numpy as _np


def single_layer(
    space,
    points,
    omega,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a modified Helmholtz single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "modified_helmholtz_single_layer_potential",  # Identifier
        [omega],  # Options
        "modified_helmholtz_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )


def double_layer(
    space,
    points,
    omega,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a modified Helmholtz double-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "modified_helmholtz_double_layer_potential",  # Identifier
        [omega],  # Options
        "modified_helmholtz_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )

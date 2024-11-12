"""Maxwell potential operators."""

import numpy as _np


def electric_field(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Maxwell electric field potential operator."""
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler
    import bempp_cl.api

    if space.identifier != "rwg0":
        raise ValueError("Space must be an RWG type function space.")

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "maxwell_electric_field_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_single_layer",  # Kernel type
        "maxwell_electric_field",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )


def magnetic_field(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Maxwell magnetic field potential operator."""
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler
    import bempp_cl.api

    if space.identifier != "rwg0":
        raise ValueError("Space must be an RWG type function space.")

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "maxwell_magnetic_field_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_single_layer",  # Kernel type
        "maxwell_magnetic_field",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )

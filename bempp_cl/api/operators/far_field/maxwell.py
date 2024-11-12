"""Maxwell far-field operators."""

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
    """Return a Maxwell electric far-field potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "maxwell_far_field_electric_field_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_far_field_single_layer",  # Kernel type
        "maxwell_electric_far_field",  # Assembly type
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
    """Return a Maxwell magnetic far-field potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "maxwell_far_field_magnetic_field_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_far_field_single_layer",  # Kernel type
        "maxwell_magnetic_far_field",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )


# def magnetic_field(
# space, points, wavenumber, parameters=None, device_interface=None, precision=None
# ):
# """Return a Maxwell magnetic field potential operator."""
# from bempp_cl.core.dense_potential_assembler import DensePotentialAssembler
# from bempp_cl.api.operators import OperatorDescriptor
# from bempp_cl.api.operators import _add_wavenumber
# from bempp_cl.api.assembly.potential_operator import PotentialOperator

# options = {}
# options["KERNEL_FUNCTION"] = "helmholtz_gradient"
# _add_wavenumber(options, wavenumber)

# return PotentialOperator(
# DensePotentialAssembler(
# space,
# OperatorDescriptor(
# "maxwell_magnetic_field_potential", options, "maxwell_magnetic_field"
# ),
# points,
# 3,
# True,
# device_interface,
# precision,
# parameters=parameters,
# )
# )

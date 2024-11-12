"""Laplace potential operators."""


def single_layer(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
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
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(space, points, operator_descriptor, device_interface, assembler, parameters)
    )

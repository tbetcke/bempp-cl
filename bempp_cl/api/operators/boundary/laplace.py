"""Interfaces to Laplace operators."""

from bempp_cl.api.operators.boundary import common as _common


def single_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace single-layer boundary operator."""
    return _common.create_operator(
        "laplace_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_single_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace double-layer boundary operator."""
    return _common.create_operator(
        "laplace_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace adjoint double-layer boundary operator."""
    return _common.create_operator(
        "laplace_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Laplace hypersingular boundary operator."""
    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    return _common.create_operator(
        "laplace_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [],
        "laplace_single_layer",
        "laplace_hypersingular",
        device_interface,
        precision,
        False,
    )


def multitrace_operator(
    grid,
    target=None,
    space_type="p1",
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """
    Simplified version of multitrace operator assembly.

    Parameters
    ----------
    grid : Grid
        Bempp grid object.
    target : Grid
        The grid for the range spaces. If target is None then
        target is set to the input grid (that is the domain
        grid).
    space_type : string
        Controls which discretisation spaces are used.
        Supported values:
        - "p1", which means that all operators are discretised
        with P1 basis functions.
        - "p1-dp0", which means that P1 basis functions are
        used for the first row and column, and DP0 for the second.
        - "p1-dual0", which means that P1 basis functions are
        used for the first row and column, and DUAL0 for the second.
    parameters : Parameters
        An optional parameters object.
    assembler : string
        The assembler type.
    device_interface : DeviceInterface
        The device interface object to be used.
    precision : string
        Either "single" or "double" for single or
        double precision mode.

    Output
    ------
    The Helmholtz multitrace operator of the form
    [[-dlp, slp], [hyp, adj_dlp]], where
    dlp : double layer boundary operator
    slp : single layer boundary operator
    hyp : hypersingular boundary operator
    adj_dlp : adjoint double layer boundary operator.

    """
    import bempp_cl.api
    from bempp_cl.api.assembly.blocked_operator import BlockedOperator

    if space_type == "p1":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = space1
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = target_space1
            target_space_dual1 = target_space1
            target_space_dual0 = target_space1
        else:
            target_space1 = space1
            target_space0 = space1
            target_space_dual1 = space1
            target_space_dual0 = space1
    elif space_type == "p1-dp0":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = bempp_cl.api.function_space(grid, "DP", 0)
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = bempp_cl.api.function_space(target, "DP", 0)
            target_space_dual1 = target_space1
            target_space_dual0 = target_space0
    elif space_type == "p1-dual":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = bempp_cl.api.function_space(grid, "DUAL", 0)
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = bempp_cl.api.function_space(target, "DUAL", 0)
            target_space_dual1 = target_space1
            target_space_dual0 = target_space0
    else:
        raise ValueError(f"Unknown space type: {space_type}")

    slp = single_layer(
        space0,
        target_space1,
        target_space_dual0,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    dlp = double_layer(
        space1,
        target_space1,
        target_space_dual0,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    hyp = hypersingular(
        space1,
        target_space0,
        target_space_dual1,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    adj_dlp = adjoint_double_layer(
        space0,
        target_space0,
        target_space_dual1,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)

    blocked[0, 0] = -dlp
    blocked[0, 1] = slp
    blocked[1, 0] = hyp
    blocked[1, 1] = adj_dlp

    return blocked

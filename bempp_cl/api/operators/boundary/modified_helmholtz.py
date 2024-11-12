"""Interfaces to modified Helmholtz operators."""

from bempp_cl.api.operators.boundary import common as _common
import numpy as _np


def single_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz single-layer boundary operator."""
    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    return _common.create_operator(
        "modified_helmholtz_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [omega],
        "modified_helmholtz_single_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz double-layer boundary operator."""
    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    return _common.create_operator(
        "modified_helmholtz_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [omega],
        "modified_helmholtz_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz adj. double-layer boundary op."""
    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    return _common.create_operator(
        "modified_helmholtz_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [omega],
        "modified_helmholtz_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        False,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    omega,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the mod. Helmholtz hypersingular boundary op."""
    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    if _np.imag(omega) != 0:
        raise ValueError("'omega' must be real.")

    return _common.create_operator(
        "modified_helmholtz_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [omega],
        "modified_helmholtz_single_layer",
        "modified_helmholtz_hypersingular",
        device_interface,
        precision,
        False,
    )

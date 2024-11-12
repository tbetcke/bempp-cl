"""Unit tests for the dense assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

WAVENUMBER = 2.5
WAVENUMBER_COMPLEX = 2.5 + 1j


def test_helmholtz_single_layer_p0(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz slp with p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p0_p0")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_single_layer_p1_disc(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz slp with disc. p1 basis."""
    from bempp_cl.api.operators.boundary.helmholtz import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_dp1_dp1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_single_layer_p1_p0(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the slp with disc. p1/p0 basis."""
    from bempp_cl.api.operators.boundary.helmholtz import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space0,
        space1,
        space1,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_dp1_p0")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_single_layer_p0_p1(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the slp with disc. p0/p1 basis."""
    from bempp_cl.api.operators.boundary.helmholtz import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space1,
        space1,
        space0,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p0_dp1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_single_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz slp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.helmholtz import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_single_layer_boundary_p1_p1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_double_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz dlp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.helmholtz import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_double_layer_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_adjoint_double_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz adjoint dlp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.helmholtz import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_adj_double_layer_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_hypersingular(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Helmholtz hypersingular operator."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.helmholtz import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        WAVENUMBER,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("helmholtz_hypersingular_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))

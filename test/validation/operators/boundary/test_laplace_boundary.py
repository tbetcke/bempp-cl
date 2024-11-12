"""Unit tests for the dense assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_laplace_single_layer_p0(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace slp with p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "DP", 0)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        parameters=default_parameters,
        device_interface=device_interface,
        precision=precision,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p0_p0")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_p1_disc(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace slp with disc. p1 basis."""
    from bempp_cl.api.operators.boundary.laplace import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_dp1_dp1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_p1_p0(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the slp with disc. p1/p0 basis."""
    from bempp_cl.api.operators.boundary.laplace import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space0,
        space1,
        space1,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_dp1_p0")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_p0_p1(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the slp with disc. p0/p1 basis."""
    from bempp_cl.api.operators.boundary.laplace import single_layer
    from bempp_cl.api import function_space

    grid = helpers.load_grid("sphere")

    space0 = function_space(grid, "DP", 0)
    space1 = function_space(grid, "DP", 1)

    discrete_op = single_layer(
        space1,
        space1,
        space0,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p0_dp1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace slp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.laplace import single_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = single_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_single_layer_boundary_p1_p1")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_double_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace dlp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.laplace import double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_double_layer_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_adjoint_double_layer_p1_cont(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace adjoint dlp with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.laplace import adjoint_double_layer

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = adjoint_double_layer(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_adj_double_layer_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_hypersingular(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace hypersingular operator."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.laplace import hypersingular

    grid = helpers.load_grid("sphere")

    space = function_space(grid, "P", 1)

    discrete_op = hypersingular(
        space,
        space,
        space,
        assembler="dense",
        precision=precision,
        device_interface=device_interface,
        parameters=default_parameters,
    ).weak_form()

    expected = helpers.load_npy_data("laplace_hypersingular_boundary")

    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_element_p0(default_parameters, helpers, precision, device_interface):
    """Test dense assembler for the Laplace slp with p0 basis on the unit triangle."""
    from bempp_cl.api import Grid, function_space
    from bempp_cl.api.operators.boundary.laplace import single_layer

    vertices = _np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=_np.float64).T
    elements = _np.array([[0, 1, 2]]).T

    grid = Grid(vertices, elements)

    space = function_space(grid, "DP", 0)

    value = (
        single_layer(
            space,
            space,
            space,
            assembler="dense",
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .A[0, 0]
    )

    expected = 0.0798

    assert _np.isclose(value, expected, rtol=1e-3)

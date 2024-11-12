"""Unit tests for Laplace potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_laplace_single_layer_potential_p0(default_parameters, helpers, device_interface, precision):
    """Test Laplace slp potential with p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "DP", 0)

    data = helpers.load_npz_data("laplace_single_layer_potential_p0")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_potential_p1(default_parameters, helpers, device_interface, precision):
    """Test Laplace slp potential with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("laplace_single_layer_potential_p1")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_laplace_double_layer_potential_p1(default_parameters, helpers, device_interface, precision):
    """Test Laplace dlp potential with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import double_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("laplace_double_layer_potential_p1")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = double_layer(
        space,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_laplace_single_layer_potential_p1_complex(default_parameters, helpers, device_interface, precision):
    """Test Laplace slp potential with p1 basis and complex coeffs."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("laplace_single_layer_potential_p1")

    points = data["points"]
    coefficients = _np.random.rand(space.global_dof_count) + 1j * _np.random.rand(space.global_dof_count)

    fun = GridFunction(space, coefficients=coefficients)
    fun_real = GridFunction(space, coefficients=_np.real(coefficients))
    fun_complex = GridFunction(space, coefficients=_np.imag(coefficients))

    op = single_layer(
        space,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    )

    expected = op.evaluate(fun_real) + 1j * op.evaluate(fun_complex)
    actual = op.evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_laplace_p1_segments(default_parameters, helpers, device_interface, precision):
    """Test P1 potential evaluation on segments."""

    from bempp_cl.api.shapes import cube
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import single_layer

    grid = cube()
    seg1 = function_space(grid, "P", 1, segments=[1, 2, 3], include_boundary_dofs=False)
    seg2 = function_space(
        grid,
        "P",
        1,
        segments=[4, 5, 6],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    seg_all = function_space(grid, "DP", 1)

    random = _np.random.RandomState(0)

    coeffs1 = random.rand(seg1.global_dof_count)
    coeffs2 = random.rand(seg2.global_dof_count)
    coeffs_all = seg1.map_to_full_grid.dot(coeffs1) + seg2.map_to_full_grid.dot(coeffs2)

    points = 1.6 * _np.ones((3, 1)) + 0.5 * _np.random.rand(3, 20)

    fun1 = GridFunction(seg1, coefficients=coeffs1)
    fun2 = GridFunction(seg2, coefficients=coeffs2)
    fun_all = GridFunction(seg_all, coefficients=coeffs_all)

    seg1_res = single_layer(
        seg1,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun1)

    seg2_res = single_layer(
        seg2,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun2)

    seg_all_res = single_layer(
        seg_all,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun_all)

    actual = seg1_res + seg2_res
    expected = seg_all_res

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_laplace_p1_segments_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test P1 potential evaluation on segments with complex coeffs."""

    from bempp_cl.api.shapes import cube
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.laplace import single_layer

    grid = cube()
    seg1 = function_space(grid, "P", 1, segments=[1, 2, 3], include_boundary_dofs=False)
    seg2 = function_space(
        grid,
        "P",
        1,
        segments=[4, 5, 6],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    seg_all = function_space(grid, "DP", 1)

    random = _np.random.RandomState(0)

    coeffs1 = random.rand(seg1.global_dof_count) + 1j * random.rand(seg1.global_dof_count)
    coeffs2 = random.rand(seg2.global_dof_count) + 1j * random.rand(seg2.global_dof_count)
    coeffs_all = seg1.map_to_full_grid.dot(coeffs1) + seg2.map_to_full_grid.dot(coeffs2)

    points = 1.6 * _np.ones((3, 1)) + 0.5 * _np.random.rand(3, 20)

    fun1 = GridFunction(seg1, coefficients=coeffs1)
    fun2 = GridFunction(seg2, coefficients=coeffs2)
    fun_all = GridFunction(seg_all, coefficients=coeffs_all)

    seg1_res = single_layer(
        seg1,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun1)

    seg2_res = single_layer(
        seg2,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun2)

    seg_all_res = single_layer(
        seg_all,
        points,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun_all)

    actual = seg1_res + seg2_res
    expected = seg_all_res

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

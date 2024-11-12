"""Unit tests for Helmholtz far-field assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

WAVENUMBER = 2.5

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_helmholtz_single_layer_far_field(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz slp far-field with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.far_field.helmholtz import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("helmholtz_single_layer_far_field_p1")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = single_layer(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_double_layer_potential_p1(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz dlp far field with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.far_field.helmholtz import double_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("helmholtz_double_layer_far_field_p1")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = double_layer(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_far_field_segments(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz far field on segments."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.far_field.helmholtz import single_layer
    from bempp_cl.api.operators.far_field.helmholtz import double_layer
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)
    points = rand.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    for op in [single_layer, double_layer]:
        for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
            new_grid = grid_from_segments(grid, seglist)

            coeffs = rand.rand(new_grid.number_of_vertices)

            space1 = function_space(
                grid,
                "P",
                1,
                segments=seglist,
                swapped_normals=swapped_normals,
                include_boundary_dofs=True,
            )
            space2 = function_space(new_grid, "P", 1, swapped_normals=swapped_normals)

            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = (
                op(
                    space1,
                    points,
                    2.5,
                    device_interface=device_interface,
                    precision=precision,
                )
                * fun1
            )
            expected = (
                op(
                    space2,
                    points,
                    2.5,
                    device_interface=device_interface,
                    precision=precision,
                )
                * fun2
            )

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_far_field_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz far field ops with complex coefficients."""
    import bempp_cl.api
    from bempp_cl.api.operators.far_field.helmholtz import single_layer
    from bempp_cl.api.operators.far_field.helmholtz import double_layer

    grid = bempp_cl.api.shapes.regular_sphere(3)

    space = bempp_cl.api.function_space(grid, "P", 1)

    random = _np.random.RandomState(0)

    points = random.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    coeffs_real = random.rand(grid.number_of_vertices)
    coeffs_imag = random.rand(grid.number_of_vertices)
    coeffs = coeffs_real + 1j * coeffs_imag

    fun_real = bempp_cl.api.GridFunction(space, coefficients=coeffs_real)
    fun_imag = bempp_cl.api.GridFunction(space, coefficients=coeffs_imag)
    fun = bempp_cl.api.GridFunction(space, coefficients=coeffs)

    for op in [single_layer, double_layer]:
        far_field_op = op(
            space,
            points,
            WAVENUMBER,
            device_interface=device_interface,
            precision=precision,
        )
        actual_real = far_field_op * fun_real
        actual_imag = far_field_op * fun_imag
        actual = actual_real + 1j * actual_imag
        expected = far_field_op * fun
        _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_far_field_segments_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Maxwell potentials on segments with complex coeffs."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.far_field.helmholtz import single_layer
    from bempp_cl.api.operators.far_field.helmholtz import double_layer
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)
    points = rand.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    for op in [single_layer, double_layer]:
        for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
            new_grid = grid_from_segments(grid, seglist)

            coeffs = rand.rand(new_grid.number_of_vertices) + 1j * rand.rand(new_grid.number_of_vertices)

            space1 = function_space(
                grid,
                "P",
                1,
                segments=seglist,
                swapped_normals=swapped_normals,
                include_boundary_dofs=True,
            )
            space2 = function_space(new_grid, "P", 1, swapped_normals=swapped_normals)

            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = (
                op(
                    space1,
                    points,
                    2.5,
                    device_interface=device_interface,
                    precision=precision,
                )
                * fun1
            )
            expected = (
                op(
                    space2,
                    points,
                    2.5,
                    device_interface=device_interface,
                    precision=precision,
                )
                * fun2
            )

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

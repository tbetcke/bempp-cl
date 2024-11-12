"""Unit tests for Maxwell far-field assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

WAVENUMBER = 2.5

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_maxwell_electric_far_field(default_parameters, helpers, device_interface, precision):
    """Test Maxwell electric far field."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.far_field.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_far_field")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = electric_field(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_magnetic_far_field(default_parameters, helpers, device_interface, precision):
    """Test Maxwell magnetic far field."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.far_field.maxwell import magnetic_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_magnetic_far_field")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = magnetic_field(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_far_field_segments(default_parameters, helpers, device_interface, precision):
    """Test Maxwell far field on segments."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.far_field.maxwell import electric_field
    from bempp_cl.api.operators.far_field.maxwell import magnetic_field
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)
    points = rand.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    for op in [electric_field, magnetic_field]:
        for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
            new_grid = grid_from_segments(grid, seglist)

            coeffs = rand.rand(new_grid.number_of_edges)

            space1 = function_space(
                grid,
                "RWG",
                0,
                segments=seglist,
                swapped_normals=swapped_normals,
                include_boundary_dofs=True,
            )
            space2 = function_space(new_grid, "RWG", 0, swapped_normals=swapped_normals)

            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = (
                op(
                    space1,
                    points,
                    2.5,
                    parameters=default_parameters,
                    precision=precision,
                    device_interface=device_interface,
                )
                * fun1
            )
            expected = (
                op(
                    space2,
                    points,
                    2.5,
                    parameters=default_parameters,
                    precision=precision,
                    device_interface=device_interface,
                )
                * fun2
            )

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_far_field_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Maxwell far field ops with complex coefficients."""
    import bempp_cl.api
    from bempp_cl.api.operators.far_field.maxwell import electric_field
    from bempp_cl.api.operators.far_field.maxwell import magnetic_field

    grid = bempp_cl.api.shapes.regular_sphere(2)

    space = bempp_cl.api.function_space(grid, "RWG", 0)

    random = _np.random.RandomState(0)

    points = random.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    coeffs_real = random.rand(grid.number_of_edges)
    coeffs_imag = random.rand(grid.number_of_edges)
    coeffs = coeffs_real + 1j * coeffs_imag

    fun_real = bempp_cl.api.GridFunction(space, coefficients=coeffs_real)
    fun_imag = bempp_cl.api.GridFunction(space, coefficients=coeffs_imag)
    fun = bempp_cl.api.GridFunction(space, coefficients=coeffs)

    for op in [electric_field, magnetic_field]:
        far_field_op = op(
            space,
            points,
            WAVENUMBER,
            parameters=default_parameters,
            precision=precision,
            device_interface=device_interface,
        )
        actual_real = far_field_op * fun_real
        actual_imag = far_field_op * fun_imag
        actual = actual_real + 1j * actual_imag
        expected = far_field_op * fun
        _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_far_field_segments_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Maxwell potentials on segments with complex coeffs."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.far_field.maxwell import electric_field
    from bempp_cl.api.operators.far_field.maxwell import magnetic_field
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)

    points = rand.randn(3, 10)
    points /= _np.linalg.norm(points, axis=0)

    for op in [electric_field, magnetic_field]:
        for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
            new_grid = grid_from_segments(grid, seglist)

            coeffs = rand.rand(new_grid.number_of_edges) + 1j * rand.rand(new_grid.number_of_edges)

            space1 = function_space(
                grid,
                "RWG",
                0,
                segments=seglist,
                swapped_normals=swapped_normals,
                include_boundary_dofs=True,
            )
            space2 = function_space(new_grid, "RWG", 0, swapped_normals=swapped_normals)

            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = (
                op(
                    space1,
                    points,
                    2.5,
                    parameters=default_parameters,
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
                    parameters=default_parameters,
                    device_interface=device_interface,
                    precision=precision,
                )
                * fun2
            )

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

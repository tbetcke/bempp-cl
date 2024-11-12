"""Unit tests for Helmholtz potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

WAVENUMBER = 2.5

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_helmholtz_single_layer_potential_p1(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz slp potential with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.helmholtz import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("helmholtz_single_layer_potential_p1")

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
    """Test Helmholtz dlp potential with p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.helmholtz import double_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("helmholtz_double_layer_potential_p1")

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


def test_helmholtz_single_layer_potential_p1_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz slp potential with p1 basis and complex coeffs."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.helmholtz import single_layer

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "P", 1)

    data = helpers.load_npz_data("helmholtz_single_layer_potential_p1")

    points = data["points"]
    coefficients = _np.random.rand(space.global_dof_count) + 1j * _np.random.rand(space.global_dof_count)

    fun = GridFunction(space, coefficients=coefficients)
    fun_real = GridFunction(space, coefficients=_np.real(coefficients))
    fun_complex = GridFunction(space, coefficients=_np.imag(coefficients))

    op = single_layer(
        space,
        points,
        WAVENUMBER,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    )

    expected = op.evaluate(fun_real) + 1j * op.evaluate(fun_complex)
    actual = op.evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_potentials_segments(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz potentials on segments."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.potential.helmholtz import single_layer
    from bempp_cl.api.operators.potential.helmholtz import double_layer
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)

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

            points = _np.array([2.3, 1.3, 1.5]).reshape(3, 1) + rand.rand(3, 5)
            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = op(space1, points, 2.5) * fun1
            expected = op(space2, points, 2.5) * fun2

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_helmholtz_potentials_segments_complex_coeffs(default_parameters, helpers, device_interface, precision):
    """Test Helmholtz potentials on segments with complex coeffs."""
    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.potential.helmholtz import single_layer
    from bempp_cl.api.operators.potential.helmholtz import double_layer
    from bempp_cl.api.grid.grid import grid_from_segments

    grid = bempp_cl.api.shapes.multitrace_cube()

    seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
    swapped_normal_lists = [{}, {6}]

    rand = _np.random.RandomState(0)

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

            points = _np.array([2.3, 1.3, 1.5]).reshape(3, 1) + rand.rand(3, 5)
            fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
            fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

            actual = op(space1, points, 2.5) * fun1
            expected = op(space2, points, 2.5) * fun2

            _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))

"""Unit tests for Maxwell potential assemblers."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

WAVENUMBER = 2.5
WAVENUMBER_COMPLEX = 2.5 + 1j


def test_maxwell_electric_field_potential_complex(default_parameters, helpers, device_interface, precision):
    """Test Maxwell efield potential with complex wavenumber."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_field_potential_complex")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = electric_field(
        space,
        points,
        WAVENUMBER_COMPLEX,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


def test_maxwell_electric_field_potential_rwg(default_parameters, helpers, device_interface, precision):
    """Test Maxwell efield potential."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.maxwell import electric_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_electric_field_potential")

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


def test_maxwell_magnetic_field_potential_rwg(default_parameters, helpers, device_interface, precision):
    """Test Maxwell magnetic potential."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.maxwell import magnetic_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_magnetic_field_potential")

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


def test_maxwell_magnetic_field_potential_complex(default_parameters, helpers, device_interface, precision):
    """Test Maxwell magnetic potential with complex wavenumber."""
    from bempp_cl.api import function_space
    from bempp_cl.api import GridFunction
    from bempp_cl.api.operators.potential.maxwell import magnetic_field

    grid = helpers.load_grid("sphere")
    space = function_space(grid, "RWG", 0)

    data = helpers.load_npz_data("maxwell_magnetic_field_potential_complex")

    coefficients = data["vec"]
    points = data["points"]
    expected = data["result"]

    fun = GridFunction(space, coefficients=coefficients)

    actual = magnetic_field(
        space,
        points,
        WAVENUMBER_COMPLEX,
        parameters=default_parameters,
        precision=precision,
        device_interface=device_interface,
    ).evaluate(fun)

    _np.testing.assert_allclose(actual, expected, rtol=helpers.default_tolerance(precision))


# def test_maxwell_potentials_segments(
#         default_parameters, helpers, device_interface, precision):
#     """Test Maxwell potentials on segments."""
#     import bempp_cl.api
#     from bempp_cl.api import function_space
#     from bempp_cl.api import GridFunction
#     from bempp_cl.api.operators.potential.maxwell import electric_field
#     from bempp_cl.api.operators.potential.maxwell import magnetic_field
#     from bempp_cl.api.grid.grid import grid_from_segments

#     grid = bempp_cl.api.shapes.multitrace_cube()

#     seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
#     swapped_normal_lists = [{}, {6}]

#     rand = _np.random.RandomState(0)

#     for op in [electric_field, magnetic_field]:
#         for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
#             new_grid = grid_from_segments(grid, seglist)

#             coeffs = rand.rand(new_grid.number_of_edges)

#             space1 = function_space(grid, "RWG", 0, segments=seglist, swapped_normals=swapped_normals,
#                 include_boundary_dofs=True)
#             space2 = function_space(new_grid, "RWG", 0, swapped_normals=swapped_normals)

#             points = _np.array([2.3, 1.3, 1.5]).reshape(3, 1) + rand.rand(3, 5)
#             fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
#             fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

#             actual = op(space1, points, 2.5) * fun1
#             expected = op(space2, points, 2.5) * fun2

#             _np.testing.assert_allclose(
#                 actual, expected, rtol=helpers.default_tolerance(precision)
#             )

# def test_maxwell_potentials_segments_complex_coeffs(
#         default_parameters, helpers, device_interface, precision):
#     """Test Maxwell potentials on segments with complex coeffs."""
#     import bempp_cl.api
#     from bempp_cl.api import function_space
#     from bempp_cl.api import GridFunction
#     from bempp_cl.api.operators.potential.maxwell import electric_field
#     from bempp_cl.api.operators.potential.maxwell import magnetic_field
#     from bempp_cl.api.grid.grid import grid_from_segments

#     grid = bempp_cl.api.shapes.multitrace_cube()

#     seglists = [[1, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]]
#     swapped_normal_lists = [{}, {6}]

#     rand = _np.random.RandomState(0)

#     for op in [electric_field, magnetic_field]:
#         for seglist, swapped_normals in zip(seglists, swapped_normal_lists):
#             new_grid = grid_from_segments(grid, seglist)

#             coeffs = rand.rand(new_grid.number_of_edges) + 1j * rand.rand(new_grid.number_of_edges)

#             space1 = function_space(grid, "RWG", 0, segments=seglist, swapped_normals=swapped_normals,
#                 include_boundary_dofs=True)
#             space2 = function_space(new_grid, "RWG", 0, swapped_normals=swapped_normals)

#             points = _np.array([2.3, 1.3, 1.5]).reshape(3, 1) + rand.rand(3, 5)
#             fun1 = bempp_cl.api.GridFunction(space1, coefficients=coeffs)
#             fun2 = bempp_cl.api.GridFunction(space2, coefficients=coeffs)

#             actual = op(space1, points, 2.5) * fun1
#             expected = op(space2, points, 2.5) * fun2

#             _np.testing.assert_allclose(
#                 actual, expected, rtol=helpers.default_tolerance(precision)
#             )

import numpy as np
from math import sqrt

import bempp_cl.api


def test_interpolation_real_callable():
    @bempp_cl.api.real_callable
    def f(x, n, d, r):
        r[:] = x[0]

    grid = bempp_cl.api.shapes.cube(h=0.5)
    space = bempp_cl.api.function_space(grid, "P", 1)
    fun = bempp_cl.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(7 / 3))


def test_interpolation_complex_callable():
    @bempp_cl.api.complex_callable
    def f(x, n, d, r):
        r[:] = x[0] + 1j * x[0]

    grid = bempp_cl.api.shapes.cube(h=0.5)
    space = bempp_cl.api.function_space(grid, "P", 1)
    fun = bempp_cl.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(14 / 3))


def test_interpolation_python_callable():
    # This array can't be assembled by Numba, so jit cannot be used
    parameters = [1, None, 1.0]

    @bempp_cl.api.complex_callable(jit=False)
    def f(x, n, d, r):
        r[:] = x[0] * parameters[0] + 1j * x[1] * parameters[2]

    grid = bempp_cl.api.shapes.cube(h=0.5)
    space = bempp_cl.api.function_space(grid, "P", 1)
    fun = bempp_cl.api.GridFunction(space, fun=f)

    assert np.isclose(fun.l2_norm(), sqrt(14 / 3))


def test_vectorized_assembly():
    """Test vectorized assembly of grid functions."""
    from bempp_cl.api.integration.triangle_gauss import rule
    from bempp_cl.api.assembly.grid_function import get_function_quadrature_information

    grid = bempp_cl.api.shapes.cube()

    p1_space = bempp_cl.api.function_space(grid, "P", 1, segments=[1, 2], swapped_normals=[1])

    direction = np.array([1, 2, 3]) / np.sqrt(14)
    k = 2

    points, _ = rule(4)
    npoints = points.shape[1]

    grid_data = p1_space.grid.data("double")

    (
        global_points,
        global_normals,
        global_domain_indices,
    ) = get_function_quadrature_information(grid_data, p1_space.support_elements, p1_space.normal_multipliers, points)

    for index, element in enumerate(p1_space.support_elements):
        element_global_points = grid_data.local2global(element, points)
        np.testing.assert_allclose(
            global_points[:, index * npoints : (1 + index) * npoints],
            element_global_points,
        )
        for local_index in range(npoints):
            np.testing.assert_allclose(
                global_normals[:, index * npoints + local_index],
                grid_data.normals[element] * p1_space.normal_multipliers[element],
            )
            assert global_domain_indices[index * npoints + local_index] == grid_data.domain_indices[element]

    @bempp_cl.api.callable(complex=True)
    def fun_non_vec(x, n, d, res):
        res[0] = np.dot(n, direction) * 1j * k * np.exp(1j * k * np.dot(x, direction))

    @bempp_cl.api.callable(complex=True, vectorized=True)
    def fun_vec(x, n, d, res):
        res[0, :] = np.dot(direction, n) * 1j * k * np.exp(1j * k * np.dot(direction, x))

    grid_fun_non_vec = bempp_cl.api.GridFunction(p1_space, fun=fun_non_vec)
    grid_fun_vec = bempp_cl.api.GridFunction(p1_space, fun=fun_vec)

    rel_diff = np.abs(grid_fun_non_vec.projections() - grid_fun_vec.projections()) / np.abs(
        grid_fun_non_vec.projections()
    )

    assert np.max(rel_diff) < 1e-14

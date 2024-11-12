"""Benchmarks for dense assembly."""

import pytest

import bempp.api

PYTESTMARK = pytest.mark.usefixtures("default_parameters", "helpers")


# pylint: disable=C0103
def laplace_potential_dense_large_p0_benchmark(benchmark, default_parameters):
    """Benchmark for Laplace potential evaluation on large sphere"""

    from bempp.api.operators.potential.laplace import single_layer
    from bempp.api import function_space
    import numpy as np

    grid = bempp.api.shapes.regular_sphere(6)
    space = function_space(grid, "DP", 0)

    npoints = 10000

    theta = np.linspace(0, 2 * np.pi, npoints)
    points = np.vstack([np.cos(theta), np.sin(theta), 3 * np.ones(npoints, dtype="float64")])

    coefficients = np.random.randn(space.global_dof_count)
    grid_fun = bempp.api.GridFunction(space, coefficients=coefficients)

    fun = lambda: single_layer(space, points, parameters=default_parameters).evaluate(grid_fun)

    benchmark(fun)


def helmholtz_potential_dense_large_p0_benchmark(benchmark, default_parameters):
    """Benchmark for Helmholtz potential evaluation on large sphere"""

    from bempp.api.operators.potential.helmholtz import single_layer
    from bempp.api import function_space
    import numpy as np

    grid = bempp.api.shapes.regular_sphere(6)
    space = function_space(grid, "DP", 0)

    npoints = 10000

    theta = np.linspace(0, 2 * np.pi, npoints)
    points = np.vstack([np.cos(theta), np.sin(theta), 3 * np.ones(npoints, dtype="float64")])

    coefficients = np.random.randn(space.global_dof_count) + 1j * np.random.randn(space.global_dof_count)

    grid_fun = bempp.api.GridFunction(space, coefficients=coefficients)

    fun = lambda: single_layer(space, points, 2.5, parameters=default_parameters).evaluate(grid_fun)

    benchmark(fun)

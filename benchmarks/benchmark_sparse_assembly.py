import pytest

import bempp.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def sparse_identity_benchmark(benchmark, default_parameters):
    """Benchmark for assembling the identity operator."""

    from bempp.api.operators.boundary.sparse import identity
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(6)
    space = function_space(grid, "DP", 1)

    fun = lambda: identity(space, space, space, parameters=default_parameters).weak_form()

    benchmark(fun)

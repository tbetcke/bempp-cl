# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


@pytest.mark.parametrize("space_info", [("RWG", 0), ("DP", 0), ("P", 1)])
def test_barycentric(space_info, helpers, precision):
    """Test barycentric space."""
    import bempp_cl.api
    import math

    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = bempp_cl.api.function_space(grid, space_info[0], space_info[1])
    space_bary = space.barycentric_representation()

    # Define a function on this space

    rand = _np.random.RandomState(0)
    coeffs = rand.randn(space.global_dof_count)

    fun = bempp_cl.api.GridFunction(space, coefficients=coeffs)
    fun_bary = bempp_cl.api.GridFunction(space_bary, coefficients=coeffs)

    assert math.isclose(fun.l2_norm(), fun_bary.l2_norm(), rel_tol=helpers.default_tolerance(precision))

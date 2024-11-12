import pytest
import numpy as np
import bempp_cl.api


@pytest.mark.parametrize(
    "gridname, args, kwargs",
    [
        ("almond", [], {}),
        ("cylinders", [], {}),
        ("reentrant_cube", [], {}),
        ("screen", [np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])], {}),
        ("cube", [], {}),
        ("ellipsoid", [], {}),
        ("reference_triangle", [], {}),
        ("cuboid", [], {}),
        ("multitrace_cube", [], {}),
        ("multitrace_sphere", [], {}),
        ("regular_sphere", [2], {}),
        ("sphere", [], {}),
    ],
)
def test_shape(gridname, args, kwargs):
    getattr(bempp_cl.api.shapes, gridname)(*args, **kwargs)

"""Unit tests for function io."""

import os
import pytest
import bempp_cl.api
import numpy as np


@pytest.fixture
def folder():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("filename", ["testoutput_cube.msh", "testoutput_cube.vtk"])
@pytest.mark.parametrize(
    "space_type",
    [
        ("DP", 0),
        ("DP", 1),
        ("P", 1),
        ("DUAL", 0),
        ("DUAL", 1),
        ("RWG", 0),
        ("SNC", 0),
        ("BC", 0),
        ("RBC", 0),
    ],
)
def test_export(filename, folder, space_type):
    """Return geometries of two element grid."""
    grid = bempp_cl.api.shapes.cube(h=0.5)
    space = bempp_cl.api.function_space(grid, *space_type)
    function = bempp_cl.api.GridFunction(space, coefficients=np.random.rand(space.global_dof_count))
    bempp_cl.api.export(os.path.join(folder, filename), grid_function=function)

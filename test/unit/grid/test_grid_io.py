"""Unit tests for grid io."""

import os
import pytest
import bempp_cl.api


@pytest.fixture
def folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "grids")


@pytest.mark.parametrize("filename", ["sphere.msh"])
def test_import(filename, folder):
    """Return geometries of two element grid."""
    bempp_cl.api.grid.io.import_grid(os.path.join(folder, filename))


@pytest.mark.parametrize("filename", ["testoutput_cube.msh", "testoutput_cube.vtk"])
def test_export(filename, folder):
    """Return geometries of two element grid."""
    grid = bempp_cl.api.shapes.cube(h=0.5)
    bempp_cl.api.export(os.path.join(folder, filename), grid=grid)

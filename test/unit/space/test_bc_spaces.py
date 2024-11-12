"""Unit tests for Buffa-Christansen spaces."""

import bempp_cl.api
import pytest


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_segments(space_type):
    """Test creation of BC space on multitrace cube."""
    grid = bempp_cl.api.shapes.multitrace_cube(h=1)
    bempp_cl.api.function_space(grid, space_type, 0, segments=[1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_swapped_normals(space_type):
    """Test creation of BC space on multitrace cube with swapped normals."""
    grid = bempp_cl.api.shapes.multitrace_cube(h=1)
    bempp_cl.api.function_space(grid, space_type, 0, segments=[6, 7, 8, 9, 10, 11], swapped_normals=[6])


@pytest.mark.parametrize("space_type", ["BC", "RBC"])
def test_segments_6_7(space_type):
    """Test creation of BC space on multitrace cube."""
    grid = bempp_cl.api.shapes.multitrace_cube(h=1)
    bempp_cl.api.function_space(grid, space_type, 0, segments=[6, 7, 8, 9, 10, 11], swapped_normals=[6])

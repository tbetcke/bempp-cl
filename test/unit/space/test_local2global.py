"""Unit tests for Space objects."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest
import bempp_cl.api


@pytest.mark.parametrize(
    "grid",
    [
        bempp_cl.api.shapes.sphere(h=0.5),
        bempp_cl.api.shapes.cube(h=0.5),
        bempp_cl.api.shapes.shapes.screen(_np.array([(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)]), h=0.5),
    ],
)
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
@pytest.mark.parametrize("include_boundary_dofs", [True, False])
@pytest.mark.parametrize("truncate_at_segment_edge", [True, False])
def test_local2global(grid, space_type, include_boundary_dofs, truncate_at_segment_edge):
    """Check that data in local2global and global2local agree."""
    if len(grid.vertices[0]) - len(grid.edges[0]) + len(grid.elements[0]) != 2:
        # grid is a screen, not a polyhedron
        if space_type[0] in ["BC", "RBC"]:
            print("BC spaces not yet supported on screens")
            return

    space = bempp_cl.api.function_space(
        grid,
        *space_type,
        include_boundary_dofs=include_boundary_dofs,
        truncate_at_segment_edge=truncate_at_segment_edge,
    )
    test_local2global = _np.full(space.local2global.shape, -1, dtype=_np.int32)
    for i, locals in enumerate(space.global2local):
        for cell, dof in locals:
            assert space.local2global[cell][dof] == i
            test_local2global[cell][dof] = i
    for i, globals in enumerate(test_local2global):
        for j, dof in enumerate(globals):
            if dof == -1:
                assert _np.isclose(space.local_multipliers[i][j], 0)
                assert space.cell_dofs(i)[j] is None

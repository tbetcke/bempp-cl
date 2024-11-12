"""Unit tests for Space objects."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest


@pytest.mark.parametrize(
    "space_info",
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
def test_segment_space(space_info, helpers, precision):
    """Test that a space on a face of a cube has fewer DOFs."""
    import bempp_cl.api
    import math

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(grid, space_info[0], space_info[1])
    fun0 = bempp_cl.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
    )
    fun1 = bempp_cl.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    assert space0.global_dof_count > space1.global_dof_count
    assert fun0.l2_norm() > fun1.l2_norm()

    space2 = bempp_cl.api.function_space(
        grid, space_info[0], space_info[1], segments=[1], truncate_at_segment_edge=True
    )
    fun2 = bempp_cl.api.GridFunction(space2, coefficients=_np.ones(space2.global_dof_count))

    if space2.is_barycentric:
        c = 6
    else:
        c = 1

    evals = fun2.evaluate_on_element_centers()
    for n, i in enumerate(grid.domain_indices):
        if i != 1:
            for cell in range(c * n, c * (n + 1)):
                assert math.isclose(
                    _np.linalg.norm(evals[:, cell]),
                    0,
                    rel_tol=helpers.default_tolerance(precision),
                )


@pytest.mark.parametrize("space_info", [("P", 1), ("DUAL", 0), ("RWG", 0), ("SNC", 0), ("BC", 0), ("RBC", 0)])
def test_segments_space_with_boundary_dofs(space_info):
    """Test that space with boundary DOFs have more DOFs if these are included."""
    import bempp_cl.api

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=False,
    )
    fun0 = bempp_cl.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    fun1 = bempp_cl.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    assert space0.global_dof_count < space1.global_dof_count
    assert fun0.l2_norm() < fun1.l2_norm()


@pytest.mark.parametrize("space_info", [("DP", 0), ("DP", 1), ("DUAL", 1)])
def test_segments_space_without_boundary_dofs(space_info, helpers, precision):
    """Test that including boundary DOFs has no effect on these spaces."""
    import bempp_cl.api
    import math

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=False,
    )
    fun0 = bempp_cl.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    fun1 = bempp_cl.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    space2 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    fun2 = bempp_cl.api.GridFunction(space2, coefficients=_np.ones(space2.global_dof_count))

    assert space0.global_dof_count == space1.global_dof_count == space2.global_dof_count
    assert math.isclose(fun0.l2_norm(), fun1.l2_norm(), rel_tol=helpers.default_tolerance(precision))
    assert math.isclose(fun0.l2_norm(), fun2.l2_norm(), rel_tol=helpers.default_tolerance(precision))


@pytest.mark.parametrize(
    "space_info",
    [("P", 1), ("DUAL", 0), ("DUAL", 1), ("RWG", 0), ("SNC", 0), ("BC", 0), ("RBC", 0)],
)
def test_truncating(space_info):
    """Test that truncating these spaces at the boundary is correct."""
    import bempp_cl.api

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )
    fun0 = bempp_cl.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=True,
        truncate_at_segment_edge=True,
    )
    fun1 = bempp_cl.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    assert space0.global_dof_count == space1.global_dof_count
    assert fun1.l2_norm() < fun0.l2_norm()


@pytest.mark.parametrize("space_info", [("DUAL", 0)])
def test_truncating_node_dual_spaces(space_info, helpers, precision):
    """Test spaces on segments."""
    import bempp_cl.api
    import math

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=True,
    )
    fun0 = bempp_cl.api.GridFunction(space0, coefficients=_np.ones(space0.global_dof_count))

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=False,
    )
    fun1 = bempp_cl.api.GridFunction(space1, coefficients=_np.ones(space1.global_dof_count))

    assert space0.global_dof_count == space1.global_dof_count
    assert math.isclose(fun0.l2_norm(), fun1.l2_norm(), rel_tol=helpers.default_tolerance(precision))


@pytest.mark.parametrize("space_info", [("BC", 0), ("RBC", 0), ("DUAL", 1)])
def test_truncating_edge_and_face_dual_spaces(space_info):
    """Test spaces on segments."""
    import bempp_cl.api

    grid = bempp_cl.api.shapes.cube(h=0.4)

    space0 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=True,
    )

    space1 = bempp_cl.api.function_space(
        grid,
        space_info[0],
        space_info[1],
        segments=[1],
        include_boundary_dofs=False,
        truncate_at_segment_edge=False,
    )

    assert space0.global_dof_count == space1.global_dof_count
    assert _np.linalg.norm(space0.mass_matrix().to_sparse().toarray()) < _np.linalg.norm(
        space1.mass_matrix().to_sparse().toarray()
    )

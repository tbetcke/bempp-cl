"""Define scalar spaces on the dual grid."""

import numpy as _np
from bempp_cl.api import log

from .scalar_spaces import (
    p1_continuous_function_space,
    p0_discontinuous_function_space,
    _numba_p0_surface_gradient,
    _numba_p1_surface_gradient,
)


def dual0_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=False,
):
    """Define a space of DP0 functions on the dual grid."""
    from .space import SpaceBuilder, invert_local2global
    from scipy.sparse import coo_matrix

    coarse_space = p1_continuous_function_space(
        grid,
        support_elements,
        segments,
        swapped_normals,
        include_boundary_dofs=include_boundary_dofs,
        truncate_at_segment_edge=truncate_at_segment_edge,
    )

    number_of_support_elements = coarse_space.number_of_support_elements

    bary_support_elements = 6 * _np.repeat(coarse_space.support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    bary_grid = grid.barycentric_refinement
    bary_support_size = len(bary_support_elements)

    support = _np.zeros(bary_grid.number_of_elements, dtype=_np.bool_)
    support[bary_support_elements] = True

    local2global = _np.zeros((bary_grid.number_of_elements, 1), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid.number_of_elements, 1), dtype="uint32")

    local2global[support] = _np.arange(bary_support_size).reshape(bary_support_size, 1)

    local_multipliers[support] = 1
    global2local = invert_local2global(local2global, local_multipliers)

    _bary_dofs = []
    _coarse_dofs = []

    support_numbers = {j: i for i, j in enumerate(coarse_space.support_elements)}

    for global_dof_index in range(coarse_space.global_dof_count):
        local_dofs = coarse_space.global2local[global_dof_index]

        for face, vertex in local_dofs:
            if support[face] or not truncate_at_segment_edge:
                face_n = support_numbers[face]
                _coarse_dofs.append(global_dof_index)
                _coarse_dofs.append(global_dof_index)
                _bary_dofs.append(6 * face_n + (2 * vertex - 1) % 6)
                _bary_dofs.append(6 * face_n + 2 * vertex)

    nentries = len(_bary_dofs)

    coarse_dofs = _np.zeros(nentries, dtype=_np.uint32)
    coarse_dofs[:] = _coarse_dofs

    bary_dofs = _np.zeros(nentries, dtype=_np.uint32)
    bary_dofs[:] = _bary_dofs

    values = _np.ones(nentries, dtype=_np.float64)

    dof_transformation = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(bary_support_size, coarse_space.global_dof_count),
        dtype=_np.float64,
    ).tocsr()

    collocation_points = _np.array([[1.0 / 3], [1.0 / 3]])

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p0_discontinuous")
        .set_identifier("p0_discontinuous")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_collocation_points(collocation_points)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p0_surface_gradient)
        .build()
    )


def dual1_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=None,
    truncate_at_segment_edge=False,
):
    """Define a space of DP1 functions on the dual grid."""
    from .space import SpaceBuilder, invert_local2global
    from scipy.sparse import coo_matrix

    if include_boundary_dofs is not None:
        log("Setting include_boundary_dofs has no effect on this space type.", "warning")

    coarse_space = p0_discontinuous_function_space(grid, support_elements, segments, swapped_normals)

    coarse_support = _np.zeros(grid.number_of_elements, dtype=_np.bool_)
    coarse_support[coarse_space.support_elements] = True

    nentries = 0
    for global_dof_index in range(coarse_space.global_dof_count):
        local_dofs = coarse_space.global2local[global_dof_index]
        element_index = local_dofs[0][0]

        # 1 at barycentre of the triangle
        nentries += 6
        # 1/2 at the centre of each edge
        for e in range(3):
            edge = coarse_space.grid.element_edges[e][element_index]
            for neighbour in coarse_space.grid.edge_neighbors[edge]:
                if not truncate_at_segment_edge:
                    coarse_support[neighbour] = True
                if coarse_support[neighbour]:
                    nentries += 2
        # 1/num_coarse_triangles_at_vertex at each vertex
        for v in range(3):
            vertex = coarse_space.grid.elements[v][element_index]
            start = coarse_space.grid.vertex_neighbors.indexptr[vertex]
            end = coarse_space.grid.vertex_neighbors.indexptr[vertex + 1]
            neighbours = coarse_space.grid.vertex_neighbors.indices[start:end]
            for neighbour in neighbours:
                if not truncate_at_segment_edge:
                    coarse_support[neighbour] = True
                if coarse_support[neighbour]:
                    nentries += 2

    support_elements = [i for i, j in enumerate(coarse_support) if j]
    support_numbers = {j: i for i, j in enumerate(support_elements)}

    coarse_dofs = _np.zeros(nentries, dtype=_np.uint32)
    bary_dofs = _np.zeros(nentries, dtype=_np.uint32)
    values = _np.zeros(nentries, dtype=_np.float64)

    count = 0
    for global_dof_index in range(coarse_space.global_dof_count):
        local_dofs = coarse_space.global2local[global_dof_index]
        element_index = local_dofs[0][0]

        # 1 at barycentre of the triangle
        if coarse_support[element_index]:
            face_n = support_numbers[element_index]
            for n in [1, 5, 7, 11, 13, 17]:
                bary_dofs[count] = 6 * 3 * face_n + n
                coarse_dofs[count] = global_dof_index
                values[count] = 1
                count += 1
        # 1/2 at the centre of each edge
        for e in range(3):
            edge = coarse_space.grid.element_edges[e][element_index]
            for neighbour in coarse_space.grid.edge_neighbors[edge]:
                if coarse_support[neighbour]:
                    face_n = support_numbers[neighbour]
                    for i, dofs in enumerate([[1, 5], [13, 17], [7, 11]]):
                        if coarse_space.grid.element_edges[i][neighbour] == edge:
                            for n in dofs:
                                bary_dofs[count] = 6 * 3 * face_n + n
                                coarse_dofs[count] = global_dof_index
                                values[count] = 0.5
                                count += 1
                            break
        # 1/num_coarse_triangles_at_vertex at each vertex
        for v in range(3):
            vertex = coarse_space.grid.elements[v][element_index]
            start = coarse_space.grid.vertex_neighbors.indexptr[vertex]
            end = coarse_space.grid.vertex_neighbors.indexptr[vertex + 1]
            neighbour_count = end - start
            neighbours = coarse_space.grid.vertex_neighbors.indices[start:end]
            for neighbour in neighbours:
                if coarse_support[neighbour]:
                    face_n = support_numbers[neighbour]
                    for i, dofs in enumerate([[0, 15], [3, 6], [9, 12]]):
                        if coarse_space.grid.elements[i][neighbour] == vertex:
                            for n in dofs:
                                bary_dofs[count] = 6 * 3 * face_n + n
                                coarse_dofs[count] = global_dof_index
                                values[count] = 1 / neighbour_count
                                count += 1
                            break

    bary_grid = grid.barycentric_refinement

    number_of_support_elements = len(support_elements)

    coarse_elements = _np.array(support_elements, dtype=_np.uint32)

    bary_support_elements = 6 * _np.repeat(coarse_elements, 6) + _np.tile(_np.arange(6), number_of_support_elements)

    bary_support_size = len(bary_support_elements)

    support = _np.zeros(bary_grid.number_of_elements, dtype=_np.bool_)
    support[bary_support_elements] = True

    local2global = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")

    local2global[support] = _np.arange(3 * bary_support_size).reshape(bary_support_size, 3)

    local_multipliers[support] = 1
    global2local = invert_local2global(local2global, local_multipliers)

    dof_transformation = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(3 * bary_support_size, coarse_space.global_dof_count),
        dtype=_np.float64,
    ).tocsr()

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_order(1)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p1_discontinuous")
        .set_identifier("p1_discontinuous")
        .set_local2global(local2global)
        .set_global2local(global2local)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p1_surface_gradient)
        .build()
    )

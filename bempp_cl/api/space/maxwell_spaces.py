"""Definition of Maxwell spaces."""

import numpy as _np
import numba as _numba


def _is_screen(grid):
    """Check if there is an edge only adjacent to one triangle."""
    for e in range(grid.edges.shape[1]):
        if len([j for i in grid.element_edges for j in i if j == e]) < 2:
            return True
    return False


def rwg0_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=True,
):
    """Define a space of RWG functions of order 0."""
    from .space import SpaceBuilder, _process_segments
    from bempp_cl.api.utils.helpers import serialise_list_of_lists

    support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

    edge_neighbors, edge_neighbors_ptr = serialise_list_of_lists(grid.edge_neighbors)

    (
        global_dof_count,
        support,
        local2global,
        local_multipliers,
    ) = _compute_rwg0_space_data(
        support,
        edge_neighbors,
        edge_neighbors_ptr,
        grid.element_edges,
        grid.number_of_elements,
        grid.number_of_edges,
        include_boundary_dofs,
        truncate_at_segment_edge,
    )

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(False)
        .set_shapeset("rwg0")
        .set_identifier("rwg0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_barycentric_representation(rwg0_barycentric_function_space)
        .set_numba_evaluator(_numba_rwg0_evaluate)
        .build()
    )


def rwg0_barycentric_function_space(coarse_space):
    """Define a space of RWG functions of order 0 over a barycentric grid."""
    from .space import SpaceBuilder
    from scipy.sparse import coo_matrix

    number_of_support_elements = coarse_space.number_of_support_elements
    bary_grid_number_of_elements = 6 * coarse_space.grid.number_of_elements

    bary_support_elements = 6 * _np.repeat(coarse_space.support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    bary_support_size = len(bary_support_elements)

    support = _np.zeros(6 * coarse_space.grid.number_of_elements, dtype=_np.bool_)
    support[bary_support_elements] = True

    normal_multipliers = _np.repeat(coarse_space.normal_multipliers, 6)

    local_coords = _np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5], [1.0 / 3, 1.0 / 3]]).T

    coeffs = (
        _np.array(
            [
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
            ]
        ),
        _np.array(
            [
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
            ]
        ),
        _np.array(
            [
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
            ]
        ),
    )

    coarse_dofs, bary_dofs, values = generate_rwg0_map(
        coarse_space.grid.data(), coarse_space.support_elements, local_coords, coeffs
    )

    local2global = _np.zeros((bary_grid_number_of_elements, 3), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid_number_of_elements, 3), dtype="uint32")

    local2global[support] = _np.arange(3 * bary_support_size).reshape(bary_support_size, 3)

    local_multipliers[support] = 1

    transform = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(3 * bary_support_size, 3 * number_of_support_elements),
        dtype=_np.float64,
    ).tocsr()

    dof_transformation = transform @ coarse_space.map_to_localised_space

    return (
        SpaceBuilder(coarse_space.grid.barycentric_refinement)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("rwg0")
        .set_identifier("rwg0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_evaluator(_numba_rwg0_evaluate)
        .build()
    )


def snc0_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=True,
):
    """Define a space of SNC functions of order 0."""
    from .space import SpaceBuilder, _process_segments
    from bempp_cl.api.utils.helpers import serialise_list_of_lists

    support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

    edge_neighbors, edge_neighbors_ptr = serialise_list_of_lists(grid.edge_neighbors)

    (
        global_dof_count,
        support,
        local2global,
        local_multipliers,
    ) = _compute_rwg0_space_data(
        support,
        edge_neighbors,
        edge_neighbors_ptr,
        grid.element_edges,
        grid.number_of_elements,
        grid.number_of_edges,
        include_boundary_dofs,
        truncate_at_segment_edge,
    )

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(False)
        .set_shapeset("snc0")
        .set_identifier("snc0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_barycentric_representation(snc0_barycentric_function_space)
        .set_numba_evaluator(_numba_snc0_evaluate)
        .set_numba_surface_curl(_numba_snc0_surface_curl)
        .build()
    )


def snc0_barycentric_function_space(coarse_space):
    """Define a space of SNC functions of order 0 over a barycentric grid."""
    from .space import SpaceBuilder
    from scipy.sparse import coo_matrix

    number_of_support_elements = coarse_space.number_of_support_elements
    bary_grid_number_of_elements = 6 * coarse_space.grid.number_of_elements

    bary_support_elements = 6 * _np.repeat(coarse_space.support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    bary_support_size = len(bary_support_elements)

    support = _np.zeros(6 * coarse_space.grid.number_of_elements, dtype=_np.bool_)
    support[bary_support_elements] = True

    normal_multipliers = _np.repeat(coarse_space.normal_multipliers, 6)

    local_coords = _np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5], [1.0 / 3, 1.0 / 3]]).T

    coeffs = (
        _np.array(
            [
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
            ]
        ),
        _np.array(
            [
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
            ]
        ),
        _np.array(
            [
                [0, 0, 1.0 / 6],
                [1.0 / 3, 0, -1.0 / 6],
                [1, -1.0 / 3, 0],
                [-1.0 / 3, 1, 0],
                [0, 1.0 / 3, -1.0 / 6],
                [0, 0, 1.0 / 6],
            ]
        ),
    )

    coarse_dofs, bary_dofs, values = generate_rwg0_map(
        coarse_space.grid.data(), coarse_space.support_elements, local_coords, coeffs
    )

    local2global = _np.zeros((bary_grid_number_of_elements, 3), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid_number_of_elements, 3), dtype="uint32")

    local2global[support] = _np.arange(3 * bary_support_size).reshape(bary_support_size, 3)

    local_multipliers[support] = 1

    transform = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(3 * bary_support_size, 3 * number_of_support_elements),
        dtype=_np.float64,
    ).tocsr()

    dof_transformation = transform @ coarse_space.map_to_localised_space

    return (
        SpaceBuilder(coarse_space.grid.barycentric_refinement)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("rwg0")
        .set_identifier("snc0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_evaluator(_numba_snc0_evaluate)
        .build()
    )


def bc_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=True,
):
    """Define a space of BC functions."""
    from .space import SpaceBuilder

    if _is_screen(grid) and include_boundary_dofs:
        raise ValueError("Boundary dofs inclusion is not implemented for BC yet")

    bary_grid = grid.barycentric_refinement

    coarse_space = rwg0_function_space(
        grid,
        support_elements,
        segments,
        swapped_normals,
        include_boundary_dofs=include_boundary_dofs,
        truncate_at_segment_edge=truncate_at_segment_edge,
    )

    (
        dof_transformation,
        support,
        normal_multipliers,
        local2global,
        local_multipliers,
    ) = _compute_bc_space_data(grid, bary_grid, coarse_space, truncate_at_segment_edge, swapped_normals)

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("rwg0")
        .set_identifier("rwg0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_evaluator(_numba_rwg0_evaluate)
        .build()
    )


def rbc_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=True,
):
    """Define a space of RBC functions."""
    from .space import SpaceBuilder

    if _is_screen(grid) and include_boundary_dofs:
        raise ValueError("Boundary dofs inclusion is not implemented for RBC yet")

    bary_grid = grid.barycentric_refinement

    coarse_space = rwg0_function_space(
        grid,
        support_elements,
        segments,
        swapped_normals,
        include_boundary_dofs=include_boundary_dofs,
        truncate_at_segment_edge=truncate_at_segment_edge,
    )

    (
        dof_transformation,
        support,
        normal_multipliers,
        local2global,
        local_multipliers,
    ) = _compute_bc_space_data(grid, bary_grid, coarse_space, truncate_at_segment_edge, swapped_normals)

    return (
        SpaceBuilder(bary_grid)
        .set_codomain_dimension(3)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("rwg0")
        .set_identifier("snc0")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_dof_transformation(dof_transformation)
        .set_numba_evaluator(_numba_snc0_evaluate)
        .build()
    )


def _compute_bc_space_data(grid, bary_grid, coarse_space, truncate_at_segment_edge, swapped_normals):
    """Generate the BC map."""
    from bempp_cl.api.grid.grid import (
        _get_barycentric_support,
        _get_bary_coefficients,
        _get_coefficients_reference_edge,
        _get_data_multipliers,
        _get_barycentric_edges_associated_to_vertex,
    )

    from scipy.sparse import coo_matrix

    support, bary_support_size, bary_support_elements = _get_barycentric_support(
        truncate_at_segment_edge, grid, bary_grid, coarse_space
    )

    bary_vertex_to_edge, local2global, edge_lengths, normal_multipliers, local_multipliers = _get_data_multipliers(
        support, bary_grid, bary_support_elements, swapped_normals, coarse_space, bary_support_size
    )

    coarse_dofs = []
    bary_dofs = []
    values = []

    for global_dof_index in range(coarse_space.global_dof_count):
        # Local numbering of the edges consituting the support.
        local_dofs = coarse_space.global2local[global_dof_index]

        # Edge within the coarse support (reference edge).
        edge_index = grid.data().element_edges[local_dofs[0][1], local_dofs[0][0]]

        # Coarse elements associated to a reference edge
        neighbors = grid.edge_neighbors[edge_index]

        # Find upper and lower coarse elements over reference edge.
        other = neighbors[1] if local_dofs[0][0] == neighbors[0] else neighbors[0]
        if coarse_space.local_multipliers[local_dofs[0][0], local_dofs[0][1]] > 0:
            lower = local_dofs[0][0]
            upper = other
        else:
            lower = other
            upper = local_dofs[0][0]

        # For the BC function, pick the left and right vertex to which the
        # barycentric elements are associated to.
        vertex1, vertex2 = grid.data().edges[:, edge_index]

        # Define the right and left vertex depending on which is the upper
        # and lower element over the reference edge.
        for local_index, vertex_index in enumerate(grid.data().elements[:, upper]):
            if vertex_index == vertex1:
                break

        if vertex2 == grid.data().elements[(local_index - 1) % 3, upper]:
            vertex1, vertex2 = vertex2, vertex1

        # Get the local coarse vertex indices (0, 1 or 2) of vertex1 and vertex2 in upper and
        # lower coarse cell
        for index, value in enumerate(grid.data().elements[:, upper]):
            if value == vertex1:
                local_vertex1 = index
                break
        else:
            local_vertex1 = -1

        for index, value in enumerate(grid.data().elements[:, lower]):
            if value == vertex2:
                local_vertex2 = index
                break
        else:
            local_vertex2 = -1

        # Numbering of barycentric cells on reference edge:
        # 6 * coarse_element_number + 2 * coarse_vertex_number
        bary_upper_minus = 6 * upper + 2 * local_vertex1
        bary_upper_plus = 6 * upper + 2 * local_vertex1 + 1
        bary_lower_minus = 6 * lower + 2 * local_vertex2
        bary_lower_plus = 6 * lower + 2 * local_vertex2 + 1

        # Starting from the upper minus cell or the lower minus cell, append in
        # anti clock-wise order the barycentric cells that follor
        vertex_edges1, sorted_edges1, number_of_cells1, ref_edge1 = _get_barycentric_edges_associated_to_vertex(
            vertex1, bary_vertex_to_edge, bary_upper_minus, bary_grid
        )
        vertex_edges2, sorted_edges2, number_of_cells2, ref_edge2 = _get_barycentric_edges_associated_to_vertex(
            vertex2, bary_vertex_to_edge, bary_lower_minus, bary_grid
        )

        aux_values, aux_bary_dofs, aux_coarse_dofs = _get_bary_coefficients(
            edge_lengths,
            vertex_edges1,
            vertex_edges2,
            sorted_edges1,
            sorted_edges2,
            bary_grid,
            local2global,
            number_of_cells1,
            number_of_cells2,
            global_dof_index,
            ref_edge1,
            ref_edge2,
        )

        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs

        aux_values, aux_bary_dofs, aux_coarse_dofs = _get_coefficients_reference_edge(
            edge_lengths,
            bary_grid,
            local2global,
            global_dof_index,
            bary_upper_minus,
            bary_upper_plus,
            bary_lower_minus,
            bary_lower_plus,
        )

        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs

    nentries = len(coarse_dofs)
    np_coarse_dofs = _np.zeros(nentries, dtype=_np.uint32)
    np_bary_dofs = _np.zeros(nentries, dtype=_np.uint32)
    np_values = _np.zeros(nentries, dtype=_np.float64)

    np_coarse_dofs[:] = coarse_dofs
    np_bary_dofs[:] = bary_dofs
    np_values[:] = values

    dof_transformation = coo_matrix(
        (np_values, (np_bary_dofs, np_coarse_dofs)),
        shape=(3 * bary_support_size, coarse_space.global_dof_count),
        dtype=_np.float64,
    ).tocsr()

    return (
        dof_transformation,
        support,
        normal_multipliers,
        local2global,
        local_multipliers,
    )


@_numba.njit(cache=True)
def _compute_rwg0_space_data(
    support,
    edge_neighbors,
    edge_neighbors_ptr,
    element_edges,
    number_of_elements,
    number_of_edges,
    include_boundary_dofs,
    truncate_at_segment_edge,
):
    """Compute the local2global map for the space."""
    local2global_map = _np.zeros((number_of_elements, 3), dtype=_np.uint32)
    local_multipliers = _np.zeros((number_of_elements, 3), dtype=_np.float64)
    edge_dofs = -_np.ones(number_of_edges, dtype=_np.int32)
    dof_count = 0
    for element in _np.flatnonzero(support):
        has_dof = False
        for local_index in range(3):
            edge_index = element_edges[local_index, element]
            if edge_dofs[edge_index] != -1:
                has_dof = True
            else:
                current_neighbors = edge_neighbors[edge_neighbors_ptr[edge_index] : edge_neighbors_ptr[1 + edge_index]]
                supported_neighbors = [e for e in current_neighbors if support[e]]
                if len(supported_neighbors) == 2:
                    if edge_dofs[edge_index]:
                        edge_dofs[edge_index] = dof_count
                        dof_count += 1
                    has_dof = True
                if len(supported_neighbors) == 1 and include_boundary_dofs:
                    if edge_dofs[edge_index]:
                        edge_dofs[edge_index] = dof_count
                        dof_count += 1
                    has_dof = True
                    if not truncate_at_segment_edge:
                        for cell in current_neighbors:
                            # Add the element to the support
                            support[cell] = True
        if not has_dof:
            # If the element has no DOFs, remove it from support
            support[element] = False

    for element_index in _np.flatnonzero(support):
        dofmap = -_np.ones(3, dtype=_np.int32)
        for local_index in range(3):
            edge_index = element_edges[local_index, element_index]
            if edge_dofs[edge_index] != -1:
                dofmap[local_index] = edge_dofs[edge_index]

                current_neighbors = edge_neighbors[edge_neighbors_ptr[edge_index] : edge_neighbors_ptr[1 + edge_index]]
                supported_neighbors = [e for e in current_neighbors if support[e]]

                if len(supported_neighbors) == 1:
                    local_multipliers[element_index, local_index] = 1
                else:
                    # Assign 1 or -1 depending on element index
                    local_multipliers[element_index, local_index] = (
                        1 if element_index == min(supported_neighbors) else -1
                    )

        # For every zero local multiplier assign an existing global dof
        # in this element. This does not change the result as zero multipliers
        # do not contribute. But it allows us not to have to distinguish between
        # existing and non existing dofs later on.
        first_nonzero = 0
        for local_index in range(3):
            if local_multipliers[element_index, local_index] != 0:
                first_nonzero = local_index
                break

        for local_index in range(3):
            if local_multipliers[element_index, local_index] == 0:
                dofmap[local_index] = dofmap[first_nonzero]
        local2global_map[element_index, :] = dofmap

    return dof_count, support, local2global_map, local_multipliers


@_numba.njit(cache=True)
def generate_rwg0_map(grid_data, support_elements, local_coords, coeffs):
    """Actually generate the sparse matrix data."""
    number_of_elements = len(support_elements)

    coarse_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    bary_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    values = _np.empty(3 * 18 * number_of_elements, dtype=_np.float64)

    # Iterate through the global dofs and fill up the
    # corresponding coefficients.

    count = 0

    for index, elem_index in enumerate(support_elements):
        # Compute all the local vertices
        local_vertices = grid_data.local2global(elem_index, local_coords)
        l1 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 4])
        l2 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 3])
        l3 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 5])
        l4 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 2])
        l5 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 1])
        l6 = _np.linalg.norm(local_vertices[:, 6] - local_vertices[:, 0])
        le1 = _np.linalg.norm(local_vertices[:, 2] - local_vertices[:, 0])
        le2 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 0])
        le3 = _np.linalg.norm(local_vertices[:, 4] - local_vertices[:, 2])

        outer_edges = [le1, le2, le3]

        dof_mult = _np.array(
            [
                [le1, l6, l5],
                [l4, le1, l5],
                [le3, l4, l2],
                [l1, le3, l2],
                [le2, l1, l3],
                [l6, le2, l3],
            ]
        )

        # Assign the dofs for the six barycentric elements

        bary_elements = _np.arange(6) + 6 * index
        for local_dof in range(3):
            coarse_dof = 3 * index + local_dof
            bary_coeffs = coeffs[local_dof]
            dof_coeffs = bary_coeffs * outer_edges[local_dof] / dof_mult
            coarse_dofs[count : count + 18] = coarse_dof
            bary_dofs[count : count + 18] = _np.arange(3 * bary_elements[0], 3 * bary_elements[0] + 18)
            values[count : count + 18] = dof_coeffs.ravel()
            count += 18
    return coarse_dofs, bary_dofs, values


@_numba.njit()
def _numba_rwg0_evaluate(
    element_index,
    shapeset_evaluate,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the basis on an element."""
    reference_values = shapeset_evaluate(local_coordinates)
    npoints = local_coordinates.shape[1]
    result = _np.empty((3, 3, npoints), dtype=_np.float64)

    edge_lengths = _np.empty(3, dtype=_np.float64)
    edge_lengths[0] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[0, element_index]]
        - grid_data.vertices[:, grid_data.elements[1, element_index]]
    )
    edge_lengths[1] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[2, element_index]]
        - grid_data.vertices[:, grid_data.elements[0, element_index]]
    )
    edge_lengths[2] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[1, element_index]]
        - grid_data.vertices[:, grid_data.elements[2, element_index]]
    )

    for index in range(3):
        result[:, index, :] = (
            local_multipliers[element_index, index]
            * edge_lengths[index]
            / grid_data.integration_elements[element_index]
            * grid_data.jacobians[element_index].dot(reference_values[:, index, :])
        )
    return result


@_numba.njit()
def _numba_snc0_evaluate(
    element_index,
    shapeset_evaluate,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the basis on an element."""
    reference_values = shapeset_evaluate(local_coordinates)
    npoints = local_coordinates.shape[1]
    result = _np.empty((3, 3, npoints), dtype=_np.float64)
    tmp = _np.empty((3, 3, npoints), dtype=_np.float64)
    normal = grid_data.normals[element_index] * normal_multipliers[element_index]

    edge_lengths = _np.empty(3, dtype=_np.float64)
    edge_lengths[0] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[0, element_index]]
        - grid_data.vertices[:, grid_data.elements[1, element_index]]
    )
    edge_lengths[1] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[2, element_index]]
        - grid_data.vertices[:, grid_data.elements[0, element_index]]
    )
    edge_lengths[2] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[1, element_index]]
        - grid_data.vertices[:, grid_data.elements[2, element_index]]
    )

    for index in range(3):
        tmp[:, index, :] = (
            local_multipliers[element_index, index]
            * edge_lengths[index]
            / grid_data.integration_elements[element_index]
            * grid_data.jacobians[element_index].dot(reference_values[:, index, :])
        )

    result[0, :, :] = normal[1] * tmp[2, :, :] - normal[2] * tmp[1, :, :]
    result[1, :, :] = normal[2] * tmp[0, :, :] - normal[0] * tmp[2, :, :]
    result[2, :, :] = normal[0] * tmp[1, :, :] - normal[1] * tmp[0, :, :]

    return result


@_numba.njit
def _numba_snc0_surface_curl(
    element_index,
    shapeset_gradient,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the curl on an element."""
    normal = grid_data.normals[element_index] * normal_multipliers[element_index]
    reference_derivatives = shapeset_gradient(local_coordinates)
    jac_inv_t = grid_data.jac_inv_trans[element_index]
    derivatives = jac_inv_t @ reference_derivatives @ jac_inv_t.T
    reference_values = (
        normal[0] * (derivatives[2, 1] - derivatives[1, 2])
        + normal[1] * (derivatives[0, 2] - derivatives[2, 0])
        + normal[2] * (derivatives[1, 0] - derivatives[0, 1])
    )

    result = _np.empty(3, dtype=_np.float64)

    edge_lengths = _np.empty(3, dtype=_np.float64)
    edge_lengths[0] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[0, element_index]]
        - grid_data.vertices[:, grid_data.elements[1, element_index]]
    )
    edge_lengths[1] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[2, element_index]]
        - grid_data.vertices[:, grid_data.elements[0, element_index]]
    )
    edge_lengths[2] = _np.linalg.norm(
        grid_data.vertices[:, grid_data.elements[1, element_index]]
        - grid_data.vertices[:, grid_data.elements[2, element_index]]
    )

    for index in range(3):
        result[index] = local_multipliers[element_index, index] * edge_lengths[index] * reference_values
    return result

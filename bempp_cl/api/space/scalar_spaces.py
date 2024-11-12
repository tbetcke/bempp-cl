"""Definition of scalar function spaces."""

from bempp_cl.helpers import timeit as _timeit

import numpy as _np
import numba as _numba
from bempp_cl.api import log


@_timeit
def p0_discontinuous_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=None,
    truncate_at_segment_edge=None,
):
    """Define a space of piecewise constant functions."""
    from .space import SpaceBuilder, _process_segments

    if include_boundary_dofs is not None:
        log("Setting include_boundary_dofs has no effect on this space type.", "warning")
    if truncate_at_segment_edge is not None:
        log(
            "Setting truncate_at_segment_edge has no effect on this space type.",
            "warning",
        )

    support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

    elements_in_support = _np.flatnonzero(support)
    support_size = len(elements_in_support)

    local2global = _np.zeros((grid.number_of_elements, 1), dtype="uint32")
    local2global[support] = _np.expand_dims(_np.arange(support_size, dtype="uint32"), 1)

    local_multipliers = _np.zeros((grid.number_of_elements, 1), dtype="float64")
    local_multipliers[support] = 1

    collocation_points = _np.array([[1.0 / 3], [1.0 / 3]])

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_shapeset("p0_discontinuous")
        .set_identifier("p0_discontinuous")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_collocation_points(collocation_points)
        .set_barycentric_representation(p0_barycentric_discontinuous_function_space)
        .set_numba_surface_gradient(_numba_p0_surface_gradient)
        .build()
    )


@_timeit
def p0_barycentric_discontinuous_function_space(coarse_space):
    """Define a space of piecewise constant functions over a barycentric grid."""
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

    coarse_dofs = _np.repeat(_np.arange(number_of_support_elements, dtype=_np.uint32), 6)
    bary_dofs = _np.arange(6 * number_of_support_elements, dtype=_np.uint32)
    values = _np.ones(6 * number_of_support_elements, dtype=_np.float64)

    local2global = _np.zeros((bary_grid_number_of_elements, 1), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid_number_of_elements, 1), dtype="uint32")

    local2global[support] = _np.arange(bary_support_size).reshape(bary_support_size, 1)

    local_multipliers[support] = 1

    transform = coo_matrix(
        (values, (bary_dofs, coarse_dofs)),
        shape=(bary_support_size, number_of_support_elements),
        dtype=_np.float64,
    ).tocsr()

    dof_transformation = transform @ coarse_space.map_to_localised_space

    collocation_points = _np.array([[1.0 / 3], [1.0 / 3]])

    return (
        SpaceBuilder(coarse_space.grid.barycentric_refinement)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p0_discontinuous")
        .set_identifier("p0_discontinuous")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_collocation_points(collocation_points)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p0_surface_gradient)
        .build()
    )


@_timeit
def p1_discontinuous_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=None,
    truncate_at_segment_edge=None,
):
    """Define a discontinuous space of piecewise linear functions."""
    from .space import SpaceBuilder, _process_segments

    if include_boundary_dofs is not None:
        log("Setting include_boundary_dofs has no effect on this space type.", "warning")
    if truncate_at_segment_edge is not None:
        log(
            "Setting truncate_at_segment_edge has no effect on this space type.",
            "warning",
        )

    support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

    elements_in_support = _np.flatnonzero(support)
    support_size = len(elements_in_support)

    local2global = _np.zeros((grid.number_of_elements, 3), dtype="uint32")
    local2global[support] = _np.arange(3 * support_size).reshape(support_size, 3)

    local_multipliers = _np.zeros((grid.number_of_elements, 3), dtype="float64")
    local_multipliers[support] = 1

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(1)
        .set_is_localised(True)
        .set_shapeset("p1_discontinuous")
        .set_identifier("p1_discontinuous")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_numba_surface_gradient(_numba_p1_surface_gradient)
        .build()
    )


@_timeit
def p1_continuous_function_space(
    grid,
    support_elements=None,
    segments=None,
    swapped_normals=None,
    include_boundary_dofs=False,
    truncate_at_segment_edge=True,
):
    """Define a space of continuous piecewise linear functions."""
    from .space import SpaceBuilder, _process_segments

    support, normal_multipliers = _process_segments(grid, support_elements, segments, swapped_normals)

    # Create list of vertex neighbors. Needed for dofmap computation

    # vertex_neighbors = [[] for _ in range(grid.number_of_vertices)]
    # for index in range(grid.number_of_elements):
    # for vertex in grid.elements[:, index]:
    # vertex_neighbors[vertex].append(index)
    # vertex_neighbors1, index_ptr1 = serialise_list_of_lists(vertex_neighbors)

    vertex_neighbors, index_ptr = grid.vertex_neighbors

    local2global, local_multipliers, support = _compute_p1_dof_map(
        grid.data(),
        support,
        include_boundary_dofs,
        truncate_at_segment_edge,
        vertex_neighbors,
        index_ptr,
    )

    return (
        SpaceBuilder(grid)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(1)
        .set_is_localised(False)
        .set_shapeset("p1_discontinuous")
        .set_identifier("p1_continuous")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_barycentric_representation(p1_barycentric_continuous_function_space)
        .set_numba_surface_gradient(_numba_p1_surface_gradient)
        .build()
    )


@_timeit
def p1_barycentric_continuous_function_space(coarse_space):
    """Define a space of piecewise constant functions over a barycentric grid."""
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

    coeffs = [
        _np.array(
            [
                [1.0, 1 / 3, 1 / 2],
                [1.0, 1 / 2, 1 / 3],
                [0.0, 1 / 3, 1 / 2],
                [0.0, 0.0, 1 / 3],
                [0.0, 1 / 3, 0.0],
                [0.0, 1 / 2, 1 / 3],
            ]
        ),
        _np.array(
            [
                [0.0, 1 / 3, 0.0],
                [0.0, 1 / 2, 1 / 3],
                [1.0, 1 / 3, 1 / 2],
                [1.0, 1 / 2, 1 / 3],
                [0.0, 1 / 3, 1 / 2],
                [0.0, 0.0, 1 / 3],
            ]
        ),
        _np.array(
            [
                [0.0, 1 / 3, 1 / 2],
                [0.0, 0.0, 1 / 3],
                [0.0, 1 / 3, 0.0],
                [0.0, 1 / 2, 1 / 3],
                [1.0, 1 / 3, 1 / 2],
                [1.0, 1 / 2, 1 / 3],
            ]
        ),
    ]

    coarse_dofs, bary_dofs, values = generate_p1_map(coarse_space.grid.data(), coarse_space.support_elements, coeffs)

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

    collocation_points = _np.array([[1.0 / 3], [1.0 / 3]])

    return (
        SpaceBuilder(coarse_space.grid.barycentric_refinement)
        .set_codomain_dimension(1)
        .set_support(support)
        .set_normal_multipliers(normal_multipliers)
        .set_order(0)
        .set_is_localised(True)
        .set_is_barycentric(True)
        .set_shapeset("p1_discontinuous")
        .set_identifier("p1_continuous")
        .set_local2global(local2global)
        .set_local_multipliers(local_multipliers)
        .set_collocation_points(collocation_points)
        .set_dof_transformation(dof_transformation)
        .set_numba_surface_gradient(_numba_p0_surface_gradient)
        .build()
    )


@_numba.njit(cache=True)
def generate_p1_map(grid_data, support_elements, coeffs):
    """Actually generate the sparse matrix data."""
    number_of_elements = len(support_elements)

    coarse_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    bary_dofs = _np.empty(3 * 18 * number_of_elements, dtype=_np.uint32)
    values = _np.empty(3 * 18 * number_of_elements, dtype=_np.float64)

    # Iterate through the global dofs and fill up the
    # corresponding coefficients.

    count = 0

    for index, elem_index in enumerate(support_elements):
        # Assign the dofs for the six barycentric elements

        bary_elements = _np.arange(6) + 6 * index
        for local_dof in range(3):
            coarse_dof = 3 * index + local_dof
            bary_coeffs = coeffs[local_dof]
            coarse_dofs[count : count + 18] = coarse_dof
            bary_dofs[count : count + 18] = _np.arange(3 * bary_elements[0], 3 * bary_elements[0] + 18)
            values[count : count + 18] = bary_coeffs.ravel()
            count += 18
    return coarse_dofs, bary_dofs, values


@_timeit
@_numba.njit(cache=True)
def _compute_p1_dof_map(
    grid_data,
    support,
    include_boundary_dofs,
    truncate_at_segment_edge,
    vertex_neighbors,
    index_ptr,
):
    """Compute the local2global and local_multipliers maps for P1 space."""

    def find_index(array, value):
        """Return first position of value in array."""
        for index, val in enumerate(array):
            if val == value:
                return index
        return -1

    elements_in_support = []
    for index, val in enumerate(support):
        if val:
            elements_in_support.append(index)

    number_of_elements = grid_data.elements.shape[1]
    number_of_vertices = grid_data.vertices.shape[1]
    local2global = -_np.ones((number_of_elements, 3), dtype=_np.int32)

    vertex_is_dof = _np.zeros(number_of_vertices, dtype=_np.bool_)
    extended_support = []

    for element_index in elements_in_support:
        for local_index in range(3):
            vertex = grid_data.elements[local_index, element_index]
            neighbors = vertex_neighbors[index_ptr[vertex] : index_ptr[vertex + 1]]
            non_support_neighbors = [n for n in neighbors if not support[n]]
            node_is_interior = len(non_support_neighbors) == 0 and not grid_data.vertex_on_boundary[vertex]
            if include_boundary_dofs or node_is_interior:
                # Just add dof
                local2global[element_index, local_index] = vertex
                vertex_is_dof[vertex] = True
            if len(non_support_neighbors) > 0 and not truncate_at_segment_edge and include_boundary_dofs:
                for en in non_support_neighbors:
                    extended_support.append(en)
                    other_local_index = find_index(grid_data.elements[:, en], vertex)
                    local2global[en, other_local_index] = vertex
                    # vertex_is_dof was already set to True in previous if.

    # We have now all the vertices that have dofs attached and local2global
    # has the vertex index if it is used and -1 otherwise.

    # Now need to convert vertex indices into actual dof indices. For subgrids
    # these two are not identical.

    support_final = _np.zeros(number_of_elements, dtype=_np.bool_)
    local2global_final = _np.zeros((number_of_elements, 3), dtype=_np.uint32)
    local_multipliers = _np.zeros((number_of_elements, 3), dtype=_np.float64)

    dofs = -_np.ones(number_of_vertices)
    used_dofs = _np.flatnonzero(vertex_is_dof)
    global_dof_count = len(used_dofs)
    dofs[used_dofs] = _np.arange(global_dof_count)

    # Iterate through all support elements and replace vertex indices by
    # dof indices

    elements_in_support.extend(set(extended_support))

    for element_index in elements_in_support:
        for local_index in range(3):
            vertex_index = local2global[element_index, local_index]
            if vertex_index == -1:
                continue
            mapped_dof = dofs[vertex_index]
            support_final[element_index] = True
            local2global_final[element_index, local_index] = mapped_dof
            local_multipliers[element_index, local_index] = 1
        # If not every local index was used in a grid we need to
        # map the non-used local dofs to some global dof. Use the
        # one with maximal index in element. The corresponding
        # multipliers are set to zero so that these artificial dofs
        # do not influence computations.
        if support_final[element_index]:
            max_dof = _np.max(local2global_final[element_index])
            for local_index in range(3):
                if local2global[element_index, local_index] == -1:
                    local2global_final[element_index, local_index] = max_dof

    return local2global_final, local_multipliers, support_final


@_numba.njit
def _numba_p0_surface_gradient(
    element_index,
    shapeset_gradient,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the surface gradient."""
    return _np.zeros((1, 3, 1, local_coordinates.shape[1]), dtype=_np.float64)


@_numba.njit
def _numba_p1_surface_gradient(
    element_index,
    shapeset_gradient,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the surface gradient."""
    reference_values = shapeset_gradient(local_coordinates)
    result = _np.empty((1, 3, 3, local_coordinates.shape[1]), dtype=_np.float64)
    for index in range(3):
        result[0, :, index, :] = grid_data.jac_inv_trans[element_index].dot(reference_values[0, :, index, :])
    return result

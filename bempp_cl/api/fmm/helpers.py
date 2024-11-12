"""FMM helper functions."""

import numpy as _np
import numba as _numba

M_INV_4PI = 1.0 / (4 * _np.pi)


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_kernel(target_points, source_points, kernel_parameters, dtype, result_type):
    """Evaluate the Laplace kernel."""
    ntargets = target_points.shape[1]
    nsources = source_points.shape[1]
    m_inv_4pi = dtype.type(M_INV_4PI)

    interactions = _np.empty(4 * ntargets * nsources, dtype=result_type)
    diff = _np.zeros((3, nsources), dtype=dtype)
    for target_point_index in range(ntargets):
        dist = _np.zeros(nsources, dtype=dtype)
        for i in range(3):
            for j in range(nsources):
                diff[i, j] = target_points[i, target_point_index] - source_points[i, j]
                dist[j] += diff[i, j] * diff[i, j]
        for j in range(nsources):
            dist[j] = _np.sqrt(dist[j])
        for j in range(nsources):
            interactions[target_point_index * 4 * nsources + 4 * j] = m_inv_4pi / dist[j]
        for i in range(3):
            for j in range(nsources):
                interactions[target_point_index * 4 * nsources + 4 * j + 1 + i] = (
                    -diff[i, j] * m_inv_4pi / (dist[j] * dist[j] * dist[j])
                )
        # Now fix zero distance case
        for j in range(nsources):
            if dist[j] == 0:
                for i in range(4):
                    interactions[target_point_index * 4 * nsources + 4 * j + i] = 0

    return interactions


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_kernel(target_points, source_points, kernel_parameters, dtype, result_type):
    """Evaluate the modified Helmholtz kernel."""
    ntargets = target_points.shape[1]
    nsources = source_points.shape[1]
    m_inv_4pi = dtype.type(M_INV_4PI)

    interactions = _np.empty(4 * ntargets * nsources, dtype=result_type)
    diff = _np.zeros((3, nsources), dtype=dtype)
    for target_point_index in range(ntargets):
        dist = _np.zeros(nsources, dtype=dtype)
        for i in range(3):
            for j in range(nsources):
                diff[i, j] = target_points[i, target_point_index] - source_points[i, j]
                dist[j] += diff[i, j] * diff[i, j]
        for j in range(nsources):
            dist[j] = _np.sqrt(dist[j])
        for j in range(nsources):
            interactions[target_point_index * 4 * nsources + 4 * j] = (
                m_inv_4pi * _np.exp(-kernel_parameters[0] * dist[j]) / dist[j]
            )
        for i in range(3):
            for j in range(nsources):
                interactions[target_point_index * 4 * nsources + 4 * j + 1 + i] = (
                    (-kernel_parameters[0] * dist[j] - 1)
                    * diff[i, j]
                    * _np.exp(-kernel_parameters[0] * dist[j])
                    * m_inv_4pi
                    / (dist[j] * dist[j] * dist[j])
                )
        # Now fix zero distance case
        for j in range(nsources):
            if dist[j] == 0:
                for i in range(4):
                    interactions[target_point_index * 4 * nsources + 4 * j + i] = 0

    return interactions


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_kernel(target_points, source_points, kernel_parameters, dtype, result_type):
    """Evaluate the Laplace kernel."""
    ntargets = target_points.shape[1]
    nsources = source_points.shape[1]
    m_inv_4pi = dtype.type(M_INV_4PI)

    tmp_real = _np.empty(nsources, dtype=dtype)
    tmp_imag = _np.empty(nsources, dtype=dtype)
    interactions_real = _np.empty(4 * ntargets * nsources, dtype=dtype)
    interactions_imag = _np.empty(4 * ntargets * nsources, dtype=dtype)
    diff = _np.zeros((3, nsources), dtype=dtype)
    for target_point_index in range(ntargets):
        dist = _np.zeros(nsources, dtype=dtype)
        for i in range(3):
            for j in range(nsources):
                diff[i, j] = target_points[i, target_point_index] - source_points[i, j]
                dist[j] += diff[i, j] * diff[i, j]
        for j in range(nsources):
            dist[j] = _np.sqrt(dist[j])
        for j in range(nsources):
            tmp_real[j] = m_inv_4pi * _np.cos(kernel_parameters[0] * dist[j]) / dist[j]
            tmp_imag[j] = m_inv_4pi * _np.sin(kernel_parameters[0] * dist[j]) / dist[j]
        if kernel_parameters[1] != 0:
            for j in range(nsources):
                tmp_real[j] *= _np.exp(-kernel_parameters[1] * dist[j])
                tmp_imag[j] *= _np.exp(-kernel_parameters[1] * dist[j])
        for j in range(nsources):
            interactions_real[target_point_index * 4 * nsources + 4 * j] = tmp_real[j]
            interactions_imag[target_point_index * 4 * nsources + 4 * j] = tmp_imag[j]
        for i in range(3):
            for j in range(nsources):
                interactions_real[target_point_index * 4 * nsources + 4 * j + 1 + i] = (
                    (-(1 + dist[j] * kernel_parameters[1]) * tmp_real[j] - kernel_parameters[0] * dist[j] * tmp_imag[j])
                    / (dist[j] * dist[j])
                    * diff[i, j]
                )
                interactions_imag[target_point_index * 4 * nsources + 4 * j + 1 + i] = (
                    (-(1 + dist[j] * kernel_parameters[1]) * tmp_imag[j] + kernel_parameters[0] * dist[j] * tmp_real[j])
                    / (dist[j] * dist[j])
                    * diff[i, j]
                )

        # Now fix zero distance case
        for j in range(nsources):
            if dist[j] == 0:
                for i in range(4):
                    interactions_real[target_point_index * 4 * nsources + 4 * j + i] = 0
                    interactions_imag[target_point_index * 4 * nsources + 4 * j + i] = 0

    return interactions_real + 1j * interactions_imag


def get_local_interaction_operator(
    grid,
    local_points,
    kernel_function,
    kernel_parameters,
    precision,
    is_complex,
    device_interface=None,
):
    """Get the local interaction operator."""
    import bempp_cl.api
    from bempp_cl.api import GLOBAL_PARAMETERS
    from bempp_cl.api.utils.helpers import get_type
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import aslinearoperator
    from scipy.sparse.linalg import LinearOperator

    npoints = local_points.shape[1]

    dtype = _np.dtype(get_type(precision).real)
    if is_complex:
        result_type = _np.dtype(get_type(precision).complex)
    else:
        result_type = dtype

    rows = 4 * npoints * grid.number_of_elements
    cols = npoints * grid.number_of_elements

    if kernel_function == "laplace":
        kernel = laplace_kernel
    elif kernel_function == "helmholtz":
        kernel = helmholtz_kernel
    elif kernel_function == "modified_helmholtz":
        kernel = modified_helmholtz_kernel

    if GLOBAL_PARAMETERS.fmm.near_field_representation == "sparse":
        data, indices, indexptr = get_local_interaction_matrix_impl(
            grid.data(precision),
            local_points.astype(dtype),
            kernel,
            _np.array(kernel_parameters, dtype=dtype),
            dtype,
            result_type,
        )
        return aslinearoperator(csr_matrix((data, indices, indexptr), shape=(rows, cols)))

    elif GLOBAL_PARAMETERS.fmm.near_field_representation == "evaluate":
        if device_interface is None:
            device_interface = bempp_cl.api.DEFAULT_DEVICE_INTERFACE
        if device_interface == "numba":
            evaluator = get_local_interaction_evaluator_numba(
                grid.data(precision),
                local_points.astype(dtype),
                kernel,
                _np.array(kernel_parameters, dtype=dtype),
                dtype,
                result_type,
            )
            return LinearOperator(shape=(rows, cols), matvec=evaluator, dtype=result_type)
        elif device_interface == "opencl":
            evaluator = get_local_interaction_evaluator_opencl(
                grid,
                local_points.astype(dtype),
                kernel_function,
                _np.array(kernel_parameters, dtype=dtype),
                dtype,
                result_type,
            )
            return LinearOperator(shape=(rows, cols), matvec=evaluator, dtype=result_type)
        else:
            raise ValueError("Device interface must be one of 'numba', 'opencl'.")
    else:
        raise ValueError("Unknown value for near_field_representation.")


def get_local_interaction_evaluator_numba(
    grid_data, local_points, kernel_function, kernel_parameters, dtype, result_type
):
    """Return an evaluator for the local interactions."""
    import bempp_cl.api

    def evaluator(coeffs):
        """Actually evaluate the near field correction."""
        with bempp_cl.api.Timer(message="Singular Corrections Evaluator."):
            return numba_evaluate_local_interactions(
                grid_data,
                coeffs,
                local_points,
                kernel_function,
                kernel_parameters,
                dtype,
                result_type,
            )

    return evaluator


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def numba_evaluate_local_interactions(
    grid_data,
    coeffs,
    local_points,
    kernel_function,
    kernel_parameters,
    dtype,
    result_type,
):
    """Get the local interaction matrix on the grid."""
    nelements = grid_data.elements.shape[1]
    npoints = local_points.shape[1]
    neighbor_indices = grid_data.element_neighbor_indices
    neighbor_indexptr = grid_data.element_neighbor_indexptr

    result = _np.zeros(4 * npoints * nelements, dtype=result_type)
    global_points = _np.zeros((nelements, 3, npoints), dtype=dtype)

    for element_index in range(nelements):
        global_points[element_index, :, :] = grid_data.local2global(element_index, local_points)

    for target_element in _numba.prange(nelements):
        nneighbors = neighbor_indexptr[1 + target_element] - neighbor_indexptr[target_element]
        source_elements = _np.sort(
            neighbor_indices[neighbor_indexptr[target_element] : neighbor_indexptr[1 + target_element]]
        )

        local_source_points = _np.empty((3, npoints * nneighbors), dtype=dtype)
        for source_element_index in range(nneighbors):
            source_element = source_elements[source_element_index]
            local_source_points[:, npoints * source_element_index : npoints * (1 + source_element_index)] = (
                global_points[source_element, :, :]
            )
        local_target_points = global_points[target_element, :, :]
        interactions = kernel_function(
            local_target_points,
            local_source_points,
            kernel_parameters,
            dtype,
            result_type,
        )

        for target_point_index in range(npoints):
            for i in range(4):
                for source_element_index in range(nneighbors):
                    source_element = source_elements[source_element_index]
                    for source_point_index in range(npoints):
                        result[4 * npoints * target_element + 4 * target_point_index + i] += (
                            interactions[
                                4 * target_point_index * nneighbors * npoints
                                + 4 * source_element_index * npoints
                                + 4 * source_point_index
                                + i
                            ]
                            * coeffs[npoints * source_element + source_point_index]
                        )

    return result


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def get_local_interaction_matrix_impl(grid_data, local_points, kernel_function, kernel_parameters, dtype, result_type):
    """Get the local interaction matrix on the grid."""
    nelements = grid_data.elements.shape[1]
    npoints = local_points.shape[1]
    neighbor_indices = grid_data.element_neighbor_indices
    neighbor_indexptr = grid_data.element_neighbor_indexptr

    data = _np.zeros(4 * npoints * npoints * len(neighbor_indices), dtype=result_type)
    indexptr = _np.zeros(4 * npoints * nelements + 1, dtype=_np.uint32)
    indices = _np.zeros(4 * npoints * npoints * len(neighbor_indices), dtype=_np.uint32)
    indexptr[-1] = 4 * npoints * npoints * len(neighbor_indices)

    global_points = _np.zeros((nelements, 3, npoints), dtype=dtype)

    for element_index in range(nelements):
        global_points[element_index, :, :] = grid_data.local2global(element_index, local_points)

    for target_element in _numba.prange(nelements):
        nneighbors = neighbor_indexptr[1 + target_element] - neighbor_indexptr[target_element]
        source_elements = _np.sort(
            neighbor_indices[neighbor_indexptr[target_element] : neighbor_indexptr[1 + target_element]]
        )

        local_source_points = _np.empty((3, npoints * nneighbors), dtype=dtype)
        for source_element_index in range(nneighbors):
            source_element = source_elements[source_element_index]
            local_source_points[:, npoints * source_element_index : npoints * (1 + source_element_index)] = (
                global_points[source_element, :, :]
            )
        local_target_points = global_points[target_element, :, :]
        interactions = kernel_function(
            local_target_points,
            local_source_points,
            kernel_parameters,
            dtype,
            result_type,
        )

        local_count = 4 * npoints * npoints * neighbor_indexptr[target_element]
        for target_point_index in range(npoints):
            for i in range(4):
                indexptr[4 * npoints * target_element + 4 * target_point_index + i] = local_count
                for source_element_index in range(nneighbors):
                    source_element = source_elements[source_element_index]
                    for source_point_index in range(npoints):
                        data[local_count] = interactions[
                            4 * target_point_index * nneighbors * npoints
                            + 4 * source_element_index * npoints
                            + 4 * source_point_index
                            + i
                        ]
                        indices[local_count] = npoints * source_element + source_point_index
                        local_count += 1

    return data, indices, indexptr


def map_space_to_points(space, local_points, weights, return_transpose=False):
    """Return mapper from grid coeffs to point evaluations."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import aslinearoperator

    grid = space.grid
    number_of_local_points = local_points.shape[1]
    number_of_vertices = number_of_local_points * grid.number_of_elements

    data, global_indices, vertex_indices = map_space_to_points_impl(
        grid.data("double"),
        space.localised_space.local2global,
        space.localised_space.local_multipliers,
        space.localised_space.normal_multipliers,
        space.support_elements,
        space.numba_evaluate,
        space.shapeset.evaluate,
        local_points,
        weights,
        space.number_of_shape_functions,
    )

    if return_transpose:
        transform = coo_matrix(
            (data, (global_indices, vertex_indices)),
            shape=(space.localised_space.global_dof_count, number_of_vertices),
        )

        return aslinearoperator(space.map_to_localised_space.T) @ aslinearoperator(transform)
    else:
        transform = coo_matrix(
            (data, (vertex_indices, global_indices)),
            shape=(number_of_vertices, space.localised_space.global_dof_count),
        )
        return aslinearoperator(transform) @ aslinearoperator(space.map_to_localised_space)


@_numba.njit
def map_space_to_points_impl(
    grid_data,
    local2global,
    local_multipliers,
    normal_multipliers,
    support_elements,
    numba_evaluate,
    shape_fun,
    local_points,
    weights,
    number_of_shape_functions,
):
    """Numba accelerated computational parts for point map."""
    number_of_local_points = local_points.shape[1]
    number_of_support_elements = len(support_elements)

    nlocal = number_of_local_points * number_of_shape_functions

    data = _np.empty(nlocal * number_of_support_elements, dtype=_np.float64)
    global_indices = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)
    vertex_indices = _np.empty(nlocal * number_of_support_elements, dtype=_np.int64)

    for elem in support_elements:
        basis_values = (
            numba_evaluate(
                elem,
                shape_fun,
                local_points,
                grid_data,
                local_multipliers,
                normal_multipliers,
            )[0, :, :]
            * weights
            * grid_data.integration_elements[elem]
        )
        data[elem * nlocal : (1 + elem) * nlocal] = basis_values.ravel()
        for index in range(number_of_shape_functions):
            vertex_indices[
                elem * nlocal + index * number_of_local_points : elem * nlocal + (1 + index) * number_of_local_points
            ] = _np.arange(elem * number_of_local_points, (1 + elem) * number_of_local_points)
        global_indices[elem * nlocal : (1 + elem) * nlocal] = _np.repeat(local2global[elem, :], number_of_local_points)

    return (data, global_indices, vertex_indices)


@_numba.njit
def grid_to_points(grid_data, local_points):
    """
    Map a grid to an array of points.

    Returns a (N, 3) point array that stores the global vertices
    associated with the local points in each triangle.
    Points are stored in consecutive order for each element
    in the support_elements list. Hence, the returned array is of the form
    [ v_1^1, v_2^1, ..., v_M^1, v_1^2, v_2^2, ...], where
    v_i^j is the ith point in the jth element in
    the support_elements list.

    Parameters
    ----------
    grid_data : GridData
        A Bempp GridData object.
    local_points : np.ndarray
        (2, M) array of local coordinates.
    """
    number_of_elements = grid_data.elements.shape[1]
    number_of_points = local_points.shape[1]

    points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    for elem in range(number_of_elements):
        points[number_of_points * elem : number_of_points * (1 + elem), :] = (
            _np.expand_dims(grid_data.vertices[:, grid_data.elements[0, elem]], 1)
            + grid_data.jacobians[elem].dot(local_points)
        ).T
    return points


def get_local_interaction_evaluator_opencl(grid, local_points, kernel_function, kernel_parameters, dtype, result_type):
    """Return an evaluator for the local interactions."""
    import pyopencl as _cl
    import bempp_cl.api
    from bempp_cl.core.opencl_kernels import get_kernel_from_name
    from bempp_cl.core.opencl_kernels import default_context, default_device

    if "laplace" in kernel_function:
        mode = "laplace"
    elif "modified_helmholtz" in kernel_function:
        mode = "modified_helmholtz"
    elif "helmholtz" in kernel_function:
        mode = "helmholtz"
    else:
        raise ValueError("Unknown value for kernel_function.")

    mf = _cl.mem_flags
    ctx = default_context()
    device = default_device()
    # vector_width = get_vector_width("double")
    npoints = local_points.shape[1]
    ncoeffs = npoints * grid.number_of_elements

    max_nneighbors = _np.max(_np.diff(grid.element_neighbors.indexptr))

    grid_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=grid.as_array.astype(dtype),
    )

    # elements_buffer = _cl.Buffer(
    #     ctx,
    #     mf.READ_ONLY | mf.COPY_HOST_PTR,
    #     hostbuf=grid.elements.ravel(order="F"),
    # )

    points_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=local_points.ravel(order="F"),
    )

    neighbor_indices_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=grid.element_neighbors.indices,
    )

    neighbor_indexptr_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid.element_neighbors.indexptr)

    coefficients_buffer = _cl.Buffer(ctx, mf.READ_ONLY, size=result_type.itemsize * ncoeffs)

    result_buffer = _cl.Buffer(ctx, mf.READ_WRITE, size=4 * result_type.itemsize * ncoeffs)

    if len(kernel_parameters) == 0:
        kernel_parameters = [0]

    kernel_parameters_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=_np.array(kernel_parameters, dtype="float64"),
    )

    options = {"MAX_POINTS": max_nneighbors * npoints, "NPOINTS": npoints}
    if result_type == "complex128":
        options["COMPLEX_KERNEL"] = None

    kernel_name = "near_field_evaluator_" + mode
    kernel = get_kernel_from_name(kernel_name, options)

    def evaluator(coeffs):
        """Actually evaluate the near-field correction."""
        result = _np.empty(4 * ncoeffs, dtype=result_type)
        with bempp_cl.api.Timer(message="Singular Corrections Evaluator"):
            with _cl.CommandQueue(ctx, device=device) as queue:
                _cl.enqueue_copy(queue, coefficients_buffer, coeffs.astype(result_type))
                _cl.enqueue_fill_buffer(
                    queue,
                    result_buffer,
                    _np.uint8(0),
                    0,
                    result_type.itemsize * ncoeffs,
                )
                kernel(
                    queue,
                    (grid.number_of_elements,),
                    (1,),
                    grid_buffer,
                    neighbor_indices_buffer,
                    neighbor_indexptr_buffer,
                    points_buffer,
                    coefficients_buffer,
                    result_buffer,
                    kernel_parameters_buffer,
                    _np.uint32(grid.number_of_elements),
                )
                _cl.enqueue_copy(queue, result, result_buffer)

        return result

    return evaluator


def debug_fmm(targets, sources, charges, mode, kernel_parameters, fmm_result):
    """Compare the result of an FMM result with the corresponding dense computation."""
    import bempp_cl.api

    dense_result = dense_interaction_evaluator(targets, sources, charges, mode, kernel_parameters).reshape(-1, 4)

    rel_error = _np.max(_np.abs(dense_result - fmm_result) / _np.abs(fmm_result))

    bempp_cl.api.log(f"FMM error: {rel_error}.")

    return dense_result


def dense_interaction_evaluator(targets, sources, charges, mode, kernel_parameters):
    """
    Dense evaluation of interaction between sources and targets.

    Parameters
    ----------
    targets : ndarray
        M x 3 array of target points.
    sources : ndarray
        N x 3 array of source points.
    charges : ndarray
        N array of charges.
    mode : string
        Either 'laplace', 'helmholtz', 'modified_helmholtz'
    kernel_parameters : ndarray
        Array with kernel parameters
    kernel_type : dtype
        Type of the kernel (numpy.float64 or numpy.complex128)

    Returns the dense evaluation of the interaction between sources
    and targets with the given charges.
    """
    if mode == "laplace":
        kernel = laplace_kernel
        kernel_type = _np.float64
    elif mode == "helmholtz":
        kernel = helmholtz_kernel
        kernel_type = _np.complex128
    elif mode == "modified_helmholtz":
        kernel = modified_helmholtz_kernel
        kernel_type = _np.float64
    else:
        raise ValueError("Unknown value for 'kernel_function'.")

    return dense_interaction_evaluator_impl(targets, sources, charges, kernel, kernel_parameters, kernel_type).reshape(
        -1, 4
    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def dense_interaction_evaluator_impl(targets, sources, charges, kernel, kernel_parameters, kernel_type):
    """
    Dense evaluation of interaction between sources and targets.

    Parameters
    ----------
    targets : ndarray
        M x 3 array of target points.
    sources : ndarray
        N x 3 array of source points.
    charges : ndarray
        N array of charges.
    kernel : Numba function object
        The kernel object (either helpers.laplace,
        helpers.helmholtz or helpers.modified_helmholtz)
    kernel_parameters : ndarray
        Array with kernel parameters
    kernel_type : dtype
        Type of the kernel (numpy.float64 or numpy.complex128)

    Returns the dense evaluation of the interaction between sources
    and targets with the given charges.
    """
    dtype = sources.dtype

    sources = sources.T.copy()
    targets = targets.T.copy()

    ntargets = targets.shape[1]
    nsources = sources.shape[1]
    result = _np.zeros(4 * ntargets, dtype=kernel_type)

    for target_index in _numba.prange(ntargets):
        current_target = targets[:, target_index].copy().reshape((3, 1))
        vals = kernel(current_target, sources, kernel_parameters, dtype, kernel_type)
        for source_index in range(nsources):
            for local_index in range(4):
                result[4 * target_index + local_index] += vals[4 * source_index + local_index] * charges[source_index]

    return result

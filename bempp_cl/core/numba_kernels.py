"""Kernels for assembly using Numba."""

import numba as _numba
import numpy as _np

M_INV_4PI = 1.0 / (4 * _np.pi)


def select_numba_kernels(operator_descriptor, mode="regular"):
    """Select the Numba kernels."""
    assembly_functions_singular = {
        "default_scalar": default_scalar_singular_kernel,
        "laplace_hypersingular": laplace_hypersingular_singular,
        "helmholtz_hypersingular": helmholtz_hypersingular_singular,
        "modified_helmholtz_hypersingular": modified_helmholtz_hypersingular_singular,
        "maxwell_electric_field": maxwell_efield_singular,
        "maxwell_magnetic_field": maxwell_mfield_singular,
    }

    assembly_functions_regular = {
        "default_scalar": default_scalar_regular_kernel,
        "laplace_hypersingular": laplace_hypersingular_regular,
        "helmholtz_hypersingular": helmholtz_hypersingular_regular,
        "modified_helmholtz_hypersingular": modified_helmholtz_hypersingular_regular,
        "maxwell_electric_field": maxwell_efield_regular_assembler,
        "maxwell_magnetic_field": maxwell_mfield_regular_assembler,
    }
    assembly_function_potential = {
        "default_scalar": default_scalar_potential_kernel,
        "maxwell_electric_field": maxwell_efield_potential,
        "maxwell_magnetic_field": maxwell_mfield_potential,
        "maxwell_magnetic_far_field": maxwell_mfield_far_field,
        "maxwell_electric_far_field": maxwell_efield_far_field,
    }

    assembly_functions_sparse = {"default_sparse": default_sparse_kernel}

    kernel_functions_regular = {
        "laplace_single_layer": laplace_single_layer_regular,
        "laplace_double_layer": laplace_double_layer_regular,
        "laplace_adjoint_double_layer": laplace_adjoint_double_layer_regular,
        "helmholtz_single_layer": helmholtz_single_layer_regular,
        "helmholtz_double_layer": helmholtz_double_layer_regular,
        "helmholtz_far_field_single_layer": helmholtz_far_field_single_layer,
        "helmholtz_far_field_double_layer": helmholtz_far_field_double_layer,
        "helmholtz_adjoint_double_layer": helmholtz_adjoint_double_layer_regular,
        "modified_helmholtz_single_layer": modified_helmholtz_single_layer_regular,
        "modified_helmholtz_double_layer": modified_helmholtz_double_layer_regular,
        "modified_helmholtz_adjoint_double_layer": modified_helmholtz_adjoint_double_layer_regular,
    }

    kernel_functions_singular = {
        "laplace_single_layer": laplace_single_layer_singular,
        "laplace_double_layer": laplace_double_layer_singular,
        "laplace_adjoint_double_layer": laplace_adjoint_double_layer_singular,
        "helmholtz_single_layer": helmholtz_single_layer_singular,
        "helmholtz_double_layer": helmholtz_double_layer_singular,
        "helmholtz_adjoint_double_layer": helmholtz_adjoint_double_layer_singular,
        "modified_helmholtz_single_layer": modified_helmholtz_single_layer_singular,
        "modified_helmholtz_double_layer": modified_helmholtz_double_layer_singular,
        "modified_helmholtz_adjoint_double_layer": modified_helmholtz_adjoint_double_layer_singular,
    }

    kernel_functions_sparse = {
        "l2_identity": l2_identity_kernel,
        "laplace_beltrami": laplace_beltrami_kernel,
        "_vector_grad_product": _vector_grad_product_kernel,
        "_curl_curl_product": _curl_curl_product_kernel,
    }

    if mode == "regular":
        return (
            assembly_functions_regular[operator_descriptor.assembly_type],
            kernel_functions_regular[operator_descriptor.kernel_type],
        )
    elif mode == "singular":
        return (
            assembly_functions_singular[operator_descriptor.assembly_type],
            kernel_functions_singular[operator_descriptor.kernel_type],
        )
    elif mode == "sparse":
        return (
            assembly_functions_sparse[operator_descriptor.assembly_type],
            kernel_functions_sparse[operator_descriptor.kernel_type],
        )
    elif mode == "potential":
        return (
            assembly_function_potential[operator_descriptor.assembly_type],
            kernel_functions_regular[operator_descriptor.kernel_type],
        )
    else:
        raise ValueError("Unknown mode.")


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def get_piola_transform(grid_data, elements, local_points):
    """Compute the Piola transform."""
    npoints = local_points.shape[1]
    nelements = len(elements)
    result = _np.zeros((nelements, 3, 3, npoints), dtype=local_points.dtype)
    vals = _np.zeros((2, 3, local_points.shape[1]), dtype=local_points.dtype)
    vals[0, :, :] = _np.vstack((local_points[0], local_points[0] - 1, local_points[0]))
    vals[1, :, :] = _np.vstack((local_points[1] - 1, local_points[1], local_points[1]))
    for element_index in _numba.prange(nelements):
        element = elements[element_index]
        for index in range(3):
            result[element_index, index, :, :] = (
                grid_data.jacobians[element] @ vals[:, index, :]
            ) / grid_data.integration_elements[element]
    return result


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def get_edge_lengths(grid_data, elements):
    """Compute the edge lengths for the given elements."""
    nelements = len(elements)
    result = _np.zeros((nelements, 3), dtype=grid_data.vertices.dtype)

    for element_index in _numba.prange(nelements):
        element = elements[element_index]
        result[element_index, 0] = _np.linalg.norm(
            grid_data.vertices[:, grid_data.elements[0, element]]
            - grid_data.vertices[:, grid_data.elements[1, element]]
        )
        result[element_index, 1] = _np.linalg.norm(
            grid_data.vertices[:, grid_data.elements[2, element]]
            - grid_data.vertices[:, grid_data.elements[0, element]]
        )
        result[element_index, 2] = _np.linalg.norm(
            grid_data.vertices[:, grid_data.elements[1, element]]
            - grid_data.vertices[:, grid_data.elements[2, element]]
        )
    return result


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def get_global_points(grid_data, elements, local_points):
    """Get global points."""
    npoints = local_points.shape[1]
    nelements = len(elements)
    output = _np.empty((3, nelements * npoints), dtype=grid_data.vertices.dtype)
    for index, element in enumerate(elements):
        output[:, npoints * index : npoints * (1 + index)] = grid_data.local2global(element, local_points)
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def get_normals(grid_data, nrepetitions, elements, multipliers):
    """Get normals to be repeated n times per element."""
    output = _np.empty((3, nrepetitions * len(elements)), dtype=grid_data.normals.dtype)
    for index, element in enumerate(elements):
        for dim in range(3):
            for n in range(nrepetitions):
                output[dim, nrepetitions * index + n] = grid_data.normals[element, dim] * multipliers[element]

    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def elements_adjacent(elements, index1, index2):
    """Check if two elements are adjacent."""
    return (
        elements[0, index1] == elements[0, index2]
        or elements[0, index1] == elements[1, index2]
        or elements[0, index1] == elements[2, index2]
        or elements[1, index1] == elements[0, index2]
        or elements[1, index1] == elements[1, index2]
        or elements[1, index1] == elements[2, index2]
        or elements[2, index1] == elements[0, index2]
        or elements[2, index1] == elements[1, index2]
        or elements[2, index1] == elements[2, index2]
    )


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_single_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Laplace single layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            output[j] += (trial_points[i, j] - test_point[i]) ** 2
    for j in range(npoints):
        output[j] = m_inv_4pi / _np.sqrt(output[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_double_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Laplace double layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_point[i]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * trial_normals[i, j]
    for j in range(npoints):
        output[j] *= -m_inv_4pi / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_adjoint_double_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Laplace adjoint double layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_point[i]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        output[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_single_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Laplace single layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            output[j] += (trial_points[i, j] - test_points[i, j]) ** 2
    for j in range(npoints):
        output[j] = m_inv_4pi / _np.sqrt(output[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_double_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Laplace double layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i, j]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * trial_normal[i]
    for j in range(npoints):
        output[j] *= -m_inv_4pi / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_adjoint_double_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Laplace adjoint double layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    output = _np.zeros(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i, j]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        output[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_single_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Helmholtz single layer for regular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    dist = _np.zeros(npoints, dtype=dtype)
    output_real = _np.zeros(npoints, dtype=dtype)
    output_imag = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            dist[j] += (trial_points[i, j] - test_point[i]) ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        output_real[j] = _np.cos(wavenumber_real * dist[j]) * m_inv_4pi / dist[j]
        output_imag[j] = _np.sin(wavenumber_real * dist[j]) * m_inv_4pi / dist[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            output_real[j] *= _np.exp(-wavenumber_imag * dist[j])
            output_imag[j] *= _np.exp(-wavenumber_imag * dist[j])
    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_double_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Helmholtz double layer for regular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    factor_real = _np.empty(npoints, dtype=dtype)
    factor_imag = _np.empty(npoints, dtype=dtype)
    output_real = _np.empty(npoints, dtype=dtype)
    output_imag = _np.empty(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    laplace_grad = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_point[i]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            laplace_grad[j] += diff[i, j] * trial_normals[i, j]
    for j in range(npoints):
        laplace_grad[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
        factor_real[j] = _np.cos(wavenumber_real * dist[j]) * laplace_grad[j]
        factor_imag[j] = _np.sin(wavenumber_real * dist[j]) * laplace_grad[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            factor_real[j] *= _np.exp(-wavenumber_imag * dist[j])
            factor_imag[j] *= _np.exp(-wavenumber_imag * dist[j])
    for j in range(npoints):
        output_real[j] = (-1 - wavenumber_imag * dist[j]) * factor_real[j] - wavenumber_real * dist[j] * factor_imag[j]
        output_imag[j] = wavenumber_real * dist[j] * factor_real[j] + factor_imag[j] * (-1 - wavenumber_imag * dist[j])

    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_adjoint_double_layer_regular(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Helmholtz adjoint double layer for regular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    factor_real = _np.empty(npoints, dtype=dtype)
    factor_imag = _np.empty(npoints, dtype=dtype)
    output_real = _np.empty(npoints, dtype=dtype)
    output_imag = _np.empty(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    laplace_grad = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = test_point[i] - trial_points[i, j]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            laplace_grad[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        laplace_grad[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
        factor_real[j] = _np.cos(wavenumber_real * dist[j]) * laplace_grad[j]
        factor_imag[j] = _np.sin(wavenumber_real * dist[j]) * laplace_grad[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            factor_real[j] *= _np.exp(-wavenumber_imag * dist[j])
            factor_imag[j] *= _np.exp(-wavenumber_imag * dist[j])
    for j in range(npoints):
        output_real[j] = (-1 - wavenumber_imag * dist[j]) * factor_real[j] - wavenumber_real * dist[j] * factor_imag[j]
        output_imag[j] = wavenumber_real * dist[j] * factor_real[j] + factor_imag[j] * (-1 - wavenumber_imag * dist[j])

    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_far_field_single_layer(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Helmholtz single layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype

    dotprod = _np.zeros(npoints, dtype=dtype)
    for dim in range(3):
        for index in range(npoints):
            dotprod[index] += test_point[dim] * trial_points[dim, index]

    output_real = _np.zeros(npoints, dtype=dtype)
    output_imag = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)

    for index in range(npoints):
        output_real[index] = m_inv_4pi * _np.cos(-kernel_parameters[0] * dotprod[index])
    for index in range(npoints):
        output_imag[index] = m_inv_4pi * _np.sin(-kernel_parameters[0] * dotprod[index])

    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_far_field_double_layer(test_point, trial_points, test_normal, trial_normals, kernel_parameters):
    """Evaluate Helmholtz single layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype

    dotprod = _np.zeros(npoints, dtype=dtype)
    for dim in range(3):
        for index in range(npoints):
            dotprod[index] += test_point[dim] * trial_points[dim, index]

    factor = _np.zeros(npoints, dtype=dtype)
    for dim in range(3):
        for index in range(npoints):
            factor[index] -= kernel_parameters[0] * test_point[dim] * trial_normals[dim, index]

    output_real = _np.zeros(npoints, dtype=dtype)
    output_imag = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)

    for index in range(npoints):
        output_real[index] = -factor[index] * m_inv_4pi * _np.sin(-kernel_parameters[0] * dotprod[index])
    for index in range(npoints):
        output_imag[index] = factor[index] * m_inv_4pi * _np.cos(-kernel_parameters[0] * dotprod[index])

    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_single_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Helmholtz single layer for regular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    rad = _np.zeros(npoints, dtype=dtype)
    output_real = _np.zeros(npoints, dtype=dtype)
    output_imag = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            rad[j] += (trial_points[i, j] - test_points[i, j]) ** 2
    for j in range(npoints):
        rad[j] = _np.sqrt(rad[j])
    for j in range(npoints):
        output_real[j] = _np.cos(wavenumber_real * rad[j]) * m_inv_4pi / rad[j]
        output_imag[j] = _np.sin(wavenumber_real * rad[j]) * m_inv_4pi / rad[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            output_real[j] *= _np.exp(-wavenumber_imag * rad[j])
            output_imag[j] *= _np.exp(-wavenumber_imag * rad[j])
    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_double_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Helmholtz double layer for singular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    factor_real = _np.empty(npoints, dtype=dtype)
    factor_imag = _np.empty(npoints, dtype=dtype)
    output_real = _np.empty(npoints, dtype=dtype)
    output_imag = _np.empty(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    laplace_grad = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i, j]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            laplace_grad[j] += diff[i, j] * trial_normal[i]
    for j in range(npoints):
        laplace_grad[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
        factor_real[j] = _np.cos(wavenumber_real * dist[j]) * laplace_grad[j]
        factor_imag[j] = _np.sin(wavenumber_real * dist[j]) * laplace_grad[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            factor_real[j] *= _np.exp(-wavenumber_imag * dist[j])
            factor_imag[j] *= _np.exp(-wavenumber_imag * dist[j])
    for j in range(npoints):
        output_real[j] = (-1 - wavenumber_imag * dist[j]) * factor_real[j] - wavenumber_real * dist[j] * factor_imag[j]
        output_imag[j] = wavenumber_real * dist[j] * factor_real[j] + factor_imag[j] * (-1 - wavenumber_imag * dist[j])
    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_adjoint_double_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Helmholtz adjoint double layer for singular kernels."""
    wavenumber_real = kernel_parameters[0]
    wavenumber_imag = kernel_parameters[1]
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    factor_real = _np.empty(npoints, dtype=dtype)
    factor_imag = _np.empty(npoints, dtype=dtype)
    output_real = _np.empty(npoints, dtype=dtype)
    output_imag = _np.empty(npoints, dtype=dtype)
    diff = _np.empty((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    laplace_grad = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = test_points[i, j] - trial_points[i, j]
            dist[j] += diff[i, j] * diff[i, j]
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for i in range(3):
        for j in range(npoints):
            laplace_grad[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        laplace_grad[j] *= m_inv_4pi / (dist[j] * dist[j] * dist[j])
        factor_real[j] = _np.cos(wavenumber_real * dist[j]) * laplace_grad[j]
        factor_imag[j] = _np.sin(wavenumber_real * dist[j]) * laplace_grad[j]
    if wavenumber_imag != 0:
        for j in range(npoints):
            factor_real[j] *= _np.exp(-wavenumber_imag * dist[j])
            factor_imag[j] *= _np.exp(-wavenumber_imag * dist[j])
    for j in range(npoints):
        output_real[j] = (-1 - wavenumber_imag * dist[j]) * factor_real[j] - wavenumber_real * dist[j] * factor_imag[j]
        output_imag[j] = wavenumber_real * dist[j] * factor_real[j] + factor_imag[j] * (-1 - wavenumber_imag * dist[j])
    return output_real + 1j * output_imag


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_single_layer_regular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Modified Helmholtz single layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            dist[j] += (trial_points[i, j] - test_points[i]) ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for j in range(npoints):
        output[j] = m_inv_4pi * ewr[j] / dist[j]
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_single_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Modified Helmholtz single layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            dist[j] += (trial_points[i, j] - test_points[i, j]) ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for j in range(npoints):
        output[j] = m_inv_4pi * ewr[j] / dist[j]
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_double_layer_regular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Modified Helmholtz double layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    diff = _np.zeros((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i]
    for i in range(3):
        for j in range(npoints):
            dist[j] += diff[i, j] ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * trial_normal[i, j]
    for j in range(npoints):
        output[j] *= (-kernel_parameters[0] * dist[j] - 1) * m_inv_4pi * ewr[j] / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_double_layer_singular(test_points, trial_points, test_normal, trial_normal, kernel_parameters):
    """Evaluate Modified Helmholtz double layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    diff = _np.zeros((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i, j]
    for i in range(3):
        for j in range(npoints):
            dist[j] += diff[i, j] ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * trial_normal[i]
    for j in range(npoints):
        output[j] *= (-kernel_parameters[0] * dist[j] - 1) * m_inv_4pi * ewr[j] / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_adjoint_double_layer_regular(
    test_points, trial_points, test_normal, trial_normal, kernel_parameters
):
    """Evaluate Modified Helmholtz adjoint double layer for regular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    diff = _np.zeros((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i]
    for i in range(3):
        for j in range(npoints):
            dist[j] += diff[i, j] ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        output[j] *= (kernel_parameters[0] * dist[j] + 1) * m_inv_4pi * ewr[j] / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_adjoint_double_layer_singular(
    test_points, trial_points, test_normal, trial_normal, kernel_parameters
):
    """Evaluate Modified Helmholtz adjoint double layer for singular kernels."""
    npoints = trial_points.shape[1]
    dtype = trial_points.dtype
    diff = _np.zeros((3, npoints), dtype=dtype)
    dist = _np.zeros(npoints, dtype=dtype)
    ewr = _np.zeros(npoints, dtype=dtype)
    output = _np.zeros(npoints, dtype=dtype)
    m_inv_4pi = dtype.type(M_INV_4PI)
    for i in range(3):
        for j in range(npoints):
            diff[i, j] = trial_points[i, j] - test_points[i, j]
    for i in range(3):
        for j in range(npoints):
            dist[j] += diff[i, j] ** 2
    for j in range(npoints):
        dist[j] = _np.sqrt(dist[j])
    for j in range(npoints):
        ewr[j] = _np.exp(-kernel_parameters[0] * dist[j])
    for i in range(3):
        for j in range(npoints):
            output[j] += diff[i, j] * test_normal[i]
    for j in range(npoints):
        output[j] *= (kernel_parameters[0] * dist[j] + 1) * m_inv_4pi * ewr[j] / (dist[j] * dist[j] * dist[j])
    return output


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def l2_identity_kernel(
    grid_data,
    nshape_test,
    nshape_trial,
    element_index,
    elements,
    quad_points,
    quad_weights,
    test_normal_multipliers,
    trial_normal_multipliers,
    test_multipliers,
    trial_multipliers,
    test_shapeset,
    trial_shapeset,
    test_basis_evaluate,
    trial_basis_evaluate,
    result,
):
    """Evaluate kernel for L2 identity."""
    element = elements[element_index]

    local_test_fun_values = test_basis_evaluate(
        element,
        test_shapeset,
        quad_points,
        grid_data,
        test_multipliers,
        test_normal_multipliers,
    )
    local_trial_fun_values = trial_basis_evaluate(
        element,
        trial_shapeset,
        quad_points,
        grid_data,
        trial_multipliers,
        trial_normal_multipliers,
    )

    nshape = nshape_test * nshape_trial
    dimension = local_test_fun_values.shape[0]
    n_quad_points = local_test_fun_values.shape[2]
    integration_element = grid_data.integration_elements[element]

    for test_index in range(nshape_test):
        for trial_index in range(nshape_trial):
            for dim_index in range(dimension):
                for quad_index in range(n_quad_points):
                    result[nshape * element_index + test_index * nshape_trial + trial_index] += (
                        local_test_fun_values[dim_index, test_index, quad_index]
                        * local_trial_fun_values[dim_index, trial_index, quad_index]
                        * quad_weights[quad_index]
                        * integration_element
                    )


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def _vector_grad_product_kernel(
    grid_data,
    nshape_test,
    nshape_trial,
    element_index,
    elements,
    quad_points,
    quad_weights,
    test_normal_multipliers,
    trial_normal_multipliers,
    test_multipliers,
    trial_multipliers,
    test_shapeset,
    trial_shapeset_gradient,
    test_basis_evaluate,
    trial_basis_gradient,
    result,
):
    element = elements[element_index]
    local_test_fun_values = test_basis_evaluate(
        element,
        test_shapeset,
        quad_points,
        grid_data,
        test_multipliers,
        test_normal_multipliers,
    )
    local_trial_fun_values = trial_basis_gradient(
        element,
        trial_shapeset_gradient,
        quad_points,
        grid_data,
        trial_multipliers,
        trial_normal_multipliers,
    )

    nshape = nshape_test * nshape_trial
    n_quad_points = len(quad_weights)
    integration_element = grid_data.integration_elements[element]

    for test_index in range(nshape_test):
        for trial_index in range(nshape_trial):
            for dim_index in range(1):
                for grad_index in range(3):
                    for quad_index in range(n_quad_points):
                        result[nshape * element_index + test_index * nshape_trial + trial_index] += (
                            local_test_fun_values[grad_index, test_index, quad_index]
                            * local_trial_fun_values[dim_index, grad_index, trial_index, quad_index]
                            * quad_weights[quad_index]
                            * integration_element
                        )


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def _curl_curl_product_kernel(
    grid_data,
    nshape_test,
    nshape_trial,
    element_index,
    elements,
    quad_points,
    quad_weights,
    test_normal_multipliers,
    trial_normal_multipliers,
    test_multipliers,
    trial_multipliers,
    test_shapeset_gradient,
    trial_shapeset_gradient,
    test_basis_curl,
    trial_basis_curl,
    result,
):
    """Evaluate kernel for L2 identity."""
    element = elements[element_index]

    local_test_fun_values = test_basis_curl(
        element,
        test_shapeset_gradient,
        quad_points,
        grid_data,
        test_multipliers,
        test_normal_multipliers,
    )

    local_trial_fun_values = trial_basis_curl(
        element,
        trial_shapeset_gradient,
        quad_points,
        grid_data,
        trial_multipliers,
        trial_normal_multipliers,
    )

    nshape = nshape_test * nshape_trial
    integration_element = grid_data.integration_elements[element]
    n_quad_points = len(quad_weights)
    for test_index in range(nshape_test):
        for trial_index in range(nshape_trial):
            for quad_index in range(n_quad_points):
                result[nshape * element_index + test_index * nshape_trial + trial_index] += (
                    local_test_fun_values[test_index]
                    * local_trial_fun_values[trial_index]
                    * quad_weights[quad_index]
                    * integration_element
                )


@_numba.jit(nopython=True, parallel=False, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_beltrami_kernel(
    grid_data,
    nshape_test,
    nshape_trial,
    element_index,
    elements,
    quad_points,
    quad_weights,
    test_normal_multipliers,
    trial_normal_multipliers,
    test_multipliers,
    trial_multipliers,
    test_shapeset_gradient,
    trial_shapeset_gradient,
    test_basis_gradient,
    trial_basis_gradient,
    result,
):
    """Evaluate kernel for Laplace-Beltrami."""

    element = elements[element_index]

    local_test_fun_values = test_basis_gradient(
        element,
        test_shapeset_gradient,
        quad_points,
        grid_data,
        test_multipliers,
        test_normal_multipliers,
    )
    local_trial_fun_values = trial_basis_gradient(
        element,
        trial_shapeset_gradient,
        quad_points,
        grid_data,
        trial_multipliers,
        trial_normal_multipliers,
    )

    nshape = nshape_test * nshape_trial
    dimension = local_test_fun_values.shape[0]
    n_quad_points = local_test_fun_values.shape[3]
    integration_element = grid_data.integration_elements[element]

    for test_index in range(nshape_test):
        for trial_index in range(nshape_trial):
            for dim_index in range(dimension):
                for grad_index in range(3):
                    for quad_index in range(n_quad_points):
                        result[nshape * element_index + test_index * nshape_trial + trial_index] += (
                            local_test_fun_values[dim_index, grad_index, test_index, quad_index]
                            * local_trial_fun_values[dim_index, grad_index, trial_index, quad_index]
                            * quad_weights[quad_index]
                            * integration_element
                        )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def default_sparse_kernel(
    grid_data,
    nshape_test,
    nshape_trial,
    elements,
    quad_points,
    quad_weights,
    test_normal_multipliers,
    trial_normal_multipliers,
    test_multipliers,
    trial_multipliers,
    test_shapeset,
    trial_shapeset,
    test_basis_evaluate,
    trial_basis_evaluate,
    kernel_evaluator,
    result,
):
    """Evaluate default sparse kernel."""
    # n_quad_points = len(quad_weights)

    nelements = len(elements)

    for element_index in _numba.prange(nelements):
        kernel_evaluator(
            grid_data,
            nshape_test,
            nshape_trial,
            element_index,
            elements,
            quad_points,
            quad_weights,
            test_normal_multipliers,
            trial_normal_multipliers,
            test_multipliers,
            trial_multipliers,
            test_shapeset,
            trial_shapeset,
            test_basis_evaluate,
            trial_basis_evaluate,
            result,
        )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def default_scalar_regular_kernel(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaulate default scalar kernel."""
    # Compute global points
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers)
    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (local_factors[index] * quad_weights[test_point_index])

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += (
                                tmp[trial_element_index * n_quad_points + quad_point_index]
                                * local_trial_fun_values[0, trial_fun_index, quad_point_index]
                                * local_test_fun_values[0, test_fun_index, test_point_index]
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_hypersingular_regular(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Laplace hypersingular kernel."""
    # Compute global points
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    # local_test_fun_values = test_shapeset(quad_points)
    # local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers)
    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(test_grid_data.normals[test_element], test_surface_gradients[:, i])
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = local_factors[index] * kernel_values[index] * quad_weights[test_point_index]

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                curl_product = test_surface_curls_trans[i] @ trial_surface_curls[trial_element_index]
                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += (
                                tmp[trial_element_index * n_quad_points + quad_point_index]
                                * curl_product[test_fun_index, trial_fun_index]
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_hypersingular_regular(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers)
    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(test_grid_data.normals[test_element], test_surface_gradients[:, i])
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (local_factors[index] * quad_weights[test_point_index])

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = trial_grid_data.normals[trial_element] * trial_normal_multipliers[trial_element]
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = test_surface_curls_trans[i] @ trial_surface_curls[trial_element_index]
                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                - wavenumber
                                * wavenumber
                                * local_test_fun_values[0, test_fun_index, test_point_index]
                                * local_trial_fun_values[0, trial_fun_index, quad_point_index]
                                * normal_prod
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_hypersingular_regular(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Modified Helmholtz hypersingular kernel."""
    # Compute global points
    wavenumber = kernel_parameters[0]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    local_test_fun_values = test_shapeset(quad_points)
    local_trial_fun_values = trial_shapeset(quad_points)
    trial_normals = get_normals(trial_grid_data, n_quad_points, trial_elements, trial_normal_multipliers)
    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    test_surface_curls_trans = _np.empty((n_test_elements, 3, 3), dtype=dtype)
    trial_surface_curls = _np.empty((n_trial_elements, 3, 3), dtype=dtype)

    for test_index in range(n_test_elements):
        test_element = test_elements[test_index]
        test_surface_gradients = test_grid_data.jac_inv_trans[test_element] @ reference_gradient
        for i in range(3):
            test_surface_curls_trans[test_index, i, :] = (
                _np.cross(test_grid_data.normals[test_element], test_surface_gradients[:, i])
                * test_normal_multipliers[test_element]
            )

    for trial_index in range(n_trial_elements):
        trial_element = trial_elements[trial_index]
        trial_surface_gradients = trial_grid_data.jac_inv_trans[trial_element] @ reference_gradient
        for i in range(3):
            trial_surface_curls[trial_index, :, i] = (
                _np.cross(
                    trial_grid_data.normals[trial_element],
                    trial_surface_gradients[:, i],
                )
                * trial_normal_multipliers[trial_element]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        test_normal = test_grid_data.normals[test_element] * test_normal_multipliers[test_element]
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]
        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                test_normal,
                trial_normals,
                kernel_parameters,
            )
            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (local_factors[index] * quad_weights[test_point_index])

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]
                trial_normal = trial_grid_data.normals[trial_element] * trial_normal_multipliers[trial_element]
                normal_prod = _np.dot(test_normal, trial_normal)
                curl_product = test_surface_curls_trans[i] @ trial_surface_curls[trial_element_index]
                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                curl_product[test_fun_index, trial_fun_index]
                                + wavenumber
                                * wavenumber
                                * local_test_fun_values[0, test_fun_index, test_point_index]
                                * local_trial_fun_values[0, trial_fun_index, quad_point_index]
                                * normal_prod
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def default_scalar_singular_kernel(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Evaluate singular kernel."""
    nelements = len(test_elements)

    for index in _numba.prange(nelements):
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(test_points[:, test_offset : test_offset + npoints])
        trial_fun_values = trial_shapeset(trial_points[:, trial_offset : trial_offset + npoints])
        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            grid_data.normals[test_element] * test_normal_multipliers[test_element],
            grid_data.normals[trial_element] * trial_normal_multipliers[trial_element],
            kernel_parameters,
        )
        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index]
                        * quad_weights[weights_offset + point_index]
                        * test_fun_values[0, test_fun_index, point_index]
                        * trial_fun_values[0, trial_fun_index, point_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element] * grid_data.integration_elements[trial_element]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def laplace_hypersingular_singular(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Evaluate Laplace hypersingular singular kernel."""
    dtype = grid_data.vertices.dtype
    nelements = len(test_elements)

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    for index in _numba.prange(nelements):
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        # test_fun_values = test_shapeset(
        #     test_points[:, test_offset : test_offset + npoints]
        # )
        # trial_fun_values = trial_shapeset(
        #     trial_points[:, trial_offset : trial_offset + npoints]
        # )

        test_surface_gradient = grid_data.jac_inv_trans[test_element] @ reference_gradient
        trial_surface_gradient = grid_data.jac_inv_trans[trial_element] @ reference_gradient

        test_normal = grid_data.normals[test_element] * test_normal_multipliers[test_element]
        trial_normal = grid_data.normals[trial_element] * trial_normal_multipliers[trial_element]

        test_surface_curl_trans = _np.empty((3, 3), dtype=dtype)
        trial_surface_curl = _np.empty((3, 3), dtype=dtype)

        for fun_index in range(3):
            test_surface_curl_trans[fun_index, :] = _np.cross(test_normal, test_surface_gradient[:, fun_index])
            trial_surface_curl[:, fun_index] = _np.cross(trial_normal, trial_surface_gradient[:, fun_index])

        surface_curl_products = test_surface_curl_trans @ trial_surface_curl

        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            test_normal,
            trial_normal,
            kernel_parameters,
        )

        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index] * quad_weights[weights_offset + point_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element]
                    * grid_data.integration_elements[trial_element]
                    * surface_curl_products[test_fun_index, trial_fun_index]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def helmholtz_hypersingular_singular(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Evaluate Helmholtz hypersingular singular kernel."""
    dtype = grid_data.vertices.dtype
    nelements = len(test_elements)

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    for index in _numba.prange(nelements):
        wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(test_points[:, test_offset : test_offset + npoints])
        trial_fun_values = trial_shapeset(trial_points[:, trial_offset : trial_offset + npoints])

        test_surface_gradient = grid_data.jac_inv_trans[test_element] @ reference_gradient
        trial_surface_gradient = grid_data.jac_inv_trans[trial_element] @ reference_gradient

        test_normal = grid_data.normals[test_element] * test_normal_multipliers[test_element]
        trial_normal = grid_data.normals[trial_element] * trial_normal_multipliers[trial_element]

        normal_product = _np.dot(test_normal, trial_normal)

        test_surface_curl_trans = _np.empty((3, 3), dtype=dtype)
        trial_surface_curl = _np.empty((3, 3), dtype=dtype)

        for fun_index in range(3):
            test_surface_curl_trans[fun_index, :] = _np.cross(test_normal, test_surface_gradient[:, fun_index])
            trial_surface_curl[:, fun_index] = _np.cross(trial_normal, trial_surface_gradient[:, fun_index])

        surface_curl_products = test_surface_curl_trans @ trial_surface_curl

        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            test_normal,
            trial_normal,
            kernel_parameters,
        )

        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index]
                        * (
                            surface_curl_products[test_fun_index, trial_fun_index]
                            - wavenumber
                            * wavenumber
                            * test_fun_values[0, test_fun_index, point_index]
                            * trial_fun_values[0, trial_fun_index, point_index]
                            * normal_product
                        )
                        * quad_weights[weights_offset + point_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element] * grid_data.integration_elements[trial_element]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def modified_helmholtz_hypersingular_singular(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Singular evaluator."""
    dtype = grid_data.vertices.dtype
    nelements = len(test_elements)

    reference_gradient = _np.array([[-1, 1, 0], [-1, 0, 1]], dtype=dtype)

    for index in _numba.prange(nelements):
        wavenumber = kernel_parameters[0]
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(test_points[:, test_offset : test_offset + npoints])
        trial_fun_values = trial_shapeset(trial_points[:, trial_offset : trial_offset + npoints])

        test_surface_gradient = grid_data.jac_inv_trans[test_element] @ reference_gradient
        trial_surface_gradient = grid_data.jac_inv_trans[trial_element] @ reference_gradient

        test_normal = grid_data.normals[test_element] * test_normal_multipliers[test_element]
        trial_normal = grid_data.normals[trial_element] * trial_normal_multipliers[trial_element]

        normal_product = _np.dot(test_normal, trial_normal)

        test_surface_curl_trans = _np.empty((3, 3), dtype=dtype)
        trial_surface_curl = _np.empty((3, 3), dtype=dtype)

        for fun_index in range(3):
            test_surface_curl_trans[fun_index, :] = _np.cross(test_normal, test_surface_gradient[:, fun_index])
            trial_surface_curl[:, fun_index] = _np.cross(trial_normal, trial_surface_gradient[:, fun_index])

        surface_curl_products = test_surface_curl_trans @ trial_surface_curl

        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            test_normal,
            trial_normal,
            kernel_parameters,
        )

        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index]
                        * (
                            surface_curl_products[test_fun_index, trial_fun_index]
                            + wavenumber
                            * wavenumber
                            * test_fun_values[0, test_fun_index, point_index]
                            * trial_fun_values[0, trial_fun_index, point_index]
                            * normal_product
                        )
                        * quad_weights[weights_offset + point_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element] * grid_data.integration_elements[trial_element]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def default_scalar_potential_kernel(
    dtype,
    result_type,
    kernel_dimension,
    points,
    x,
    grid_data,
    quad_points,
    quad_weights,
    number_of_shape_functions,
    shapeset_evaluate,
    kernel_function,
    kernel_parameters,
    normal_multipliers,
    support_elements,
):
    """Implement a scalar potential kernel."""
    result = _np.zeros((kernel_dimension, points.shape[1]), dtype=result_type)
    n_support_elements = len(support_elements)
    number_of_quad_points = len(quad_weights)
    number_of_points = points.shape[1]

    global_points = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=dtype)

    tmp = _np.zeros(number_of_quad_points * n_support_elements, dtype=result_type)

    for element_index, element in enumerate(support_elements):
        global_points[
            :,
            number_of_quad_points * element_index : number_of_quad_points * (1 + element_index),
        ] = grid_data.local2global(element, quad_points)

    normals = get_normals(grid_data, number_of_quad_points, support_elements, normal_multipliers)

    fun_values = shapeset_evaluate(quad_points)

    test_normal = _np.array([0.0, 0.0, 0.0], dtype=dtype)  # Just need a dummy test normal

    for element_index, element in enumerate(support_elements):
        for quad_point_index in range(number_of_quad_points):
            for fun_index in range(number_of_shape_functions):
                tmp[number_of_quad_points * element_index + quad_point_index] += (
                    grid_data.integration_elements[element]
                    * quad_weights[quad_point_index]
                    * fun_values[0, fun_index, quad_point_index]
                    * x[number_of_shape_functions * element + fun_index]
                )

    for point_index in _numba.prange(number_of_points):
        test_point = points[:, point_index]

        kernel_values = _np.atleast_2d(
            kernel_function(test_point, global_points, test_normal, normals, kernel_parameters)
        )

        for dim in range(kernel_dimension):
            point_result = result_type.type(0)
            for trial_index in range(number_of_quad_points * n_support_elements):
                point_result += kernel_values[dim, trial_index] * tmp[trial_index]
            result[dim, point_index] = point_result

    return result


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_efield_regular_assembler(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Maxwell electric field kernel."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    test_basis_functions = get_piola_transform(test_grid_data, test_elements, quad_points)
    trial_basis_functions = get_piola_transform(trial_grid_data, trial_elements, quad_points)

    test_edge_lengths = get_edge_lengths(test_grid_data, test_elements)
    trial_edge_lengths = get_edge_lengths(trial_grid_data, trial_elements)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]

        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index]
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                None,
                None,
                kernel_parameters,
            )

            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (local_factors[index] * quad_weights[test_point_index])

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]

                divergence_product = 4 / (
                    test_grid_data.integration_elements[test_element]
                    * trial_grid_data.integration_elements[trial_element]
                )

                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                -1j
                                * wavenumber
                                * test_basis_functions[i, test_fun_index, :, test_point_index].dot(
                                    trial_basis_functions[
                                        trial_element_index,
                                        trial_fun_index,
                                        :,
                                        quad_point_index,
                                    ]
                                )
                                - divergence_product / (1j * wavenumber)
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                        * test_edge_lengths[i, test_fun_index]
                        * trial_edge_lengths[trial_element_index, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_efield_singular(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Singular evaluator."""
    nelements = len(test_elements)

    test_edge_lengths = get_edge_lengths(grid_data, test_elements)
    trial_edge_lengths = get_edge_lengths(grid_data, trial_elements)

    for index in _numba.prange(nelements):
        wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(test_points[:, test_offset : test_offset + npoints])
        trial_fun_values = trial_shapeset(trial_points[:, trial_offset : trial_offset + npoints])

        test_fun_values = get_piola_transform(
            grid_data,
            [test_element],
            test_points[:, test_offset : test_offset + npoints],
        )[0]
        trial_fun_values = get_piola_transform(
            grid_data,
            [trial_element],
            trial_points[:, trial_offset : trial_offset + npoints],
        )[0]

        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            None,
            None,
            kernel_parameters,
        )

        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index]
                        * (
                            -1j
                            * wavenumber
                            * (
                                test_fun_values[test_fun_index, :, point_index].dot(
                                    trial_fun_values[trial_fun_index, :, point_index]
                                )
                            )
                            - 4
                            / (
                                1j
                                * wavenumber
                                * grid_data.integration_elements[test_element]
                                * grid_data.integration_elements[trial_element]
                            )
                        )
                        * quad_weights[weights_offset + point_index]
                        * test_edge_lengths[index, test_fun_index]
                        * trial_edge_lengths[index, trial_fun_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element] * grid_data.integration_elements[trial_element]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_mfield_singular(
    grid_data,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    test_normal_multipliers,
    trial_normal_multipliers,
    nshape_test,
    nshape_trial,
    test_shapeset,
    trial_shapeset,
    kernel_evaluator,
    kernel_parameters,
    result,
):
    """Singular evaluator."""
    nelements = len(test_elements)

    test_edge_lengths = get_edge_lengths(grid_data, test_elements)
    trial_edge_lengths = get_edge_lengths(grid_data, trial_elements)

    for index in _numba.prange(nelements):
        wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
        test_element = test_elements[index]
        trial_element = trial_elements[index]
        test_offset = test_offsets[index]
        trial_offset = trial_offsets[index]
        weights_offset = weights_offsets[index]
        npoints = number_of_quad_points[index]
        test_local_points = test_points[:, test_offset : test_offset + npoints]
        trial_local_points = trial_points[:, trial_offset : trial_offset + npoints]
        test_global_points = grid_data.local2global(test_element, test_local_points)
        trial_global_points = grid_data.local2global(trial_element, trial_local_points)
        test_fun_values = test_shapeset(test_points[:, test_offset : test_offset + npoints])
        trial_fun_values = trial_shapeset(trial_points[:, trial_offset : trial_offset + npoints])

        test_fun_values = get_piola_transform(
            grid_data,
            [test_element],
            test_points[:, test_offset : test_offset + npoints],
        )[0]
        trial_fun_values = get_piola_transform(
            grid_data,
            [trial_element],
            trial_points[:, trial_offset : trial_offset + npoints],
        )[0]

        kernel_values = kernel_evaluator(
            test_global_points,
            trial_global_points,
            None,
            None,
            kernel_parameters,
        )

        for test_fun_index in range(nshape_test):
            for trial_fun_index in range(nshape_trial):
                for point_index in range(npoints):
                    diff = test_global_points[:, point_index] - trial_global_points[:, point_index]
                    dist = _np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                    result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] += (
                        kernel_values[point_index]
                        * (1j * wavenumber * dist - 1)
                        / (dist * dist)
                        * diff.dot(
                            _np.cross(
                                test_fun_values[test_fun_index, :, point_index],
                                trial_fun_values[trial_fun_index, :, point_index],
                            )
                        )
                        * quad_weights[weights_offset + point_index]
                        * test_edge_lengths[index, test_fun_index]
                        * trial_edge_lengths[index, trial_fun_index]
                    )
                result[nshape_trial * nshape_test * index + test_fun_index * nshape_trial + trial_fun_index] *= (
                    grid_data.integration_elements[test_element] * grid_data.integration_elements[trial_element]
                )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_mfield_regular_assembler(
    test_grid_data,
    trial_grid_data,
    nshape_test,
    nshape_trial,
    test_elements,
    trial_elements,
    test_multipliers,
    trial_multipliers,
    test_global_dofs,
    trial_global_dofs,
    test_normal_multipliers,
    trial_normal_multipliers,
    quad_points,
    quad_weights,
    kernel_evaluator,
    kernel_parameters,
    grids_identical,
    test_shapeset,
    trial_shapeset,
    result,
):
    """Evaluate Maxwell magnetic field kernel."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = test_grid_data.vertices.dtype
    result_type = result.dtype
    n_quad_points = len(quad_weights)
    n_test_elements = len(test_elements)
    n_trial_elements = len(trial_elements)

    trial_global_points = get_global_points(trial_grid_data, trial_elements, quad_points)

    test_basis_functions = get_piola_transform(test_grid_data, test_elements, quad_points)
    trial_basis_functions = get_piola_transform(trial_grid_data, trial_elements, quad_points)

    test_edge_lengths = get_edge_lengths(test_grid_data, test_elements)
    trial_edge_lengths = get_edge_lengths(trial_grid_data, trial_elements)

    factors = _np.empty(n_quad_points * n_trial_elements, dtype=trial_global_points.dtype)
    for trial_element_index in range(n_trial_elements):
        for trial_point_index in range(n_quad_points):
            factors[n_quad_points * trial_element_index + trial_point_index] = (
                quad_weights[trial_point_index]
                * trial_grid_data.integration_elements[trial_elements[trial_element_index]]
            )

    for i in _numba.prange(n_test_elements):
        test_element = test_elements[i]
        local_result = _np.zeros((n_trial_elements, nshape_test, nshape_trial), dtype=result_type)
        test_global_points = test_grid_data.local2global(test_element, quad_points)
        local_factors = _np.empty(n_trial_elements * n_quad_points, dtype=test_global_points.dtype)
        tmp = _np.empty(n_trial_elements * n_quad_points, dtype=result_type)
        is_adjacent = _np.zeros(n_trial_elements, dtype=_np.bool_)

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            if grids_identical and elements_adjacent(test_grid_data.elements, test_element, trial_element):
                is_adjacent[trial_element_index] = True

        for index in range(n_trial_elements * n_quad_points):
            local_factors[index] = factors[index] * test_grid_data.integration_elements[test_element]

        for test_point_index in range(n_quad_points):
            test_global_point = test_global_points[:, test_point_index].copy()
            kernel_values = kernel_evaluator(
                test_global_point,
                trial_global_points,
                None,
                None,
                kernel_parameters,
            )

            for index in range(n_trial_elements * n_quad_points):
                tmp[index] = kernel_values[index] * (local_factors[index] * quad_weights[test_point_index])

            diff = test_global_point.reshape(3, 1) - trial_global_points
            dist = _np.zeros(n_trial_elements * n_quad_points, dtype=dtype)

            for dim in range(3):
                for col in range(n_trial_elements * n_quad_points):
                    dist[col] += diff[dim, col] * diff[dim, col]
            dist = _np.sqrt(dist)

            for trial_element_index in range(n_trial_elements):
                if is_adjacent[trial_element_index]:
                    continue
                trial_element = trial_elements[trial_element_index]

                for test_fun_index in range(nshape_test):
                    for trial_fun_index in range(nshape_trial):
                        for quad_point_index in range(n_quad_points):
                            ldist = dist[trial_element_index * n_quad_points + quad_point_index]
                            local_result[trial_element_index, test_fun_index, trial_fun_index] += tmp[
                                trial_element_index * n_quad_points + quad_point_index
                            ] * (
                                diff[
                                    :,
                                    trial_element_index * n_quad_points + quad_point_index,
                                ].dot(
                                    _np.cross(
                                        test_basis_functions[i, test_fun_index, :, test_point_index],
                                        trial_basis_functions[
                                            trial_element_index,
                                            trial_fun_index,
                                            :,
                                            quad_point_index,
                                        ],
                                    )
                                )
                                * (1j * wavenumber * ldist - 1)
                                / (ldist * ldist)
                            )

        for trial_element_index in range(n_trial_elements):
            trial_element = trial_elements[trial_element_index]
            for test_fun_index in range(nshape_test):
                for trial_fun_index in range(nshape_trial):
                    result[
                        test_global_dofs[test_element, test_fun_index],
                        trial_global_dofs[trial_element, trial_fun_index],
                    ] += (
                        local_result[trial_element_index, test_fun_index, trial_fun_index]
                        * test_multipliers[test_element, test_fun_index]
                        * trial_multipliers[trial_element, trial_fun_index]
                        * test_edge_lengths[i, test_fun_index]
                        * trial_edge_lengths[trial_element_index, trial_fun_index]
                    )


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_efield_potential(
    dtype,
    result_type,
    kernel_dimension,
    points,
    x,
    grid_data,
    quad_points,
    quad_weights,
    number_of_shape_functions,
    shapeset_evaluate,
    kernel_function,
    kernel_parameters,
    normal_multipliers,
    support_elements,
):
    """Implement the Maxwell electric field potential."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = grid_data.vertices.dtype
    result = _np.zeros((kernel_dimension, points.shape[1]), dtype=result_type)
    n_support_elements = len(support_elements)
    number_of_quad_points = len(quad_weights)
    number_of_points = points.shape[1]

    global_points = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=dtype)

    basis_functions = get_piola_transform(grid_data, support_elements, quad_points)

    edge_lengths = get_edge_lengths(grid_data, support_elements)

    tmp1 = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=result_type)
    tmp2 = _np.zeros(number_of_quad_points * n_support_elements, dtype=result_type)

    for element_index, element in enumerate(support_elements):
        global_points[
            :,
            number_of_quad_points * element_index : number_of_quad_points * (1 + element_index),
        ] = grid_data.local2global(element, quad_points)

    for element_index, element in enumerate(support_elements):
        for quad_point_index in range(number_of_quad_points):
            for fun_index in range(number_of_shape_functions):
                factor = (
                    quad_weights[quad_point_index]
                    * x[number_of_shape_functions * element + fun_index]
                    * edge_lengths[element_index, fun_index]
                )
                tmp1[:, number_of_quad_points * element_index + quad_point_index] += (
                    factor
                    * basis_functions[element_index, fun_index, :, quad_point_index]
                    * grid_data.integration_elements[element]
                )
                tmp2[number_of_quad_points * element_index + quad_point_index] += 2 * factor

    for point_index in _numba.prange(number_of_points):
        test_point = points[:, point_index].copy()

        kernel_values = kernel_function(test_point, global_points, None, None, kernel_parameters)
        diff = test_point.reshape(3, 1) - global_points
        dist = _np.zeros(number_of_quad_points * n_support_elements, dtype=dtype)
        for dim in range(3):
            for index in range(number_of_quad_points * n_support_elements):
                dist[index] += diff[dim, index] * diff[dim, index]
        dist = _np.sqrt(dist)

        for dim in range(kernel_dimension):
            point_result = 0
            for trial_index in range(number_of_quad_points * n_support_elements):
                ldist = dist[trial_index]
                point_result += kernel_values[trial_index] * (
                    1j * wavenumber * tmp1[dim, trial_index]
                    - diff[dim, trial_index]
                    * (1j * wavenumber * ldist - 1)
                    * tmp2[trial_index]
                    / (1j * wavenumber * ldist * ldist)
                )
            result[dim, point_index] = point_result

    return result


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_mfield_potential(
    dtype,
    result_type,
    kernel_dimension,
    points,
    x,
    grid_data,
    quad_points,
    quad_weights,
    number_of_shape_functions,
    shapeset_evaluate,
    kernel_function,
    kernel_parameters,
    normal_multipliers,
    support_elements,
):
    """Implement the Maxwell magnetic field potential."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = grid_data.vertices.dtype
    result = _np.zeros((kernel_dimension, points.shape[1]), dtype=result_type)
    n_support_elements = len(support_elements)
    number_of_quad_points = len(quad_weights)
    number_of_points = points.shape[1]

    global_points = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=dtype)

    basis_functions = get_piola_transform(grid_data, support_elements, quad_points)

    edge_lengths = get_edge_lengths(grid_data, support_elements)

    tmp = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=result_type)

    for element_index, element in enumerate(support_elements):
        global_points[
            :,
            number_of_quad_points * element_index : number_of_quad_points * (1 + element_index),
        ] = grid_data.local2global(element, quad_points)

    for element_index, element in enumerate(support_elements):
        for quad_point_index in range(number_of_quad_points):
            for fun_index in range(number_of_shape_functions):
                factor = (
                    quad_weights[quad_point_index]
                    * x[number_of_shape_functions * element + fun_index]
                    * edge_lengths[element_index, fun_index]
                )
                tmp[:, number_of_quad_points * element_index + quad_point_index] += (
                    factor
                    * basis_functions[element_index, fun_index, :, quad_point_index]
                    * grid_data.integration_elements[element]
                )

    for point_index in _numba.prange(number_of_points):
        test_point = points[:, point_index].copy()

        kernel_values = kernel_function(test_point, global_points, None, None, kernel_parameters)
        diff = test_point.reshape(3, 1) - global_points
        dist = _np.zeros(number_of_quad_points * n_support_elements, dtype=dtype)
        for dim in range(3):
            for index in range(number_of_quad_points * n_support_elements):
                dist[index] += diff[dim, index] * diff[dim, index]
        dist = _np.sqrt(dist)

        for trial_index in range(number_of_quad_points * n_support_elements):
            ldist = dist[trial_index]
            val = kernel_values[trial_index] * (1j * wavenumber * ldist - 1) * tmp[:, trial_index] / (ldist * ldist)
            result[0, point_index] += diff[1, trial_index] * val[2] - diff[2, trial_index] * val[1]
            result[1, point_index] += diff[2, trial_index] * val[0] - diff[0, trial_index] * val[2]
            result[2, point_index] += diff[0, trial_index] * val[1] - diff[1, trial_index] * val[0]

    return result


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_efield_far_field(
    dtype,
    result_type,
    kernel_dimension,
    points,
    x,
    grid_data,
    quad_points,
    quad_weights,
    number_of_shape_functions,
    shapeset_evaluate,
    kernel_function,
    kernel_parameters,
    normal_multipliers,
    support_elements,
):
    """Implement the Maxwell electric far-field potential."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = grid_data.vertices.dtype
    result = _np.zeros((kernel_dimension, points.shape[1]), dtype=result_type)
    n_support_elements = len(support_elements)
    number_of_quad_points = len(quad_weights)
    number_of_points = points.shape[1]

    global_points = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=dtype)

    basis_functions = get_piola_transform(grid_data, support_elements, quad_points)

    edge_lengths = get_edge_lengths(grid_data, support_elements)

    tmp1 = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=result_type)
    tmp2 = _np.zeros(number_of_quad_points * n_support_elements, dtype=result_type)

    for element_index, element in enumerate(support_elements):
        global_points[
            :,
            number_of_quad_points * element_index : number_of_quad_points * (1 + element_index),
        ] = grid_data.local2global(element, quad_points)

    for element_index, element in enumerate(support_elements):
        for quad_point_index in range(number_of_quad_points):
            for fun_index in range(number_of_shape_functions):
                factor = (
                    quad_weights[quad_point_index]
                    * x[number_of_shape_functions * element + fun_index]
                    * edge_lengths[element_index, fun_index]
                )
                tmp1[:, number_of_quad_points * element_index + quad_point_index] += (
                    factor
                    * basis_functions[element_index, fun_index, :, quad_point_index]
                    * grid_data.integration_elements[element]
                )
                tmp2[number_of_quad_points * element_index + quad_point_index] += 2 * factor

    for point_index in _numba.prange(number_of_points):
        test_point = points[:, point_index].copy()

        kernel_values = kernel_function(test_point, global_points, None, None, kernel_parameters)
        for dim in range(kernel_dimension):
            point_result = 0
            for trial_index in range(number_of_quad_points * n_support_elements):
                point_result += kernel_values[trial_index] * (
                    1j * wavenumber * tmp1[dim, trial_index] - test_point[dim] * tmp2[trial_index]
                )
            result[dim, point_index] = point_result

    return result


@_numba.jit(nopython=True, parallel=True, error_model="numpy", fastmath=True, boundscheck=False)
def maxwell_mfield_far_field(
    dtype,
    result_type,
    kernel_dimension,
    points,
    x,
    grid_data,
    quad_points,
    quad_weights,
    number_of_shape_functions,
    shapeset_evaluate,
    kernel_function,
    kernel_parameters,
    normal_multipliers,
    support_elements,
):
    """Implement the Maxwell magnetic far-field potential."""
    wavenumber = kernel_parameters[0] + 1j * kernel_parameters[1]
    dtype = grid_data.vertices.dtype
    result = _np.zeros((kernel_dimension, points.shape[1]), dtype=result_type)
    n_support_elements = len(support_elements)
    number_of_quad_points = len(quad_weights)
    number_of_points = points.shape[1]

    global_points = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=dtype)

    basis_functions = get_piola_transform(grid_data, support_elements, quad_points)

    edge_lengths = get_edge_lengths(grid_data, support_elements)

    tmp = _np.zeros((3, number_of_quad_points * n_support_elements), dtype=result_type)

    for element_index, element in enumerate(support_elements):
        global_points[
            :,
            number_of_quad_points * element_index : number_of_quad_points * (1 + element_index),
        ] = grid_data.local2global(element, quad_points)

    for element_index, element in enumerate(support_elements):
        for quad_point_index in range(number_of_quad_points):
            for fun_index in range(number_of_shape_functions):
                factor = (
                    quad_weights[quad_point_index]
                    * x[number_of_shape_functions * element + fun_index]
                    * edge_lengths[element_index, fun_index]
                )
                tmp[:, number_of_quad_points * element_index + quad_point_index] += (
                    factor
                    * basis_functions[element_index, fun_index, :, quad_point_index]
                    * grid_data.integration_elements[element]
                )

    for point_index in _numba.prange(number_of_points):
        test_point = points[:, point_index].copy()

        kernel_values = kernel_function(test_point, global_points, None, None, kernel_parameters)

        for trial_index in range(number_of_quad_points * n_support_elements):
            val = kernel_values[trial_index] * 1j * wavenumber * tmp[:, trial_index]
            result[0, point_index] += test_point[1] * val[2] - test_point[2] * val[1]
            result[1, point_index] += test_point[2] * val[0] - test_point[0] * val[2]
            result[2, point_index] += test_point[0] * val[1] - test_point[1] * val[0]

    return result

"""Actual implementation of Numba assemblers."""

import numpy as _np


def singular_assembler(
    device_interface,
    operator_descriptor,
    grid,
    domain,
    dual_to_range,
    test_points,
    trial_points,
    quad_weights,
    test_elements,
    trial_elements,
    test_offsets,
    trial_offsets,
    weights_offsets,
    number_of_quad_points,
    kernel_options,
    result,
):
    """Numba assembler for the singular part of integral operators."""
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.numba_kernels import select_numba_kernels

    numba_assembly_function, numba_kernel_function = select_numba_kernels(operator_descriptor, mode="singular")

    # Perform Numba assembly always in double precision
    # precision = operator_descriptor.precision
    precision = "double"
    dtype = get_type(precision).real

    numba_assembly_function(
        grid.data(precision),
        test_points,
        trial_points,
        quad_weights,
        test_elements,
        trial_elements,
        test_offsets,
        trial_offsets,
        weights_offsets,
        number_of_quad_points,
        dual_to_range.normal_multipliers,
        domain.normal_multipliers,
        dual_to_range.number_of_shape_functions,
        domain.number_of_shape_functions,
        dual_to_range.shapeset.evaluate,
        domain.shapeset.evaluate,
        numba_kernel_function,
        _np.array(kernel_options, dtype=dtype),
        result,
    )


def dense_assembler(device_interface, operator_descriptor, domain, dual_to_range, parameters, result):
    """Numba based dense assembler."""
    from bempp_cl.core.numba_kernels import select_numba_kernels
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.api.integration.triangle_gauss import rule

    (
        numba_assembly_function_regular,
        numba_kernel_function_regular,
    ) = select_numba_kernels(operator_descriptor, mode="regular")

    order = parameters.quadrature.regular
    quad_points, quad_weights = rule(order)

    # Perform Numba assembly always in double precision
    # precision = operator_descriptor.precision
    precision = "double"

    data_type = get_type(precision).real

    test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
    trial_indices, trial_color_indexptr = domain.get_elements_by_color()
    number_of_test_colors = len(test_color_indexptr) - 1
    # number_of_trial_colors = len(trial_color_indexptr) - 1

    # rows = dual_to_range.global_dof_count
    # cols = domain.global_dof_count

    nshape_test = dual_to_range.number_of_shape_functions
    nshape_trial = domain.number_of_shape_functions
    grids_identical = domain.grid == dual_to_range.grid

    for test_color_index in range(number_of_test_colors):
        numba_assembly_function_regular(
            dual_to_range.grid.data(precision),
            domain.grid.data(precision),
            nshape_test,
            nshape_trial,
            test_indices[test_color_indexptr[test_color_index] : test_color_indexptr[1 + test_color_index]],
            trial_indices,
            dual_to_range.local_multipliers.astype(data_type),
            domain.local_multipliers.astype(data_type),
            dual_to_range.local2global,
            domain.local2global,
            dual_to_range.normal_multipliers,
            domain.normal_multipliers,
            quad_points.astype(data_type),
            quad_weights.astype(data_type),
            numba_kernel_function_regular,
            _np.array(operator_descriptor.options, dtype=data_type),
            grids_identical,
            dual_to_range.shapeset.evaluate,
            domain.shapeset.evaluate,
            result,
        )


def potential_assembler(device_interface, space, operator_descriptor, points, parameters):
    """Return an evaluator function to evaluate a potential."""
    from bempp_cl.core.numba_kernels import select_numba_kernels
    from bempp_cl.api.integration.triangle_gauss import rule
    from bempp_cl.api.utils.helpers import get_type

    (numba_assembly_function, numba_kernel_function_regular) = select_numba_kernels(
        operator_descriptor, mode="potential"
    )

    quad_points, quad_weights = rule(parameters.quadrature.regular)

    # Perform Numba assembly always in double precision
    # precision = operator_descriptor.precision
    precision = "double"

    dtype = _np.dtype(get_type(precision).real)

    if operator_descriptor.is_complex:
        result_type = _np.dtype(get_type(precision).complex)
    else:
        result_type = dtype

    kernel_dimension = operator_descriptor.kernel_dimension

    points_transformed = points.astype(dtype)
    grid_data = space.grid.data(precision)

    kernel_parameters = _np.array(operator_descriptor.options, dtype=dtype)

    def evaluator(x):
        """Actually evaluate the potential."""
        return numba_assembly_function(
            dtype,
            result_type,
            kernel_dimension,
            points_transformed,
            x.astype(result_type),
            grid_data,
            quad_points.astype(precision),
            quad_weights.astype(precision),
            space.number_of_shape_functions,
            space.shapeset.evaluate,
            numba_kernel_function_regular,
            kernel_parameters,
            space.normal_multipliers,
            space.support_elements,
        )

    return evaluator

"""
Actual implementation of OpenCL assemblers.
"""

import numpy as _np
import pyopencl as _cl

WORKGROUP_SIZE_GALERKIN = 16
WORKGROUP_SIZE_POTENTIAL = 128


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
    """Assemble singular part of integral operators with OpenCL."""
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.opencl_kernels import get_kernel_from_operator_descriptor
    from bempp_cl.core.opencl_kernels import default_context, default_device

    mf = _cl.mem_flags
    ctx = default_context()
    device = default_device()

    precision = operator_descriptor.precision
    dtype = get_type(precision).real

    options = {
        "WORKGROUP_SIZE": WORKGROUP_SIZE_GALERKIN,
        "TEST": dual_to_range.shapeset.identifier,
        "TRIAL": domain.shapeset.identifier,
        "NUMBER_OF_TEST_SHAPE_FUNCTIONS": dual_to_range.number_of_shape_functions,
        "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS": domain.number_of_shape_functions,
    }

    if operator_descriptor.is_complex:
        options["COMPLEX_KERNEL"] = None

    kernel = get_kernel_from_operator_descriptor(operator_descriptor, options, "singular")

    # Initialize OpenCL Buffers

    grid_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grid.as_array.astype(dtype))
    test_normals_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.normal_multipliers)
    trial_normals_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.normal_multipliers)
    test_points_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_points.astype(dtype))
    trial_points_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_points.astype(dtype))
    quad_weights_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quad_weights.astype(dtype))
    test_elements_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_elements)
    trial_elements_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_elements)
    test_offsets_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_offsets)
    trial_offsets_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_offsets)
    weights_offsets_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights_offsets)

    local_quad_points = number_of_quad_points // WORKGROUP_SIZE_GALERKIN

    local_quad_points_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=local_quad_points)

    result_buffer = _cl.Buffer(ctx, mf.WRITE_ONLY, size=result.nbytes)

    if not kernel_options:
        kernel_options = [0.0]

    kernel_options_array = _np.array(kernel_options, dtype=dtype)

    kernel_options_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel_options_array)

    number_of_singular_indices = len(test_elements)

    with _cl.CommandQueue(ctx, device=device) as queue:
        kernel(
            queue,
            (number_of_singular_indices,),
            (WORKGROUP_SIZE_GALERKIN,),
            grid_buffer,
            test_normals_buffer,
            trial_normals_buffer,
            test_points_buffer,
            trial_points_buffer,
            quad_weights_buffer,
            test_elements_buffer,
            trial_elements_buffer,
            test_offsets_buffer,
            trial_offsets_buffer,
            weights_offsets_buffer,
            local_quad_points_buffer,
            result_buffer,
            kernel_options_buffer,
            g_times_l=True,
        )
        _cl.enqueue_copy(queue, result, result_buffer)


def dense_assembler(device_interface, operator_descriptor, domain, dual_to_range, parameters, result):
    """Assemble dense with OpenCL."""
    import bempp_cl.api
    from bempp_cl.api.integration.triangle_gauss import rule
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.opencl_kernels import get_kernel_from_operator_descriptor
    from bempp_cl.core.opencl_kernels import (
        default_context,
        default_device,
        get_vector_width,
    )

    if bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE == "gpu":
        device_type = "gpu"
    elif bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE == "cpu":
        device_type = "cpu"
    else:
        raise RuntimeError(f"Unknown device type {bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE}")

    mf = _cl.mem_flags
    ctx = default_context(device_type)
    device = default_device(device_type)

    precision = operator_descriptor.precision
    dtype = get_type(precision).real
    kernel_options = operator_descriptor.options

    quad_points, quad_weights = rule(parameters.quadrature.regular)

    test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
    trial_indices, trial_color_indexptr = domain.get_elements_by_color()

    number_of_test_colors = len(test_color_indexptr) - 1
    number_of_trial_colors = len(trial_color_indexptr) - 1

    options = {
        "NUMBER_OF_QUAD_POINTS": len(quad_weights),
        "TEST": dual_to_range.shapeset.identifier,
        "TRIAL": domain.shapeset.identifier,
        "TRIAL_NUMBER_OF_ELEMENTS": domain.number_of_support_elements,
        "TEST_NUMBER_OF_ELEMENTS": dual_to_range.number_of_support_elements,
        "NUMBER_OF_TEST_SHAPE_FUNCTIONS": dual_to_range.number_of_shape_functions,
        "NUMBER_OF_TRIAL_SHAPE_FUNCTIONS": domain.number_of_shape_functions,
    }

    if operator_descriptor.is_complex:
        options["COMPLEX_KERNEL"] = None

    main_kernel = get_kernel_from_operator_descriptor(operator_descriptor, options, "regular", device_type=device_type)
    remainder_kernel = get_kernel_from_operator_descriptor(
        operator_descriptor,
        options,
        "regular",
        force_novec=True,
        device_type=device_type,
    )

    test_indices_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_indices)
    trial_indices_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=trial_indices)

    test_normals_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.normal_multipliers)
    trial_normals_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.normal_multipliers)
    test_grid_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=dual_to_range.grid.as_array.astype(dtype),
    )
    trial_grid_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.grid.as_array.astype(dtype))

    test_elements_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=dual_to_range.grid.elements.ravel(order="F"),
    )

    trial_elements_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=domain.grid.elements.ravel(order="F"),
    )

    test_local2global_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dual_to_range.local2global)

    trial_local2global_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=domain.local2global)

    test_multipliers_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=dual_to_range.local_multipliers.astype(dtype),
    )

    trial_multipliers_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=domain.local_multipliers.astype(dtype),
    )

    quad_points_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=quad_points.ravel(order="F").astype(dtype),
    )

    quad_weights_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quad_weights.astype(dtype))

    result_buffer = _cl.Buffer(ctx, mf.READ_WRITE, size=result.nbytes)

    if not kernel_options:
        kernel_options = [0.0]

    kernel_options_array = _np.array(kernel_options, dtype=dtype)

    kernel_options_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel_options_array)

    vector_width = get_vector_width(precision, device_type=device_type)

    def kernel_runner(
        queue,
        test_offset,
        trial_offset,
        test_number_of_indices,
        trial_number_of_indices,
    ):
        """Actually run the kernel for a given range."""
        remainder_size = trial_number_of_indices % vector_width
        main_size = trial_number_of_indices - remainder_size

        buffers = [
            test_indices_buffer,
            trial_indices_buffer,
            test_normals_buffer,
            trial_normals_buffer,
            test_grid_buffer,
            trial_grid_buffer,
            test_elements_buffer,
            trial_elements_buffer,
            test_local2global_buffer,
            trial_local2global_buffer,
            test_multipliers_buffer,
            trial_multipliers_buffer,
            quad_points_buffer,
            quad_weights_buffer,
            result_buffer,
            kernel_options_buffer,
            _np.int32(dual_to_range.global_dof_count),
            _np.int32(domain.global_dof_count),
            _np.uint8(domain.grid != dual_to_range.grid),
        ]

        if main_size > 0:
            main_kernel(
                queue,
                (test_number_of_indices, main_size // vector_width),
                (1, 1),
                *buffers,
                global_offset=(test_offset, trial_offset),
            )

        if remainder_size > 0:
            remainder_kernel(
                queue,
                (test_number_of_indices, remainder_size),
                (1, 1),
                *buffers,
                global_offset=(test_offset, trial_offset + main_size),
            )

    with _cl.CommandQueue(ctx, device=device) as queue:
        _cl.enqueue_fill_buffer(queue, result_buffer, _np.uint8(0), 0, result.nbytes)
        for test_index in range(number_of_test_colors):
            test_offset = test_color_indexptr[test_index]
            n_test_indices = test_color_indexptr[1 + test_index] - test_color_indexptr[test_index]
            for trial_index in range(number_of_trial_colors):
                n_trial_indices = trial_color_indexptr[1 + trial_index] - trial_color_indexptr[trial_index]
                trial_offset = trial_color_indexptr[trial_index]
                kernel_runner(queue, test_offset, trial_offset, n_test_indices, n_trial_indices)
        _cl.enqueue_copy(queue, result, result_buffer)


def potential_assembler(device_interface, space, operator_descriptor, points, parameters):
    """Assemble dense with OpenCL."""
    import bempp_cl.api
    from bempp_cl.api.integration.triangle_gauss import rule
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.opencl_kernels import get_kernel_from_name
    from bempp_cl.core.opencl_kernels import get_kernel_from_operator_descriptor
    from bempp_cl.core.opencl_kernels import (
        default_context,
        default_device,
        get_vector_width,
    )

    if bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE == "gpu":
        device_type = "gpu"
    elif bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE == "cpu":
        device_type = "cpu"
    else:
        raise RuntimeError(f"Unknown device type {bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE}")

    mf = _cl.mem_flags
    ctx = default_context(device_type)
    device = default_device(device_type)

    quad_points, quad_weights = rule(parameters.quadrature.regular)

    precision = operator_descriptor.precision
    dtype = get_type(precision).real
    kernel_options = operator_descriptor.options
    kernel_dimension = operator_descriptor.kernel_dimension

    if operator_descriptor.is_complex:
        result_type = _np.dtype(get_type(precision).complex)
    else:
        result_type = dtype
    result_type = _np.dtype(result_type)

    indices = space.support_elements
    nelements = len(indices)
    vector_width = get_vector_width(precision, device_type=device_type)
    npoints = points.shape[1]
    remainder_size = nelements % WORKGROUP_SIZE_POTENTIAL
    main_size = nelements - remainder_size

    main_kernel = None
    remainder_kernel = None
    sum_kernel = None

    options = {
        "NUMBER_OF_QUAD_POINTS": len(quad_weights),
        "SHAPESET": space.shapeset.identifier,
        "NUMBER_OF_SHAPE_FUNCTIONS": space.number_of_shape_functions,
        "WORKGROUP_SIZE": WORKGROUP_SIZE_POTENTIAL // vector_width,
    }

    if operator_descriptor.is_complex:
        options["COMPLEX_KERNEL"] = None
        options["COMPLEX_COEFFICIENTS"] = None
        options["COMPLEX_RESULT"] = None

    if main_size > 0:
        main_kernel = get_kernel_from_operator_descriptor(
            operator_descriptor, options, "potential", device_type=device_type
        )
        sum_kernel = get_kernel_from_name("sum_for_potential_novec", options, precision, device_type=device_type)

    if remainder_size > 0:
        options["WORKGROUP_SIZE"] = remainder_size
        remainder_kernel = get_kernel_from_operator_descriptor(
            operator_descriptor,
            options,
            "potential",
            force_novec=True,
            device_type=device_type,
        )

    indices_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)

    normals_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=space.normal_multipliers)

    points_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=points.ravel(order="F").astype(dtype),
    )

    grid_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=space.grid.as_array.astype(dtype))

    # elements_buffer = _cl.Buffer(
    #     ctx,
    #     mf.READ_ONLY | mf.COPY_HOST_PTR,
    #     hostbuf=space.grid.elements.ravel(order="F"),
    # )

    quad_points_buffer = _cl.Buffer(
        ctx,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=quad_points.ravel(order="F").astype(dtype),
    )

    quad_weights_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=quad_weights.astype(dtype))

    result_buffer = _cl.Buffer(ctx, mf.READ_WRITE, size=result_type.itemsize * kernel_dimension * npoints)

    coefficients_buffer = _cl.Buffer(ctx, mf.READ_ONLY, size=result_type.itemsize * space.map_to_full_grid.shape[0])

    if main_size > 0:
        sum_size = kernel_dimension * npoints * (nelements // WORKGROUP_SIZE_POTENTIAL) * result_type.itemsize
        sum_buffer = _cl.Buffer(ctx, mf.READ_WRITE, size=sum_size)

    if not kernel_options:
        kernel_options = [0.0]

    kernel_options_array = _np.array(kernel_options, dtype=dtype)

    kernel_options_buffer = _cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel_options_array)

    def evaluator(x):
        """Evaluate a potential."""
        result = _np.empty(kernel_dimension * npoints, dtype=result_type)

        with _cl.CommandQueue(ctx, device=device) as queue:
            _cl.enqueue_copy(queue, coefficients_buffer, x.astype(result_type))
            _cl.enqueue_fill_buffer(
                queue,
                result_buffer,
                _np.uint8(0),
                0,
                kernel_dimension * npoints * result_type.itemsize,
            )

            if main_size > 0:
                _cl.enqueue_fill_buffer(queue, sum_buffer, _np.uint8(0), 0, sum_size)
                queue.finish()

                main_kernel(
                    queue,
                    (npoints, main_size // vector_width),
                    (1, WORKGROUP_SIZE_POTENTIAL // vector_width),
                    grid_buffer,
                    indices_buffer,
                    normals_buffer,
                    points_buffer,
                    coefficients_buffer,
                    quad_points_buffer,
                    quad_weights_buffer,
                    sum_buffer,
                    kernel_options_buffer,
                )

                sum_kernel(
                    queue,
                    (kernel_dimension * npoints,),
                    (1,),
                    sum_buffer,
                    result_buffer,
                    _np.uint32(nelements // WORKGROUP_SIZE_POTENTIAL),
                )

            if remainder_size > 0:
                remainder_kernel(
                    queue,
                    (npoints, remainder_size),
                    (1, remainder_size),
                    grid_buffer,
                    indices_buffer,
                    normals_buffer,
                    points_buffer,
                    coefficients_buffer,
                    quad_points_buffer,
                    quad_weights_buffer,
                    result_buffer,
                    kernel_options_buffer,
                    global_offset=(0, main_size),
                )

            _cl.enqueue_copy(queue, result, result_buffer)

        return result

    return evaluator

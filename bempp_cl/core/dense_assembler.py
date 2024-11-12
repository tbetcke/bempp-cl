"""Dense assembly."""

import numpy as _np
from bempp_cl.api.assembly import assembler as _assembler


class DenseAssembler(_assembler.AssemblerBase):
    """Implementation of a dense assembler for integral operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a dense assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(self, operator_descriptor, device_interface, precision, *args, **kwargs):
        """Dense assembly of the integral operator."""
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            DenseDiscreteBoundaryOperator,
        )
        from bempp_cl.api.utils.helpers import promote_to_double_precision

        if self.domain.requires_dof_transformation or self.dual_to_range.requires_dof_transformation:
            raise ValueError("Spaces that require dof transformations not supported for dense assembly.")

        # (
        # numba_assembly_function_regular,
        # numba_kernel_function_regular,
        # ) = select_numba_kernels(operator_descriptor, mode="regular")

        # (
        # numba_assembly_function_singular,
        # numba_kernel_function_singular,
        # ) = select_numba_kernels(operator_descriptor, mode="singular")

        mat = assemble_dense(
            self.domain,
            self.dual_to_range,
            self.parameters,
            operator_descriptor,
            device_interface,
        )

        if self.parameters.assembly.always_promote_to_double:
            mat = promote_to_double_precision(mat)

        return DenseDiscreteBoundaryOperator(mat)


def assemble_dense(domain, dual_to_range, parameters, operator_descriptor, device_interface):
    """Assembles the operator and returns a dense matrix."""
    import bempp_cl.api
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.dispatcher import dense_assembler_dispatcher
    from bempp_cl.core.singular_assembler import assemble_singular_part

    precision = operator_descriptor.precision

    rows = dual_to_range.global_dof_count
    cols = domain.global_dof_count

    if operator_descriptor.is_complex:
        result_type = get_type(precision).complex
    else:
        result_type = get_type(precision).real

    result = _np.zeros((rows, cols), dtype=result_type)

    with bempp_cl.api.Timer(message=f"Regular assembler:{operator_descriptor.identifier}:{device_interface}"):
        dense_assembler_dispatcher(
            device_interface,
            operator_descriptor,
            domain,
            dual_to_range,
            parameters,
            result,
        )

    grids_identical = domain.grid == dual_to_range.grid

    if grids_identical:
        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        singular_rows, singular_cols, singular_values = assemble_singular_part(
            domain.localised_space,
            dual_to_range.localised_space,
            parameters,
            operator_descriptor,
            device_interface,
        )

        rows = test_local2global[singular_rows]
        cols = trial_local2global[singular_cols]
        values = singular_values * trial_multipliers[singular_cols] * test_multipliers[singular_rows]

        _np.add.at(result, (rows, cols), values)

    return result


# @_timeit
# def assemble_dense(
# domain,
# dual_to_range,
# parameters,
# operator_descriptor,
# numba_assembly_function_regular,
# numba_kernel_function_regular,
# numba_assembly_function_singular,
# numba_kernel_function_singular,
# ):
# """
# Really assemble the operator.
# Assembles the complete operator (near-field and far-field)
# Returns a dense matrix.
# """
# from bempp_cl.api.integration.triangle_gauss import rule as regular_rule
# from bempp_cl.api import log
# from bempp_cl.core.numba.singular_assembler import assemble_singular_part
# from bempp_cl.api.utils.helpers import get_type
# import bempp_cl.api

# order = parameters.quadrature.regular
# quad_points, quad_weights = regular_rule(order)

# test_indices, test_color_indexptr = dual_to_range.get_elements_by_color()
# trial_indices, trial_color_indexptr = domain.get_elements_by_color()
# number_of_test_colors = len(test_color_indexptr) - 1
# number_of_trial_colors = len(trial_color_indexptr) - 1

# rows = dual_to_range.global_dof_count
# cols = domain.global_dof_count

# nshape_test = dual_to_range.number_of_shape_functions
# nshape_trial = domain.number_of_shape_functions

# precision = operator_descriptor.precision

# data_type = get_type(precision).real
# if operator_descriptor.is_complex:
# result_type = get_type(precision).complex
# else:
# result_type = get_type(precision).real

# result = _np.zeros((rows, cols), dtype=result_type)

# grids_identical = domain.grid == dual_to_range.grid

# dense_assembler_dispatcher(
# device_interface,
# operator_descriptor,
# domain,
# dual_to_range,


# with bempp_cl.api.Timer(message="Start Numba dense assembler."):
# for test_color_index in range(number_of_test_colors):
# numba_assembly_function_regular(
# dual_to_range.grid.data(precision),
# domain.grid.data(precision),
# nshape_test,
# nshape_trial,
# test_indices[
# test_color_indexptr[test_color_index] : test_color_indexptr[
# 1 + test_color_index
# ]
# ],
# trial_indices,
# dual_to_range.local_multipliers.astype(data_type),
# domain.local_multipliers.astype(data_type),
# dual_to_range.local2global,
# domain.local2global,
# dual_to_range.normal_multipliers,
# domain.normal_multipliers,
# quad_points.astype(data_type),
# quad_weights.astype(data_type),
# numba_kernel_function_regular,
# _np.array(operator_descriptor.options, dtype=data_type),
# grids_identical,
# dual_to_range.shapeset.evaluate,
# domain.shapeset.evaluate,
# result,
# )

# if grids_identical:
# # Need to treat singular contribution

# trial_local2global = domain.local2global.ravel()
# test_local2global = dual_to_range.local2global.ravel()
# trial_multipliers = domain.local_multipliers.ravel()
# test_multipliers = dual_to_range.local_multipliers.ravel()

# singular_rows, singular_cols, singular_values = assemble_singular_part(
# domain.localised_space,
# dual_to_range.localised_space,
# parameters,
# operator_descriptor,
# numba_assembly_function_singular,
# numba_kernel_function_singular,
# )

# rows = test_local2global[singular_rows]
# cols = trial_local2global[singular_cols]
# values = (
# singular_values
# * trial_multipliers[singular_cols]
# * test_multipliers[singular_rows]
# )

# _np.add.at(result, (rows, cols), values)

# return result

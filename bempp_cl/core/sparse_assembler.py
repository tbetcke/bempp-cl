"""Sparse assembly."""

import numpy as _np

from bempp_cl.api.assembly import assembler as _assembler
from bempp_cl.helpers import timeit as _timeit


class SparseAssembler(_assembler.AssemblerBase):
    """Implementation of a sparse assembler."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a dense assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(self, operator_descriptor, device_interface, precision, *args, **kwargs):
        """Sparse assembly of operators."""
        from bempp_cl.api.utils.helpers import promote_to_double_precision
        from bempp_cl.api.space.space import return_compatible_representation
        from .numba_kernels import select_numba_kernels
        from scipy.sparse import coo_matrix
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )

        domain, dual_to_range = return_compatible_representation(self.domain, self.dual_to_range)
        row_grid_dofs = dual_to_range.grid_dof_count
        col_grid_dofs = domain.grid_dof_count

        if domain.grid != dual_to_range.grid:
            raise ValueError("For sparse operators the domain and dual_to_range grids must be identical.")

        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        numba_assembly_function, numba_kernel_function = select_numba_kernels(operator_descriptor, mode="sparse")

        rows, cols, values = assemble_sparse(
            domain.localised_space,
            dual_to_range.localised_space,
            self.parameters,
            operator_descriptor,
            numba_assembly_function,
            numba_kernel_function,
        )
        global_rows = test_local2global[rows]
        global_cols = trial_local2global[cols]
        global_values = values * trial_multipliers[cols] * test_multipliers[rows]

        if self.parameters.assembly.always_promote_to_double:
            values = promote_to_double_precision(values)

        mat = coo_matrix(
            (global_values, (global_rows, global_cols)),
            shape=(row_grid_dofs, col_grid_dofs),
        ).tocsr()

        if domain.requires_dof_transformation:
            mat = mat @ domain.dof_transformation

        if dual_to_range.requires_dof_transformation:
            mat = dual_to_range.dof_transformation.T @ mat

        return SparseDiscreteBoundaryOperator(mat)


@_timeit
def assemble_sparse(
    domain,
    dual_to_range,
    parameters,
    operator_descriptor,
    numba_assembly_function,
    numba_kernel_function,
):
    """Actually assemble the operator."""
    import bempp_cl.api
    from bempp_cl.api.integration.triangle_gauss import rule as regular_rule
    from bempp_cl.api.utils.helpers import get_type

    order = parameters.quadrature.regular
    quad_points, quad_weights = regular_rule(order)

    support = domain.support * dual_to_range.support

    elements = _np.flatnonzero(support)
    number_of_elements = len(elements)

    nshape_test = dual_to_range.number_of_shape_functions
    nshape_trial = domain.number_of_shape_functions

    # Always assemble in double precision for sparse ops
    # precision = operator_descriptor.precision

    precision = "double"

    if operator_descriptor.is_complex:
        result_type = get_type(precision).complex
    else:
        result_type = get_type(precision).real

    result = _np.zeros(nshape_test * nshape_trial * number_of_elements, dtype=result_type)

    if operator_descriptor.identifier == "laplace_beltrami":
        dual_to_range_shapeset_evaluation = dual_to_range.shapeset.gradient
        domain_shapeset_evaluation = domain.shapeset.gradient
        dual_to_range_numba_evaluation = dual_to_range.numba_surface_gradient
        domain_numba_evaluation = domain.numba_surface_gradient
    elif operator_descriptor.identifier == "_vector_grad_product":
        dual_to_range_shapeset_evaluation = dual_to_range.shapeset.evaluate
        domain_shapeset_evaluation = domain.shapeset.gradient
        dual_to_range_numba_evaluation = dual_to_range.numba_evaluate
        domain_numba_evaluation = domain.numba_surface_gradient
    elif operator_descriptor.identifier == "_curl_curl_product":
        dual_to_range_shapeset_evaluation = dual_to_range.shapeset.gradient
        domain_shapeset_evaluation = domain.shapeset.gradient
        dual_to_range_numba_evaluation = dual_to_range.numba_surface_curl
        domain_numba_evaluation = domain.numba_surface_curl
    else:
        dual_to_range_shapeset_evaluation = dual_to_range.shapeset.evaluate
        domain_shapeset_evaluation = domain.shapeset.evaluate
        dual_to_range_numba_evaluation = dual_to_range.numba_evaluate
        domain_numba_evaluation = domain.numba_evaluate

    with bempp_cl.api.Timer() as t:  # noqa: F841
        numba_assembly_function(
            domain.grid.data(precision),
            nshape_test,
            nshape_trial,
            elements,
            quad_points,
            quad_weights,
            dual_to_range.normal_multipliers,
            domain.normal_multipliers,
            dual_to_range.local_multipliers,
            domain.local_multipliers,
            dual_to_range_shapeset_evaluation,
            domain_shapeset_evaluation,
            dual_to_range_numba_evaluation,
            domain_numba_evaluation,
            numba_kernel_function,
            result,
        )

    irange = _np.arange(nshape_test)
    jrange = _np.arange(nshape_trial)

    i_ind = _np.tile(_np.repeat(irange, nshape_trial), len(elements)) + _np.repeat(
        elements * nshape_test,
        nshape_test * nshape_trial,
    )

    j_ind = _np.tile(_np.tile(jrange, nshape_test), len(elements)) + _np.repeat(
        elements * nshape_trial,
        nshape_test * nshape_trial,
    )

    return i_ind, j_ind, result

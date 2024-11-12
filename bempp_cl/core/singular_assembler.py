"""Singular assembly."""

import numpy as _np

from bempp_cl.api.assembly import assembler as _assembler
from bempp_cl.api.integration import duffy_galerkin as _duffy_galerkin

import collections as _collections


class SingularAssembler(_assembler.AssemblerBase):
    """Assembler for the singular part of boundary integral operators."""

    def __init__(self, domain, dual_to_range, parameters=None):
        """Instantiate the assembler."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(self, operator_descriptor, device_interface, precision, *args, **kwargs):
        """Assemble the singular part."""
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from bempp_cl.api.utils.helpers import promote_to_double_precision
        from scipy.sparse import coo_matrix, csr_matrix
        from bempp_cl.api.space.space import return_compatible_representation

        domain, dual_to_range = return_compatible_representation(self.domain, self.dual_to_range)
        row_dof_count = dual_to_range.global_dof_count
        col_dof_count = domain.global_dof_count
        row_grid_dofs = dual_to_range.grid_dof_count
        col_grid_dofs = domain.grid_dof_count

        if domain.grid != dual_to_range.grid:
            return SparseDiscreteBoundaryOperator(csr_matrix((row_dof_count, col_dof_count), dtype="float64"))

        trial_local2global = domain.local2global.ravel()
        test_local2global = dual_to_range.local2global.ravel()
        trial_multipliers = domain.local_multipliers.ravel()
        test_multipliers = dual_to_range.local_multipliers.ravel()

        rows, cols, values = assemble_singular_part(
            domain.localised_space,
            dual_to_range.localised_space,
            self.parameters,
            operator_descriptor,
            device_interface,
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


def assemble_singular_part(domain, dual_to_range, parameters, operator_descriptor, device_interface):
    """Actually assemble the Numba kernel."""
    from bempp_cl.api.utils.helpers import get_type
    from bempp_cl.core.dispatcher import singular_assembler_dispatcher
    import bempp_cl.api

    precision = operator_descriptor.precision
    kernel_options = operator_descriptor.options
    is_complex = operator_descriptor.is_complex

    grid = domain.grid
    order = parameters.quadrature.singular

    rule = _SingularQuadratureRuleInterfaceGalerkin(grid, order, dual_to_range.support, domain.support)

    number_of_test_shape_functions = dual_to_range.number_of_shape_functions
    number_of_trial_shape_functions = domain.number_of_shape_functions

    [
        test_points,
        trial_points,
        quad_weights,
        test_elements,
        trial_elements,
        test_offsets,
        trial_offsets,
        weights_offsets,
        number_of_quad_points,
    ] = rule.get_arrays()

    if is_complex:
        result_type = get_type(precision).complex
    else:
        result_type = get_type(precision).real

    result = _np.zeros(
        number_of_test_shape_functions * number_of_trial_shape_functions * len(test_elements),
        dtype=result_type,
    )

    with bempp_cl.api.Timer(message=(f"Singular assembler:{operator_descriptor.identifier}:{device_interface}")):
        singular_assembler_dispatcher(
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
        )

    irange = _np.arange(number_of_test_shape_functions)
    jrange = _np.arange(number_of_trial_shape_functions)

    i_ind = _np.tile(_np.repeat(irange, number_of_trial_shape_functions), len(rule.trial_indices)) + _np.repeat(
        rule.test_indices * number_of_test_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    j_ind = _np.tile(_np.tile(jrange, number_of_test_shape_functions), len(rule.trial_indices)) + _np.repeat(
        rule.trial_indices * number_of_trial_shape_functions,
        number_of_test_shape_functions * number_of_trial_shape_functions,
    )

    return (i_ind, j_ind, result)


_SingularQuadratureRule = _collections.namedtuple("_QuadratureRule", "test_points trial_points weights")


class _SingularQuadratureRuleInterfaceGalerkin(object):
    """Interface for a singular quadrature rule."""

    def __init__(self, grid, order, test_support, trial_support):
        """Initialize singular quadrature rule."""
        self._grid = grid
        self._order = order
        self._test_indices = None
        self._trial_indices = None

        self._coincident_rule = _SingularQuadratureRule(*_duffy_galerkin.rule(order, "coincident"))

        self._edge_adjacent_rule = _SingularQuadratureRule(*_duffy_galerkin.rule(order, "edge_adjacent"))

        self._vertex_adjacent_rule = _SingularQuadratureRule(*_duffy_galerkin.rule(order, "vertex_adjacent"))

        # Iterate through the singular pairs and only add those that are
        # in the support of the space.

        self._index_count = {}

        self._coincident_indices = _np.flatnonzero(test_support * trial_support)
        self._index_count["coincident"] = len(self._coincident_indices)

        # test_support and trial_support are boolean arrays.
        # * operation corresponds to and op between the arrays.

        edge_adjacent_pairs = _np.flatnonzero(
            test_support[grid.edge_adjacency[0, :]] * trial_support[grid.edge_adjacency[1, :]]
        )

        self._edge_adjacency = grid.edge_adjacency[:, edge_adjacent_pairs]

        vertex_adjacent_pairs = _np.flatnonzero(
            test_support[grid.vertex_adjacency[0, :]] * trial_support[grid.vertex_adjacency[1, :]]
        )

        self._vertex_adjacency = grid.vertex_adjacency[:, vertex_adjacent_pairs]

        self._index_count["edge_adjacent"] = self._edge_adjacency.shape[1]
        self._index_count["vertex_adjacent"] = self._vertex_adjacency.shape[1]

        self._index_count["all"] = (
            self._index_count["coincident"] + self._index_count["edge_adjacent"] + self._index_count["vertex_adjacent"]
        )

    @property
    def order(self):
        """Return the order."""
        return self._order

    @property
    def coincident_rule(self):
        """Return coincident rule."""
        return self._coincident_rule

    @property
    def edge_adjacent_rule(self):
        """Return edge adjacent rule."""
        return self._edge_adjacent_rule

    @property
    def vertex_adjacent_rule(self):
        """Return vertex adjacent rule."""
        return self._vertex_adjacent_rule

    @property
    def grid(self):
        """Return the grid."""
        return self._grid

    @property
    def edge_adjacency(self):
        """Return the grid edge adjacency information."""
        return self._edge_adjacency

    @property
    def vertex_adjacency(self):
        """Return vertex adjacency."""
        return self._vertex_adjacency

    @property
    def number_of_elements(self):
        """Return the number of elements of the underlying grid."""
        return self.grid.number_of_elements

    @property
    def index_count(self):
        """Return the index count."""
        return self._index_count

    @property
    def test_indices(self):
        """Return the test indicies of all singular contributions."""
        return self._test_indices

    @property
    def trial_indices(self):
        """Return the trial indicies of all singular contributions."""
        return self._trial_indices

    def number_of_points(self, adjacency):
        """Return the number of quadrature points for given adjacency."""
        return _duffy_galerkin.number_of_quadrature_points(self.order, adjacency)

    def get_arrays(self):
        """Return the arrays."""
        test_indices, trial_indices = self._vectorize_indices()
        test_points, trial_points = self._vectorize_points()
        weights = self._vectorize_weights()
        test_offsets, trial_offsets, weights_offsets = self._vectorize_offsets()
        number_of_quad_points = self._get_number_of_quad_points()

        self._test_indices = test_indices
        self._trial_indices = trial_indices

        arrays = [
            test_points,
            trial_points,
            weights,
            test_indices,
            trial_indices,
            test_offsets,
            trial_offsets,
            weights_offsets,
            number_of_quad_points,
        ]

        return arrays

    def _collect_remapped_quad_points_for_edge_adjacent_rule(self, quad_points):
        """
        Remap quad points for edge adjacent quadrature rules.

        Given a 2xN array of quadrature points, return all possible
        combinations of remapped rules, according to the following
        order
        0: edge (index 0, 1)
        1: edge (index 1, 0)
        2: edge (index 1, 2)
        3: edge (index 2, 1)
        4: edge (index 0, 2)
        5: edge (index 2, 0)
        """
        return _np.hstack(
            [
                _duffy_galerkin.remap_points_shared_edge(quad_points, 0, 1),
                _duffy_galerkin.remap_points_shared_edge(quad_points, 1, 0),
                _duffy_galerkin.remap_points_shared_edge(quad_points, 1, 2),
                _duffy_galerkin.remap_points_shared_edge(quad_points, 2, 1),
                _duffy_galerkin.remap_points_shared_edge(quad_points, 0, 2),
                _duffy_galerkin.remap_points_shared_edge(quad_points, 2, 0),
            ]
        )

    def _collect_remapped_quad_points_for_vertex_adjacent_rule(self, quad_points):
        """
        Remap quad points for vertex adjacent quadrature rules.

        Given a 2xN array of quadrature points, return all possible
        combinations of remapped rules, according to the following
        order
        0: vertex (index 0)
        1: vertex (index 1)
        2: vertex (index 2)
        """
        return _np.hstack(
            [
                _duffy_galerkin.remap_points_shared_vertex(quad_points, 0),
                _duffy_galerkin.remap_points_shared_vertex(quad_points, 1),
                _duffy_galerkin.remap_points_shared_vertex(quad_points, 2),
            ]
        )

    def _compute_edge_offsets(self):
        """Compute offsets for the edge based rule."""
        ncoincident = self.number_of_points("coincident")
        nedge_adjacent = self.number_of_points("edge_adjacent")

        # Offset values is a 3 x 3 matrix such that
        # the value (i, j) is the offset of the rule
        # associted with the (i, j) remap case,
        # where i and j are 0, 1, or 2. The diagonal
        # elements are not needed, so just set to -1.

        offset_values = _np.array([[-1, 0, 4], [1, -1, 2], [5, 3, -1]])

        edge_offsets = ncoincident + nedge_adjacent * offset_values

        return edge_offsets

    def _compute_vertex_offsets(self):
        """Compute offsets for the vertex based rules."""
        ncoincident = self.number_of_points("coincident")
        nedge_adjacent = self.number_of_points("edge_adjacent")
        nvertex_adjacent = self.number_of_points("vertex_adjacent")

        vertex_offsets = ncoincident + 6 * nedge_adjacent + nvertex_adjacent * _np.arange(3, dtype="uint32")

        return vertex_offsets

    def _vectorize_indices(self):
        """Return vector of test and trial indices for sing. integration."""
        test_indices = _np.empty(self.index_count["all"], dtype="uint32")
        trial_indices = _np.empty(self.index_count["all"], dtype="uint32")

        for array, index in zip([test_indices, trial_indices], [0, 1]):
            count = self._index_count["coincident"]
            array[:count] = self._coincident_indices

            array[count : (count + self.index_count["edge_adjacent"])] = self.edge_adjacency[index, :]

            count += self.index_count["edge_adjacent"]

            array[count:] = self.vertex_adjacency[index, :]

        return test_indices, trial_indices

    def _get_number_of_quad_points(self):
        """Compute an array of local numbers of integration points."""
        n = self.index_count["all"]
        number_of_quad_points = _np.empty(self.index_count["all"], dtype="uint32")

        number_of_quad_points[: self.index_count["coincident"]] = self.number_of_points("coincident")
        number_of_quad_points[
            self.index_count["coincident"] : (self.index_count["coincident"] + self.index_count["edge_adjacent"])
        ] = self.number_of_points("edge_adjacent")
        number_of_quad_points[n - self.index_count["vertex_adjacent"] :] = self.number_of_points("vertex_adjacent")

        return number_of_quad_points

    def _vectorize_points(self):
        """Return an array of all quadrature points for all different rules."""
        test_points = _np.hstack(
            [
                self.coincident_rule.test_points,
                self._collect_remapped_quad_points_for_edge_adjacent_rule(self.edge_adjacent_rule.test_points),
                self._collect_remapped_quad_points_for_vertex_adjacent_rule(self.vertex_adjacent_rule.test_points),
            ]
        )

        trial_points = _np.hstack(
            [
                self.coincident_rule.trial_points,
                self._collect_remapped_quad_points_for_edge_adjacent_rule(self.edge_adjacent_rule.trial_points),
                self._collect_remapped_quad_points_for_vertex_adjacent_rule(self.vertex_adjacent_rule.trial_points),
            ]
        )

        return test_points, trial_points

    def _vectorize_weights(self):
        """Vectorize the quadrature weights."""
        weights = _np.hstack(
            [
                self.coincident_rule.weights,
                self.edge_adjacent_rule.weights,
                self.vertex_adjacent_rule.weights,
            ]
        )

        return weights

    def _vectorize_offsets(self):
        """Vectorize the offsets."""
        edge_offsets = self._compute_edge_offsets()
        vertex_offsets = self._compute_vertex_offsets()

        test_offsets = _np.empty(self.index_count["all"], dtype="uint32")
        trial_offsets = _np.empty(self.index_count["all"], dtype="uint32")
        weights_offsets = _np.empty(self.index_count["all"], dtype="uint32")

        test_offsets[: self.index_count["coincident"]] = 0

        test_offsets[
            self.index_count["coincident"] : (self.index_count["coincident"] + self.index_count["edge_adjacent"])
        ] = edge_offsets[self.edge_adjacency[2, :], self.edge_adjacency[3, :]]

        test_offsets[(self.index_count["coincident"] + self.index_count["edge_adjacent"]) :] = vertex_offsets[
            self.vertex_adjacency[2, :]
        ]

        trial_offsets[: self.index_count["coincident"]] = _np.zeros(self.index_count["coincident"])

        trial_offsets[
            self.index_count["coincident"] : (self.index_count["coincident"] + self.index_count["edge_adjacent"])
        ] = edge_offsets[self.edge_adjacency[4, :], self.edge_adjacency[5, :]]

        trial_offsets[(self.index_count["coincident"] + self._index_count["edge_adjacent"]) :] = vertex_offsets[
            self.vertex_adjacency[3, :]
        ]

        weights_offsets[: self.index_count["coincident"]] = 0
        weights_offsets[
            self.index_count["coincident"] : (self.index_count["coincident"] + self.index_count["edge_adjacent"])
        ] = self.number_of_points("coincident")
        weights_offsets[(self.index_count["coincident"] + self.index_count["edge_adjacent"]) :] = self.number_of_points(
            "coincident"
        ) + self.number_of_points("edge_adjacent")

        return test_offsets, trial_offsets, weights_offsets

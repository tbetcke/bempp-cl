"""Generic access to spaces."""

import numpy as _np
import numba as _numba


def function_space(grid, kind, degree, scatter=True, **kwargs):
    """
    Initialize a function space.

    Parameters
    ----------
    grid : bempp_cl.Grid
        The grid that the space is defined on.
    kind : str
        The space type
    degree : int
        The polynomial degree of the space
    support_elements : np.array
        The element indices of elements that make up the part
        of the mesh on which the space is defined.
    segments : list
        The segment numbers of the part of the mesh on which the space is defined.
    swapped_normals : bool
        TODO
    scatter : bool
        TODO
    include_boundary_dofs : bool
        Should degrees of freedom on the boundary of the grid segments be included?
    truncate_at_segment_edge : bool
        Should basis functions be truncated at the edge of the grid segment? If this is set to true,
        continuous spaces will no longer be continuous across the segment edge.
    """
    from bempp_cl.api.utils import pool

    from . import scalar_spaces
    from . import scalar_dual_spaces
    from . import maxwell_spaces

    space_f = None

    if "support_elements" in kwargs and "segments" in kwargs:
        raise ValueError("Only one of 'support_elements' and 'segments' must be nonzero.")

    if kind == "DP":
        if degree == 0:
            space_f = scalar_spaces.p0_discontinuous_function_space
        if degree == 1:
            space_f = scalar_spaces.p1_discontinuous_function_space

    if kind == "P":
        if degree == 1:
            space_f = scalar_spaces.p1_continuous_function_space

    if kind == "DUAL":
        if degree == 0:
            space_f = scalar_dual_spaces.dual0_function_space
        if degree == 1:
            space_f = scalar_dual_spaces.dual1_function_space

    if kind == "RWG" or kind == "RT":
        if degree == 0:
            space_f = maxwell_spaces.rwg0_function_space

    if kind == "SNC" or kind == "NC":
        if degree == 0:
            space_f = maxwell_spaces.snc0_function_space

    if kind == "BC":
        if degree == 0:
            space_f = maxwell_spaces.bc_function_space

    if kind == "RBC":
        if degree == 0:
            space_f = maxwell_spaces.rbc_function_space

    if space_f is None:
        raise ValueError("Requested space not implemented.")

    space = space_f(grid, **kwargs)

    if scatter and pool.is_initialised() and not pool.is_worker():
        pool.execute(
            _space_scatter_worker,
            grid.id,
            space.id,
            kind,
            degree,
            kwargs,
        )
        space._is_scattered = True

    return space


class SpaceBuilder(object):
    """Configure and builds a space object."""

    def __init__(self, grid):
        """Set all parameters to None."""
        self._grid = grid

        self._codomain_dimension = None
        self._order = None
        self._shapeset = None
        self._local2global_map = None
        self._local_multipliers = None
        self._identifier = None
        self._support = None
        self._normal_multipliers = None
        self._barycentric_representation = None
        self._numba_evaluator = None
        self._numba_surface_gradient = None
        self._numba_surface_curl = None
        self._dof_transformation = None
        self._collocation_points = None
        self._is_localised = None
        self._global2local_map = None

        self._is_barycentric = False
        self._space = None
        self._requires_dof_transformation = False

    def set_codomain_dimension(self, codomain_dimension):
        """Set the codomain dimension."""
        self._codomain_dimension = codomain_dimension
        return self

    def set_support(self, support):
        """Set the support."""
        self._support = support
        return self

    def set_normal_multipliers(self, normal_multipliers):
        """Set normal multipliers."""
        self._normal_multipliers = normal_multipliers
        return self

    def set_order(self, order):
        """Set the order of the space."""
        self._order = order
        return self

    def set_shapeset(self, shapeset):
        """Set the shapeset string."""
        self._shapeset = shapeset
        return self

    def set_local2global(self, local2global):
        """Set the local2global map."""
        self._local2global_map = local2global
        return self

    def set_global2local(self, global2local):
        """Set the global2local map."""
        self._global2local_map = global2local
        return self

    def set_local_multipliers(self, local_multipliers):
        """Set the local multipliers."""
        self._local_multipliers = local_multipliers
        return self

    def set_identifier(self, identifier):
        """Set the identifier."""
        self._identifier = identifier
        return self

    def set_is_localised(self, is_localised):
        """Set to true if space is localised."""
        self._is_localised = is_localised
        return self

    def set_dof_transformation(self, dof_transformation):
        """Set the dof transformation."""
        self._dof_transformation = dof_transformation
        self._requires_dof_transformation = True

        return self

    def set_is_barycentric(self, is_barycentric):
        """Call to define space as barycentric."""
        self._is_barycentric = is_barycentric
        return self

    def set_numba_evaluator(self, basis_evaluator):
        """Hand over Numba method that evaluates the basis."""
        self._numba_evaluator = basis_evaluator
        return self

    def set_numba_surface_gradient(self, surface_gradient):
        """Hand over Numba method that evaluates surface gradient."""
        self._numba_surface_gradient = surface_gradient
        return self

    def set_numba_surface_curl(self, surface_curl):
        """Hand over Numba method that evaluates surface curl."""
        self._numba_surface_curl = surface_curl
        return self

    def set_barycentric_representation(self, barycentric_representation):
        """Set barycentric representation."""
        self._barycentric_representation = barycentric_representation
        return self

    def set_collocation_points(self, collocation_points):
        """Define the collocation points."""
        self._collocation_points = collocation_points
        return self

    def build(self):
        """Build a space object."""
        from scipy.sparse import identity

        if self._space is not None:
            return self._space

        # Check if enough information was provided.
        if self._codomain_dimension is None:
            raise ValueError("Codomain dimension not defined." "")

        if self._order is None:
            raise ValueError("order not defined.")

        if self._shapeset is None:
            raise ValueError("shapeset not defined.")

        if self._local2global_map is None:
            raise ValueError("local2global map not defined.")

        if self._local_multipliers is None:
            raise ValueError("Local multipliers not defined.")

        if self._identifier is None:
            raise ValueError("identifier not defined.")

        if self._is_localised is None:
            raise ValueError("is_localised property not defined.")

        self._space = FunctionSpace.__new__(FunctionSpace)

        if self._support is None:
            self._support = _np.ones(self._grid.number_of_elements, dtype=_np.bool)

        if self._normal_multipliers is None:
            self._normal_multipliers = _np.ones(self._grid.number_of_elements, dtype=_np.int32)

        if self._dof_transformation is None:
            ndofs = 1 + _np.max(self._local2global_map)
            self._dof_transformation = identity(ndofs, dtype=_np.float64)

        if self._numba_evaluator is None:
            self._numba_evaluator = _numba_evaluate

        if self._barycentric_representation is None and self._is_barycentric:
            self._barycentric_representation = self._space

        self._space.__init__(
            self._grid,
            self._codomain_dimension,
            self._order,
            self._shapeset,
            self._local2global_map,
            self._global2local_map,
            self._local_multipliers,
            self._identifier,
            self._is_localised,
            self._support,
            self._normal_multipliers,
            self._requires_dof_transformation,
            self._is_barycentric,
            self._barycentric_representation,
            self._dof_transformation,
            self._numba_evaluator,
            self._numba_surface_gradient,
            self._numba_surface_curl,
            self._collocation_points,
        )

        return self._space


class FunctionSpace(object):
    """
    Main class for function spaces.

    This class is not meant to be initialized on its
    own. Rather there are functions for each type
    of space that configure this object through a
    builder pattern.

    """

    def __init__(
        self,
        grid,
        codomain_dimension,
        order,
        shapeset,
        local2global_map,
        global2local_map,
        local_multipliers,
        identifier,
        is_localised,
        support,
        normal_multipliers,
        requires_dof_transformation,
        is_barycentric,
        barycentric_representation,
        dof_transformation,
        numba_evaluator,
        numba_surface_gradient,
        numba_surface_curl,
        collocation_points,
    ):
        """Initialize the space."""
        from .shapesets import Shapeset
        from scipy.sparse import coo_matrix
        from bempp_cl.api.utils.helpers import create_unique_id

        self._grid = grid
        self._grid_id = self._grid.id
        self._codomain_dimension = codomain_dimension
        self._order = order
        self._shapeset = Shapeset(shapeset)
        self._local2global_map = local2global_map
        self._global2local_map = global2local_map
        self._local_multipliers = local_multipliers
        self._identifier = identifier
        self._support = support
        self._requires_dof_transformation = requires_dof_transformation
        self._is_barycentric = is_barycentric
        self._barycentric_representation = barycentric_representation
        self._dof_transformation = dof_transformation
        self._numba_evaluate = numba_evaluator
        self._numba_surface_gradient = numba_surface_gradient
        self._numba_surface_curl = numba_surface_curl
        self._normal_multipliers = normal_multipliers
        self._number_of_support_elements = _np.count_nonzero(self._support)
        self._support_elements = _np.flatnonzero(self._support).astype("uint32")
        self._collocation_points = collocation_points

        self._id = create_unique_id()
        self._hash_string = None
        self._color_map = None
        self._mass_matrix = None
        self._inverse_mass_matrix = None
        self._sorted_indices = None
        self._indexptr = None
        self._is_scattered = False

        # Number of dofs for the space defined over the grid
        # This is different from the global_dof_count, which
        # takes the dof transformation matrix into account.

        number_of_grid_dofs = 1 + _np.max(self._local2global_map)
        self._grid_dof_count = number_of_grid_dofs

        # Number of shape functions
        nshape_fun = self.number_of_shape_functions

        # Generate map to localised space

        self._map_to_localised_space = coo_matrix(
            (
                self._local_multipliers[self._support].ravel(),
                (
                    _np.arange(nshape_fun * self._number_of_support_elements),
                    self._local2global_map[self._support].ravel(),
                ),
            ),
            shape=(nshape_fun * self._number_of_support_elements, self._grid_dof_count),
            dtype="float64",
        ).tocsr()

        # Generate map to full grid
        # This works like the map to the localised grid.
        # But if the space is defined over a subgrid it
        # maps to the localised space on the full grid.

        self._map_to_full_grid = coo_matrix(
            (
                self._local_multipliers[self._support].ravel(),
                (
                    nshape_fun * _np.repeat(self._support_elements, nshape_fun)
                    + _np.tile(_np.arange(nshape_fun), self._number_of_support_elements),
                    self._local2global_map[self._support].ravel(),
                ),
            ),
            shape=(nshape_fun * self._grid.number_of_elements, number_of_grid_dofs),
            dtype="float64",
        ).tocsr()

        # Create localised space.

        # First check if space is already a localised space.
        # For this we need that all multipliers are 1 and all global
        # dofs are each only associated with one local dof.

        if is_localised:
            self._localised_space = self
        else:
            # Create a new localised space
            self._localised_space = make_localised_space(self)

    @property
    def grid(self):
        """Return the grid."""
        return self._grid

    @property
    def grid_id(self):
        """Return id of base grid."""
        return self._grid_id

    @property
    def codomain_dimension(self):
        """Return the codomain dimension."""
        return self._codomain_dimension

    @property
    def global_dof_count(self):
        """Return the global dof count."""
        return self._dof_transformation.shape[1]

    @property
    def grid_dof_count(self):
        """Return the grid dof count."""
        return self._grid_dof_count

    @property
    def order(self):
        """Return order of the space."""
        return self._order

    def cell_dofs(self, cell_index):
        """Return the DOF numbers associated with the cell."""
        return [
            None if _np.isclose(j, 0) else i
            for i, j in zip(self.local2global[cell_index], self.local_multipliers[cell_index])
        ]

    @property
    def local2global(self):
        """Return local to global map."""
        return self._local2global_map

    @property
    def local_multipliers(self):
        """Return the multipliers for each local dof."""
        return self._local_multipliers

    @property
    def normal_multipliers(self):
        """Return the normal multipliers for each grid element."""
        return self._normal_multipliers

    @property
    def global2local(self):
        """Return global to local map."""
        if self._global2local_map is None:
            self._global2local_map = invert_local2global(self.local2global, self.local_multipliers)
        return self._global2local_map

    @property
    def number_of_shape_functions(self):
        """Return the number of shape functions on each element."""
        return self._shapeset.number_of_shape_functions

    @property
    def number_of_support_elements(self):
        """Return the number of elements that form the support."""
        return self._number_of_support_elements

    @property
    def support_elements(self):
        """Return the list of elements on which space is supported."""
        return self._support_elements

    @property
    def identifier(self):
        """Return the identifier."""
        return self._identifier

    @property
    def support(self):
        """Return support of the space."""
        return self._support

    @property
    def id(self):
        """Return id string of the space."""
        return self._id

    @property
    def shapeset(self):
        """Return the shapeset."""
        return self._shapeset

    @property
    def localised_space(self):
        """Return the elementwise defined space."""
        return self._localised_space

    @property
    def is_localised(self):
        """Return true if space is localised."""
        return self.localised_space == self

    @property
    def color_map(self):
        """
        Return a coloring of the grid associated with the space.

        The coloring is defined such that if two elements have
        the same color then their associated global degrees of
        freedom do not intersect. This is important for the
        efficient summation of global dofs during the assembly.
        """
        if self._color_map is None:
            self._compute_color_map()
        return self._color_map

    @property
    def map_to_localised_space(self):
        """Return a sparse matrix that maps dofs to localised space."""
        return self._map_to_localised_space

    @property
    def map_to_full_grid(self):
        """Return a sparse matrix that maps dofs to localised space on full grid."""
        return self._map_to_full_grid

    @property
    def dof_transformation(self):
        """Return transformation from global dofs to space dofs."""
        return self._dof_transformation

    @property
    def requires_dof_transformation(self):
        """Return true if the dof transformation matrix is not the identity."""
        return self._requires_dof_transformation

    @property
    def is_barycentric(self):
        """Return true if space is defined over barycentric grid."""
        return self._is_barycentric

    @property
    def hash(self):
        """Return hash string for space comparison."""
        if self._hash_string is None:
            self._hash_string = self._generate_hash()

        return self._hash_string

    @property
    def numba_evaluate(self):
        """Return the basis evaluator."""
        return self._numba_evaluate

    @property
    def numba_surface_gradient(self):
        """Return the surface gradient evaluator."""
        if self._numba_surface_gradient is None:
            raise ValueError("No surface gradient define for this space.")
        return self._numba_surface_gradient

    @property
    def numba_surface_curl(self):
        """Return the surface curl evaluator."""
        if self._numba_surface_curl is None:
            raise ValueError("No surface curl define for this space.")
        return self._numba_surface_curl

    @property
    def has_surface_gradient(self):
        """Return True if surface gradient is defined."""
        return self._numba_surface_gradient is not None

    @property
    def has_surface_curl(self):
        """Return True if surface curl is defined."""
        return self._numba_surface_curl is not None

    @property
    def collocation_points(self):
        """Return collocation points."""
        return self._collocation_points

    def barycentric_representation(self):
        """Return barycentric_representation if it exists."""
        import collections

        # Lazy evaluation. By default only a function object is created.
        # On first call this is executed to produce the space.
        if isinstance(self._barycentric_representation, collections.abc.Callable):
            self._barycentric_representation = self._barycentric_representation(self)
        return self._barycentric_representation

    def map_to_points(self, quadrature_order=None, return_transpose=False):
        """
        Return a map from function space coefficients to point evaluations.

        Creates a mapping from function space coefficients to Green's fct.
        coefficients. Needed mainly for FMM evaluations. The point definition
        is the quadrature order of the underlying quadrature rule. If
        'return_transpose' is true then then transpose of the operator is returned.
        """
        return map_space_to_points(self, quadrature_order=quadrature_order, return_transpose=return_transpose)

    def get_elements_by_color(self):
        """
        Return color sorted elements and their index positions.

        This method returns a tuple (sorted_indices, indexptr) so that
        all element indices with color i are contained in
        sorted_indices[indexptr[i]:indexptr[i+1]].

        """
        if self._sorted_indices is None:
            self._sort_elements_by_color()
        return (self._sorted_indices, self._indexptr)

    def evaluate(self, element_index, local_coordinates):
        """
        Evaluate the basis on an element.

        Returns an array of the form
        (codomain_dimension, number_of_shape_functions, number_of_eval_points)
        that contains the basis functions evaluated at the given points.
        """
        return self.numba_evaluate(
            element_index,
            self.shapeset.evaluate,
            local_coordinates,
            self.grid.data(),
            self.local_multipliers,
            self.normal_multipliers,
        )

    def surface_gradient(self, element_index, local_coordinates):
        """Return the surface gradient."""
        return self.numba_surface_gradient(
            element_index,
            self.shapeset.gradient,
            local_coordinates,
            self.grid.data(),
            self.local_multipliers,
            self.normal_multipliers,
        )

    def surface_curl(self, element_index, element, local_coordinates):
        """Return the surface gradient."""
        return self.numba_surface_curl(
            element_index, element, self.shapeset.gradient, local_coordinates, self.grid.data()
        )

    def mass_matrix(self):
        """Return the mass matrix associated with this space."""
        if self._mass_matrix is None:
            from bempp_cl.api.operators.boundary.sparse import identity

            self._mass_matrix = identity(self, self, self).weak_form()

        return self._mass_matrix

    def inverse_mass_matrix(self):
        """Return the inverse mass matrix for this space."""
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        if self._inverse_mass_matrix is None:
            self._inverse_mass_matrix = InverseSparseDiscreteBoundaryOperator(self.mass_matrix())
        return self._inverse_mass_matrix

    def _generate_hash(self):
        """Generate a hash for the space object."""
        from hashlib import md5

        dof_transform_csr = self.dof_transformation.tocsr().sorted_indices()

        md5_gen = md5()

        md5_gen.update(self.local2global.tobytes())
        md5_gen.update(self.support_elements.tobytes())
        md5_gen.update(self.normal_multipliers.tobytes())
        md5_gen.update(self.local_multipliers.tobytes())
        md5_gen.update(dof_transform_csr.indices)
        md5_gen.update(dof_transform_csr.indptr)
        md5_gen.update(dof_transform_csr.data)

        return self.identifier + "_" + self.grid_id + "_" + md5_gen.hexdigest()

    def is_compatible(self, other):
        """Check if space is compatible with other space."""
        return self == other

    def _set_id(self, new_id):
        """Assign a new id string to the space."""
        self._id = new_id

    def _compute_color_map(self):
        """Compute the color map."""
        self._color_map = -_np.ones(self.grid.number_of_elements, dtype=_np.int32)
        for element_index in self.support_elements:
            neighbors = {element_index}
            global_dofs = self.local2global[element_index]
            for dof in global_dofs:
                for elem, _ in self.global2local[dof]:
                    neighbors.add(elem)
            neighbors.remove(element_index)

            neighbor_colors = self._color_map[list(neighbors)]
            self._color_map[element_index] = next(
                color for color in range(self.number_of_support_elements) if color not in neighbor_colors
            )

    def _sort_elements_by_color(self):
        """Implement elements by color computation."""
        sorted_indices = _np.empty(self.number_of_support_elements, dtype="uint32")
        ncolors = 1 + max(self.color_map)
        indexptr = _np.zeros(1 + ncolors, dtype="uint32")

        count = 0
        for index, color in enumerate(_np.arange(ncolors, dtype="uint32")):
            colors = _np.where(self.color_map == color)[0]
            colors_length = len(colors)
            sorted_indices[count : count + colors_length] = colors
            count += colors_length
            indexptr[index + 1] = count
        self._sorted_indices, self._indexptr = sorted_indices, indexptr

    def __eq__(self, other):
        """Check if spaces are compatible."""
        return check_if_compatible(self, other)


def return_compatible_representation(*args):
    """Return representation of spaces on same grid."""
    # Check if at least one space is barycentric.
    is_barycentric = any([space.is_barycentric for space in args])

    if not is_barycentric:
        return args
    else:
        # Convert spaces
        converted = [space.barycentric_representation() for space in args]
        if not all(converted):
            raise ValueError("Not all spaces have a valid barycentric representation.")
        return converted


def check_if_compatible(space1, space2):
    """Return true if two spaces are compatible."""
    if space1.id == space2.id:
        return True

    try:
        new_space1, new_space2 = return_compatible_representation(space1, space2)
        return new_space1.hash == new_space2.hash
    except:  # noqa: E722
        return False


def map_space_to_points(space, quadrature_order=None, return_transpose=False):
    """Return mapper from grid coeffs to point evaluations."""
    import bempp_cl.api
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import aslinearoperator
    from bempp_cl.api.integration.triangle_gauss import rule

    grid = space.grid

    if quadrature_order is None:
        quadrature_order = bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular

    local_points, weights = rule(quadrature_order)

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
            shape=(space.localised_space.grid_dof_count, number_of_vertices),
        )

        return (
            aslinearoperator(space.dof_transformation.T)
            @ aslinearoperator(space.map_to_localised_space.T)
            @ aslinearoperator(transform)
        )
    else:
        transform = coo_matrix(
            (data, (vertex_indices, global_indices)),
            shape=(number_of_vertices, space.localised_space.grid_dof_count),
        )
        return (
            aslinearoperator(transform)
            @ aslinearoperator(space.map_to_localised_space)
            @ aslinearoperator(space.dof_transformation)
        )


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
    """Run Numba accelerated computational parts for point map."""
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


def _process_segments(grid, support_elements, segments, swapped_normals):
    """Process information from support_elements and segments vars."""
    if support_elements is not None and segments is not None:
        raise ValueError("Only one of 'support_elements' and 'segments' must be nonzero.")

    if swapped_normals is None:
        swapped_normals = {}

    number_of_elements = grid.number_of_elements
    normal_multipliers = _np.zeros(number_of_elements, dtype=_np.int32)

    for element_index in range(number_of_elements):
        if grid.domain_indices[element_index] in swapped_normals:
            normal_multipliers[element_index] = -1
        else:
            normal_multipliers[element_index] = 1

    if support_elements is not None:
        support = _np.full(number_of_elements, False, dtype=bool)
        support[support_elements] = True
    elif segments is not None:
        support = _np.full(number_of_elements, False, dtype=bool)
        for element_index in range(number_of_elements):
            if grid.domain_indices[element_index] in segments:
                support[element_index] = True
    else:
        support = _np.full(number_of_elements, True, dtype=bool)

    return support, normal_multipliers


def invert_local2global(local2global_map, local_multipliers):
    """Obtain the global to local dof map from the local to global map."""
    global_dof_count = 1 + _np.max(local2global_map)
    number_of_elements = len(local2global_map)

    global2local_map = [[] for _ in range(global_dof_count)]

    for elem_index in range(number_of_elements):
        for local_index, dof in enumerate(local2global_map[elem_index]):
            if local_multipliers[elem_index, local_index] != 0:
                global2local_map[dof].append((elem_index, local_index))

    for index, elem in enumerate(global2local_map):
        global2local_map[index] = tuple(elem)

    return global2local_map


def make_localised_space(space):
    """Return the associated localised space."""
    number_of_elements = space.grid.number_of_elements
    support_size = space.number_of_support_elements
    local_size = space.number_of_shape_functions

    local2global_map = _np.zeros((number_of_elements, local_size), dtype="uint32")
    local2global_map[space.support] = _np.arange(local_size * support_size, dtype="uint32").reshape(
        (support_size, local_size)
    )

    local_multipliers = _np.zeros((number_of_elements, local_size), dtype="float64")
    local_multipliers[space.support] = 1

    surface_gradient = space.numba_surface_gradient if space.has_surface_gradient else None

    surface_curl = space.numba_surface_curl if space.has_surface_curl else None

    global2local_map = invert_local2global(local2global_map, local_multipliers)

    return (
        SpaceBuilder(space.grid)
        .set_codomain_dimension(space.codomain_dimension)
        .set_support(space.support)
        .set_normal_multipliers(space.normal_multipliers)
        .set_order(space.order)
        .set_shapeset(space.shapeset.identifier)
        .set_is_localised(True)
        .set_identifier(space.identifier + "_localised")
        .set_local2global(local2global_map)
        .set_global2local(global2local_map)
        .set_local_multipliers(local_multipliers)
        .set_numba_evaluator(space.numba_evaluate)
        .set_numba_surface_gradient(surface_gradient)
        .set_numba_surface_curl(surface_curl)
        .set_is_barycentric(space.is_barycentric)
        .build()
    )


def _space_scatter_worker(
    grid_id,
    space_id,
    kind,
    degree,
    kwargs,
):
    """Copy a space to a worker."""
    import bempp_cl.api
    from bempp_cl.api.utils import pool

    grid = pool.get_data(grid_id)
    space = bempp_cl.api.function_space(
        grid,
        kind,
        degree,
        **kwargs,
    )
    space._set_id(space_id)
    pool.insert_data(space_id, space)
    bempp_cl.api.log(f"Copied space {space.id} to worker {pool.get_id()}.", "debug")


@_numba.njit
def _numba_evaluate(
    element_index,
    shapeset_evaluate,
    local_coordinates,
    grid_data,
    local_multipliers,
    normal_multipliers,
):
    """Evaluate the basis on an element."""
    shapeset_values = shapeset_evaluate(local_coordinates)
    nshapeset = shapeset_values.shape[1]
    for index in range(nshapeset):
        shapeset_values[:, index, :] *= local_multipliers[element_index, index]
    return shapeset_values

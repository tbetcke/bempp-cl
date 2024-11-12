# pylint: disable-msg=too-many-arguments
"""Definition of Grid functions in Bempp.

Documentation of the functionality of this module and examples
can be found at https://bempp.com/handbook/api/grid_functions.html
"""

import numba as _numba
import numpy as _np


def callable(*args, complex=False, jit=True, parameterized=False, vectorized=False):
    """Wrap a callable for Bempp."""
    if vectorized:

        def wrap(f):
            """Wrap a functions (vectorised)."""
            if parameterized:

                def wrapper_callable(x, n, domain_index, res, parameters):
                    """Call the wrapped callable."""
                    f(x, n, domain_index, res, parameters)

            else:

                def wrapper_callable(x, n, domain_index, res, parameters):
                    """Call the wrapped callable."""
                    f(x, n, domain_index, res)

            wrapper_callable.bempp_type = "complex" if complex else "real"
            wrapper_callable.bempp_vectorized = True

            return wrapper_callable

    else:
        if complex:
            wrap_type = _numba.complex128[:]
        else:
            wrap_type = _numba.float64[:]

        signature_parameterized = _numba.void(_numba.float64[:], _numba.float64[:], _numba.uint32, wrap_type, wrap_type)
        signature_not_parameterized = _numba.void(_numba.float64[:], _numba.float64[:], _numba.uint32, wrap_type)

        if parameterized:
            signature = signature_parameterized
        else:
            signature = signature_not_parameterized

        def wrap(f):
            """Wrap a function."""
            if not jit:
                if parameterized:

                    def wrapper_callable(x, n, domain_index, res, parameters):
                        """Call using object mode."""
                        with _numba.objmode():
                            f(x, n, domain_index, res, parameters)

                else:

                    def wrapper_callable(x, n, domain_index, res, parameters):
                        """Call using object mode."""
                        with _numba.objmode():
                            f(x, n, domain_index, res)

            else:
                f_jit = _numba.njit(signature)(f)
                if parameterized:

                    def wrapper_callable(x, n, domain_index, res, parameters):
                        """Call a wrapped function."""
                        f_jit(x, n, domain_index, res, parameters)

                else:

                    def wrapper_callable(x, n, domain_index, res, parameters):
                        """Call a wrapped function."""
                        f_jit(x, n, domain_index, res)

            njit_wrapper = _numba.njit(signature_parameterized)(wrapper_callable)

            njit_wrapper.bempp_type = "complex" if complex else "real"
            njit_wrapper.bempp_vectorized = False

            return njit_wrapper

    if not args:
        return wrap
    else:
        return wrap(args[0])


def real_callable(*args, jit=True):
    """Wrap function as a real Numba callable."""
    return callable(*args, complex=False, jit=jit)


def complex_callable(*args, jit=True):
    """Wrap function as a complex Numba callable."""
    return callable(*args, complex=True, jit=jit)


class GridFunction(object):
    """
    Representation of functions on a grid.

    Attributes
    ----------
    coefficients : np.ndarray
        Return or set the vector of coefficients.
    component_count : int
        Return the number of components of the grid
        function values.
    space : bemp.api.space.Space
        Return the space over which the GridFunction is defined.
    grid : bempp_cl.api.grid.Grid
        Return the underlying grid.
    parameters : bempp_cl.api.ParameterList
        Return the set of parameters.
    representation : string
        Return 'primal' if the coefficients of the Gridfunction
        are known. Return 'dual' if only the coefficients in the
        dual space are known.

    """

    def __init__(
        self,
        space,
        dual_space=None,
        fun=None,
        coefficients=None,
        projections=None,
        parameters=None,
        function_parameters=None,
    ):
        """
        Construct a grid function.

        A grid function can be initialized in three different ways.

        1. By providing a Python callable. Any Python callable of the
           following form is valid.::

                callable(x,n,domain_index,result)

           Here, x, n, and result are all numpy arrays. x contains the current
           evaluation point, n the associated outward normal direction and
           result is a numpy array that will store the result of the Python
           callable. The variable domain_index stores the index of the
           subdomain on which x lies (default 0). This makes it possible to
           define different functions for different subdomains.

           The following example defines input data that is the inner product
           of the coordinate x with the normal direction n.::

                fun(x,n,domain_index,result):
                    result[0] =  np.dot(x,n)

        2. By providing a vector of coefficients at the nodes. This is
           preferable if the coefficients of the data are coming from an
           external code.

        3. By providing a vector of projection data and a corresponding
           dual space.

        Parameters
        ----------
        space : bempp_cl.api.space.Space
            The space over which the GridFunction is defined.
        dual_space : bempp_cl.api.Space
            A representation of the dual space. If not specified
            then space == dual_space is assumed (optional).
        fun : callable
            A Python function from which the GridFunction is constructed
            (optional).
        coefficients : np.ndarray
            A 1-dimensional array with the coefficients of the GridFunction
            at the interpolatoin points of the space (optional).
        projections : np.ndarray
            A 1-dimensional array with the projections of the GridFunction
            onto a dual space (optional).
        parameters : bempp_cl.api.ParameterList
            A ParameterList object used for the assembly of
            the GridFunction (optional).
        function_parameters : np.ndarray
            Numpy array containing optional parameters for the callable
            to discretise.

        Notes
        -----
        * Only one of projections, coefficients, or fun is allowed as
          parameter.

        Examples
        --------
        To create a GridFunction from a Python callable my_fun use

        >>> grid_function = GridFunction(space, fun=my_fun)

        To create a GridFunction from a vector of coefficients coeffs use

        >>> grid_function = GridFunction(space,coefficients=coeffs)

        To create a GridFunction from a vector of projections proj use

        >>> grid_function = GridFunction(
                space,dual_space=dual_space, projections=proj)

        """
        from bempp_cl.api.utils.helpers import assign_parameters
        from bempp_cl.api.space.space import return_compatible_representation

        self._space = None
        self._dual_space = None
        self._coefficients = None
        self._grid_coefficients = None
        self._projections = None
        self._representation = None

        if dual_space is None:
            dual_space = space

        self._space, self._dual_space = space, dual_space

        # Now check that space and dual are defined over same grid
        # with the same normal directions. If one space is barycentric,
        # need to take this into account.

        comp_domain, comp_dual = return_compatible_representation(space, dual_space)
        self._comp_domain = comp_domain
        self._comp_dual = comp_dual

        if not comp_domain.grid == comp_dual.grid or not _np.all(
            comp_domain.normal_multipliers == comp_dual.normal_multipliers
        ):
            raise ValueError("Space and dual space must be defined on the " + "same grid with same normal directions.")

        self._parameters = assign_parameters(parameters)

        if sum(1 for e in [fun, coefficients, projections] if e is not None) != 1:
            raise ValueError("Exactly one of 'fun', 'coefficients' or 'projections' " + "must be nonzero.")

        if coefficients is not None:
            self._coefficients = coefficients
            self._representation = "primal"

        if projections is not None:
            self._projections = projections
            self._representation = "dual"

        if fun is not None:
            from bempp_cl.api.integration.triangle_gauss import rule

            points, weights = rule(self._parameters.quadrature.regular)

            if function_parameters is None:
                function_parameters = _np.array([], dtype="float64")

            if fun.bempp_type == "real":
                dtype = "float64"
            else:
                dtype = "complex128"
                if function_parameters is not None:
                    function_parameters = function_parameters.astype("complex128")

            grid_projections = _np.zeros(comp_dual.grid_dof_count, dtype=dtype)
            if not fun.bempp_vectorized:
                # Callable is not vectorized

                # Create a Numba callable from the function
                _project_function(
                    fun,
                    comp_dual.grid.data("double"),
                    comp_dual.support_elements,
                    comp_dual.local2global,
                    comp_dual.local_multipliers,
                    comp_dual.normal_multipliers,
                    comp_dual.numba_evaluate,
                    comp_dual.shapeset.evaluate,
                    points,
                    weights,
                    comp_domain.codomain_dimension,
                    grid_projections,
                    function_parameters,
                )
            else:
                # Callable is vectorized

                (
                    global_quad_points,
                    global_normals,
                    global_domain_indices,
                ) = get_function_quadrature_information(
                    comp_dual.grid.data("double"),
                    comp_dual.support_elements,
                    comp_dual.normal_multipliers,
                    points,
                )
                function_data = _np.empty(
                    (comp_domain.codomain_dimension, global_quad_points.shape[1]),
                    dtype=dtype,
                )
                fun(
                    global_quad_points,
                    global_normals,
                    global_domain_indices,
                    function_data,
                    function_parameters,
                )
                _project_function_vectorized(
                    function_data,
                    comp_dual.grid.data("double"),
                    comp_dual.support_elements,
                    comp_dual.local2global,
                    comp_dual.local_multipliers,
                    comp_dual.normal_multipliers,
                    comp_dual.numba_evaluate,
                    comp_dual.shapeset.evaluate,
                    points,
                    weights,
                    comp_domain.codomain_dimension,
                    grid_projections,
                    function_parameters,
                )

            self._projections = comp_dual.dof_transformation.T @ grid_projections

            self._representation = "dual"

    @property
    def space(self):
        """Return space."""
        return self._space

    @property
    def dtype(self):
        """Return type."""
        if self._coefficients is not None:
            return self._coefficients.dtype
        if self._projections is not None:
            return self._projections.dtype

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    @property
    def dual_space(self):
        """Return dual space."""
        return self._dual_space

    @property
    def representation(self):
        """
        Return the representation of the grid function.

        If the grid function is given in terms of coefficients in the
        domain space return 'primal'. If the function is given by its
        projections return 'dual'.
        """
        return self._representation

    @property
    def coefficients(self):
        """Return coefficient vector."""
        if self._coefficients is None:
            from bempp_cl.api.utils.helpers import get_inverse_mass_matrix

            op = get_inverse_mass_matrix(self.space, self.dual_space)
            self._coefficients = op @ self._projections
            self._representation = "primal"

        return self._coefficients

    @property
    def grid_coefficients(self):
        """Return grid coefficients."""
        if self._grid_coefficients is None:
            self._grid_coefficients = self.space.dof_transformation @ self.coefficients

        return self._grid_coefficients

    @property
    def real(self):
        """Return a new grid function consisting of the real part of this function."""
        import numpy as np
        import bempp_cl.api

        if self.representation == "primal":
            return bempp_cl.api.GridFunction(
                space=self.space,
                dual_space=self.dual_space,
                coefficients=np.real(self.coefficients),
            )
        else:
            return bempp_cl.api.GridFunction(
                space=self.space,
                dual_space=self.dual_space,
                projections=np.real(self.projections()),
            )

    @property
    def imag(self):
        """Return a new grid function consisting of the imaginary part of this function."""
        import numpy as np
        import bempp_cl.api

        if self.representation == "primal":
            return bempp_cl.api.GridFunction(
                space=self.space,
                dual_space=self.dual_space,
                coefficients=np.imag(self.coefficients),
            )
        else:
            return bempp_cl.api.GridFunction(
                space=self.space,
                dual_space=self.dual_space,
                projections=np.imag(self.projections()),
            )

    @property
    def component_count(self):
        """Return number of components."""
        return self.space.codomain_dimension

    def projections(self, dual_space=None):
        """
        Compute the vector of projections onto the given dual space.

        Parameters
        ----------
        dual_space : bempp_cl.api.space.Space
            A representation of the dual space. If not specified
            then fun.dual_space is used.

        Returns
        -------
        out : np.ndarray
            A vector of projections onto the dual space.

        """
        from bempp_cl.api.utils.helpers import get_mass_matrix

        if dual_space is None:
            dual_space = self.dual_space

        if dual_space == self._dual_space and self._projections is not None:
            return self._projections

        ident = get_mass_matrix(self.space, dual_space)
        return ident * self.coefficients

    def project_to_space(self, space):
        """Return an L^2 projection on another space."""
        from bempp_cl.api.utils.helpers import get_mass_matrix

        ident = get_mass_matrix(self.space, space)

        return GridFunction(space, projections=ident @ self.coefficients)

    def plot(self, mode=None, transformation=None):
        """
        Plot the grid function.

        Attributes
        ----------
        mode : string
            One of 'element' or 'node'. If 'element' is chosen
            the color is determined by the mid-point of the faces
            of the grid. For 'vertices' the vertex values are
            chosen
        transformation : string or object
            One of 'real', 'imag', 'abs', 'log_abs' or
            'abs_squared' or a callable object.
            Describes the data transformation
            before plotting. For functions with vector values
            only 'abs', 'log_abs' or 'abs_squared' are allowed.
            If a callable object is given this is applied instead.
            It is important that the callable returns numpy arrays
            with the same number of dimensions as before.

        """
        from bempp_cl.api.external.viewers import visualize

        visualize(self, mode, transformation)

    def evaluate(self, element_index, local_coordinates):
        """Evaluate grid function on a single element."""
        # Get global dof ids and weights
        global_dofs = self.space.local2global[element_index]
        element_values = self.space.evaluate(element_index, local_coordinates)
        return _np.tensordot(element_values, self.grid_coefficients[global_dofs], axes=([1], [0]))

    def evaluate_on_element_centers(self):
        """Evaluate the grid function on all element centers."""
        local_coordinates = _np.array([[1.0 / 3], [1.0 / 3]])

        values = _np.zeros((self.component_count, self.space.grid.number_of_elements), dtype=self.dtype)

        for index in self.space.support_elements:
            local_values = self.evaluate(index, local_coordinates)
            values[:, index] = local_values.flat
        return values

    def evaluate_on_vertices(self):
        """
        Evaluate the grid function on all vertices.

        If a function is discontinuous across elements a weighted average
        of the element values at the vertices is taken.
        """
        grid = self.space.grid
        local_coordinates = _np.array([[0, 1, 0], [0, 0, 1]], dtype="float64")

        values = _np.zeros((self.component_count, grid.number_of_vertices), dtype=self.dtype)

        # Sum up the areas of all elements adjacent to the vertices
        vertex_areas = _np.zeros(grid.number_of_vertices, dtype="float64")

        vertex_used = _np.full(grid.number_of_vertices, False)

        for element_index in self.space.support_elements:
            local_values = self.evaluate(element_index, local_coordinates)
            for i in range(3):
                index = grid.elements[i, element_index]
                vertex_used[index] = True
                element_area = grid.volumes[element_index]
                vertex_areas[index] += element_area
                values[:, index] += local_values[:, i] * element_area

        values[:, vertex_used] /= vertex_areas[vertex_used]
        return values

    def integrate(self):
        """Integrate grid function over a grid."""
        from bempp_cl.api.integration.triangle_gauss import rule

        points, weights = rule(self._parameters.quadrature.regular)

        return _integrate(
            self.grid_coefficients,
            self.space.grid.data("double"),
            self.space.support_elements,
            self.space.local2global,
            self.space.local_multipliers,
            self.space.normal_multipliers,
            self.space.numba_evaluate,
            self.space.shapeset.evaluate,
            points,
            weights,
            self.component_count,
            self.space.number_of_shape_functions,
        )

    def l2_norm(self):
        """Calculate the L^2 norm of the function."""
        import numpy as np

        # L2-Norm on the whole space
        mass = self.space.mass_matrix()
        vec = self.coefficients
        return np.sqrt(np.abs(vec.conjugate().T.dot(mass.dot(vec))))

    def __add__(self, other):
        """Add two grid functions."""
        if self.space != other.space:
            raise ValueError("Spaces are not identical.")

        if self.representation == "dual" and other.representation == "dual":
            if self.dual_space == other.dual_space:
                return GridFunction(
                    self.space,
                    projections=self.projections() + other.projections(),
                    dual_space=self.dual_space,
                )

        return GridFunction(self.space, coefficients=self.coefficients + other.coefficients)

    def __mul__(self, alpha):
        """Multiply a grid function."""
        import numpy as np

        if np.isscalar(alpha):
            if self.representation == "dual":
                return GridFunction(
                    self.space,
                    projections=alpha * self._projections,
                    dual_space=self.dual_space,
                    parameters=self.parameters,
                )
            else:
                return GridFunction(
                    self.space,
                    coefficients=alpha * self.coefficients,
                    parameters=self.parameters,
                )
        else:
            return NotImplemented

    def __rmul__(self, alpha):
        """Multiply a grid function."""
        import numpy as np

        if np.isscalar(alpha):
            return self * alpha
        else:
            return NotImplemented

    def __div__(self, alpha):
        """Divide a grid function."""
        return self * (1.0 / alpha)

    def __truediv__(self, alpha):
        """Divide a grid function."""
        return self.__div__(alpha)

    def __neg__(self):
        """Negate a grid function."""
        return self.__mul__(-1.0)

    def __sub__(self, other):
        """Subtract grid functions."""
        if self.space != other.space:
            raise ValueError("Spaces are not identical.")

        return self + (-other)

    @classmethod
    def from_random(cls, space):
        """Create a random grid function normalized to unit norm."""
        from numpy.random import randn

        ndofs = space.global_dof_count
        fun = cls(space, coefficients=randn(ndofs))
        return fun / fun.l2_norm()

    @classmethod
    def from_ones(cls, space):
        """Create a grid function with all coefficients set to one."""
        from numpy import ones

        ndofs = space.global_dof_count

        return cls(space, coefficients=ones(ndofs))

    @classmethod
    def from_zeros(cls, space):
        """Create a grid function with all coefficients set to one."""
        from numpy import zeros

        ndofs = space.global_dof_count

        return cls(space, coefficients=zeros(ndofs))

    # @classmethod
    # def from_grid_interpolation(cls, grid, mode, callable):
    # """
    # Obtain a grid function from interpolation on the grid.

    # Parameters
    # ----------
    # grid : Bempp Grid object
    # The grid to be used
    # mode : string
    # Either "elements" or "vertices". If mode is
    # "elements" the callable is interpolated on
    # element centers. If mode is "vertices" the
    # callable is interpolated on vertices.
    # callable : callable object
    # A Python callable of the form:
    # values = callable(points). 'points' is
    # a float64 Numpy array of shape (N, 3),
    # where N is the number of interpolation points.
    # values is a scalar Numpy array of interpolation
    # values.

    # Returns a grid function of space type ("DP", 0) for elementwise
    # interpolation and ("P", 1) for vertex interpolation.

    # """
    # import bempp_cl.api

    # if mode == 'elements':
    # space = bempp_cl.api.function_space(grid, "DP", 0)
    # points = grid.centroids
    # elif mode == 'vertices':
    # space = bempp_cl.api.function_space(grid, "P", 1)
    # points = grid.vertices.T
    # else:
    # raise ValueError("'mode' must be one of 'elements' or 'vertices'.")

    # values = callable(points)
    # return bempp_cl.api.GridFunction(space, coefficients=values)


@_numba.njit
def _integrate(
    coefficients,
    grid_data,
    support_elements,
    local2global,
    local_multipliers,
    normal_multipliers,
    evaluate_on_element,
    shapeset_evaluate,
    points,
    weights,
    codomain_dimension,
    number_of_shape_functions,
):
    """Integrate a grid function over a grid."""
    result = _np.zeros(codomain_dimension, dtype=coefficients.dtype)

    for index in support_elements:
        element_vals = evaluate_on_element(
            index,
            shapeset_evaluate,
            points,
            grid_data,
            local_multipliers,
            normal_multipliers,
        )

        result += (
            _np.sum(
                _np.sum(
                    (element_vals * weights)
                    * (coefficients[local2global[index]] * local_multipliers[index]).reshape(
                        number_of_shape_functions, 1
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
            * grid_data.integration_elements[index]
        )

    return result


@_numba.njit
def get_function_quadrature_information(grid_data, support_elements, normal_multipliers, quad_points):
    """Return vectorized version of quad_points, normals and domain_indices."""
    nelements = len(support_elements)
    nlocal = quad_points.shape[1]
    npoints = nlocal * nelements

    global_quad_points = _np.empty((3, npoints), dtype=_np.float64)
    global_normals = _np.empty((3, npoints), dtype=_np.float64)
    global_domain_indices = _np.empty(npoints, dtype=_np.float64)

    for index, element in enumerate(support_elements):
        global_quad_points[:, nlocal * index : nlocal * (1 + index)] = grid_data.local2global(element, quad_points)
        for local_index in range(nlocal):
            global_normals[:, nlocal * index + local_index] = grid_data.normals[element] * normal_multipliers[element]
            global_domain_indices[nlocal * index + local_index] = grid_data.domain_indices[element]

    return (global_quad_points, global_normals, global_domain_indices)


# Must be used in jit mode as fun might just be a Python callable and not numba compiled.
@_numba.njit
def _project_function(
    fun,
    grid_data,
    support_elements,
    local2global,
    local_multipliers,
    normal_multipliers,
    evaluate_on_element,
    shapeset_evaluate,
    points,
    weights,
    codomain_dimension,
    projections,
    function_parameters,
):
    """Project a Numba callable onto a grid."""
    npoints = points.shape[1]
    global_points = _np.empty((3, npoints), dtype=_np.float64)
    fvalues = _np.empty((codomain_dimension, npoints), dtype=projections.dtype)
    fun_result = _np.empty(codomain_dimension, dtype=projections.dtype)
    point = _np.empty(3, dtype=_np.float64)

    for index in support_elements:
        element_vals = evaluate_on_element(
            index,
            shapeset_evaluate,
            points,
            grid_data,
            local_multipliers,
            normal_multipliers,
        )

        for j in range(3):
            global_points[j] = (
                (1.0 - points[0] - points[1]) * grid_data.vertices[j, grid_data.elements[0, index]]
                + points[0] * grid_data.vertices[j, grid_data.elements[1, index]]
                + points[1] * grid_data.vertices[j, grid_data.elements[2, index]]
            )

        for j in range(npoints):
            point = global_points[:, j]

            fun(
                point,
                grid_data.normals[index] * normal_multipliers[index],
                grid_data.domain_indices[index],
                fun_result,
                function_parameters,
            )
            fvalues[:, j] = fun_result

        for local_fun_index in range(element_vals.shape[1]):
            projections[local2global[index, local_fun_index]] += (
                _np.sum(_np.sum(element_vals[:, local_fun_index, :] * fvalues * weights, axis=0))
                * grid_data.integration_elements[index]
            )


@_numba.njit
def _project_function_vectorized(
    function_data,
    grid_data,
    support_elements,
    local2global,
    local_multipliers,
    normal_multipliers,
    evaluate_on_element,
    shapeset_evaluate,
    points,
    weights,
    codomain_dimension,
    projections,
    function_parameters,
):
    """Project a Numba callable onto a grid."""
    npoints = points.shape[1]

    for index, element in enumerate(support_elements):
        element_vals = evaluate_on_element(
            element,
            shapeset_evaluate,
            points,
            grid_data,
            local_multipliers,
            normal_multipliers,
        )

        for local_fun_index in range(element_vals.shape[1]):
            projections[local2global[element, local_fun_index]] += (
                _np.sum(
                    _np.sum(
                        element_vals[:, local_fun_index, :]
                        * function_data[:, index * npoints : (1 + index) * npoints]
                        * weights,
                        axis=0,
                    )
                )
                * grid_data.integration_elements[element]
            )

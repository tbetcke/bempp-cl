"""Implementation of boundary operators."""


class BoundaryOperator(object):
    """A base class for boundary operators."""

    def __init__(self, domain, range_, dual_to_range, parameters):
        """Construct. Should only be called through derived class."""
        self._domain = domain
        self._range = range_
        self._dual_to_range = dual_to_range
        self._parameters = parameters
        self._cached = None
        self._range_map = None

    @property
    def domain(self):
        """Return the domain space of the operator."""
        return self._domain

    @property
    def range(self):
        """Return the range space of the operator."""
        return self._range

    @property
    def dual_to_range(self):
        """Return the dual to range space of the operator."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return the parameters associated with the operator."""
        return self._parameters

    def weak_form(self):
        """Return the weak form (assemble if necessary)."""
        if not self._cached:
            self._cached = self._assemble()

        return self._cached

    def strong_form(self):
        """Return a discrete operator that maps into the range space."""
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix

        if self._range_map is None:
            # This is the most frequent case and we cache the mass
            # matrix from the space object.
            self._range_map = get_inverse_mass_matrix(self.range, self.dual_to_range)

        return self._range_map * self.weak_form()

    def __mul__(self, other):
        """Return product with a scalar, grid function or other operator."""
        import numpy as np
        from bempp_cl.api import GridFunction

        if np.isscalar(other):
            return _ScaledBoundaryOperator(self, other)
        elif isinstance(other, BoundaryOperator):
            return _ProductBoundaryOperator(self, other)
        elif isinstance(other, GridFunction):
            if not self.domain.is_compatible(other.space):
                raise ValueError("Operator domain space does not match GridFunction space.")
            return GridFunction(
                self.range,
                projections=self.weak_form() * other.coefficients,
                dual_space=self.dual_to_range,
            )
        else:
            return NotImplemented

    def __matmul__(self, other):
        """Multiply by another operator."""
        return self.__mul__(other)

    def __rmul__(self, other):
        """Multiply by another operator."""
        import numpy as np

        if np.isscalar(other):
            return _ScaledBoundaryOperator(self, other)
        else:
            return NotImplemented

    def __add__(self, other):
        """Return sum of two boundary operators."""
        return _SumBoundaryOperator(self, other)

    def __neg__(self):
        """Negate an operator."""
        return self.__mul__(-1.0)

    def __sub__(self, other):
        """Subtract two operators."""
        return self.__add__(-other)


class BoundaryOperatorWithAssembler(BoundaryOperator):
    """Implements a boundary operator together with an assembler."""

    # pylint: disable=too-many-arguments
    def __init__(self, domain, range_, dual_to_range, assembler, operator_descriptor, transpose=False):
        """Initialize a boundary operator with assembler."""
        super().__init__(domain, range_, dual_to_range, assembler.parameters)

        self._assembler = assembler
        self._operator_descriptor = operator_descriptor
        self.transpose_ = transpose

    @property
    def assembler(self):
        """Return the assembler associated with this operator."""
        return self._assembler

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _assemble(self):
        """Assemble the operator."""
        if self.transpose_:
            return self.assembler.assemble(self.descriptor).transpose()
        return self.assembler.assemble(self.descriptor)

    def _transpose(self, _range):
        return BoundaryOperatorWithAssembler(
            self._dual_to_range, self._domain, _range, self._assembler, self._operator_descriptor, True
        )


class _SumBoundaryOperator(BoundaryOperator):
    """Return the sum of two boundary operators."""

    def __init__(self, op1, op2):
        """Construct the sum of two boundary operators."""
        if (
            not op1.domain.is_compatible(op2.domain)
            or not op1.range.is_compatible(op2.range)
            or not op1.dual_to_range.is_compatible(op2.dual_to_range)
        ):
            raise ValueError("Spaces not compatible.")

        super(_SumBoundaryOperator, self).__init__(op1.domain, op1.range, op1.dual_to_range, None)

        self._op1 = op1
        self._op2 = op2

    def _assemble(self):
        """Implement the weak form."""
        return self._op1.weak_form() + self._op2.weak_form()


class _ScaledBoundaryOperator(BoundaryOperator):
    """Scale a boundary operator."""

    def __init__(self, op, alpha):
        """Construct a scaled boundary operator."""
        super(_ScaledBoundaryOperator, self).__init__(op.domain, op.range, op.dual_to_range, op.parameters)

        self._op = op
        self._alpha = alpha

    def _assemble(self):
        """Implement the weak form."""
        return self._op.weak_form() * self._alpha


class _ProductBoundaryOperator(BoundaryOperator):
    """Multiply two boundary operators."""

    def __init__(self, op1, op2):
        """Construct a product boundary operator."""
        if not op2.range.is_compatible(op1.domain):
            raise ValueError(
                "Range space of second operator must be compatible to " + "domain space of first operator."
            )

        super(_ProductBoundaryOperator, self).__init__(op2.domain, op1.range, op1.dual_to_range, None)

        self._op1 = op1
        self._op2 = op2

    def _assemble(self):
        """Implement the weak form."""
        return self._op1.weak_form() * self._op2.strong_form()


class ZeroBoundaryOperator(BoundaryOperator):
    """A boundary operator that represents a zero operator.

    Parameters
    ----------
    domain : bempp_cl.api.space.Space
        Domain space of the operator.
    range_ : bempp_cl.api.space.Space
        Range space of the operator.
    dual_to_range : bempp_cl.api.space.Space
        Dual space to the range space.

    """

    def __init__(self, domain, range_, dual_to_range):
        import bempp_cl.api

        super(ZeroBoundaryOperator, self).__init__(domain, range_, dual_to_range, bempp_cl.api.GLOBAL_PARAMETERS)

    def _assemble(self):
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            ZeroDiscreteBoundaryOperator,
        )

        return ZeroDiscreteBoundaryOperator(self.dual_to_range.global_dof_count, self.domain.global_dof_count)

    def __iadd__(self, other):
        """Add."""
        if self.domain != other.domain or self.range != other.range or self.dual_to_range != other.dual_to_range:
            raise ValueError("Spaces not compatible.")

        return other

    def __isub__(self, other):
        """Subtract."""
        if self.domain != other.domain or self.range != other.range or self.dual_to_range != other.dual_to_range:
            raise ValueError("Spaces not compatible.")

        return -other


class MultiplicationOperator(BoundaryOperator):
    """
    Returns a weak multiplication operator with a grid function.

    Given a grid function g, this operator assembles a sparse
    matrix that implements weak multiplication with g in a given
    basis.
    """

    def __init__(
        self,
        grid_function,
        domain,
        range_,
        dual_to_range,
        parameters=None,
        mode="component",
    ):
        """
        Initialize the multiplication operator.

        This class initializes a multiplication operator mult from a given
        grid function g, such that the result h = mult @ f for a given
        grid function f is the result of projecting g * f onto the
        space `dual_to_range`.

        The number of components of the shapesets of grid_function.space,
        domain and dual_to_range must be compatible. For example, grid_function
        could be scalar, and space and dual_to_range vectorial with 3 components.


        Parameters
        ----------
        grid_function : GridFunction object
            The grid function object from which the multiplication operator is
            built. It needs to allow representation in primal (coordinate) form.
        domain : Space object
            The domain space of the operator.
        range_ : Space object
            The range space of the operator
        dual_to_range : Space object
            The dual space on which the result of the multiplication is projected.
        parameters : Parameters object
            The parameters associated with this operator.
        mode : string
            Either 'component' or 'inner'. If mode is 'component' (default) then
            the multiplication is componentwise at each quadrature point. Hence,
            if the shapesets of grid_function.space and domain.space both have
            dimension 3, then the result of the multiplication is also of dimension
            3, and the dual_to_range space needs to be compatible to it. If mode
            is 'inner', then inner product is taken of the values of grid_function
            at the quadrature point and the basis functions in space at the
            quadrature points.

        """
        from bempp_cl.api.utils.helpers import assign_parameters

        self._mode = mode
        self._grid_fun = grid_function

        # Check compatibility

        dim = grid_function.component_count
        dual_dim = dual_to_range.codomain_dimension
        dimensions_compatible = False
        if mode == "component":
            dimensions_compatible = dim == domain.codomain_dimension and dim == dual_dim
        elif mode == "inner":
            dimensions_compatible = dim == domain.codomain_dimension and dual_dim == 1
        else:
            raise ValueError("Unknown value for 'mode'. Allowed: 'component', 'inner'")
        if not dimensions_compatible:
            raise ValueError("Incompatible codomain dimensions.")

        super().__init__(domain, range_, dual_to_range, assign_parameters(parameters))

    def _assemble(self):
        """Assemble the operator."""
        from bempp_cl.api.space.space import return_compatible_representation
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from bempp_cl.api.integration.triangle_gauss import rule
        from scipy.sparse import coo_matrix

        import numpy as _np

        points, weights = rule(self._parameters.quadrature.regular)

        comp_trial, comp_test, comp_fun = return_compatible_representation(
            self.domain, self.dual_to_range, self._grid_fun.space
        )

        grid = comp_trial.grid

        if self._mode == "component":
            op = _np.multiply
        elif self._mode == "inner":
            op = lambda x, y: _np.sum(x * y.reshape[:, _np.newaxis, :], axis=0, keepdims=True)

        elements = (
            set(comp_test.support_elements)
            .intersection(set(comp_trial.support_elements))
            .intersection(set(comp_fun.support_elements))
        )

        elements = _np.flatnonzero(comp_trial.support * comp_test.support * comp_fun.support)

        number_of_elements = len(elements)
        nshape_trial = comp_trial.shapeset.number_of_shape_functions
        nshape_test = comp_test.shapeset.number_of_shape_functions
        nshape = nshape_trial * nshape_test

        if _np.iscomplexobj(self._grid_fun.coefficients):
            dtype = "complex128"
        else:
            dtype = "float64"

        data = _np.zeros(number_of_elements * nshape_trial * nshape_test, dtype=dtype)

        for index, elem_index in enumerate(elements):
            scale_vals = self._grid_fun.evaluate(elem_index, points) * weights * grid.integration_elements[index]
            domain_vals = comp_trial.evaluate(elem_index, points)
            trial_vals = op(domain_vals, scale_vals)
            test_vals = _np.conj(comp_test.evaluate(elem_index, points))
            res = _np.tensordot(test_vals, trial_vals, axes=([0, 2], [0, 2]))
            data[nshape * index : nshape * (1 + index)] = res.ravel()

        irange = _np.arange(nshape_test)
        jrange = _np.arange(nshape_trial)

        rows = _np.tile(_np.repeat(irange, nshape_trial), number_of_elements) + _np.repeat(
            elements * nshape_test, nshape
        )

        cols = _np.tile(_np.tile(jrange, nshape_test), number_of_elements) + _np.repeat(elements * nshape_trial, nshape)

        new_rows = comp_test.local2global.ravel()[rows]
        new_cols = comp_trial.local2global.ravel()[cols]

        nrows = comp_test.dof_transformation.shape[0]
        ncols = comp_trial.dof_transformation.shape[0]

        mat = coo_matrix((data, (new_rows, new_cols)), shape=(nrows, ncols)).tocsr()

        if comp_trial.requires_dof_transformation:
            mat = mat @ self.domain.dof_transformation

        if comp_test.requires_dof_transformation:
            mat = self.dual_to_range.dof_transformation.T @ mat

        return SparseDiscreteBoundaryOperator(mat)

"""Definition of blocked operator structures."""

import numpy as _np
from bempp_cl.api.assembly.discrete_boundary_operator import _DiscreteOperatorBase


def _sum(op1, op2):
    """Sum of two operators, allowing for one of them to be None."""
    if op1 is None:
        return op2
    elif op2 is None:
        return op1
    else:
        return op1 + op2


def _prod(op1, op2):
    """Product of two operators, allowing for one of them to be None."""
    if op1 is None or op2 is None:
        return None
    else:
        return op1 * op2


class BlockedOperatorBase(object):
    """Base class for blocked operators."""

    def __init__(self):
        """Create base class for blocked operators."""
        self._weak_form = None
        self._range_map = None
        self._cached = None

    def weak_form(self):
        """Return cached weak form (assemble if necessary)."""
        if not self._cached:
            self._cached = self._assemble()
        return self._cached

    def strong_form(self):
        """Return a discrete operator that maps into the range space.

        Parameters
        ----------
        recompute : bool
            Usually the strong form is cached. If this parameter is set to
            `true` the strong form is recomputed.
        """
        from bempp_cl.api.utils.helpers import get_inverse_mass_matrix

        if self._range_map is None:
            nrows = len(self.range_spaces)

            _range_ops = _np.empty((nrows, nrows), dtype="O")

            for index in range(nrows):
                _range_ops[index, index] = get_inverse_mass_matrix(
                    self.range_spaces[index], self.dual_to_range_spaces[index]
                )

            self._range_map = BlockedDiscreteOperator(_range_ops)

        return self._range_map * self.weak_form()

    @property
    def ndims(self):
        """Return number of block rows and block columns."""
        return (len(self.range_spaces), len(self.domain_spaces))

    @property
    def range_spaces(self):
        """Return a list of range spaces of the blocked operator."""
        raise NotImplementedError()

    @property
    def dual_to_range_spaces(self):
        """Return a list of dual to range spaces of the blocked operator."""
        raise NotImplementedError()

    @property
    def domain_spaces(self):
        """Return a list of domain spaces of the blocked operator."""
        raise NotImplementedError()

    def __add__(self, other):
        """Add two blocked operators."""
        if not isinstance(other, BlockedOperatorBase):
            return NotImplementedError

        return SumBlockedOperator(self, other)

    def __neg__(self):
        """Multiply the blocked operator with -1."""
        return self.__mul__(-1)

    def __sub__(self, other):
        """Subtract another blocked operator from this blocked operator."""
        return self.__add__(-other)

    def __mul__(self, other):
        """Multiply two blocked operators."""
        from collections.abc import Iterable

        if _np.isscalar(other):
            # Multiplication with scalar
            return ScaledBlockedOperator(self, other)
        elif isinstance(other, BlockedOperatorBase):
            # Multiplication with another blocked operator.
            return ProductBlockedOperator(self, other)
        elif isinstance(other, Iterable):
            # Multiplication with a list of grid functions.
            from bempp_cl.api.assembly.grid_function import GridFunction

            list_input = list(other)
            if len(list_input) != self.ndims[1]:
                raise ValueError(
                    "Length of input list is {0}.".format(len(list_input))
                    + ". But domain dimension of blocked operator"
                    + "is {0}".format(self.ndims[1])
                )
            for item in list_input:
                if not isinstance(item, GridFunction):
                    raise ValueError("All items in the input list must be grid functions.")
            weak_op = self.weak_form()
            x_in = coefficients_from_grid_functions_list(list_input)
            res = weak_op * x_in

            # Now assemble the output grid functions back together.
            output_list = grid_function_list_from_projections(res, self.range_spaces, self.dual_to_range_spaces)
            return output_list

        else:
            return NotImplemented

    def __matmul__(self, other):
        """Multiply two operators."""
        return self.__mul__(other)

    def __rmul__(self, other):
        """Multiply two operators."""
        if _np.isscalar(other):
            return self.__mul__(other)
        else:
            return NotImplemented


class BlockedOperator(BlockedOperatorBase):
    """Construct an m x n blocked boundary operator."""

    def __init__(self, m, n):
        """Construct an (m x n) blocked boundary operator."""
        super().__init__()

        self._operators = _np.empty((m, n), dtype=object)
        self._rows = m * [False]
        self._cols = n * [False]

        self._dual_to_range_spaces = m * [None]
        self._range_spaces = m * [None]
        self._domain_spaces = n * [None]

        super().__init__()

    def __getitem__(self, key):
        """Return a component operator with a key of the form (i, j)."""
        import bempp_cl.api

        if self._operators[key] is None:
            return bempp_cl.api.ZeroBoundaryOperator(
                self.domain_spaces[key[1]],
                self.range_spaces[key[0]],
                self.dual_to_range_spaces[key[0]],
            )
        else:
            return self._operators[key]

    def __setitem__(self, key, operator):
        """Set a component operator with a key of the form (i, j)."""
        row = key[0]
        col = key[1]

        if self.range_spaces[row] is not None:
            if operator.range != self.range_spaces[row]:
                raise ValueError("Range space not compatible with " + "self.range_spaces[{0}]".format(row))

        if self.dual_to_range_spaces[row] is not None:
            if operator.dual_to_range != self.dual_to_range_spaces[row]:
                raise ValueError(
                    "Dual to range space not compatible with " + "self.dual_to_range_spaces[{0}]".format(row)
                )

        if self.domain_spaces[col] is not None:
            if operator.domain != self.domain_spaces[col]:
                raise ValueError("Domain space not compatible with " + "self.domain_spaces[{0}]".format(col))

        self._range_spaces[row] = operator.range
        self._dual_to_range_spaces[row] = operator.dual_to_range
        self._domain_spaces[col] = operator.domain
        self._operators[key] = operator
        self._rows[row] = True
        self._cols[col] = True

    def _fill_complete(self):
        """Check if each row and  column contain at least one operator."""
        return (False not in self._cols) and (False not in self._rows)

    def _assemble(self):
        """Implement the weak form."""
        if not self._fill_complete():
            raise ValueError("Each row and column must have at least one operator")

        ops = _np.empty((self.ndims[0], self.ndims[1]), dtype="O")

        for i in range(self.ndims[0]):
            for j in range(self.ndims[1]):
                if self._operators[i, j] is not None:
                    ops[i, j] = self._operators[i, j].weak_form()

        return BlockedDiscreteOperator(ops)

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._domain_spaces)


class GeneralizedBlockedOperator(BlockedOperatorBase):
    """
    Construct a generalized blocked operator.

    A generalized blocked operator has as components either

    - Simple operators
    - Blocked operators
    - Generalized blocked operators
    - Arrays of simple/blocked/generalized blocked operators

    """

    def __init__(self, array):
        """
        Initialize the operator.

        The input array must be a two-dimensional iterable that
        specifies the components. As long as the components make sense
        in terms of compatibility of spaces, the input will be
        accepted.

        """
        from bempp_cl.api.assembly.boundary_operator import BoundaryOperator
        from collections.abc import Iterable

        def make_blocked(operator):
            """Turn a BoundaryOperator into a 1x1 blocked operator."""
            blocked_operator = BlockedOperator(1, 1)
            blocked_operator[0, 0] = operator
            return blocked_operator

        self._ops = []
        self._components_per_row = None
        self._components_per_column = None

        # First iterate through the array and transform each component into a
        # generalized blocked operator.

        for row in array:
            current_row = []
            for elem in row:
                if isinstance(elem, Iterable):
                    current_row.append(GeneralizedBlockedOperator(elem))
                elif isinstance(elem, BoundaryOperator):
                    current_row.append(make_blocked(elem))
                elif isinstance(elem, BlockedOperatorBase):
                    current_row.append(elem)
                else:
                    raise ValueError("Cannot process element of type: {0}".format(type(elem)))
            self._ops.append(current_row)

            all_domain_spaces = []
            all_range_spaces = []
            all_dual_to_range_spaces = []

            for row in self._ops:
                range_spaces = row[0].range_spaces
                dual_to_range_spaces = row[0].dual_to_range_spaces
                domain_spaces = []
                for elem in row:
                    if elem.range_spaces != range_spaces:
                        raise ValueError("Incompatible range spaces detected.")
                    if elem.dual_to_range_spaces != dual_to_range_spaces:
                        raise ValueError("Incompatible dual to range spaces detected.")
                    domain_spaces.extend(elem.domain_spaces)
                all_range_spaces.extend(range_spaces)
                all_dual_to_range_spaces.extend(dual_to_range_spaces)
                if all_domain_spaces:
                    # We have already processed one row
                    # and compare domain spaces to it.
                    if domain_spaces != all_domain_spaces:
                        raise ValueError("Incompatible domain spaces detected.")
                else:
                    # We are at the first row.
                    all_domain_spaces = domain_spaces

            self._domain_spaces = tuple(all_domain_spaces)
            self._dual_to_range_spaces = tuple(all_dual_to_range_spaces)
            self._range_spaces = tuple(all_range_spaces)

            super().__init__()

    def _assemble(self):
        """Implement the weak form."""
        assembled_list = []
        for row in self._ops:
            assembled_row = []
            for elem in row:
                assembled_row.append(elem.weak_form())
            assembled_list.append(assembled_row)
        return GeneralizedDiscreteBlockedOperator(assembled_list)

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._domain_spaces)


class MultitraceOperatorFromAssembler(BlockedOperatorBase):
    """A multitrace operator from an assembler."""

    def __init__(
        self,
        domain_spaces,
        range_spaces,
        dual_to_range_spaces,
        assembler,
        multitrace_operator_descriptor,
    ):
        """Generate operator from assembler."""
        self._domain_spaces = tuple(domain_spaces)
        self._range_spaces = tuple(range_spaces)
        self._dual_to_range_spaces = tuple(dual_to_range_spaces)
        self._assembler = assembler
        self._descriptor = multitrace_operator_descriptor

        super().__init__()

    def _assemble(self, *args, **kwargs):
        """Assemble the operator."""
        return self._assembler.assemble(self._descriptor, *args, **kwargs)

    @property
    def range_spaces(self):
        """Return a list of range spaces of the blocked operator."""
        return self._range_spaces

    @property
    def dual_to_range_spaces(self):
        """Return a list of dual to range spaces of the blocked operator."""
        return self._dual_to_range_spaces

    @property
    def domain_spaces(self):
        """Return a list of domain spaces of the blocked operator."""
        return self._domain_spaces


class SumBlockedOperator(BlockedOperatorBase):
    """Represents the sum of two blocked boundary operators."""

    def __init__(self, op1, op2):
        """Construct the object from two blocked operators."""
        if (
            op1.domain_spaces != op2.domain_spaces
            or op1.range_spaces != op2.range_spaces
            or op1.dual_to_range_spaces != op2.dual_to_range_spaces
        ):
            raise ValueError("Incompatible spaces for summation.")

        self._op1 = op1
        self._op2 = op2

        super().__init__()

    def _assemble(self):
        return self._op1.weak_form() + self._op2.weak_form()

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op1.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op1.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op1.domain_spaces)


class ProductBlockedOperator(BlockedOperatorBase):
    """Represents the Product of two blocked boundary operators."""

    def __init__(self, op1, op2):
        """Construct the blocked operator product op1 * op2."""
        if op2.range_spaces != op1.domain_spaces:
            raise ValueError("Incompatible spaces for multiplication.")

        self._op1 = op1
        self._op2 = op2

        super().__init__()

    def _assemble(self):
        return self._op1.weak_form() * self._op2.strong_form()

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op1.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op1.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op2.domain_spaces)


class ScaledBlockedOperator(BlockedOperatorBase):
    """Represents the scalar multiplication of a blocked operator."""

    def __init__(self, op, alpha):
        """Construct the scaled operator alpha * op."""
        self._op = op
        self._alpha = alpha

        super(ScaledBlockedOperator, self).__init__()

    def _assemble(self):
        """Assemble operator."""
        return self._alpha * self._op.weak_form()

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._op.range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._op.dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._op.domain_spaces)


class GeneralizedDiscreteBlockedOperator(_DiscreteOperatorBase):
    """A discrete generalized blocked operator."""

    def __init__(self, operators):
        """Initialize a generalized blocked operator."""
        from bempp_cl.api.utils.data_types import combined_type

        self._operators = operators

        shape = [0, 0]
        # Get column dimension
        for elem in operators[0]:
            shape[1] += elem.shape[1]
        # Get row dimension
        for row in operators:
            shape[0] += row[0].shape[0]

        shape = tuple(shape)

        # Get dtype

        dtype = operators[0][0].dtype
        for row in operators:
            for elem in row:
                dtype = combined_type(dtype, elem.dtype)

        # Sanity check of dimensions

        for row in operators:
            row_dim = row[0].shape[0]
            column_dim = 0
            for elem in row:
                if elem.shape[0] != row_dim:
                    raise ValueError("Incompatible dimensions detected.")
                column_dim += elem.shape[1]
            if column_dim != shape[1]:
                raise ValueError("Incompatible dimensions detected.")

        super().__init__(dtype, shape)

    def to_dense(self):
        """Return dense matrix."""
        rows = []
        for row in self._operators:
            rows.append([op.to_dense() for op in row])
        return _np.block(rows)

    def _matmat(self, other):
        """Implement the matrix/vector product."""
        from bempp_cl.api.utils.data_types import combined_type

        row_count = 0
        output = _np.zeros(
            (self.shape[0], other.shape[1]),
            dtype=combined_type(self.dtype, other.dtype),
        )

        for row in self._operators:
            row_dim = row[0].shape[0]
            column_count = 0
            for elem in row:
                output[row_count : row_count + row_dim, :] += (
                    elem @ other[column_count : column_count + elem.shape[1], :]
                )
                column_count += elem.shape[1]
            row_count += row_dim

        return output


class BlockedDiscreteOperator(_DiscreteOperatorBase):
    """Implementation of a discrete blocked boundary operator."""

    def __init__(self, ops):
        """
        Construct an operator from a two dimensional Numpy array of operators.

        ops is a list of list containing discrete boundary operators or None.
        A None entry is equivalent to a zero discrete boundary operator.

        """
        # pylint: disable=too-many-branches
        from bempp_cl.api.utils.data_types import combined_type
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            ZeroDiscreteBoundaryOperator,
        )

        if not isinstance(ops, _np.ndarray):
            ops = _np.array(ops)

        rows = ops.shape[0]
        cols = ops.shape[1]

        self._ndims = (rows, cols)

        self._operators = _np.empty((rows, cols), dtype=object)
        self._rows = -_np.ones(rows, dtype=int)
        self._cols = -_np.ones(cols, dtype=int)

        for i in range(rows):
            for j in range(cols):
                if ops[i, j] is None:
                    continue
                if self._rows[i] != -1:
                    if ops[i, j].shape[0] != self._rows[i]:
                        raise ValueError("Block row {0} has incompatible ".format(i) + " operator sizes.")
                else:
                    self._rows[i] = ops[i, j].shape[0]

                if self._cols[j] != -1:
                    if ops[i, j].shape[1] != self._cols[j]:
                        raise ValueError("Block column {0} has incompatible".format(j) + "operator sizes.")
                else:
                    self._cols[j] = ops[i, j].shape[1]
                self._operators[i, j] = ops[i, j]

        if not self._fill_complete():
            raise ValueError("Each row and column must contain at least one operator.")

        for i in range(rows):
            for j in range(cols):
                if self._operators[i, j] is None:
                    self._operators[i, j] = ZeroDiscreteBoundaryOperator(self._rows[i], self._cols[j])

        shape = (_np.sum(self._rows), _np.sum(self._cols))

        dtype = "float32"
        for obj in self._operators.ravel():
            if obj is not None:
                dtype = combined_type(dtype, obj.dtype)

        super().__init__(dtype, shape)

    def __getitem__(self, key):
        """Return the object at position (i, j)."""
        return self._operators[key]

    def _fill_complete(self):
        if (-1 in self._rows) or (-1 in self._cols):
            return False
        return True

    def _matvec(self, x):
        from bempp_cl.api.utils.data_types import combined_type

        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")

        if x.ndim > 1:
            return self._matmat(x)

        row_dim = 0
        res = _np.zeros(self.shape[0], dtype=combined_type(self.dtype, x.dtype))

        for i in range(self._ndims[0]):
            col_dim = 0
            local_res = res[row_dim : row_dim + self._rows[i]]
            for j in range(self._ndims[1]):
                local_x = x[col_dim : col_dim + self._cols[j]]
                op_is_complex = _np.iscomplexobj(self._operators[i, j].dtype.type(1))
                if _np.iscomplexobj(x) and not op_is_complex:
                    local_res += self._operators[i, j].dot(_np.real(local_x)) + 1j * self._operators[i, j].dot(
                        _np.imag(local_x)
                    )
                else:
                    local_res += self._operators[i, j].dot(local_x)
                col_dim += self._cols[j]
            row_dim += self._rows[i]

        return res

    def _matmat(self, x):
        from bempp_cl.api.utils.data_types import combined_type

        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")

        row_dim = 0
        res = _np.zeros((self.shape[0], x.shape[1]), dtype=combined_type(self.dtype, x.dtype))

        for i in range(self._ndims[0]):
            col_dim = 0
            local_res = res[row_dim : row_dim + self._rows[i], :]
            for j in range(self._ndims[1]):
                local_x = x[col_dim : col_dim + self._cols[j], :]
                op_is_complex = _np.iscomplexobj(self._operators[i, j].dtype.type(1))
                if _np.iscomplexobj(x) and not op_is_complex:
                    local_res[:] += self._operators[i, j].dot(_np.real(local_x)) + 1j * self._operators[i, j].dot(
                        _np.imag(local_x)
                    )
                else:
                    local_res[:] += self._operators[i, j].dot(local_x)
                col_dim += self._cols[j]
            row_dim += self._rows[i]
        return res

    def _get_row_dimensions(self):
        return self._rows

    def _get_column_dimensions(self):
        return self._cols

    def to_dense(self):
        """Return dense matrix."""
        rows = []
        for i in range(self._ndims[0]):
            row = [self[i, j].to_dense() for j in range(self._ndims[1])]
            rows.append(_np.hstack(row))
        return _np.vstack(rows)

    def _transpose(self):
        """Implement the transpose."""
        raise NotImplementedError()

    def _adjoint(self):
        """Implement the adjoint."""
        raise NotImplementedError()

    row_dimensions = property(_get_row_dimensions)
    column_dimensions = property(_get_column_dimensions)


# pylint: disable=invalid-name
def coefficients_from_grid_functions_list(grid_funs):
    """
    Return a vector of coefficients of a list of grid functions.

    Given a list [f0, f1, f2, ...] this function returns a
    single Numpy vector containing [f1.coefficients, f2.coefficients, ...].

    Parameters
    ----------
    grid_funs : list of GridFunction objects
        A list containing the grid functions
    """
    vec_len = 0
    input_type = _np.dtype("float32")
    for item in grid_funs:
        input_type = _np.promote_types(input_type, item.coefficients.dtype)
        vec_len += item.space.global_dof_count
    res = _np.zeros(vec_len, dtype=input_type)
    pos = 0
    for item in grid_funs:
        dof_count = item.space.global_dof_count
        res[pos : pos + dof_count] = item.coefficients
        pos += dof_count
    return res


def projections_from_grid_functions_list(grid_funs, projection_spaces):
    """
    Return a vector of projections of a list of grid functions.

    Given a list [f0, f1, f2, ...] this function returns a
    single Numpy vector containing
    [f0.projections(projection_spaces[0]),
     f1.projections(projection_spaces[1])],
     ...].

    Parameters
    ----------
    grid_funs : list of GridFunction objects
        A list containing the grid functions
    projection_spaces : list of projection spaces
        A list of projection spaces. Must have the same
        length as grid_funs.
    """
    projections = []
    for item, proj_space in zip(grid_funs, projection_spaces):
        projections.append(item.projections(proj_space))
    vec_len = 0
    input_type = _np.dtype("float32")
    for item in projections:
        input_type = _np.promote_types(input_type, item.dtype)
        vec_len += len(item)
    res = _np.zeros(vec_len, dtype=input_type)
    pos = 0
    for item in projections:
        dof_count = len(item)
        res[pos : pos + dof_count] = item
        pos += dof_count
    return res


# pylint: disable=invalid-name
def grid_function_list_from_coefficients(coefficients, spaces):
    """
    Create a list of grid functions from a long vector of coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        One-dimensional array of coefficients
    spaces : list of Space objects
        The sum of the global dofs of the spaces must be equal to the
        length of the coefficients vector.
    """
    from bempp_cl.api import GridFunction

    pos = 0
    res_list = []
    for space in spaces:
        dof_count = space.global_dof_count
        res_list.append(GridFunction(space, coefficients=coefficients[pos : pos + dof_count]))
        pos += dof_count
    return res_list


# pylint: disable=invalid-name
def grid_function_list_from_projections(projections, spaces, dual_spaces=None):
    """
    Create a list of grid functions from a long vector of projections.

    Parameters
    ----------
    coefficients : np.ndarray
        One-dimensional array of coefficients
    spaces : list of Space objects
        The sum of the global dofs of the spaces must be equal to the
        length of the coefficients vector.
    dual_spaces : list of Space objects
        The associated dual spaces. If None use the spaces as dual spaces.
    """
    from bempp_cl.api import GridFunction

    pos = 0
    res_list = []
    if dual_spaces is None:
        dual_spaces = spaces
    if len(spaces) != len(dual_spaces):
        raise ValueError("spaces must have the same length as dual_spaces")
    for space, dual in zip(spaces, dual_spaces):
        dof_count = space.global_dof_count
        res_list.append(GridFunction(space, projections=projections[pos : pos + dof_count], dual_space=dual))
        pos += dof_count
    return res_list

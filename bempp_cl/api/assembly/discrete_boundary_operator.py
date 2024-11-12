"""Data structures for assembled boundary operators."""

import warnings
import numpy as _np
from bempp_cl.helpers import timeit as _timeit
from scipy.sparse.linalg.interface import LinearOperator as _LinearOperator

# Disable warnings for differing overridden parameters
# pylint: disable=W0221


class _DiscreteOperatorBase(_LinearOperator):
    """Discrete boundary operator base."""

    def __init__(self, dtype, shape):
        """Construct. Should not be called directly."""
        super().__init__(dtype, shape)

    def __add__(self, other):
        """Sum of two operators."""
        if isinstance(other, _DiscreteOperatorBase):
            return _SumDiscreteOperator(self, other)
        else:
            return super().__add__(other)

    def __neg__(self):
        """Negation."""
        return _ScaledDiscreteOperator(self, -1)

    def __sub__(self, other):
        """Subtraction."""
        return self.__add__(-other)

    def dot(self, other):
        """Product with other objects."""
        if isinstance(other, _DiscreteOperatorBase):
            return _ProductDiscreteOperator(self, other)
        elif _np.isscalar(other):
            return _ScaledDiscreteOperator(self, other)
        else:
            return super().dot(other)

    def __mul__(self, other):
        """Product with other objects."""
        return self.dot(other)

    def __rmul__(self, other):
        """Reverse product."""
        if _np.isscalar(other):
            return _ScaledDiscreteOperator(self, other)
        else:
            return NotImplemented

    @property
    def A(self):
        """Return dense matrix."""
        warnings.warn(
            "operator.A is deprecated and will be removed in a " "future version. Use operator.to_dense() instead.",
            DeprecationWarning,
        )
        return self.to_dense()

    def to_dense(self):
        """Return dense matrix."""
        raise NotImplementedError()

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        raise NotImplementedError()


class _ScaledDiscreteOperator(_DiscreteOperatorBase):
    """Return a scaled operator."""

    def __init__(self, op, alpha):
        dtype = _np.result_type(op.dtype, type(alpha))
        self._op = op
        self._alpha = alpha
        super().__init__(dtype, op.shape)

    def _matvec(self, x):
        """Matvec."""
        return self._alpha * (self._op @ x)

    def to_dense(self):
        """Return dense matrix."""
        return self._alpha * self._op.to_dense()

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        return self._alpha * self._op.to_sparse()


class _SumDiscreteOperator(_DiscreteOperatorBase):
    """Return a sum operator."""

    def __init__(self, op1, op2):
        """Construct."""
        if op1.shape != op2.shape:
            raise ValueError(f"Operators have incompatible shapes {op1.shape} != {op2.shape}")

        self._op1 = op1
        self._op2 = op2

        dtype = _np.result_type(op1.dtype, op2.dtype)

        super().__init__(dtype, op1.shape)

    def _matvec(self, x):
        """Evaluate matvec."""
        return self._op1 @ x + self._op2 @ x

    def to_dense(self):
        """Return dense matrix."""
        return self._op1.to_dense() + self._op2.to_dense()

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        return self._op1.to_sparse() + self._op2.to_sparse()


class _ProductDiscreteOperator(_DiscreteOperatorBase):
    """Product of two operators."""

    def __init__(self, op1, op2):
        """Construct the product of two operators."""
        if op1.shape[1] != op2.shape[0]:
            raise ValueError(f"Incompatible dimensions shapes for multiplication with {op1.shape} and {op2.shape}")

        self._op1 = op1
        self._op2 = op2

        dtype = _np.result_type(op1.dtype, op2.dtype)

        super().__init__(dtype, (op1.shape[0], op2.shape[1]))

    def _matvec(self, x):
        """Evaluate matvec."""
        return self._op1 @ (self._op2 @ x)

    def to_dense(self):
        """Return dense matrix."""
        return self._op1.to_dense() @ self._op2.to_dense()

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        return self._op1.to_sparse() @ self._op2.to_sparse()


class GenericDiscreteBoundaryOperator(_DiscreteOperatorBase):
    """Discrete boundary operator that implements a matvec routine."""

    def __init__(self, evaluator):
        """Construct discrete boundary operator."""
        super().__init__(evaluator.dtype, evaluator.shape)
        self._evaluator = evaluator
        self._is_complex = self.dtype == "complex128" or self.dtype == "complex64"

    def _matvec(self, x):
        """Mutliply the operator by a vector."""
        if self._is_complex:
            return self._evaluator.matvec(x)
        if _np.iscomplexobj(x):
            return self._evaluator.matvec(_np.real(x)) + 1j * self._evaluator.matvec(_np.imag(x))
        else:
            return self._evaluator.matvec(x)

    def to_dense(self):
        """Return dense matrix."""
        return self @ _np.eye(self.shape[1])

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        raise ValueError("This matrix is not sparse.")


class DenseDiscreteBoundaryOperator(_DiscreteOperatorBase):
    """
    Main class for the discrete form of dense nonlocal operators.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    """

    def __init__(self, impl):
        """Construct. Should not be called by the user."""
        self._impl = impl
        super().__init__(impl.dtype, impl.shape)

    def _matmat(self, x):
        """Multiply the operator by a matrix or operator."""
        if _np.iscomplexobj(x) and not _np.iscomplexobj(self.to_dense()):
            return self.to_dense().dot(_np.real(x).astype(self.dtype)) + 1j * self.to_dense().dot(
                _np.imag(x).astype(self.dtype)
            )
        return self.to_dense().dot(x.astype(self.dtype))

    def __add__(self, other):
        """Add two operators."""
        if isinstance(other, DenseDiscreteBoundaryOperator):
            return DenseDiscreteBoundaryOperator(self.to_dense() + other.to_dense())
        else:
            return super().__add__(other)

    def __neg__(self):
        """Negate the operator."""
        return DenseDiscreteBoundaryOperator(-self.to_dense())

    def __mul__(self, other):
        """Multiply."""
        return self.dot(other)

    def dot(self, other):
        """Form the product with another object."""
        if isinstance(other, DenseDiscreteBoundaryOperator):
            return DenseDiscreteBoundaryOperator(self.to_dense().dot(other.to_dense()))
        if _np.isscalar(other):
            if self._impl.dtype in ["float32", "complex64"]:
                # Necessary to ensure that scalar multiplication does not change
                # precision to double precision.
                if _np.iscomplexobj(other):
                    return DenseDiscreteBoundaryOperator(self.to_dense() * _np.dtype("complex64").type(other))
                else:
                    return DenseDiscreteBoundaryOperator(self.to_dense() * _np.dtype("float32").type(other))
            else:
                return DenseDiscreteBoundaryOperator(self.to_dense() * other)
        return super().dot(other)

    def __rmul__(self, other):
        """Multiply."""
        if _np.isscalar(other):
            return DenseDiscreteBoundaryOperator(self.to_dense() * other)
        else:
            return NotImplemented

    def _transpose(self):
        """Transpose of the operator."""
        return DenseDiscreteBoundaryOperator(self.to_dense().T)

    def _adjoint(self):
        """Adjoint of the operator."""
        return DenseDiscreteBoundaryOperator(self.to_dense().conjugate().transpose())

    def to_dense(self):
        """Return dense matrix."""
        return self._impl

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        raise ValueError("This matrix is not sparse.")


class DiagonalOperator(_DiscreteOperatorBase):
    """
    Main class for discrete diagonal operators.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    """

    def __init__(self, values, shape=None):
        """Construct. Should not be called by the user."""
        self._values = values.ravel()
        if shape is None:
            shape = (len(values), len(values))
        super().__init__(values.dtype, shape)

    def get_diagonal(self):
        """Get the diagonal values."""
        return self._values

    def _matvec(self, x):
        """Multiply the operator by a vector."""
        vec_shape = x.shape

        return (self.get_diagonal() * x.ravel()).reshape(vec_shape)

    def __add__(self, other):
        """Add."""
        if self.shape != other.shape:
            raise ValueError(f"Incompatible dimensions: {self.shape} != {other.shape}")
        if isinstance(other, DiagonalOperator):
            return DiagonalOperator(self.get_diagonal() + other.get_diagonal())
        else:
            return super().__add__(other)

    def __neg__(self):
        """Negate."""
        return DiagonalOperator(-self.get_diagonal())

    def __mul__(self, other):
        """Multiply."""
        return self.dot(other)

    def dot(self, other):
        """Product with other objects."""
        if _np.isscalar(other):
            if self.get_diagonal().dtype in ["float32", "complex64"]:
                # Necessary to ensure that scalar multiplication does not change
                # precision to double precision.
                if _np.iscomplexobj(other):
                    return DiagonalOperator(self.get_diagonal() * _np.dtype("complex64").type(other))
                else:
                    return DiagonalOperator(self.get_diagonal() * _np.dtype("float32").type(other))
            else:
                return DiagonalOperator(self.get_diagonal() * other)
        return super().dot(other)

    def __rmul__(self, other):
        """Multiply."""
        if _np.isscalar(other):
            return DiagonalOperator(self.get_diagonal() * other)
        else:
            return NotImplemented

    def _transpose(self):
        """Transpose of the operator."""
        return self

    def _adjoint(self):
        """Adjoint of the operator."""
        return DiagonalOperator(self._values.conjugate())

    def to_dense(self):
        """Return dense matrix."""
        return _np.diag(self._values)

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        from scipy.sparse import diags

        return diags(self._values)


class SparseDiscreteBoundaryOperator(_DiscreteOperatorBase):
    """
    Main class for the discrete form of sparse operators.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    """

    def __init__(self, impl):
        """Construct. Should not be called by the user."""
        super(SparseDiscreteBoundaryOperator, self).__init__(impl.dtype, impl.shape)
        self._impl = impl

    def _matmat(self, vec):
        """Multiply the operator with a numpy vector or matrix x."""
        if self.dtype == "float64" and _np.iscomplexobj(vec):
            return self.to_sparse() * _np.real(vec) + 1j * (self.to_sparse() * _np.imag(vec))
        return self.to_sparse() * vec

    def _transpose(self):
        """Return the transpose of the discrete operator."""
        return SparseDiscreteBoundaryOperator(self.to_sparse().transpose())

    def _adjoint(self):
        """Return the adjoint of the discrete operator."""
        return SparseDiscreteBoundaryOperator(self.to_sparse().transpose().conjugate())

    def __add__(self, other):
        """Add."""
        if isinstance(other, SparseDiscreteBoundaryOperator):
            return SparseDiscreteBoundaryOperator(self.to_sparse() + other.to_sparse())
        else:
            return super().__add__(other)

    def __neg__(self):
        """Negate."""
        return SparseDiscreteBoundaryOperator(-self.to_sparse())

    def __mul__(self, other):
        """Multiply."""
        if isinstance(other, SparseDiscreteBoundaryOperator):
            return SparseDiscreteBoundaryOperator(self.to_sparse() * other.to_sparse())
        else:
            return super().__mul__(other)

    def dot(self, other):
        """Product with other objects."""
        if _np.isscalar(other):
            return SparseDiscreteBoundaryOperator(self.to_sparse() * other)
        else:
            return super().dot(other)

    def __rmul__(self, other):
        """Multiply."""
        if _np.isscalar(other):
            return SparseDiscreteBoundaryOperator(self.to_sparse() * other)
        else:
            return NotImplemented

    @property
    def A(self):
        """Return dense matrix."""
        warnings.warn(
            "operator.A is deprecated and will be removed in a " "future version. Use operator.to_dense() instead.",
            DeprecationWarning,
        )
        return self.to_sparse()

    def to_dense(self):
        """Return dense matrix."""
        return self._impl.todense()

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        return self._impl


class InverseSparseDiscreteBoundaryOperator(_DiscreteOperatorBase):
    """
    Apply the (pseudo-)inverse of a sparse operator.

    This class uses a Sparse LU-Decomposition
    (in the case of a square matrix) or a sparse normal
    equation to provide the application of an inverse to
    a sparse operator.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    Parameters
    ----------
    operator : bempp_cl.api.SparseDiscreteBoundaryOperator
        Sparse operator to be inverted.

    """

    def __init__(self, operator):
        """Create a inverse sparse operator."""
        self._solver = _Solver(operator)
        self._operator = operator
        super().__init__(self._solver.dtype, self._solver.shape)

    def _matmat(self, vec):
        """Implemententation of matvec."""
        return self._solver.solve(vec)

    def to_dense(self):
        """Return dense matrix."""
        eye = _np.eye(self.shape[1])
        return self @ eye

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        raise ValueError("This matrix is not sparse.")


class ZeroDiscreteBoundaryOperator(_DiscreteOperatorBase):
    """A discrete operator that represents a zero operator.

    This class derives from
    :class:`scipy.sparse.linalg.interface.LinearOperator`
    and thereby implements the SciPy LinearOperator protocol.

    Parameters
    ----------
    rows : int
        The number of rows in the operator.
    columns : int
        The number of columns in the operator.

    """

    def __init__(self, rows, columns):
        """Construct a zero operator."""
        super(ZeroDiscreteBoundaryOperator, self).__init__(_np.dtype("float64"), (rows, columns))

    def _matmat(self, x):
        """Multiply operator by a matrix."""
        return _np.zeros((self.shape[0], x.shape[1]), dtype="float64")

    def to_dense(self):
        """Return dense matrix."""
        return _np.zeros(self.shape)

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        from scipy.sparse import csc_matrix

        return csc_matrix(self.shape, dtype="float64")


class DiscreteRankOneOperator(_DiscreteOperatorBase):
    """Creates a discrete rank one operator.

    This class represents a rank one operator given
    by column * row, where column is column is
    interpreted as a (m, 1) array and row as
    a (1, n) array.

    Parameters
    ----------
    column : np.array
        A column vector
    row : np.array
        A row vector

    """

    def __init__(self, column, row):
        """Construct a discrete rank one operator."""
        if row.dtype == "complex128" or column.dtype == "complex128":
            dtype = "complex128"
        else:
            dtype = "float64"

        self._row = row.ravel()
        self._column = column.ravel()

        shape = (len(self._column), len(self._row))
        super().__init__(dtype, shape)

    def _matvec(self, x):
        """Multiply operator by a vector."""
        if x.ndim > 1:
            return _np.outer(self._column, _np.dot(self._row, x))
        else:
            return self._column * _np.dot(self._row, x)

    def _rmatvec(self, x):
        """Multiply."""
        # pylint: disable=protected-access
        return self._adjoint()._matvec(x)

    def _transpose(self):
        """Find the transpose."""
        return DiscreteRankOneOperator(self._row, self._column)

    def _adjoint(self):
        """Find the adjoint."""
        return DiscreteRankOneOperator(self._row.conjugate(), self._column.conjugate())

    def to_dense(self):
        """Return dense matrix."""
        return _np.outer(self._column, self._row)

    def to_sparse(self):
        """Return sparse matrix if operator is sparse."""
        raise ValueError("This matrix is not sparse.")


def as_matrix(operator):
    """
    Convert a discrte operator into a dense matrix.

    Parameters
    ----------
    operator : scipy.sparse.linalg.interface.LinearOperator
        The linear operator to be converted into a matrix.


    Notes
    -----
    Note that this function may be slow depending on how the original
    discrete operator was stored. In the case of a dense assembly simple
    the underlying NumPy matrix is returned. For sparse matrices the
    corresponding Scipy sparse matrix is returned. Otherwise, the operator needs
    to be converted to an array, which can take a long time depending on
    the assembler type.

    """
    return operator.to_dense()


class _Solver(object):  # pylint: disable=too-few-public-methods
    """Actual solve of a sparse linear system."""

    # pylint: disable=too-many-locals
    @_timeit
    def __init__(self, operator):
        """Create a solver."""
        from scipy.sparse import csc_matrix

        if isinstance(operator, SparseDiscreteBoundaryOperator):
            mat = operator.to_sparse()
        elif isinstance(operator, csc_matrix):
            mat = operator
        else:
            raise ValueError(
                "op must be either of type "
                + "SparseDiscreteBoundaryOperator or of type "
                + "csc_matrix. Actual type: "
                + str(type(operator))
            )

        from scipy.sparse.linalg import splu

        self._solve_fun = None
        self._shape = (mat.shape[1], mat.shape[0])
        self._dtype = mat.dtype

        use_mkl_pardiso = False

        # pylint: disable=bare-except
        try:
            # pylint: disable=E0401
            from mkl_pardiso_solve import PardisoInterface

            # pylint: disable=invalid-name
            solver_interface = PardisoInterface
            actual_mat = mat.tocsr()
            use_mkl_pardiso = True
        except:  # noqa: E722
            solver_interface = splu
            actual_mat = mat

        if mat.shape[0] == mat.shape[1]:
            # Square matrix case
            solver = solver_interface(actual_mat)
            self._solve_fun = solver.solve
        elif mat.shape[0] > mat.shape[1]:
            # Thin matrix case
            mat_hermitian = actual_mat.conjugate().transpose()
            if use_mkl_pardiso:
                solver = solver_interface((mat_hermitian * mat).tocsr())
            else:
                solver = solver_interface((mat_hermitian * mat).tocsc())
            self._solve_fun = lambda x: solver.solve(mat_hermitian * x)
        else:
            # Thick matrix case

            mat_hermitian = actual_mat.conjugate().transpose()
            if use_mkl_pardiso:
                solver = solver_interface((mat * mat_hermitian).tocsr())
            else:
                solver = solver_interface((mat * mat_hermitian).tocsc())
            self._solve_fun = lambda x: mat_hermitian * solver.solve(x)

    @_timeit
    def solve(self, rhs):
        """Solve with right-hand side mat."""
        if self._dtype == "float64" and _np.iscomplexobj(rhs):
            return self.solve(_np.real(rhs)) + 1j * self.solve(_np.imag(rhs))

        return self._solve_fun(rhs)

    @property
    def shape(self):
        """Return the shape of the inverse operator."""
        return self._shape

    @property
    def dtype(self):
        """Return the dtype."""
        return self._dtype

"""Definition of potential operators."""


class PotentialOperator(object):
    """Provides an interface to potential operators.

    This class is not supposed to be instantiated directly.
    """

    def __init__(self, potential_evaluator):
        """Construct. Should not be called by the user."""
        self._evaluator = potential_evaluator

    def evaluate(self, grid_fun):
        """
        Apply the potential operator to a grid function.

        Parameters
        ----------
        grid_fun : bempp_cl.api.GridFunction
            A GridFunction object that represents the boundary density to
            which the potential is applied to.

        """
        return self._evaluator.evaluate(grid_fun.coefficients)

    def _is_compatible(self, other):
        """Check compatibility with other potential operator."""
        import numpy as np

        return (
            self.component_count == other.component_count
            and np.linalg.norm(self.evaluation_points - other.evaluation_points, ord=np.inf) == 0
            and self.space.is_compatible(other.space)
        )

    def __add__(self, obj):
        """Add."""
        if not self._is_compatible(obj):
            raise ValueError("Potential operators not compatible.")

        return _SumPotentialOperator(self, obj)

    def __mul__(self, obj):
        """Multiply."""
        import numpy as np
        from bempp_cl.api import GridFunction

        if np.isscalar(obj):
            return _ScaledPotentialOperator(self, obj)
        elif isinstance(obj, GridFunction):
            return self.evaluate(obj)
        else:
            return NotImplemented

    def __matmul__(self, obj):
        """Multiply."""
        return self.__mul__(obj)

    def __rmul__(self, obj):
        """Reverse multiply."""
        import numpy as np

        if np.isscalar(obj):
            return _ScaledPotentialOperator(self, obj)
        else:
            return NotImplemented

    def __neg__(self):
        """Negate."""
        return self.__mul__(-1.0)

    def __sub__(self, other):
        """Subtract."""
        return self.__add__(-other)

    @property
    def space(self):
        """Return the underlying function space."""
        return self._evaluator.space

    @property
    def component_count(self):
        """Return number of components of the potential (1 for scalar potentials)."""
        return self._evaluator.kernel_dimension

    @property
    def evaluation_points(self):
        """Return the evaluation points."""
        return self._evaluator.points


class _ScaledPotentialOperator(PotentialOperator):
    """Scaled potential operator."""

    def __init__(self, op, alpha):
        self._op = op
        self._alpha = alpha

    def evaluate(self, grid_fun):
        """
        Apply the potential operator to a grid function.

        Parameters
        ----------
        grid_fun : bempp_cl.api.GridFunction
            A GridFunction object that represents the boundary density to
            which the potential is applied to.

        """
        return self._alpha * self._op.evaluate(grid_fun)

    @property
    def space(self):
        """Return the underlying function space."""
        return self._op.space

    @property
    def component_count(self):
        """Return number of components of the potential (1 for scalar potentials)."""
        return self._op.component_count

    @property
    def evaluation_points(self):
        """Return the evaluation points."""
        return self._op.points


class _SumPotentialOperator(PotentialOperator):
    """Sum of two potential operators."""

    def __init__(self, op1, op2):
        """Create sum of two potential operators."""
        if not op1._is__compatible(op2):
            raise ValueError("Potential operators are not compatible.")

        self._op1 = op1
        self._op2 = op2

    def evaluate(self, grid_fun):
        """
        Apply the potential operator to a grid function.

        Parameters
        ----------
        grid_fun : bempp_cl.api.GridFunction
            A GridFunction object that represents the boundary density to
            which the potential is applied to.

        """
        return self._op1.evaluate(grid_fun) + self._op2.evaluate(grid_fun)

    @property
    def space(self):
        """Return the underlying function space."""
        return self._op1.space

    @property
    def component_count(self):
        """Return number of components of the potential (1 for scalar potentials)."""
        return self._op1.component_count

    @property
    def evaluation_points(self):
        """Return the evaluation points."""
        return self._op1.points

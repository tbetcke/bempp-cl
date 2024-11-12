"""Definition of shapesets on the reference element."""

import numba as _numba
import numpy as _np


class Shapeset(object):
    """Definintion of a shapeset on a reference element."""

    def __init__(self, identifier):
        """Define a shapeset from its identifier."""
        data = _SHAPESETS[identifier]

        self._evaluate = data["evaluate"]
        self._gradient = data["gradient"]
        self._number_of_shape_functions = data["number_of_shape_functions"]
        self._identifier = data["identifier"]
        self._dimension = data["dimension"]

    @property
    def evaluate(self):
        """Evaluate the shapeset."""
        return self._evaluate

    @property
    def gradient(self):
        """Evaluate the gradient of the shapeset."""
        return self._gradient

    @property
    def number_of_shape_functions(self):
        """Return the number of shape functions."""
        return self._number_of_shape_functions

    @property
    def identifier(self):
        """Return identifier of this shapeset."""
        return self._identifier

    @property
    def dimension(self):
        """Return the dimension of the shapeset."""
        return self._dimension


@_numba.njit
def _p0_shapeset_evaluate(local_coordinates):
    """Evaluate P0 shapeset."""
    return _np.ones((1, 1, local_coordinates.shape[1]), dtype=local_coordinates.dtype)


@_numba.njit
def _p0_shapeset_gradient(local_coordinates):
    """Evaluate P0 gradient."""
    return _np.zeros((1, 2, 1, local_coordinates.shape[1]), dtype=local_coordinates.dtype)


@_numba.njit
def _p1_disc_shapeset_evaluate(local_coordinates):
    """Evaluate P1 discontinuous shapeset."""
    return _np.expand_dims(
        _np.vstack(
            (
                1 - local_coordinates[0, :] - local_coordinates[1, :],
                local_coordinates[0, :],
                local_coordinates[1, :],
            )
        ),
        0,
    )


@_numba.njit
def _p1_disc_shapeset_gradient(local_coordinates):
    """Evaluate P1 discontinuous shapeset gradient."""
    dtype = local_coordinates.dtype
    grad = _np.zeros((1, 2, 3, local_coordinates.shape[1]), dtype=dtype)
    grad[0, 0, :, :] = _np.vstack(
        (
            -_np.ones(local_coordinates.shape[1], dtype=dtype),
            _np.ones(local_coordinates.shape[1], dtype=dtype),
            _np.zeros(local_coordinates.shape[1], dtype=dtype),
        )
    )
    grad[0, 1, :, :] = _np.vstack(
        (
            -_np.ones(local_coordinates.shape[1], dtype=dtype),
            _np.zeros(local_coordinates.shape[1], dtype=dtype),
            _np.ones(local_coordinates.shape[1], dtype=dtype),
        )
    )
    return grad


@_numba.njit
def _rwg0_shapeset_evaluate(local_coordinates):
    """Evaluate RWG 0 shapeset."""
    dtype = local_coordinates.dtype
    vals = _np.zeros((2, 3, local_coordinates.shape[1]), dtype=dtype)
    vals[0, :, :] = _np.vstack((local_coordinates[0], local_coordinates[0] - 1, local_coordinates[0]))
    vals[1, :, :] = _np.vstack((local_coordinates[1] - 1, local_coordinates[1], local_coordinates[1]))
    return vals


@_numba.njit
def _rwg0_shapeset_gradient(local_coordinates):
    """Evaluate RWG 0 shapeset gradient."""
    dtype = local_coordinates.dtype
    npoints = local_coordinates.shape[1]
    grad = _np.zeros((2, 2, 3, npoints), dtype=_np.dtype)
    # Mixed derivates (derivative in direction j of
    # component i with i neq j) are zero.
    grad[0, 0, :, :] = _np.ones((3, npoints), dtype=dtype)
    grad[1, 1, :, :] = _np.ones((3, npoints), dtype=dtype)
    return grad


@_numba.njit
def _snc0_shapeset_gradient(local_coordinates):
    """Evaluate SNC 0 shapeset gradient."""
    dtype = local_coordinates.dtype
    temp = _np.eye(2, dtype=dtype)
    grad = _np.zeros((2, 2), dtype=dtype)
    grad[0] = -temp[1]
    grad[1] = temp[0]
    return grad


_SHAPESETS = {
    "p0_discontinuous": {
        "evaluate": _p0_shapeset_evaluate,
        "gradient": _p0_shapeset_gradient,
        "number_of_shape_functions": 1,
        "identifier": "p0_discontinuous",
        "dimension": 1,
    },
    "p1_discontinuous": {
        "evaluate": _p1_disc_shapeset_evaluate,
        "gradient": _p1_disc_shapeset_gradient,
        "number_of_shape_functions": 3,
        "identifier": "p1_discontinuous",
        "dimension": 1,
    },
    "rwg0": {
        "evaluate": _rwg0_shapeset_evaluate,
        "gradient": _rwg0_shapeset_gradient,
        "number_of_shape_functions": 3,
        "identifier": "rwg0",
        "dimension": 2,
    },
    "snc0": {
        "evaluate": _rwg0_shapeset_evaluate,
        "gradient": _snc0_shapeset_gradient,
        "number_of_shape_functions": 3,
        "identifier": "snc0",
        "dimension": 2,
    },
}

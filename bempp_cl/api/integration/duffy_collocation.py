"""Duffy type rules for collocation integrals."""

import numpy as _np


def duffy_rule_on_reference_triangle(order):
    """Return points and weights for Duffy rule on unit triangle.

    The triangle has coordinates (0, 0), (1, 0), and (1, 1). The
    singularity is assumed to be at (0, 0).
    """
    from bempp_cl.api.integration.gauss import rule

    gauss_points, gauss_weights = rule(order)
    npoints = len(gauss_weights)

    points = _np.empty((2, npoints * npoints), dtype=_np.float64)
    weights = _np.empty(npoints * npoints, dtype=_np.float64)

    count = 0
    for index1 in range(npoints):
        for index2 in range(npoints):
            points[:, count] = [
                gauss_points[index1],
                gauss_points[index1] * gauss_points[index2],
            ]
            weights[count] = gauss_weights[index1] * gauss_weights[index2] * gauss_points[index1]
            count += 1
    return points, weights


def singular_collocation_rule_piecewise_const(order):
    """Singular collocation integral for one singularity on unit triangle barycenter."""
    duffy_points, duffy_weights = duffy_rule_on_reference_triangle(order)
    npoints = len(duffy_weights)

    points = _np.empty((2, 3 * npoints), dtype=_np.float64)
    weights = _np.empty(3 * npoints, dtype=_np.float64)

    triangle_points = [
        _np.array([[1.0 / 3, 1.0 / 3], [0.0, 0], [1.0, 0]]).T,
        _np.array([[1.0 / 3, 1.0 / 3], [1, 0], [0, 1]]).T,
        _np.array([[1.0 / 3, 1.0 / 3], [0, 1], [0, 0]]).T,
    ]

    for index in range(3):
        v0 = triangle_points[index][:, 0].reshape(2, 1)
        A = _np.hstack(
            [
                (triangle_points[index][:, 1] - triangle_points[index][:, 0]).reshape(2, 1),
                (triangle_points[index][:, 2] - triangle_points[index][:, 1]).reshape(2, 1),
            ]
        )
        points[:, index * npoints : (1 + index) * npoints] = v0 + A @ duffy_points
        weights[index * npoints : (1 + index) * npoints] = _np.abs(_np.linalg.det(A)) * duffy_weights

    return points, weights

"""Unit tests for the interpolation module."""

import numpy as np

from bempp_cl.api.utils import interpolation


def test_evaluation_of_laplace_kernel_on_interp_points():
    """Test the evaluation of a Laplace kernel on interp. points."""

    order = 5

    nodes, weights = interpolation.chebychev_nodes_and_weights_second_kind(order)

    nnodes = len(nodes)

    lboundx = np.array([-2.0, -3.0, -1.0])
    uboundx = np.array([-1.0, -2.0, -0.5])

    lboundy = np.array([0.0, 0.5, 1.0])
    uboundy = np.array([1.0, 1.5, 2.0])

    pointsx = interpolation.chebychev_tensor_points_3d(lboundx, uboundx, nodes)

    pointsy = interpolation.chebychev_tensor_points_3d(lboundy, uboundy, nodes)

    expected = np.empty((nnodes**3, nnodes**3), dtype="float64")
    for i in range(nnodes**3):
        for j in range(nnodes**3):
            expected[i, j] = 1.0 / (4 * np.pi * np.linalg.norm(pointsx[i] - pointsy[j]))

    actual = interpolation.evaluate_kernel_on_interpolation_points("laplace", lboundx, uboundx, lboundy, uboundy, nodes)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_evaluation_tensor_interp_polynomial():
    """Test the evaluation of a tensor interpolation polynomial."""

    order = 10

    nodes, weights = interpolation.chebychev_nodes_and_weights_second_kind(order)

    x, y, z = np.meshgrid(nodes, nodes, nodes)

    values = np.cos(x * y * z)

    eval_grid = np.linspace(-1, 1, 8)
    eval_x, eval_y, eval_z = np.meshgrid(eval_grid, eval_grid, eval_grid)

    evaluation_points = np.vstack([eval_x.flatten(), eval_y.flatten(), eval_z.flatten()]).T

    expected = np.cos(eval_x.flatten() * eval_y.flatten() * eval_z.flatten())

    actual = interpolation.evaluate_tensor_interp_polynomial(nodes, weights, values, evaluation_points)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)

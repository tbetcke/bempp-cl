"""Helper routines for Chebychev interpolation."""

import numba as _numba
import numpy as _np

# pylint: disable = C0103


class ChebychevInterpolation(object):
    """Class providing methods for Chebychev interpolation."""

    def __init__(self, order):
        self._order = order
        self._nterms = 1 + order

        self._nodes = None
        self._weights = None

        self._nodes, self._weights = chebychev_nodes_and_weights_second_kind(order)

        self._cheb_diff_mat = chebychev_differentiation_matrix(self._nodes, self._weights)

    @property
    def nodes(self):
        """Return nodes."""
        return self._nodes

    @property
    def weights(self):
        """Return weights."""
        return self._weights

    @property
    def differentiation_matrix(self):
        """Return Chebychev differentiation matrix."""
        return self._cheb_diff_mat

    def evaluate(self, values, evaluation_points):
        """
        Evaluate interpolation polynomial.

        Evaluates the interpolation polynomial with data
        given in the array values at the points in
        the array evaluation_points.
        """
        return evaluate_interp_polynomial(self.nodes, self.weights, values, evaluation_points)

    def differentiate(self, values):
        """Differentiate the polynomial defined by values."""
        return self.differentiation_matrix.dot(values)


def chebychev_nodes_and_weights_second_kind(order):
    """
    Return the Chebychev nodes and weights for a given order.

    This function computes the Chebychev nodes of the second
    kind defined as x_j = cos (pi * j / order) for j = 0 .. order.

    The corresponding weights w_j are required for the evaluation
    of the associated interpolation polynomial via barycentric
    interpolation. For the Chebychev nodes of the second kind
    it holds that w_j = (-1)^j * d_j with d_j = 1/2 for j = 0
    or j = order and d_j = 1 otherwise.
    """
    if order == 0:
        return _np.array([0.0]), _np.array([1.0])

    nodes = _np.cos(_np.pi * _np.arange(1 + order) / order)
    weights = _np.ones(1 + order, dtype="float64")
    weights[0] = weights[order] = 0.5
    weights *= _np.array([-1], dtype="float64") ** _np.arange(1 + order)

    return nodes, weights


# def evaluate_kernel_on_interpolation_points(
#    kernel_type,
#    lboundx,
#    uboundx,
#    lboundy,
#    uboundy,
#    nodes,
#    device_interface=None,
#    precision=None,
#    **kwargs
# ):
#    """
#    Return a kernel evaluation on Chebychev points.

#    This routine supports the evaluation of Laplace and
#    Helmholtz kernels of the form k(x, y) on 3d Tensor grids.
#    It returns a matrix (k(x,_i, y_j)), where the x_i
#    are points in the box defined by lboundx and uboundx and
#    y_i are points in the box defined by lboundy and uboundy.

#    The x_i and y_i are obtained from the 1d array of nodes by
#    iterating the nodes array over the three space dimensions
#    and scaling them appropriately to fit in the boxes.
#    The points are generated in the same order as the following
#    code snippet (scaling excluded for simplicity):
#
#    npoints = len(nodes)
#    tensor_points = _np.empty((npoints**3, 3), _np.float64)
#    for i in range(npoints):
#        for j in range(npoints):
#            for k in range(npoints):
#                tensor_points[i * npoints**2 + j * npoints + k, :] = \
#                    nodes[i], nodes[j], nodes[k]
#
#    Hence, the inner dimension is the fastest changing one.
#
#    Attributes
#    ----------
#    kernel_type : string
#        Either 'laplace' or 'helmholtz' to specify whether to
#        evaluate a Laplace or a Helmholtz kernel.
#    lboundx : Numpy array
#        An array specifying the lower bound for the x-component
#        box.
#    uboundx : Numpy array
#        An array specifying the upper bound for the x-component box.
#    lboundy : Numpy array
#        An array specifying the lower bound for the y-component
#        box.
#    uboundy : Numpy array
#        An array specifying the upper bound for the y-component box.
#    nodes : array
#        An array of 1d nodes defined in the interval [-1, 1] which are
#        used as basis for the tensor points.
#    kwargs : keyword arguments
#        The currently supported keyword argument is wavenumber
#        to specify the wavenumber for Helmholtz kernels.
#    """

# from bempp_cl.core import cl_helpers
# import bempp_cl.api

# nnodes = len(nodes)

# if kernel_type not in {"laplace", "helmholtz"}:
#     raise ValueError("'kernel_type' must be one of 'laplace' or 'helmholtz'.")

# if device_interface is None:
#     device_interface = bempp_cl.api.default_device()

# if precision is None:
#     precision = bempp_cl.api.get_precision(device_interface)

# real_type = cl_helpers.get_type(precision).real

# if kernel_type == "laplace":
#     dtype = cl_helpers.get_type(precision).real

# if kernel_type == "helmholtz":
#     dtype = cl_helpers.get_type(precision).complex

# result = cl_helpers.DeviceBuffer(
#     (nnodes ** 3, nnodes ** 3),
#     dtype,
#     device_interface.context,
#     access_mode="write_only",
# )

# nodes = cl_helpers.DeviceBuffer.from_array(
#     nodes, device_interface, dtype=real_type, access_mode="read_only"
# )

# options = dict()
# options["X_XMIN"] = lboundx[0]
# options["X_YMIN"] = lboundx[1]
# options["X_ZMIN"] = lboundx[2]

# options["Y_XMIN"] = lboundy[0]
# options["Y_YMIN"] = lboundy[1]
# options["Y_ZMIN"] = lboundy[2]

# options["X_XMAX"] = uboundx[0]
# options["X_YMAX"] = uboundx[1]
# options["X_ZMAX"] = uboundx[2]

# options["Y_XMAX"] = uboundy[0]
# options["Y_YMAX"] = uboundy[1]
# options["Y_ZMAX"] = uboundy[2]
# options["NNODES"] = nnodes

# if kernel_type == "helmholtz":
#     options["WAVENUMBER"] = kwargs["wavenumber"]

# cl_kernel_source = cl_helpers.kernel_source_from_identifier(
#     kernel_type + "_kernel_evaluator" + "_novec", options
# )

# cl_kernel = cl_helpers.Kernel(cl_kernel_source, device_interface.context, precision)

# event = cl_kernel.run(
#     device_interface, (nnodes ** 3, nnodes ** 3), (1, 1), nodes, result
# )
# event.wait()

# return result.get_host_copy(device_interface)


def evaluate_kernel_on_interpolation_points(
    kernel_type,
    lboundx,
    uboundx,
    lboundy,
    uboundy,
    nodes,
    device_interface=None,
    precision="double",
    **kwargs,
):
    """
    Return a kernel evaluation on Chebychev points.

    This routine supports the evaluation of Laplace and
    Helmholtz kernels of the form k(x, y) on 3d Tensor grids.
    It returns a matrix (k(x,_i, y_j)), where the x_i
    are points in the box defined by lboundx and uboundx and
    y_i are points in the box defined by lboundy and uboundy.

    The x_i and y_i are obtained from the 1d array of nodes by
    iterating the nodes array over the three space dimensions
    and scaling them appropriately to fit in the boxes.
    The points are generated in the same order as the following
    code snippet (scaling excluded for simplicity):

    npoints = len(nodes)
    tensor_points = _np.empty((npoints**3, 3), _np.float64)
    for i in range(npoints):
        for j in range(npoints):
            for k in range(npoints):
                tensor_points[i * npoints**2 + j * npoints + k, :] = \
                    nodes[i], nodes[j], nodes[k]

    Hence, the inner dimension is the fastest changing one.

    Attributes
    ----------
    kernel_type : string
        Either 'laplace' or 'helmholtz' to specify whether to
        evaluate a Laplace or a Helmholtz kernel.
    lboundx : Numpy array
        An array specifying the lower bound for the x-component
        box.
    uboundx : Numpy array
        An array specifying the upper bound for the x-component box.
    lboundy : Numpy array
        An array specifying the lower bound for the y-component
        box.
    uboundy : Numpy array
        An array specifying the upper bound for the y-component box.
    nodes : array
        An array of 1d nodes defined in the interval [-1, 1] which are
        used as basis for the tensor points.
    kwargs : keyword arguments
        The currently supported keyword argument is wavenumber
        to specify the wavenumber for Helmholtz kernels.
    """
    pointsx = chebychev_tensor_points_3d(lboundx, uboundx, nodes)
    pointsy = chebychev_tensor_points_3d(lboundy, uboundy, nodes)

    if kernel_type == "laplace":
        return evaluate_laplace_kernel_on_interpolation_points(pointsx, pointsy)

    if kernel_type == "helmholtz":
        return evaluate_helmholtz_kernel_on_interpolation_points(pointsx, pointsy, kwargs["wavenumber"])

    raise ValueError(f"Unknown kernel: {kernel_type}")


@_numba.njit(cache=True)
def evaluate_laplace_kernel_on_interpolation_points(pointsx, pointsy):
    """Evaluate the Laplace kernel at the given points."""
    values = _np.empty((pointsx.shape[0], pointsy.shape[0]), _np.float64)
    for i, x in enumerate(pointsx):
        for j, y in enumerate(pointsy):
            values[i, j] = 1 / (4 * _np.pi * _np.linalg.norm(x - y))
    return values


@_numba.njit(cache=True)
def evaluate_helmholtz_kernel_on_interpolation_points(pointsx, pointsy, wavenumber):
    """Evaluate the Laplace kernel at the given points."""
    values = _np.empty((pointsx.shape[0], pointsy.shape[0]), _np.complex128)
    for i, x in enumerate(pointsx):
        for j, y in enumerate(pointsy):
            values[i, :] = _np.exp(1j * wavenumber * _np.linalg.norm(x - y)) / (4 * _np.pi * _np.linalg.norm(x - y))
    return values


@_numba.njit(cache=True)
def chebychev_tensor_points_3d(lbound, ubound, nodes):
    """Create Chebychev points associated with a 3d tensor grid."""
    points_i = lbound[0] + 0.5 * (ubound[0] - lbound[0]) * (1 + nodes)
    points_j = lbound[1] + 0.5 * (ubound[1] - lbound[1]) * (1 + nodes)
    points_k = lbound[2] + 0.5 * (ubound[2] - lbound[2]) * (1 + nodes)

    npoints = len(nodes)
    tensor_points = _np.empty((npoints**3, 3), _np.float64)
    for i in range(npoints):
        for j in range(npoints):
            for k in range(npoints):
                tensor_points[i * npoints**2 + j * npoints + k, :] = (
                    points_i[i],
                    points_j[j],
                    points_k[k],
                )
    return tensor_points


@_numba.njit(cache=True)
def evaluate_interp_polynomial(nodes, weights, values, evaluation_points):
    """
    Evaluate an interpolation polynomial.

    This function uses barycentric evaluation for
    stability. The data values are assumed to be real.
    """
    npoints = len(evaluation_points)
    nterms = len(nodes)

    numerator = _np.zeros(npoints, _np.float64)
    denominator = _np.zeros(npoints, _np.float64)
    exact = -_np.ones(npoints, _np.int32)

    for index in range(nterms):
        xdiff = evaluation_points - nodes[index]
        temp = weights[index] / xdiff
        numerator += temp * values[index]
        denominator += temp
        exact[xdiff == 0] = index
    result = numerator / denominator
    indices = _np.where(exact > -1)[0]
    result[indices] = values[exact[indices]]
    return result


def evaluate_tensor_interp_polynomial(nodes, weights, values, evaluation_points):
    """
    Evaluate a tensor interpolation polynomial.

    This function evaluates a tensor chebychev basis with
    specified weights at a given set of evaluation points.

    Attributes
    ----------
    nodes : Numpy array
        1d array of interpolation points
    weights : Numpy array
        Associated array of barycentric weights
    values : Numpy array
        3d array of values at interpolation points.
        The dimensions specify the values along the
        (x, y, z) axis with z being the inner most axis.
        For k interpolation nodes there are k values along
        each dimension.
    evaluation_points : Numpy array
        (N x 3) array of N evaluation points in 3 dimension.
    """
    npoints = len(evaluation_points)
    nterms = len(nodes)

    output = _np.empty(npoints, _np.float64)

    for point_index, point in enumerate(evaluation_points):
        local_values = values.copy()
        for dim in range(2, -1, -1):
            numerator = _np.zeros(dim * [nterms], _np.float64)
            exact_index = -1
            denominator = 0.0
            for index in range(nterms):
                xdiff = point[dim] - nodes[index]
                if xdiff == 0.0:
                    exact_index = index
                    break
                temp = weights[index] / xdiff
                numerator += temp * local_values[..., index]
                denominator += temp
            if exact_index > -1:
                local_values = local_values[..., exact_index]
            else:
                local_values = numerator / denominator
        output[point_index] = local_values

    return output


def chebychev_differentiation_matrix(nodes, weights):
    """Return a Chebychev differentiation matrix."""
    nterms = len(nodes)
    if nterms == 1:
        return _np.array([0.0])

    node_vec = nodes.reshape(nterms, 1).copy()
    weights_vec = weights.reshape(nterms, 1).copy()
    weights_vec[0] *= 4
    weights_vec[nterms - 1] *= 4
    X = _np.tile(node_vec, (1, nterms))
    delta = X - X.T
    cheb_mat = _np.dot(weights_vec, 1.0 / weights_vec.T) / (delta + _np.eye(nterms))
    cheb_mat -= _np.diag(_np.sum(cheb_mat.T, axis=0))
    return cheb_mat

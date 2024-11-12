"""Duffy transformation rules for singular integration for Galerkin integrals."""

import numpy as _np


def number_of_quadrature_points(order, adjacency):
    """Return the number of quadrature points for given adjacency.

    Possible cases for adjacency are "coincident", "edge_adjacent"
    or "vertex_adjacent".

    """
    if adjacency == "coincident":
        npoints = 6 * order**4
    elif adjacency == "edge_adjacent":
        npoints = 5 * order**4
    elif adjacency == "vertex_adjacent":
        npoints = 2 * order**4
    else:
        raise ValueError("adjacency must be one of 'coincident', " + "'edge_adjacent', 'vertex_adjacent'")

    return npoints


def rule(order, adjacency):
    """
    Create a singular quadrature rule.

    Possible cases for adjacency are "coincident",
    "edge_adjacent" or "vertex_adjacent".

    """
    from .gauss import rule as gauss_rule

    if adjacency not in ["coincident", "edge_adjacent", "vertex_adjacent"]:
        raise ValueError("Unknown adjacency.")

    xreg, wreg = gauss_rule(order)
    number_of_1d_points = len(wreg)
    # Create tensor Gauss points

    tensor_points = _np.empty((2, number_of_1d_points**2), dtype="float64")
    tensor_weights = _np.empty(number_of_1d_points**2, dtype="float64")

    for i in range(number_of_1d_points):
        for j in range(number_of_1d_points):
            tensor_points[0, i * number_of_1d_points + j] = xreg[j]
            tensor_points[1, i * number_of_1d_points + j] = xreg[i]
            tensor_weights[i * number_of_1d_points + j] = wreg[i] * wreg[j]

    number_of_reg_points = number_of_1d_points**2

    if adjacency == "coincident":
        number_of_points = 6 * number_of_reg_points**2
        points_test = _np.empty((2, number_of_points), dtype="float64", order="F")
        points_trial = _np.empty((2, number_of_points), dtype="float64", order="F")
        weights = _np.empty(number_of_points, dtype="float64")

        index = 0
        for test_ind in range(number_of_reg_points):
            for trial_ind in range(number_of_reg_points):
                ptest = tensor_points[:, test_ind]
                ptrial = tensor_points[:, trial_ind]

                xsi = ptest[0]
                eta1 = ptest[1]
                eta2 = ptrial[0]
                eta3 = ptrial[1]

                eta123 = eta1 * eta2 * eta3
                eta12 = eta1 * eta2

                weight = tensor_weights[test_ind] * tensor_weights[trial_ind] * xsi * xsi * xsi * eta1 * eta1 * eta2

                # Region 1
                points_test[0, index] = xsi
                points_test[1, index] = xsi * (1.0 - eta1 + eta12)
                points_trial[0, index] = xsi * (1.0 - eta123)
                points_trial[1, index] = xsi * (1.0 - eta1)
                weights[index] = weight
                index += 1

                # Region 2
                points_test[0, index] = xsi * (1.0 - eta123)
                points_test[1, index] = xsi * (1.0 - eta1)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * (1.0 - eta1 + eta12)
                weights[index] = weight
                index += 1

                # Region 3
                points_test[0, index] = xsi
                points_test[1, index] = xsi * (eta1 - eta12 + eta123)
                points_trial[0, index] = xsi * (1.0 - eta12)
                points_trial[1, index] = xsi * (eta1 - eta12)
                weights[index] = weight
                index += 1

                # Region 4
                points_test[0, index] = xsi * (1.0 - eta12)
                points_test[1, index] = xsi * (eta1 - eta12)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * (eta1 - eta12 + eta123)
                weights[index] = weight
                index += 1

                # Region 5
                points_test[0, index] = xsi * (1.0 - eta123)
                points_test[1, index] = xsi * (eta1 - eta123)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * (eta1 - eta12)
                weights[index] = weight
                index += 1

                # Region 6
                points_test[0, index] = xsi
                points_test[1, index] = xsi * (eta1 - eta12)
                points_trial[0, index] = xsi * (1.0 - eta123)
                points_trial[1, index] = xsi * (eta1 - eta123)
                weights[index] = weight
                index += 1

    if adjacency == "edge_adjacent":
        number_of_points = 5 * number_of_reg_points**2
        points_test = _np.empty((2, number_of_points), dtype="float64", order="F")
        points_trial = _np.empty((2, number_of_points), dtype="float64", order="F")
        weights = _np.empty(number_of_points, dtype="float64")

        index = 0
        for test_ind in range(number_of_reg_points):
            for trial_ind in range(number_of_reg_points):
                ptest = tensor_points[:, test_ind]
                ptrial = tensor_points[:, trial_ind]

                xsi = ptest[0]
                eta1 = ptest[1]
                eta2 = ptrial[0]
                eta3 = ptrial[1]

                eta123 = eta1 * eta2 * eta3
                eta12 = eta1 * eta2

                weight = tensor_weights[test_ind] * tensor_weights[trial_ind] * xsi * xsi * xsi * eta1 * eta1

                # Region 1
                points_test[0, index] = xsi
                points_test[1, index] = xsi * eta1 * eta3
                points_trial[0, index] = xsi * (1.0 - eta12)
                points_trial[1, index] = xsi * eta1 * (1.0 - eta2)
                weights[index] = weight
                index += 1

                # Region 2
                points_test[0, index] = xsi
                points_test[1, index] = xsi * eta1
                points_trial[0, index] = xsi * (1.0 - eta123)
                points_trial[1, index] = xsi * eta1 * eta2 * (1 - eta3)
                weights[index] = weight * eta2
                index += 1

                # Region 3
                points_test[0, index] = xsi * (1.0 - eta12)
                points_test[1, index] = xsi * eta1 * (1.0 - eta2)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * eta123
                weights[index] = weight * eta2
                index += 1

                # Region 4
                points_test[0, index] = xsi * (1.0 - eta123)
                points_test[1, index] = xsi * eta12 * (1.0 - eta3)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * eta1
                weights[index] = weight * eta2
                index += 1

                # Region 5
                points_test[0, index] = xsi * (1.0 - eta123)
                points_test[1, index] = xsi * eta1 * (1.0 - eta2 * eta3)
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * eta12
                weights[index] = weight * eta2
                index += 1

    if adjacency == "vertex_adjacent":
        number_of_points = 2 * number_of_reg_points**2
        points_test = _np.empty((2, number_of_points), dtype="float64", order="F")
        points_trial = _np.empty((2, number_of_points), dtype="float64", order="F")
        weights = _np.empty(number_of_points, dtype="float64")

        index = 0
        for test_ind in range(number_of_reg_points):
            for trial_ind in range(number_of_reg_points):
                ptest = tensor_points[:, test_ind]
                ptrial = tensor_points[:, trial_ind]

                xsi = ptest[0]
                eta1 = ptest[1]
                eta2 = ptrial[0]
                eta3 = ptrial[1]

                weight = tensor_weights[test_ind] * tensor_weights[trial_ind] * xsi * xsi * xsi * eta2

                # Region 1
                points_test[0, index] = xsi
                points_test[1, index] = xsi * eta1
                points_trial[0, index] = xsi * eta2
                points_trial[1, index] = xsi * eta2 * eta3
                weights[index] = weight
                index += 1

                # Region 2
                points_test[0, index] = xsi * eta2
                points_test[1, index] = xsi * eta2 * eta3
                points_trial[0, index] = xsi
                points_trial[1, index] = xsi * eta1
                weights[index] = weight
                index += 1

    # Points above are for a different unit triangle than Bempp uses.
    # Fix this here.

    points_test[0, :] -= points_test[1, :]
    points_trial[0, :] -= points_trial[1, :]

    return points_test, points_trial, weights


def remap_points_shared_vertex(points, vertex_id):
    """
    Remap triangle points for vertex adjacency.

    By default the Duffy rules assume that the two triangles meet
    at vertex 0. This method transforms the Duffy integration
    points depending on which vertex they meet.
    """
    if vertex_id == 0:
        return points

    if vertex_id == 1:
        new_points = _np.zeros_like(points)
        new_points[0, :] = 1.0 - points[0, :] - points[1, :]
        new_points[1, :] = points[1, :]
        return new_points

    if vertex_id == 2:
        new_points = _np.zeros_like(points)
        new_points[0, :] = points[0, :]
        new_points[1, :] = 1.0 - points[0, :] - points[1, :]
        return new_points


def remap_points_shared_edge(points, shared_vertex1, shared_vertex2):
    """
    Remap triangle points for edge adjacency.

    By default the Duffy rules assume that the two triangles meet
    at edge 0. This method transforms the Duffy integration
    points depending on which edge they meet.
    """
    # Get the shared vertices

    v0 = shared_vertex1
    v1 = shared_vertex2

    # Create reference triangle
    ref_vertices = _np.zeros((2, 3), dtype="float64")
    ref_vertices[0, 1] = 1
    ref_vertices[1, 2] = 1

    new_vertices = _np.zeros((2, 3), dtype="float64")
    new_vertices[:, 0] = ref_vertices[:, v0]
    new_vertices[:, 1] = ref_vertices[:, v1]
    new_vertices[:, 2] = ref_vertices[:, 3 - v0 - v1]

    A = _np.zeros((2, 2), dtype="float64")
    A[:, 0] = new_vertices[:, 1] - new_vertices[:, 0]
    A[:, 1] = new_vertices[:, 2] - new_vertices[:, 0]

    # Linear transformation from reference points to
    # new points

    new_points = A.dot(points) + new_vertices[:, 0].reshape(2, 1)

    return _np.require(new_points, requirements=["F"])

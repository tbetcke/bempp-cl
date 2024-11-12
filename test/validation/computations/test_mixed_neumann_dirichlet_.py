import bempp_cl.api
import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_laplace_mixed_neumann_dirichlet(helpers, precision):
    """Test solution for mixed Laplace/Dirichlet problem."""

    grid = helpers.load_grid("cube_grid")
    dirichlet_segments = [1, 3]
    neumann_segments = [2, 4, 5, 6]

    global_neumann_space = bempp_cl.api.function_space(grid, "DP", 0)
    global_dirichlet_space = bempp_cl.api.function_space(grid, "P", 1)

    neumann_space_dirichlet_segment = bempp_cl.api.function_space(grid, "DP", 0, segments=dirichlet_segments)

    neumann_space_neumann_segment = bempp_cl.api.function_space(grid, "DP", 0, segments=neumann_segments)

    dirichlet_space_dirichlet_segment = bempp_cl.api.function_space(
        grid,
        "P",
        1,
        segments=dirichlet_segments,
        include_boundary_dofs=True,
        truncate_at_segment_edge=False,
    )

    dirichlet_space_neumann_segment = bempp_cl.api.function_space(grid, "P", 1, segments=neumann_segments)

    dual_dirichlet_space = bempp_cl.api.function_space(
        grid, "P", 1, segments=dirichlet_segments, include_boundary_dofs=True
    )

    slp_DD = bempp_cl.api.operators.boundary.laplace.single_layer(
        neumann_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    dlp_DN = bempp_cl.api.operators.boundary.laplace.double_layer(
        dirichlet_space_neumann_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    adlp_ND = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_dirichlet_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    hyp_NN = bempp_cl.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    slp_DN = bempp_cl.api.operators.boundary.laplace.single_layer(
        neumann_space_neumann_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    dlp_DD = bempp_cl.api.operators.boundary.laplace.double_layer(
        dirichlet_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    id_DD = bempp_cl.api.operators.boundary.sparse.identity(
        dirichlet_space_dirichlet_segment,
        dirichlet_space_dirichlet_segment,
        neumann_space_dirichlet_segment,
    )

    adlp_NN = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    id_NN = bempp_cl.api.operators.boundary.sparse.identity(
        neumann_space_neumann_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    hyp_ND = bempp_cl.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_dirichlet_segment,
        neumann_space_neumann_segment,
        dirichlet_space_neumann_segment,
    )

    blocked = bempp_cl.api.BlockedOperator(2, 2)

    blocked[0, 0] = slp_DD
    blocked[0, 1] = -dlp_DN
    blocked[1, 0] = adlp_ND
    blocked[1, 1] = hyp_NN

    @bempp_cl.api.real_callable
    def dirichlet_data(x, n, domain_index, res):
        res[0] = 1

    @bempp_cl.api.real_callable
    def neumann_data(x, n, domain_index, res):
        res[0] = 1

    dirichlet_grid_fun = bempp_cl.api.GridFunction(
        dirichlet_space_dirichlet_segment,
        fun=dirichlet_data,
        dual_space=dual_dirichlet_space,
    )

    neumann_grid_fun = bempp_cl.api.GridFunction(
        neumann_space_neumann_segment,
        fun=neumann_data,
        dual_space=dirichlet_space_neumann_segment,
    )

    rhs_fun1 = (0.5 * id_DD + dlp_DD) * dirichlet_grid_fun - slp_DN * neumann_grid_fun
    rhs_fun2 = -hyp_ND * dirichlet_grid_fun + (0.5 * id_NN - adlp_NN) * neumann_grid_fun

    (neumann_solution, dirichlet_solution), _ = bempp_cl.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2])

    neumann_imbedding_dirichlet_segment = bempp_cl.api.operators.boundary.sparse.identity(
        neumann_space_dirichlet_segment, global_neumann_space, global_neumann_space
    )

    neumann_imbedding_neumann_segment = bempp_cl.api.operators.boundary.sparse.identity(
        neumann_space_neumann_segment, global_neumann_space, global_neumann_space
    )

    dirichlet_imbedding_dirichlet_segment = bempp_cl.api.operators.boundary.sparse.identity(
        dirichlet_space_dirichlet_segment,
        global_dirichlet_space,
        global_dirichlet_space,
    )

    dirichlet_imbedding_neumann_segment = bempp_cl.api.operators.boundary.sparse.identity(
        dirichlet_space_neumann_segment, global_dirichlet_space, global_dirichlet_space
    )

    dirichlet = (
        dirichlet_imbedding_dirichlet_segment * dirichlet_grid_fun
        + dirichlet_imbedding_neumann_segment * dirichlet_solution
    )

    neumann = (
        neumann_imbedding_neumann_segment * neumann_grid_fun + neumann_imbedding_dirichlet_segment * neumann_solution
    )

    data = helpers.load_npz_data("mixed_dirichlet_neumann_sol")

    expected_dirichlet = data["dirichlet_vals"]
    expected_neumann = data["neumann_vals"]

    actual_dirichlet = dirichlet.evaluate_on_element_centers()
    actual_neumann = neumann.evaluate_on_element_centers()

    np.testing.assert_allclose(actual_dirichlet, expected_dirichlet, rtol=helpers.default_tolerance(precision))

    np.testing.assert_allclose(actual_neumann, expected_neumann, rtol=helpers.default_tolerance(precision))

    # rel_diff_dirichlet = np.abs(actual_dirichlet - expected_dirichlet) / np.abs(
    # expected_dirichlet
    # )

    # rel_diff_neumann = np.abs(actual_neumann - expected_neumann) / np.abs(
    # expected_neumann
    # )
    # print(np.max(rel_diff_dirichlet))

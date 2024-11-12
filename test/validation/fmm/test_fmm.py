"""Unit tests for the FMM assembler."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import pytest
import numpy as np
import bempp_cl.api

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")

TOL = 2e-3
bempp_cl.api.GLOBAL_PARAMETERS.fmm.expansion_order = 10
# bempp_cl.api.GLOBAL_PARAMETERS.fmm.ncrit = 100


@pytest.fixture
def grid(helpers):
    return helpers.load_grid("fmm_grid")


@pytest.fixture
def grid1(helpers):
    return helpers.load_grid("fmm_grid1")


@pytest.fixture
def grid2(helpers):
    return helpers.load_grid("fmm_grid2")


def test_laplace_boundary_fmm(helpers, grid):
    """Test Laplace boundary operators."""
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")

    for filename, operator in [
        ("fmm_laplace_single", bempp_cl.api.operators.boundary.laplace.single_layer),
        ("fmm_laplace_double", bempp_cl.api.operators.boundary.laplace.double_layer),
        (
            "fmm_laplace_adjoint",
            bempp_cl.api.operators.boundary.laplace.adjoint_double_layer,
        ),
        ("fmm_laplace_hyper", bempp_cl.api.operators.boundary.laplace.hypersingular),
    ]:
        fmm = operator(space, space, space, assembler="fmm").weak_form()
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_laplace_potential_fmm(helpers, grid):
    """Test Laplace potential operators."""
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")
    grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
    points = helpers.load_npy_data("fmm_potential_points")

    for filename, operator in [
        (
            "fmm_laplace_potential_single",
            bempp_cl.api.operators.potential.laplace.single_layer,
        ),
        (
            "fmm_laplace_potential_double",
            bempp_cl.api.operators.potential.laplace.double_layer,
        ),
    ]:
        fmm = operator(space, points, assembler="fmm").evaluate(grid_fun)
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_helmholtz_boundary_fmm(helpers, grid, skip):
    """Test Helmholtz boundary operators."""
    if skip == "ci":
        pytest.skip()
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")
    wavenumber = 1.5

    for filename, operator in [
        ("fmm_helmholtz_single", bempp_cl.api.operators.boundary.helmholtz.single_layer),
        ("fmm_helmholtz_double", bempp_cl.api.operators.boundary.helmholtz.double_layer),
        (
            "fmm_helmholtz_adjoint",
            bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer,
        ),
        ("fmm_helmholtz_hyper", bempp_cl.api.operators.boundary.helmholtz.hypersingular),
    ]:
        fmm = operator(space, space, space, wavenumber, assembler="fmm").weak_form()
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_helmholtz_boundary_fmm_complex(helpers, skip):
    """Test Helmholtz boundary operators with complex wavenumber."""
    if skip == "ci":
        pytest.skip()
    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = rand.rand(space.global_dof_count)
    wavenumber = 1.5 + 0.5j

    for operator in [
        bempp_cl.api.operators.boundary.helmholtz.single_layer,
        bempp_cl.api.operators.boundary.helmholtz.double_layer,
        bempp_cl.api.operators.boundary.helmholtz.hypersingular,
    ]:
        fmm = operator(space, space, space, wavenumber, assembler="fmm").weak_form()
        dense = operator(space, space, space, wavenumber, assembler="dense").weak_form()

        np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_helmholtz_potential_fmm(helpers, grid, skip):
    """Test Helmholtz potential operators."""
    if skip == "ci":
        pytest.skip()
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")
    grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
    points = helpers.load_npy_data("fmm_potential_points")
    wavenumber = 1.5

    for filename, operator in [
        (
            "fmm_helmholtz_potential_single",
            bempp_cl.api.operators.potential.helmholtz.single_layer,
        ),
        (
            "fmm_helmholtz_potential_double",
            bempp_cl.api.operators.potential.helmholtz.double_layer,
        ),
    ]:
        fmm = operator(space, points, wavenumber, assembler="fmm").evaluate(grid_fun)
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_helmholtz_potential_fmm_complex(helpers, skip):
    """Test Helmholtz boundary operators with complex wavenumber."""
    if skip == "ci":
        pytest.skip()
    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = rand.rand(space.global_dof_count)
    grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
    wavenumber = 1.5 + 0.5j

    points = np.array([2.0, 1.0, 0.0]).reshape(3, 1)

    for operator in [
        bempp_cl.api.operators.potential.helmholtz.single_layer,
        bempp_cl.api.operators.potential.helmholtz.double_layer,
    ]:
        fmm = operator(space, points, wavenumber, assembler="fmm")
        dense = operator(space, points, wavenumber, assembler="dense")

        np.testing.assert_allclose(dense.evaluate(grid_fun), fmm.evaluate(grid_fun), rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_modified_helmholtz_boundary_fmm(helpers, grid, skip):
    """Test modified Helmholtz boundary operators."""
    if skip == "ci":
        pytest.skip()

    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")
    wavenumber = 1.5

    for filename, operator in [
        (
            "fmm_modified_helmholtz_single",
            bempp_cl.api.operators.boundary.modified_helmholtz.single_layer,
        ),
        (
            "fmm_modified_helmholtz_double",
            bempp_cl.api.operators.boundary.modified_helmholtz.double_layer,
        ),
        (
            "fmm_modified_helmholtz_adjoint",
            bempp_cl.api.operators.boundary.modified_helmholtz.adjoint_double_layer,
        ),
        (
            "fmm_modified_helmholtz_hyper",
            bempp_cl.api.operators.boundary.modified_helmholtz.hypersingular,
        ),
    ]:
        fmm = operator(space, space, space, wavenumber, assembler="fmm").weak_form()
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_modified_helmholtz_potential_fmm(helpers, grid, skip):
    """Test modified Helmholtz potential operators."""
    if skip == "ci":
        pytest.skip()

    space = bempp_cl.api.function_space(grid, "P", 1)
    vec = helpers.load_npy_data("fmm_p1_vec")
    grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
    points = helpers.load_npy_data("fmm_potential_points")
    wavenumber = 1.5

    for filename, operator in [
        (
            "fmm_modified_potential_helmholtz_single",
            bempp_cl.api.operators.potential.modified_helmholtz.single_layer,
        ),
        (
            "fmm_modified_potential_helmholtz_double",
            bempp_cl.api.operators.potential.modified_helmholtz.double_layer,
        ),
    ]:
        fmm = operator(space, points, wavenumber, assembler="fmm").evaluate(grid_fun)
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_boundary_fmm(helpers, grid, skip):
    """Test Maxwell boundary operators."""
    if skip == "ci":
        pytest.skip()
    rwg = bempp_cl.api.function_space(grid, "RWG", 0)
    snc = bempp_cl.api.function_space(grid, "SNC", 0)
    vec = helpers.load_npy_data("fmm_rwg_vec")
    wavenumber = 1.5

    for filename, operator in [
        ("fmm_maxwell_electric", bempp_cl.api.operators.boundary.maxwell.electric_field),
        ("fmm_maxwell_magnetic", bempp_cl.api.operators.boundary.maxwell.magnetic_field),
    ]:
        fmm = operator(rwg, rwg, snc, wavenumber, assembler="fmm").weak_form()
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_potential_fmm(helpers, grid, skip):
    """Test Maxwell potential operators."""
    if skip == "ci":
        pytest.skip()
    rwg = bempp_cl.api.function_space(grid, "RWG", 0)
    vec = helpers.load_npy_data("fmm_rwg_vec")
    grid_fun = bempp_cl.api.GridFunction(rwg, coefficients=vec)
    points = helpers.load_npy_data("fmm_potential_points")
    wavenumber = 1.5

    for filename, operator in [
        (
            "fmm_maxwell_potential_electric",
            bempp_cl.api.operators.potential.maxwell.electric_field,
        ),
        (
            "fmm_maxwell_potential_magnetic",
            bempp_cl.api.operators.potential.maxwell.magnetic_field,
        ),
    ]:
        fmm = operator(rwg, points, wavenumber, assembler="fmm").evaluate(grid_fun)
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_fmm_two_grids_laplace(helpers, grid1, grid2):
    """Test the FMM for Laplace between two different grids."""
    p1_space1 = bempp_cl.api.function_space(grid1, "P", 1)
    p1_space2 = bempp_cl.api.function_space(grid2, "P", 1)
    vec = helpers.load_npy_data("fmm_two_mesh_vec")

    for filename, operator in [
        (
            "fmm_two_mesh_laplace_single",
            bempp_cl.api.operators.boundary.laplace.single_layer,
        ),
        (
            "fmm_two_mesh_laplace_hyper",
            bempp_cl.api.operators.boundary.laplace.hypersingular,
        ),
    ]:
        fmm = operator(p1_space1, p1_space2, p1_space2, assembler="fmm").weak_form()
        dense = helpers.load_npy_data(filename)
        np.testing.assert_allclose(dense, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_boundary_fmm_complex(helpers):
    """Test Maxwell boundary operators with complex wavenumber."""
    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = bempp_cl.api.function_space(grid, "RWG", 0)
    dual_space = bempp_cl.api.function_space(grid, "SNC", 0)
    vec = rand.rand(space.global_dof_count)
    wavenumber = 1.5 + 0.5j

    for operator in [
        bempp_cl.api.operators.boundary.maxwell.electric_field,
        bempp_cl.api.operators.boundary.maxwell.magnetic_field,
    ]:
        fmm = operator(space, space, dual_space, wavenumber, assembler="fmm").weak_form()
        dense = operator(space, space, dual_space, wavenumber, assembler="dense").weak_form()

        np.testing.assert_allclose(dense @ vec, fmm @ vec, rtol=TOL)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_potential_fmm_complex(helpers):
    """Test Maxwell boundary operators with complex wavenumber."""
    rand = np.random.RandomState(0)
    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = bempp_cl.api.function_space(grid, "RWG", 0)
    vec = rand.rand(space.global_dof_count)
    grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
    wavenumber = 1.5 + 0.5j

    points = np.array([2.0, 1.0, 0.0]).reshape(3, 1)

    for operator in [
        bempp_cl.api.operators.potential.maxwell.electric_field,
        bempp_cl.api.operators.potential.maxwell.magnetic_field,
    ]:
        fmm = operator(space, points, wavenumber, assembler="fmm")
        dense = operator(space, points, wavenumber, assembler="dense")

        np.testing.assert_allclose(dense.evaluate(grid_fun), fmm.evaluate(grid_fun), rtol=TOL)

    bempp_cl.api.clear_fmm_cache()

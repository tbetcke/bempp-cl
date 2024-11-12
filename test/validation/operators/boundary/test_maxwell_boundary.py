"""Unit tests for modified Helmholtz operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_maxwell_electric_field_sphere(default_parameters, helpers, device_interface, precision):
    """Test Maxwell electric field on sphere."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    if precision == "single":
        rtol = 1e-5
        atol = 1e-7
    else:
        rtol = 1e-10
        atol = 1e-14

    expected = helpers.load_npy_data("maxwell_electric_field_boundary")
    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=rtol, atol=atol)


def test_maxwell_electric_field_rbc_bc_sphere(default_parameters, helpers, device_interface, precision, skip):
    """Test Maxwell electric field on sphere with RBC/BC basis."""
    if skip == "ci":
        pytest.skip()

    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "BC", 0)
    space2 = function_space(grid, "RBC", 0)

    rand = _np.random.RandomState(0)
    vec = rand.rand(space1.global_dof_count)

    bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = True

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="fmm",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    actual = discrete_op @ vec

    bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = False

    if precision == "single":
        rtol = 5e-5
        atol = 5e-6
    else:
        rtol = 1e-10
        atol = 1e-14

    mat = helpers.load_npy_data("maxwell_electric_field_boundary_rbc_bc")

    expected = mat @ vec

    _np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_electric_field_bc_sphere(default_parameters, helpers, device_interface, precision, skip):
    """Test Maxwell electric field on sphere with BC basis."""
    if skip == "ci":
        pytest.skip()

    import bempp_cl.api
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.maxwell import electric_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "BC", 0)
    space2 = function_space(grid, "SNC", 0)

    rand = _np.random.RandomState(0)
    vec = rand.rand(space1.global_dof_count)

    bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = True

    discrete_op = electric_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="fmm",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    actual = discrete_op @ vec

    bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation = False

    if precision == "single":
        rtol = 1e-4
        atol = 5e-6
    else:
        rtol = 1e-10
        atol = 1e-14

    mat = helpers.load_npy_data("maxwell_electric_field_boundary_bc")
    expected = mat @ vec
    _np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    bempp_cl.api.clear_fmm_cache()


def test_maxwell_magnetic_field_sphere(default_parameters, helpers, device_interface, precision):
    """Test Maxwell magnetic field on sphere."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.maxwell import magnetic_field

    grid = helpers.load_grid("sphere")

    space1 = function_space(grid, "RWG", 0)
    space2 = function_space(grid, "SNC", 0)

    discrete_op = magnetic_field(
        space1,
        space1,
        space2,
        2.5,
        assembler="dense",
        device_interface=device_interface,
        precision=precision,
        parameters=default_parameters,
    ).weak_form()

    if precision == "single":
        rtol = 1e-5
        atol = 1e-7
    else:
        rtol = 1e-10
        atol = 1e-14

    expected = helpers.load_npy_data("maxwell_magnetic_field_boundary")

    _np.testing.assert_allclose(discrete_op.to_dense(), expected, rtol=rtol, atol=atol)

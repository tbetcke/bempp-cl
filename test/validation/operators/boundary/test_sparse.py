"""Unit tests for sparse operators."""

# pylint: disable=redefined-outer-name
# pylint: disable=C0103

import numpy as _np
import pytest

pytestmark = pytest.mark.usefixtures("default_parameters", "helpers")


def test_sparse_identity_p0_p0(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with p0/p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_p0_p0")

    space = function_space(grid, "DP", 0)

    actual = (
        identity(
            space,
            space,
            space,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    _np.testing.assert_allclose(actual, expected, helpers.default_tolerance(precision))


def test_sparse_identity_p0_p1(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with p0/p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_p0_p1")

    p0 = function_space(grid, "DP", 0)
    p1 = function_space(grid, "P", 1)

    actual = (
        identity(
            p1,
            p1,
            p0,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    _np.testing.assert_allclose(actual, expected, helpers.default_tolerance(precision))


def test_sparse_identity_p1_p0(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with p1/p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_p1_p0")

    p0 = function_space(grid, "DP", 0)
    p1 = function_space(grid, "P", 1)

    actual = (
        identity(
            p0,
            p1,
            p1,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    _np.testing.assert_allclose(actual, expected, helpers.default_tolerance(precision))


def test_sparse_identity_p1_p1(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with p1/p1 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_p1_p1")

    p1 = function_space(grid, "P", 1)

    actual = (
        identity(
            p1,
            p1,
            p1,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    _np.testing.assert_allclose(actual, expected, helpers.default_tolerance(precision))


def test_sparse_identity_snc_rwg(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with snc/rwg basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_snc_rwg")

    rwg = function_space(grid, "RWG", 0)
    snc = function_space(grid, "SNC", 0)

    actual = (
        identity(
            rwg,
            rwg,
            snc,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    if precision == "single":
        atol = 1e-7
    else:
        atol = 1e-14

    _np.testing.assert_allclose(actual, expected, atol=atol)


def test_sparse_identity_snc_bc(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with snc/bc basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("sphere")
    expected = helpers.load_npy_data("sparse_identity_snc_bc")

    bc = function_space(grid, "BC", 0)
    snc = function_space(grid, "SNC", 0)

    actual = (
        identity(
            bc,
            bc,
            snc,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    if precision == "single":
        atol = 1e-7
    else:
        atol = 1e-14

    _np.testing.assert_allclose(actual, expected, atol=atol)


def test_sparse_identity_bc_bc_on_screen(default_parameters, helpers, device_interface, precision):
    """Test singular assembler for the sparse L^2 identity with p0/p0 basis."""
    from bempp_cl.api import function_space
    from bempp_cl.api.operators.boundary.sparse import identity

    grid = helpers.load_grid("screen")
    expected = helpers.load_npy_data("sparse_identity_bc_bc_screen")

    space = function_space(grid, "BC", 0)

    actual = (
        identity(
            space,
            space,
            space,
            parameters=default_parameters,
            device_interface=device_interface,
            precision=precision,
        )
        .weak_form()
        .to_dense()
    )

    print(expected)
    print(actual)

    _np.testing.assert_allclose(actual, expected, helpers.default_tolerance(precision))

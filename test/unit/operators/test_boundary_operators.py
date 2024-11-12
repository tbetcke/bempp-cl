"""Unit tests for the dense assembler."""

import pytest
import bempp_cl.api
from bempp_cl.api import function_space
from bempp_cl.api.operators.boundary import (
    laplace,
    helmholtz,
    modified_helmholtz,
    maxwell,
    sparse,
)

scalar_spaces = [("DP", 0), ("DP", 1), ("P", 1)]
div_spaces = [("RWG", 0)]
curl_spaces = [("SNC", 0)]


@pytest.mark.parametrize("operator", [sparse.identity])
@pytest.mark.parametrize("type0", scalar_spaces)
@pytest.mark.parametrize("type1", scalar_spaces)
def test_sparse_operators(operator, type0, type1):
    """Test dense assembler for the Laplace operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space1, space1)
    op.weak_form()


@pytest.mark.parametrize("operator", [sparse.identity])
@pytest.mark.parametrize("type0", div_spaces)
@pytest.mark.parametrize("type1", curl_spaces)
def test_maxwell_sparse_operators(operator, type0, type1):
    """Test dense assembler for the Laplace operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space0, space1)
    op.weak_form()


@pytest.mark.parametrize(
    "operator",
    [laplace.single_layer, laplace.double_layer, laplace.adjoint_double_layer],
)
@pytest.mark.parametrize("type0", scalar_spaces)
@pytest.mark.parametrize("type1", scalar_spaces)
def test_laplace_operators(operator, type0, type1):
    """Test dense assembler for the Laplace operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space1, space1, assembler="dense")
    op.weak_form()


@pytest.mark.parametrize(
    "operator",
    [helmholtz.single_layer, helmholtz.double_layer, helmholtz.adjoint_double_layer],
)
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
@pytest.mark.parametrize("type0", scalar_spaces)
@pytest.mark.parametrize("type1", scalar_spaces)
def test_helmholtz_operators(operator, wavenumber, type0, type1):
    """Test dense assembler for the Helmholtz operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space1, space1, wavenumber, assembler="dense")
    op.weak_form()


@pytest.mark.parametrize(
    "operator",
    [
        modified_helmholtz.single_layer,
        modified_helmholtz.double_layer,
        modified_helmholtz.adjoint_double_layer,
    ],
)
@pytest.mark.parametrize("wavenumber", [2.5])
@pytest.mark.parametrize("type0", scalar_spaces)
@pytest.mark.parametrize("type1", scalar_spaces)
def test_modified_helmholtz_operators(operator, wavenumber, type0, type1):
    """Test dense assembler for the modified Helmholtz operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space1, space1, wavenumber, assembler="dense")
    op.weak_form()


def test_hypersingular_operators():
    """Test the hypersingular operators."""

    grid = bempp_cl.api.shapes.regular_sphere(0)

    space = function_space(grid, "P", 1)

    bempp_cl.api.operators.boundary.laplace.hypersingular(space, space, space, assembler="dense").weak_form()
    bempp_cl.api.operators.boundary.helmholtz.hypersingular(space, space, space, 1.5, assembler="dense").weak_form()
    bempp_cl.api.operators.boundary.modified_helmholtz.hypersingular(
        space, space, space, 1.5, assembler="dense"
    ).weak_form()


@pytest.mark.parametrize(
    ("type0", "type1"),
    [[("DP", 0), ("P", 1)], [("P", 1), ("DP", 0)], [("DP", 0), ("DP", 0)]],
)
def test_hypersingular_fails_for_wrong_space(type0, type1):
    """Expected failure for wrong spaces."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    with pytest.raises(ValueError):
        laplace.hypersingular(space0, space1, space1, assembler="dense").weak_form()
    with pytest.raises(ValueError):
        helmholtz.hypersingular(space0, space1, space1, 1.5, assembler="dense").weak_form()
    with pytest.raises(ValueError):
        modified_helmholtz.hypersingular(space0, space1, space1, 1.5, assembler="dense").weak_form()


@pytest.mark.parametrize("operator", [maxwell.magnetic_field, maxwell.electric_field])
@pytest.mark.parametrize("wavenumber", [2.5, 2.5 + 1j])
@pytest.mark.parametrize("type0", div_spaces)
@pytest.mark.parametrize("type1", curl_spaces)
def test_maxwell_operators(operator, wavenumber, type0, type1):
    """Test dense assembler for the Maxwell operators."""
    grid = bempp_cl.api.shapes.regular_sphere(0)
    space0 = function_space(grid, *type0)
    space1 = function_space(grid, *type1)

    op = operator(space0, space0, space1, wavenumber, assembler="dense")
    op.weak_form()

    # Check that we cannot swap curl and div conforming spaces for Maxwell

    with pytest.raises(ValueError):
        op = operator(space1, space1, space0, wavenumber, assembler="dense")

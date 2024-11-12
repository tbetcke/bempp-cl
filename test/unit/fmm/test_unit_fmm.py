"""Unit tests for the dense assembler."""

import numpy as np
import pytest
import bempp_cl.api
from bempp_cl.api import function_space, check_for_fmm
from bempp_cl.api.operators.boundary import laplace, helmholtz


def test_laplace_single_layer(has_exafmm):
    """Test dense assembler for the Laplace operators."""
    if not has_exafmm and not check_for_fmm():
        pytest.skip("ExaFMM must be installed to run this test.")

    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = function_space(grid, "DP", 0)

    op1 = laplace.single_layer(space, space, space, assembler="dense")
    op2 = laplace.single_layer(space, space, space, assembler="fmm")

    fun = bempp_cl.api.GridFunction(space, coefficients=np.random.rand(space.global_dof_count))

    assert np.allclose((op1 * fun).coefficients, (op2 * fun).coefficients)

    bempp_cl.api.clear_fmm_cache()


@pytest.mark.parametrize("wavenumber", [2.5])  # , 2.5 + 1j])
def test_helmholtz_single_layer(has_exafmm, wavenumber):
    """Test dense assembler for the Laplace operators."""
    if not has_exafmm and not check_for_fmm():
        pytest.skip("ExaFMM must be installed to run this test.")

    grid = bempp_cl.api.shapes.regular_sphere(2)
    space = function_space(grid, "DP", 0)

    op1 = helmholtz.single_layer(space, space, space, wavenumber, assembler="dense")
    op2 = helmholtz.single_layer(space, space, space, wavenumber, assembler="fmm")

    fun = bempp_cl.api.GridFunction(space, coefficients=np.random.rand(space.global_dof_count))

    assert np.allclose((op1 * fun).coefficients, (op2 * fun).coefficients)

    bempp_cl.api.clear_fmm_cache()

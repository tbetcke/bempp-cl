import pytest
import bempp_cl.api
import numpy as np
from bempp_cl.api.operators.boundary import laplace
from bempp_cl.api.assembly.blocked_operator import BlockedOperator, BlockedDiscreteOperator
from scipy.sparse.linalg.interface import LinearOperator


@pytest.mark.parametrize("cols", range(4))
def test_blocked_matvec(cols):
    grid = bempp_cl.api.shapes.sphere(h=1)
    space = bempp_cl.api.function_space(grid, "P", 1)

    ndofs = space.global_dof_count

    block01 = laplace.single_layer(space, space, space)
    block10 = laplace.adjoint_double_layer(space, space, space)
    block21 = laplace.double_layer(space, space, space)

    op = BlockedOperator(3, 2)
    op[0, 1] = block01
    op[1, 0] = block10
    op[2, 1] = block21

    if cols == 0:
        vec = np.random.rand(2 * ndofs)
    else:
        vec = np.random.rand(2 * ndofs, cols)

    result1 = op.weak_form() * vec

    assert np.allclose(block01.weak_form() * vec[ndofs:], result1[:ndofs])
    assert np.allclose(block10.weak_form() * vec[:ndofs], result1[ndofs : 2 * ndofs])
    assert np.allclose(block21.weak_form() * vec[ndofs:], result1[2 * ndofs :])


@pytest.mark.parametrize("cols", range(4))
def test_blocked_matvec_linear_operator(cols):
    grid = bempp_cl.api.shapes.sphere(h=1)
    space = bempp_cl.api.function_space(grid, "P", 1)

    ndofs = space.global_dof_count

    block01 = laplace.single_layer(space, space, space).weak_form()
    block10 = laplace.adjoint_double_layer(space, space, space).weak_form()
    block21 = LinearOperator([ndofs, ndofs], matvec=lambda x: np.array([x[i] * i for i in range(ndofs)]))

    op = BlockedDiscreteOperator([[None, block01], [block10, None], [None, block21]])

    if cols == 0:
        vec = np.random.rand(2 * ndofs)
    else:
        vec = np.random.rand(2 * ndofs, cols)

    result1 = op * vec

    assert np.allclose(block01 * vec[ndofs:], result1[:ndofs])
    assert np.allclose(block10 * vec[:ndofs], result1[ndofs : 2 * ndofs])
    assert np.allclose(block21 * vec[ndofs:], result1[2 * ndofs :])


def test_blocked_matvec_only_linear_operator():
    grid = bempp_cl.api.shapes.sphere(h=1)
    space = bempp_cl.api.function_space(grid, "P", 1)

    ndofs = space.global_dof_count

    block01 = laplace.single_layer(space, space, space).weak_form()
    block10 = laplace.adjoint_double_layer(space, space, space).weak_form()
    block21 = LinearOperator([ndofs, ndofs], matvec=lambda x: x * np.arange(ndofs))

    op = BlockedDiscreteOperator([[None, block01], [block10, None], [None, block21]])

    vec = np.random.rand(2 * ndofs)

    result1 = op * vec

    assert np.allclose(block01 * vec[ndofs:], result1[:ndofs])
    assert np.allclose(block10 * vec[:ndofs], result1[ndofs : 2 * ndofs])
    assert np.allclose(block21 * vec[ndofs:], result1[2 * ndofs :])

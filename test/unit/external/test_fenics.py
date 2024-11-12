"""Unit tests for the FEniCS interface."""

import pytest
import numpy as np


def test_p1_trace(has_dolfin):
    """Test the trace of a P1 DOLFIN function."""
    if has_dolfin:
        import dolfin
    else:
        try:
            import dolfin
        except ImportError:
            pytest.skip("DOLFIN must be installed to run this test")
    import bempp_cl.api
    from bempp_cl.api.external.fenics import fenics_to_bempp_trace_data

    fenics_mesh = dolfin.UnitCubeMesh(2, 2, 2)
    fenics_space = dolfin.FunctionSpace(fenics_mesh, "CG", 1)

    bempp_space, trace_matrix = fenics_to_bempp_trace_data(fenics_space)

    fenics_coeffs = np.random.rand(fenics_space.dim())
    bempp_coeffs = trace_matrix @ fenics_coeffs

    fenics_fun = dolfin.Function(fenics_space)
    fenics_fun.vector()[:] = fenics_coeffs
    bempp_fun = bempp_cl.api.GridFunction(bempp_space, coefficients=bempp_coeffs)

    for cell in bempp_space.grid.entity_iterator(0):
        mid = cell.geometry.centroid
        bempp_val = bempp_fun.evaluate(cell.index, np.array([[1 / 3], [1 / 3]]))

        fenics_val = np.zeros(1)
        fenics_fun.eval(fenics_val, mid)

        assert np.allclose(bempp_val.T[0], fenics_val)


def test_nc1_trace(has_dolfin):
    """Test the trace of a (N1curl, 1) DOLFIN function."""
    if has_dolfin:
        import dolfin
    else:
        try:
            import dolfin
        except ImportError:
            pytest.skip("DOLFIN must be installed to run this test")
    import bempp_cl.api
    from bempp_cl.api.external.fenics import fenics_to_bempp_trace_data

    fenics_mesh = dolfin.UnitCubeMesh(2, 2, 2)
    fenics_space = dolfin.FunctionSpace(fenics_mesh, "N1curl", 1)

    bempp_space, trace_matrix = fenics_to_bempp_trace_data(fenics_space)

    fenics_coeffs = np.random.rand(fenics_space.dim())
    bempp_coeffs = trace_matrix @ fenics_coeffs

    fenics_fun = dolfin.Function(fenics_space)
    fenics_fun.vector()[:] = fenics_coeffs
    bempp_fun = bempp_cl.api.GridFunction(bempp_space, coefficients=bempp_coeffs)

    for cell in bempp_space.grid.entity_iterator(0):
        mid = cell.geometry.centroid
        normal = cell.geometry.normal
        bempp_val = bempp_fun.evaluate(cell.index, np.array([[1 / 3], [1 / 3]]))

        fenics_val = np.zeros(3)
        fenics_fun.eval(fenics_val, mid)
        crossed = np.cross(fenics_val, normal)

        assert np.allclose(bempp_val.T[0], crossed)

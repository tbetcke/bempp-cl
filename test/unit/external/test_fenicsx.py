"""Unit tests for the FEniCSx interface."""

import pytest
import numpy as np
import bempp_cl.api
from bempp_cl.api.external.fenicsx import fenics_to_bempp_trace_data, FenicsOperator


def test_p1_trace(has_dolfinx):
    """Test the trace of a P1 DOLFINx function."""
    try:
        from mpi4py import MPI
        import dolfinx
    except ImportError:
        if has_dolfinx:
            raise ImportError("DOLFINx is not installed")
        pytest.skip("DOLFINx must be installed to run this test")

    fenics_mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    fenics_space = dolfinx.fem.functionspace(fenics_mesh, ("CG", 1))

    bempp_space, trace_matrix = fenics_to_bempp_trace_data(fenics_space)

    fenics_coeffs = np.random.rand(fenics_space.dofmap.index_map.size_global).astype(np.complex64)
    bempp_coeffs = trace_matrix @ fenics_coeffs

    fenics_fun = dolfinx.fem.Function(fenics_space)
    fenics_fun.x.array[:] = fenics_coeffs
    bempp_fun = bempp_cl.api.GridFunction(bempp_space, coefficients=bempp_coeffs)

    try:
        tree = dolfinx.geometry.bb_tree(fenics_mesh, 3)
    except AttributeError:
        # Support older FEniCSx
        tree = dolfinx.geometry.BoundingBoxTree(fenics_mesh, 3)

    midpoint_tree = dolfinx.geometry.create_midpoint_tree(
        fenics_mesh, 3, np.array(range(fenics_mesh.topology.connectivity(3, 0).num_nodes))
    )

    for cell in bempp_space.grid.entity_iterator(0):
        mid = cell.geometry.centroid
        bempp_val = bempp_fun.evaluate(cell.index, np.array([[1 / 3], [1 / 3]]))

        fenics_cell = dolfinx.geometry.compute_closest_entity(tree, midpoint_tree, fenics_mesh, mid)[0]
        fenics_val = fenics_fun.eval([mid.T], [fenics_cell])
        assert np.isclose(bempp_val[0, 0], fenics_val[0])


def test_fenics_operator(has_dolfinx):
    """Test that a FEniCS operator can be assembled."""
    try:
        import ufl
    except ImportError:
        if has_dolfinx:
            raise ImportError("UFL is not installed")
        pytest.skip("UFL must be installed to run this test")
    try:
        from mpi4py import MPI
        from dolfinx.fem import functionspace
        from dolfinx.mesh import create_unit_cube
    except ImportError:
        if has_dolfinx:
            raise ImportError("DOLFINx is not installed")
        pytest.skip("DOLFINx must be installed to run this test")

    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    space = functionspace(mesh, ("CG", 1))

    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)

    op = FenicsOperator(ufl.inner(u, v) * ufl.dx)
    weak_op = op.weak_form()
    weak_op * np.ones(space.dofmap.index_map.size_global)

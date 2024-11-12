# Script to be run with legacy Bempp to generate the comparison data.

import bempp_cl.api
import numpy as np

import os.path
import sys

# run `python generate_matrices.py REGENERATE` to regenerate everything
REGENERATE = "REGENERATE" in sys.argv

bempp_cl.api.enable_console_logging()


def generate_bem_matrix(dual_to_range, domain, fname, operator, wavenumber=None):
    """Generate test matrix."""
    print("Generating " + fname)

    if wavenumber is None:
        mat = operator(domain, domain, dual_to_range, use_projection_spaces=False).weak_form().A
    else:
        mat = operator(domain, domain, dual_to_range, wavenumber, use_projection_spaces=False).weak_form().A

    if REGENERATE or not os.path.exists(fname + ".npy"):
        np.save(fname, mat)


def generate_sparse_bem_matrix(dual_to_range, domain, fname, operator):
    """Generate test matrix."""
    print("Generating " + fname)

    mat = operator(domain, domain, dual_to_range).weak_form().sparse_operator.todense()
    if REGENERATE or not os.path.exists(fname + ".npy"):
        np.save(fname, mat)


def generate_potential(domain, fname, operator, wavenumber=None):
    """Generate off-surface potential data."""

    npoints = 10
    rand = np.random.RandomState(0)

    points = 2.5 * np.ones((3, 1), dtype="float64") + rand.rand(3, npoints)

    vec = np.random.rand(domain.global_dof_count)
    fun = bempp_cl.api.GridFunction(domain, coefficients=vec)

    if wavenumber is None:
        pot = operator(domain, points)
    else:
        pot = operator(domain, points, wavenumber)

    result = pot.evaluate(fun)

    if REGENERATE or not os.path.exists(fname + ".npz"):
        np.savez(fname, result=result, points=points, vec=vec)


def generate_far_field(domain, fname, operator, wavenumber=None):
    """Generate far-field data."""

    npoints = 10
    rand = np.random.RandomState(0)

    points = 2.5 * np.ones((3, 1), dtype="float64") + rand.rand(3, npoints)
    points /= np.linalg.norm(points, axis=0)

    vec = np.random.rand(domain.global_dof_count)
    fun = bempp_cl.api.GridFunction(domain, coefficients=vec)

    if wavenumber is None:
        pot = operator(domain, points)
    else:
        pot = operator(domain, points, wavenumber)

    result = pot.evaluate(fun)

    if REGENERATE or not os.path.exists(fname + ".npz"):
        np.savez(fname, result=result, points=points, vec=vec)


parameters = bempp_cl.api.global_parameters
parameters.assembly.boundary_operator_assembly_type = "dense"
parameters.assembly.potential_operator_assembly_type = "dense"

regular_order = 4
singular_order = 6

wavenumber = 2.5
wavenumber_complex = 2.5 + 1j

parameters.quadrature.near.double_order = regular_order
parameters.quadrature.near.single_order = regular_order
parameters.quadrature.medium.double_order = regular_order
parameters.quadrature.medium.single_order = regular_order
parameters.quadrature.far.double_order = regular_order
parameters.quadrature.far.single_order = regular_order

parameters.quadrature.double_singular = singular_order

grid_structured = bempp_cl.api.import_grid("structured_grid.msh")
rwg_structured = bempp_cl.api.function_space(grid_structured, "RWG", 0)
snc_structured = bempp_cl.api.function_space(grid_structured, "SNC", 0)

generate_bem_matrix(
    snc_structured,
    rwg_structured,
    "maxwell_electric_field_structured_boundary",
    bempp_cl.api.operators.boundary.maxwell.electric_field,
    wavenumber,
)

grid = bempp_cl.api.import_grid("sphere.msh")

p0 = bempp_cl.api.function_space(grid, "DP", 0)
dp1 = bempp_cl.api.function_space(grid, "DP", 1)
p1 = bempp_cl.api.function_space(grid, "P", 1)
rwg = bempp_cl.api.function_space(grid, "RWG", 0)
brwg = bempp_cl.api.function_space(grid, "B-RWG", 0)
snc = bempp_cl.api.function_space(grid, "SNC", 0)
bsnc = bempp_cl.api.function_space(grid, "B-SNC", 0)
bc = bempp_cl.api.function_space(grid, "BC", 0)
rbc = bempp_cl.api.function_space(grid, "RBC", 0)

print("Generating Laplace BEM matrices.")

generate_bem_matrix(
    p0,
    p0,
    "laplace_single_layer_boundary_p0_p0",
    bempp_cl.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    p0,
    dp1,
    "laplace_single_layer_boundary_p0_dp1",
    bempp_cl.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    dp1,
    p0,
    "laplace_single_layer_boundary_dp1_p0",
    bempp_cl.api.operators.boundary.laplace.single_layer,
)
generate_bem_matrix(
    dp1,
    dp1,
    "laplace_single_layer_boundary_dp1_dp1",
    bempp_cl.api.operators.boundary.laplace.single_layer,
)
generate_bem_matrix(
    p1,
    p1,
    "laplace_single_layer_boundary_p1_p1",
    bempp_cl.api.operators.boundary.laplace.single_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_double_layer_boundary",
    bempp_cl.api.operators.boundary.laplace.double_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_adj_double_layer_boundary",
    bempp_cl.api.operators.boundary.laplace.adjoint_double_layer,
)

generate_bem_matrix(
    p1,
    p1,
    "laplace_hypersingular_boundary",
    bempp_cl.api.operators.boundary.laplace.hypersingular,
)

################################

print("Generating Helmholtz BEM matrices.")

generate_bem_matrix(
    p0,
    p0,
    "helmholtz_single_layer_boundary_p0_p0",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber,
)

generate_bem_matrix(
    p0,
    dp1,
    "helmholtz_single_layer_boundary_p0_dp1",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber,
)

generate_bem_matrix(
    dp1,
    p0,
    "helmholtz_single_layer_boundary_dp1_p0",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber,
)
generate_bem_matrix(
    dp1,
    dp1,
    "helmholtz_single_layer_boundary_dp1_dp1",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber,
)
generate_bem_matrix(
    p1,
    p1,
    "helmholtz_single_layer_boundary_p1_p1",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_double_layer_boundary",
    bempp_cl.api.operators.boundary.helmholtz.double_layer,
    wavenumber,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_adj_double_layer_boundary",
    bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer,
    wavenumber,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_hypersingular_boundary",
    bempp_cl.api.operators.boundary.helmholtz.hypersingular,
    wavenumber,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_single_layer_boundary",
    bempp_cl.api.operators.boundary.helmholtz.single_layer,
    wavenumber_complex,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_double_layer_boundary",
    bempp_cl.api.operators.boundary.helmholtz.double_layer,
    wavenumber_complex,
)
generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_adj_double_layer_boundary",
    bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer,
    wavenumber_complex,
)

generate_bem_matrix(
    p1,
    p1,
    "helmholtz_complex_hypersingular_boundary",
    bempp_cl.api.operators.boundary.helmholtz.hypersingular,
    wavenumber_complex,
)

##################################

print("Generate modified Helmholtz BEM matrices.")

omega = 2.5

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_single_layer_boundary",
    bempp_cl.api.operators.boundary.modified_helmholtz.single_layer,
    omega,
)

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_double_layer_boundary",
    bempp_cl.api.operators.boundary.modified_helmholtz.double_layer,
    omega,
)
generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_adj_double_layer_boundary",
    bempp_cl.api.operators.boundary.modified_helmholtz.adjoint_double_layer,
    omega,
)

generate_bem_matrix(
    p1,
    p1,
    "modified_helmholtz_hypersingular_boundary",
    bempp_cl.api.operators.boundary.modified_helmholtz.hypersingular,
    omega,
)

#####################################

print("Generate Maxwell BEM matrices.")

generate_bem_matrix(
    snc,
    rwg,
    "maxwell_electric_field_boundary",
    bempp_cl.api.operators.boundary.maxwell.electric_field,
    wavenumber,
)

generate_bem_matrix(
    bsnc,
    bc,
    "maxwell_electric_field_boundary_bc",
    bempp_cl.api.operators.boundary.maxwell.electric_field,
    wavenumber,
)

generate_bem_matrix(
    rbc,
    bc,
    "maxwell_electric_field_boundary_rbc_bc",
    bempp_cl.api.operators.boundary.maxwell.electric_field,
    wavenumber,
)

generate_bem_matrix(
    snc,
    rwg,
    "maxwell_magnetic_field_boundary",
    bempp_cl.api.operators.boundary.maxwell.magnetic_field,
    wavenumber,
)

generate_bem_matrix(
    snc,
    rwg,
    "maxwell_electric_field_complex_boundary",
    bempp_cl.api.operators.boundary.maxwell.electric_field,
    wavenumber_complex,
)

generate_bem_matrix(
    snc,
    rwg,
    "maxwell_magnetic_field_complex_boundary",
    bempp_cl.api.operators.boundary.maxwell.magnetic_field,
    wavenumber_complex,
)

generate_sparse_bem_matrix(p0, p0, "sparse_identity_p0_p0", bempp_cl.api.operators.boundary.sparse.identity)

generate_sparse_bem_matrix(p1, p1, "sparse_identity_p1_p1", bempp_cl.api.operators.boundary.sparse.identity)

generate_sparse_bem_matrix(p0, p1, "sparse_identity_p0_p1", bempp_cl.api.operators.boundary.sparse.identity)

generate_sparse_bem_matrix(p1, p0, "sparse_identity_p1_p0", bempp_cl.api.operators.boundary.sparse.identity)

generate_sparse_bem_matrix(snc, rwg, "sparse_identity_snc_rwg", bempp_cl.api.operators.boundary.sparse.identity)

generate_sparse_bem_matrix(bsnc, bc, "sparse_identity_snc_bc", bempp_cl.api.operators.boundary.sparse.identity)

generate_potential(
    p0,
    "laplace_single_layer_potential_p0",
    bempp_cl.api.operators.potential.laplace.single_layer,
)
generate_potential(
    p1,
    "laplace_single_layer_potential_p1",
    bempp_cl.api.operators.potential.laplace.single_layer,
)
generate_potential(
    p1,
    "laplace_double_layer_potential_p1",
    bempp_cl.api.operators.potential.laplace.double_layer,
)

generate_potential(
    p1,
    "helmholtz_single_layer_potential_p1",
    bempp_cl.api.operators.potential.helmholtz.single_layer,
    wavenumber,
)
generate_potential(
    p1,
    "helmholtz_double_layer_potential_p1",
    bempp_cl.api.operators.potential.helmholtz.double_layer,
    wavenumber,
)
generate_potential(
    p1,
    "helmholtz_single_layer_potential_complex_p1",
    bempp_cl.api.operators.potential.helmholtz.single_layer,
    wavenumber_complex,
)
generate_potential(
    p1,
    "helmholtz_double_layer_potential_complex_p1",
    bempp_cl.api.operators.potential.helmholtz.double_layer,
    wavenumber_complex,
)

generate_potential(
    rwg,
    "maxwell_electric_field_potential",
    bempp_cl.api.operators.potential.maxwell.electric_field,
    wavenumber,
)

generate_potential(
    bc,
    "maxwell_electric_field_potential_bc",
    bempp_cl.api.operators.potential.maxwell.electric_field,
    wavenumber,
)

generate_potential(
    rwg,
    "maxwell_electric_field_potential_complex",
    bempp_cl.api.operators.potential.maxwell.electric_field,
    wavenumber_complex,
)

generate_potential(
    rwg,
    "maxwell_magnetic_field_potential",
    bempp_cl.api.operators.potential.maxwell.magnetic_field,
    wavenumber,
)

generate_potential(
    rwg,
    "maxwell_magnetic_field_potential_complex",
    bempp_cl.api.operators.potential.maxwell.magnetic_field,
    wavenumber_complex,
)

# Start far field

generate_far_field(
    p1,
    "helmholtz_single_layer_far_field_p1",
    bempp_cl.api.operators.far_field.helmholtz.single_layer,
    wavenumber,
)

generate_far_field(
    p1,
    "helmholtz_double_layer_far_field_p1",
    bempp_cl.api.operators.far_field.helmholtz.double_layer,
    wavenumber,
)
generate_far_field(
    p1,
    "helmholtz_single_layer_far_field_complex_p1",
    bempp_cl.api.operators.far_field.helmholtz.single_layer,
    wavenumber_complex,
)
generate_far_field(
    p1,
    "helmholtz_double_layer_far_field_complex_p1",
    bempp_cl.api.operators.far_field.helmholtz.double_layer,
    wavenumber_complex,
)

generate_far_field(
    rwg,
    "maxwell_electric_far_field",
    bempp_cl.api.operators.far_field.maxwell.electric_field,
    wavenumber,
)

generate_far_field(
    rwg,
    "maxwell_electric_far_field_complex",
    bempp_cl.api.operators.far_field.maxwell.electric_field,
    wavenumber_complex,
)

generate_far_field(
    rwg,
    "maxwell_magnetic_far_field",
    bempp_cl.api.operators.far_field.maxwell.magnetic_field,
    wavenumber,
)

generate_far_field(
    rwg,
    "maxwell_magnetic_far_field_complex",
    bempp_cl.api.operators.far_field.maxwell.magnetic_field,
    wavenumber_complex,
)

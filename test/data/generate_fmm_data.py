# Script to be run with legacy Bempp to generate the comparison data.

import bempp_cl.api
import numpy as np

import os.path
import sys

# run `python generate_fmm_data.py REGENERATE` to regenerate everything
REGENERATE = "REGENERATE" in sys.argv
data = {}


def generate_vector(size, filename):
    global data
    if filename not in data:
        if REGENERATE or not os.path.exists(filename + ".npy"):
            rand = np.random.RandomState(0)
            vec = rand.rand(size)
            np.save(filename, vec)
        else:
            vec = np.load(filename + ".npy")
        data[filename] = vec
    return data[filename]


def generate_points(size, filename):
    global data
    if filename not in data:
        if REGENERATE or not os.path.exists(filename + ".npy"):
            rand = np.random.RandomState(0)
            points = np.vstack([2 * np.ones(size, dtype="float64"), rand.randn(size), rand.randn(size)])
            np.save(filename, points)
        else:
            points = np.load(filename + ".npy")
        data[filename] = points
    return data[filename]


def save_matvec_result(operator, space1, space2, space3, vec, filename, *args):
    if REGENERATE or not os.path.exists(filename + ".npy"):
        print("Generating " + filename)
        dense_mat = operator(space1, space2, space3, *args, assembler="dense").weak_form()
        np.save(filename, dense_mat @ vec)
    else:
        print("Skipping " + filename + " (already generated)")


def save_potential_eval_result(operator, space, vec, points, filename, *args):
    if REGENERATE or not os.path.exists(filename + ".npy"):
        print("Generating " + filename)
        grid_fun = bempp_cl.api.GridFunction(space, coefficients=vec)
        result = operator(space, points, *args, assembler="dense").evaluate(grid_fun)
        np.save(filename, result)
    else:
        print("Skipping " + filename + " (already generated)")


grid = bempp_cl.api.shapes.ellipsoid(1, 0.5, 0.3, h=0.1)
bempp_cl.api.export("fmm_grid.msh", grid=grid)
space = bempp_cl.api.function_space(grid, "P", 1)

vec = generate_vector(space.global_dof_count, "fmm_p1_vec")
points = generate_points(30, "fmm_potential_points.npy")

# Generate P1 boundary operator results
for filename, operator in [
    ("fmm_laplace_single", bempp_cl.api.operators.boundary.laplace.single_layer),
    ("fmm_laplace_double", bempp_cl.api.operators.boundary.laplace.double_layer),
    ("fmm_laplace_adjoint", bempp_cl.api.operators.boundary.laplace.adjoint_double_layer),
    ("fmm_laplace_hyper", bempp_cl.api.operators.boundary.laplace.hypersingular),
]:
    save_matvec_result(operator, space, space, space, vec, filename)

for filename, operator in [
    ("fmm_helmholtz_single", bempp_cl.api.operators.boundary.helmholtz.single_layer),
    ("fmm_helmholtz_double", bempp_cl.api.operators.boundary.helmholtz.double_layer),
    (
        "fmm_helmholtz_adjoint",
        bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer,
    ),
    ("fmm_helmholtz_hyper", bempp_cl.api.operators.boundary.helmholtz.hypersingular),
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
    save_matvec_result(operator, space, space, space, vec, filename, 1.5)

# Generate P1 potential operator results
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
    save_potential_eval_result(operator, space, vec, points, filename)

for filename, operator in [
    (
        "fmm_helmholtz_potential_single",
        bempp_cl.api.operators.potential.helmholtz.single_layer,
    ),
    (
        "fmm_helmholtz_potential_double",
        bempp_cl.api.operators.potential.helmholtz.double_layer,
    ),
    (
        "fmm_modified_potential_helmholtz_single",
        bempp_cl.api.operators.potential.modified_helmholtz.single_layer,
    ),
    (
        "fmm_modified_potential_helmholtz_double",
        bempp_cl.api.operators.potential.modified_helmholtz.double_layer,
    ),
]:
    save_potential_eval_result(operator, space, vec, points, filename, 1.5)


rwg = bempp_cl.api.function_space(grid, "RWG", 0)
snc = bempp_cl.api.function_space(grid, "SNC", 0)

vec2 = generate_vector(rwg.global_dof_count, "fmm_rwg_vec")

# Generate Maxwell boundary operator results
for filename, operator in [
    ("fmm_maxwell_electric", bempp_cl.api.operators.boundary.maxwell.electric_field),
    ("fmm_maxwell_magnetic", bempp_cl.api.operators.boundary.maxwell.magnetic_field),
]:
    save_matvec_result(operator, rwg, rwg, snc, vec2, filename, 1.5)

# Generate Maxwell potential operator results
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
    save_potential_eval_result(operator, rwg, vec2, points, filename, 1.5)


# Generate two grid data
grid1 = bempp_cl.api.shapes.ellipsoid(0.5, 0.5, 0.3, h=0.1)
grid2 = bempp_cl.api.shapes.sphere(r=1.5, h=0.1)
bempp_cl.api.export("fmm_grid1.msh", grid=grid1)
bempp_cl.api.export("fmm_grid2.msh", grid=grid2)

p1_space1 = bempp_cl.api.function_space(grid1, "P", 1)
p1_space2 = bempp_cl.api.function_space(grid2, "P", 1)

vec = generate_vector(p1_space1.global_dof_count, "fmm_two_mesh_vec")

for filename, operator in [
    ("fmm_two_mesh_laplace_single", bempp_cl.api.operators.boundary.laplace.single_layer),
    ("fmm_two_mesh_laplace_hyper", bempp_cl.api.operators.boundary.laplace.hypersingular),
]:
    save_matvec_result(operator, p1_space1, p1_space2, p1_space2, vec, filename)

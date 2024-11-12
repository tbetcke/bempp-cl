# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Bempp OpenCL performance benchmarks
# This is a test notebook to benchmark the Bempp performance on different devices. The default figures reported here
# are obtained under Ubuntu 20.04 Linux on an Intel Core i9-9980HK 8 Core CPU with a base clock of 2.4GHz and a maximum turbo
# frequency of 5GHz. The GPU device is an NVIDIA Quadro RTX 3000 GPU with 6GB Ram.
#
# As OpenCL CPU driver we test both POCL (in Version 1.5) and the Intel OpenCL CPU driver, both with default vectorization options.
#
# We are benchmarking the following operator types
#
# * Boundary Operators:
#
#     * Laplace single and double layer boundary operator
#     * Helmholtz single and double layer boundary operator
#     * Maxwell electric and magnetic field boundary operator
#
#
# * Domain Potential Operators:
#
#     * Laplace single and double layer potential operator
#     * Helmholtz single and double layer potential operator
#     * Maxwell electric and magnetic field domain potential operator
#
#
# We are testing all operators in single and double precision. For the GPU we only perform single precision tests as it is significantly
# slower in double precision.
#
# As mesh we use a uniform sphere with 8192 elements. As wavenumber in the Helmholtz and Maxwell tests we use the value $k=1.0$. This has no effect on the performance. As scalar function spaces we use spaces of continuous $P1$ functions. For Maxwell we use RWG functions of order 0.
#
# ## General Setup
# In this section we define the general objects that we need in all benchmarks.

import bempp_cl.api
import numpy as np
import pandas as pd

grid = bempp_cl.api.shapes.regular_sphere(5)
p1_space = bempp_cl.api.function_space(grid, "P", 1)
rwg_space = bempp_cl.api.function_space(grid, "RWG", 0)
snc_space = bempp_cl.api.function_space(grid, "SNC", 0)


# +
try:
    get_ipython()
except NameError:
    raise RuntimeError("Must run using IPython")


def benchmark_boundary_operator(operator, precision):
    """Benchmark an operator with given precision"""
    result = get_ipython().run_line_magic("timeit", "-o -r 2 -n 2 operator(precision).weak_form()")  # noqa: F821
    return result.best


def benchmark_potential_operator(operator, fun, precision):
    """Benchmark an operator with given precision"""
    result = get_ipython().run_line_magic("timeit", "-o -r 2 -n 2 res = operator(precision) @ fun")  # noqa: F821
    return result.best


# -

# ## Boundary Operators
# We first define the boundary operators that we want to test. We make them dependent only on a *precision* argument.

# +
from bempp_cl.api.operators import boundary

k = 1.0

operators = [
    lambda precision: boundary.laplace.single_layer(p1_space, p1_space, p1_space, precision=precision),
    lambda precision: boundary.laplace.double_layer(p1_space, p1_space, p1_space, precision=precision),
    lambda precision: boundary.helmholtz.single_layer(p1_space, p1_space, p1_space, k, precision=precision),
    lambda precision: boundary.helmholtz.double_layer(p1_space, p1_space, p1_space, k, precision=precision),
    lambda precision: boundary.maxwell.electric_field(rwg_space, rwg_space, snc_space, k, precision=precision),
    lambda precision: boundary.maxwell.magnetic_field(rwg_space, rwg_space, snc_space, k, precision=precision),
]
# -

# We setup a Pandas data frame to conveniently store the different timings.

# +
driver_labels = ["Portable Computing Language", "Intel(R) OpenCL"]
precision_labels = ["single", "double"]

operator_labels = [
    "laplace single layer bnd",
    "laplace double layer bnd",
    "helmholtz single layer bnd",
    "helmholtz double layer bnd",
    "maxwell electric bnd",
    "maxwell magnetic bnd",
]
df = pd.DataFrame(index=operator_labels, columns=pd.MultiIndex.from_product([driver_labels, precision_labels]))
# -

# We assemble each operator once to make sure that all Numba functions are compiled.

for precision in ["single", "double"]:
    if precision == "single":
        bempp_cl.api.VECTORIZATION_MODE = "vec16"
    else:
        bempp_cl.api.VECTORIZATION_MODE = "vec8"
    for driver_name in driver_labels:
        bempp_cl.api.set_default_cpu_device_by_name(driver_name)
        for op in operators:
            op(precision).weak_form()

# Now let's run the actual benchmarks.

for precision in ["single", "double"]:
    if precision == "single":
        bempp_cl.api.VECTORIZATION_MODE = "vec16"
    else:
        bempp_cl.api.VECTORIZATION_MODE = "vec8"
    for driver_name in driver_labels:
        print(f"Driver: {driver_name}")
        bempp_cl.api.set_default_cpu_device_by_name(driver_name)
        for label, op in zip(operator_labels, operators):
            df.loc[label, (driver_name, precision)] = benchmark_boundary_operator(op, precision)

results_boundary_operators = df

# ## Potential Operators
# We are going to evaluate the potentials at 10000 evaluation points. The points are normalised to lie on a sphere with radius .5 As evaluation function we use a simple constant function.

npoints = 10000
rand = np.random.RandomState(0)
points = rand.randn(3, npoints)
points /= np.linalg.norm(points, axis=0)

# Let us define the operators and functions.

# +
from bempp_cl.api.operators import potential

k = 1.0

operators = [
    lambda precision: potential.laplace.single_layer(p1_space, points, precision=precision),
    lambda precision: potential.laplace.double_layer(p1_space, points, precision=precision),
    lambda precision: potential.helmholtz.single_layer(p1_space, points, k, precision=precision),
    lambda precision: potential.helmholtz.double_layer(p1_space, points, k, precision=precision),
    lambda precision: potential.maxwell.electric_field(rwg_space, points, k, precision=precision),
    lambda precision: potential.maxwell.magnetic_field(rwg_space, points, k, precision=precision),
]

functions = [
    bempp_cl.api.GridFunction.from_ones(p1_space),
    bempp_cl.api.GridFunction.from_ones(p1_space),
    bempp_cl.api.GridFunction.from_ones(p1_space),
    bempp_cl.api.GridFunction.from_ones(p1_space),
    bempp_cl.api.GridFunction.from_ones(rwg_space),
    bempp_cl.api.GridFunction.from_ones(rwg_space),
]
# -

# We assemble each operator once to compile all functions.

for precision in ["single", "double"]:
    if precision == "single":
        bempp_cl.api.VECTORIZATION_MODE = "vec16"
    else:
        bempp_cl.api.VECTORIZATION_MODE = "vec8"
    for driver_name in driver_labels:
        bempp_cl.api.set_default_cpu_device_by_name(driver_name)
        for op, fun in zip(operators, functions):
            res = op(precision) @ fun

# Let's create the data structure to store the results.

# +
driver_labels = ["Portable Computing Language", "Intel(R) OpenCL", "NVIDIA CUDA"]
precision_labels = ["single", "double"]

operator_labels = [
    "laplace single layer pot",
    "laplace double layer pot",
    "helmholtz single layer pot",
    "helmholtz double layer pot",
    "maxwell electric pot",
    "maxwell magnetic pot",
]
df = pd.DataFrame(index=operator_labels, columns=pd.MultiIndex.from_product([driver_labels, precision_labels]))
# -

# Finally, we run the actual tests.

# +
bempp_cl.api.set_default_gpu_device_by_name("NVIDIA CUDA")

for precision in ["single", "double"]:
    if precision == "single":
        bempp_cl.api.VECTORIZATION_MODE = "vec16"
    else:
        bempp_cl.api.VECTORIZATION_MODE = "vec8"
    for driver_name in driver_labels:
        print(f"Driver: {driver_name}")
        if driver_name == "NVIDIA CUDA":
            bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
            bempp_cl.api.VECTORIZATION_MODE = "novec"
        else:
            bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"
            bempp_cl.api.set_default_cpu_device_by_name(driver_name)
        for label, op, fun in zip(operator_labels, operators, functions):
            df.loc[label, (driver_name, precision)] = benchmark_potential_operator(op, fun, precision)
# -

results_potential_operators = df

# ## Output

print(results_boundary_operators)
print(results_potential_operators)

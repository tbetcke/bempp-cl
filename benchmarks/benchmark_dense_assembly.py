"""Benchmarks for dense assembly."""

import pytest

import bempp.api

PYTESTMARK = pytest.mark.usefixtures("default_parameters", "helpers")


# pylint: disable=C0103
def laplace_single_layer_dense_benchmark(benchmark, default_parameters):
    """Benchmark for Laplace assembly on a small sphere"""

    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(4)
    space = function_space(grid, "DP", 0)

    fun = lambda: single_layer(space, space, space, assembler="dense", parameters=default_parameters).weak_form()

    benchmark(fun)


def laplace_single_layer_dense_large_benchmark(benchmark, default_parameters):
    """Benchmark for Laplace assembly on a larger sphere"""

    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(5)
    space = function_space(grid, "DP", 0)

    fun = lambda: single_layer(space, space, space, assembler="dense", parameters=default_parameters).weak_form()

    benchmark(fun)


def laplace_single_layer_dense_p1_disc_benchmark(benchmark, default_parameters):
    """Benchmark for Laplace assembly with disc p1 functions"""

    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(4)
    space = function_space(grid, "DP", 1)

    fun = lambda: single_layer(space, space, space, assembler="dense", parameters=default_parameters).weak_form()

    benchmark(fun)


def laplace_single_layer_dense_p1_cont_benchmark(benchmark, default_parameters):
    """Benchmark for Laplace assembly with cont p1 functions"""

    from bempp.api.operators.boundary.laplace import single_layer
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(4)
    space = function_space(grid, "P", 1)

    fun = lambda: single_layer(space, space, space, assembler="dense", parameters=default_parameters).weak_form()

    benchmark(fun)


def helmholtz_single_layer_dense_p1_cont_large_benchmark(benchmark, default_parameters):
    """Helmholtz benchmark with P1 functions on large grid."""
    from bempp.api.operators.boundary.helmholtz import single_layer
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(5)
    space = function_space(grid, "P", 1)

    wavenumber = 2.5

    fun = lambda: single_layer(
        space,
        space,
        space,
        wavenumber,
        assembler="dense",
        parameters=default_parameters,
    ).weak_form()

    benchmark(fun)


def maxwell_electric_field_dense_large_benchmark(benchmark, default_parameters):
    """Maxwell electric field benchmark on large grid."""
    from bempp.api.operators.boundary.maxwell import electric_field
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(5)
    rwg = function_space(grid, "RWG", 0)
    snc = function_space(grid, "SNC", 0)

    wavenumber = 2.5

    fun = lambda: electric_field(
        rwg,
        rwg,
        snc,
        wavenumber,
        assembler="dense",
        parameters=default_parameters,
    ).weak_form()

    benchmark(fun)


def maxwell_magnetic_field_dense_large_benchmark(benchmark, default_parameters):
    """Maxwell magnetic field benchmark on large grid."""
    from bempp.api.operators.boundary.maxwell import magnetic_field
    from bempp.api import function_space

    grid = bempp.api.shapes.regular_sphere(5)
    rwg = function_space(grid, "RWG", 0)
    snc = function_space(grid, "SNC", 0)

    wavenumber = 2.5

    fun = lambda: magnetic_field(
        rwg,
        rwg,
        snc,
        wavenumber,
        assembler="dense",
        parameters=default_parameters,
    ).weak_form()

    benchmark(fun)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Weakly imposing a Dirichlet boundary condition
#
# This tutorial shows how to implement the weak imposition of a Dirichlet boundary condition, as proposed in the paper <a href='https://bempp.com/publications.html#Betcke2019'>Boundary Element Methods with Weakly Imposed Boundary Conditions (2019)</a>.
#
# First, we import Bempp and NumPy.

import bempp_cl.api
import numpy as np

# Next, we define the grid for our problem, and the function spaces that we will use. In this example, we use a sphere with P1 and DUAL0 function spaces.

h = 0.3
grid = bempp_cl.api.shapes.sphere(h=h)
p1 = bempp_cl.api.function_space(grid, "P", 1)
dual0 = bempp_cl.api.function_space(grid, "DUAL", 0)

# Next, we define the blocked operators proposed in the paper:
# $$\left(\left(\begin{array}{cc}-\mathsf{K}&\mathsf{V}\\\mathsf{W}&\mathsf{K}'\end{array}\right)+\left(\begin{array}{cc}\tfrac12\mathsf{Id}&0\\\beta\mathsf{Id}&-\tfrac12\mathsf{Id}\end{array}\right)\right)\left(\begin{array}{c}u\\\lambda\end{array}\right)=\left(\begin{array}{c}g_\text{D}\\\beta g_\text{D}\end{array}\right),$$
# where $\beta>0$ is a parameter of our choice. In this example, we use $\beta=0.1$.

# +
beta = 0.1
multi = bempp_cl.api.BlockedOperator(2, 2)
multi[0, 0] = -bempp_cl.api.operators.boundary.laplace.double_layer(p1, p1, dual0, assembler="fmm")
multi[0, 1] = bempp_cl.api.operators.boundary.laplace.single_layer(dual0, p1, dual0, assembler="fmm")
multi[1, 0] = bempp_cl.api.operators.boundary.laplace.hypersingular(p1, dual0, p1, assembler="fmm")
multi[1, 1] = bempp_cl.api.operators.boundary.laplace.adjoint_double_layer(dual0, dual0, p1, assembler="fmm")

diri = bempp_cl.api.BlockedOperator(2, 2)
diri[0, 0] = 0.5 * bempp_cl.api.operators.boundary.sparse.identity(p1, p1, dual0)
diri[1, 0] = beta * bempp_cl.api.operators.boundary.sparse.identity(p1, dual0, p1)
diri[1, 1] = -0.5 * bempp_cl.api.operators.boundary.sparse.identity(dual0, dual0, p1)
# -

# Next, we define the function $g_\text{D}$, and define the right hand side.
#
# Here, we use $$g_\text{D}=\sin(\pi x)\sin(\pi y)\sinh(\sqrt2\pi z),$$ as in section 5 of the paper.


# +
@bempp_cl.api.real_callable
def f(x, n, d, res):
    res[0] = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sinh(np.sqrt(2) * np.pi * x[2])


f_fun = bempp_cl.api.GridFunction(p1, fun=f)

rhs = [2 * diri[0, 0] * f_fun, diri[1, 0] * f_fun]
# -

# Now we solve the system. We set `use_strong_form=True` to activate mass matrix preconditioning.

sol, info, it_count = bempp_cl.api.linalg.gmres(multi + diri, rhs, return_iteration_count=True, use_strong_form=True)
print(f"Solution took {it_count} iterations")

# For this problem, we know the analytic solution. We compute the error in the $\mathcal{B}_\text{D}$ norm:


# +
@bempp_cl.api.real_callable
def g(x, n, d, res):
    grad = np.array(
        [
            np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sinh(np.sqrt(2) * np.pi * x[2]) * np.pi,
            np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.sinh(np.sqrt(2) * np.pi * x[2]) * np.pi,
            np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.cosh(np.sqrt(2) * np.pi * x[2]) * np.pi * np.sqrt(2),
        ]
    )
    res[0] = np.dot(grad, n)


g_fun = bempp_cl.api.GridFunction(dual0, fun=g)

e_fun = [sol[0] - f_fun, sol[1] - g_fun]

error = 0
# V norm
slp = bempp_cl.api.operators.boundary.laplace.single_layer(dual0, p1, dual0, assembler="fmm")
hyp = bempp_cl.api.operators.boundary.laplace.hypersingular(p1, dual0, p1, assembler="fmm")
error += np.sqrt(np.dot(e_fun[1].coefficients.conjugate(), (slp * e_fun[1]).projections(dual0)))
error += np.sqrt(np.dot(e_fun[0].coefficients.conjugate(), (hyp * e_fun[0]).projections(p1)))
# D part
error += beta**0.5 * e_fun[0].l2_norm()

print(f"Error: {error}")
# -

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Computing the capacity of a cube with a re-entrant corner
#
# ## Background
# The capacity $\text{cap}(\Omega)$ of an isolated conductor $\Omega\subset\mathbb{R}^3$ with boundary $\Gamma$ measures its ability to store charges. It is defined as the ratio of the total surface equilibrium charge relative to its surface potential value. To compute the capacity, we need to solve the following exterior Laplace problem for the equilibrium potential $u$ with unit surface value:
# $$
# \begin{align}
# -\Delta u &= 0\quad\text{in }\Omega^\text{+},\\
# u &= 1\quad\text{on }\Gamma,\\
# |u(\mathbf{x})| &=\mathcal{O}\left(|\mathbf{x}|^{-1}\right)\quad\text{as }|\mathbf{x}|\rightarrow\infty.
# \end{align}
# $$
# Here $\Omega^\text{+}$ is the domain exterior to $\Omega$.
# The total surface charge of an isolated conductor is given by Gauss law as
# $$
# \text{cap}(\Omega)=-\epsilon_0\int_{\Gamma}\frac{\partial u}{\partial\nu}(\mathbf{x})\,\mathrm{d}\mathbf{x}.
# $$
# $\nu(\mathbf{x})$ is the outward pointing normal direction for $\mathbf{x}\in\Gamma$, and $\epsilon_0$ is the electric constant with value $\epsilon_0\approx 8.854\times 10^{-12}\,{\rm F/m}$. In the following we will use the normalized capacity $\text{cap}^*(\Omega)=-\frac{1}{4\pi}\int_{\Gamma}\frac{\partial u}{\partial\nu}\,d\mathbf{x}$. The normalized capacity has the value $1$ for the unit sphere.
#
# Using Green's representation theorem and noting that the exterior Laplace double layer potential is zero for constant densities, we can represent the solution $u$ as
# $$
# u(\mathbf{x}) = -\int_{\Gamma} g(\mathbf{x},\mathbf{y})\phi(\mathbf{y})\,\mathrm{d}\mathbf{y} \quad\text{for all }\mathbf{x}\in\Omega^\text{+},
# $$
# where $\phi:={\partial u}/{\partial\nu}$ is the normal derivative of the exterior solution $u$ and $g(\mathbf{x},\mathbf{y}):=\frac{1}{4\pi|\mathbf{x}-\mathbf{y}|}$ is the Green's function of the 3D Laplacian. By taking boundary traces, we arrive at the following boundary integral equation of the first kind.
# $$
# 1 = -\int_{\Gamma} g(\mathbf{x},\mathbf{y})\phi(\mathbf{y})\,\mathrm{d}\mathbf{y} =: -\mathsf{V}\phi(\mathbf{x})\quad\text{for all }\mathbf{x}\in\Gamma.
# $$
# The normalized capacity is now simply given by
# $$
# \text{cap}^*(\Omega) = -\frac{1}{4\pi}\int_\Gamma \phi(\mathbf{x}) \,\mathrm{d}\mathbf{x}.
# $$
#
# ## Implementation
# We start with the usual imports.

import bempp_cl.api
import numpy as np

# The grid re-entrant cube is predefined in the shapes module. By default it refines towards the singular corner. As function space on the grid we choose a simple space of piecewise constant functions.

grid = bempp_cl.api.shapes.reentrant_cube(h=0.02, refinement_factor=1)
space = bempp_cl.api.function_space(grid, "DP", 0)

# Next, we define the right-hand side.


# +
@bempp_cl.api.real_callable
def one_fun(x, n, domain_index, res):
    res[0] = 1


rhs = bempp_cl.api.GridFunction(space, fun=one_fun)
# -

# The following code defines the left-hand side single-layer boundary operator.

op = bempp_cl.api.operators.boundary.laplace.single_layer(space, space, space)

# We use GMRES to solve the system. To improve convergence we use a strong form discretisation that automatically preconditions with the mass matrix.

# +
sol, _, iteration_count = bempp_cl.api.linalg.gmres(op, rhs, use_strong_form=True, return_iteration_count=True)

print("Number of iterations: {0}".format(iteration_count))
# -

# To obtain the capacity we simply integrate the solution across the boundary.

normalized_capacity = 1.0 / (4 * np.pi) * sol.integrate()[0]
print("The normalized capacity is {0}.".format(normalized_capacity))

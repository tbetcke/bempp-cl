"""Bempp direct solver interface."""

# pylint: disable=invalid-name


def compute_lu_factors(A):
    """
    Precompute the LU factors of a dense operator A.

    This function returns a tuple of LU factors of A.
    This tuple can be used in the `lu_factor` attribute
    of the lu function so that the LU decomposition is
    not recomputed at each call to lu.

    """
    from bempp_cl.api import as_matrix
    from scipy.linalg import lu_factor

    return lu_factor(as_matrix(A.weak_form()))


def lu(A, b, lu_factor=None):
    """Perform an LU solve.

    This function takes an operator and a grid function,
    converts the operator into a dense matrix and solves
    the system via LU decomposition. The result is again
    returned as a grid function.

    Parameters
    ----------
    A : bempp_cl.api.BoundaryOperator
         The left-hand side boundary operator
    b : bempp_cl.api.GridFunction
         The right-hand side grid function
    lu_decomp : tuple
         Optionally pass the tuple (lu, piv)
         obtained by the scipy method scipy.linalg.lu_factor

    """
    from bempp_cl.api import GridFunction
    from scipy.linalg import solve, lu_solve
    from bempp_cl.api.assembly.blocked_operator import BlockedOperatorBase
    from bempp_cl.api.assembly.blocked_operator import projections_from_grid_functions_list
    from bempp_cl.api.assembly.blocked_operator import grid_function_list_from_coefficients

    if isinstance(A, BlockedOperatorBase):
        vec = projections_from_grid_functions_list(b, A.dual_to_range_spaces)
        if lu_factor is not None:
            sol = lu_solve(lu_factor, vec)
        else:
            mat = A.weak_form().to_dense()
            sol = solve(mat, vec)
        return grid_function_list_from_coefficients(sol, A.domain_spaces)
    else:
        vec = b.projections(A.dual_to_range)
        if lu_factor is not None:
            sol = lu_solve(lu_factor, vec)
        else:
            mat = A.weak_form().to_dense()
            sol = solve(mat, vec)
        return GridFunction(A.domain, coefficients=sol)

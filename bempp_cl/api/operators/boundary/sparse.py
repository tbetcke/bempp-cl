"""Interfaces to Laplace operators."""

from bempp_cl.api.operators.boundary import common as _common


def identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the L^2 identity operator."""
    return _common.create_operator(
        "l2_identity",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "l2_identity",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def _vector_grad_product(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the inner product between a vector field and grad of a P1 basis function."""
    return _common.create_operator(
        "_vector_grad_product",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "_vector_grad_product",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def _curl_curl_product(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble inner product between the surface curl of 2 SNC basis functions."""
    return _common.create_operator(
        "_curl_curl_product",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "_curl_curl_product",
        "default_sparse",
        device_interface,
        precision,
        False,
    )


def multitrace_identity(multitrace_operator, parameters=None, device_interface=None, precision=None):
    """
    Create a multitrace identity operator.

    Parameters
    ----------
    multitrace_operator : Bempp Operator
        A 2 x 2 multitrace operator object whose spaces are used to define
        the identity operator.

    Output
    ------
    A block-diagonal multitrace identity operator.

    """
    from bempp_cl.api.assembly.blocked_operator import BlockedOperator

    domain0, domain1 = multitrace_operator.domain_spaces
    dual_to_range0, dual_to_range1 = multitrace_operator.dual_to_range_spaces
    range0, range1 = multitrace_operator.range_spaces

    blocked_operator = BlockedOperator(2, 2)

    blocked_operator[0, 0] = identity(domain0, range0, dual_to_range0)
    blocked_operator[1, 1] = identity(domain1, range1, dual_to_range1)

    return blocked_operator


def mte_operators(domains_, ranges_, dual_to_ranges_, kappa):
    """
    Create basic sparse operators to assemble Pade approximate MtE operators.

    Parameters
    ----------
    domains_
        Domain spaces
    ranges_
        Range spaces
    dual_to_ranges_
        Dual spaces

    Output
    ------
    Basic sparse operators to assemble Pade approximate MtE operators.

    IP
        Identity operator built with piecewise linear basis functions
    IC
        Identity operator built with Hcurl conforming basis functions
    N
        Scaled curlcurl operator
    L
        Grad u dot v operator
    LT
        Transpose of L

    """
    IP = identity(domains_[1], ranges_[1], dual_to_ranges_[1])
    IC = identity(domains_[0], ranges_[0], dual_to_ranges_[0])
    N = (1.0 / kappa) ** 2 * _curl_curl_product(domains_[0], ranges_[0], dual_to_ranges_[0])
    LT = _vector_grad_product(domains_[1], ranges_[0], dual_to_ranges_[0])
    L = LT._transpose(LT._domain)
    return IP, IC, N, LT, L


def lambda_1(mte_operators, beta, kappa_eps):
    """Create and return block Pade approximate operator to Lambda1 = (I+Delta)^(1/2).

    Parameters
    ----------
    mte_operators
        Basic sparse Pade approximate MtE operators.
    beta
        Array containing the division between Pade coefficients: A_j/B_j
    kappa
        Damped wavenumber kappa_eps = kappa + i eps

    Output
    ------
    Block operator that approximates (I+Delta)^(1/2)
    """
    from scipy.sparse import bmat
    from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator

    IP, IC, N, LT, L = mte_operators
    return InverseSparseDiscreteBoundaryOperator(
        bmat(
            [
                [(IC - beta * N).weak_form().to_sparse(), (beta * LT).weak_form().to_sparse()],
                [L.weak_form().to_sparse(), kappa_eps**2 * IP.weak_form().to_sparse()],
            ],
            "csc",
        )
    )


def lambda_2(mte_operators):
    """Create and return Lambda2 = (I-curlcurl) operator.

    Parameters
    ----------
    mte_operators

    Output
    ------
    IC-N = (I-curlcurl)

    """
    IP, IC, N, LT, L = mte_operators
    return IC - N


def sigma_identity(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """
    Evaluate the sigma identity operator.

    For Galerkin methods this operator is equivalent to .5 * identity. For
    collocation methods the value may differ from .5 on piecewise smooth
    domains.
    """
    from bempp_cl.api.utils.helpers import assign_parameters

    parameters = assign_parameters(parameters)

    if parameters.assembly.discretization_type == "galerkin":
        return 0.5 * identity(
            domain,
            range_,
            dual_to_range,
            parameters=parameters,
            device_interface=device_interface,
            precision=precision,
        )
    elif parameters.assembly.discretization_type == "collocation":
        raise ValueError("Not yet implemented.")


def laplace_beltrami(
    domain,
    range_,
    dual_to_range,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the negative Laplace-Beltrami operator."""
    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    return _common.create_operator(
        "laplace_beltrami",
        domain,
        range_,
        dual_to_range,
        parameters,
        "sparse",
        [],
        "laplace_beltrami",
        "default_sparse",
        device_interface,
        precision,
        False,
    )

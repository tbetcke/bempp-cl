"""Interfaces to Maxwell operators."""

import numpy as _np

from bempp_cl.api.operators.boundary import common as _common


def electric_field(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the electric field boundary operator."""
    if domain.identifier != "rwg0":
        raise ValueError("Domain space must be an RWG type function space.")

    if dual_to_range.identifier != "snc0":
        raise ValueError("Dual to range space must be an SNC type function space.")

    return _common.create_operator(
        "maxwell_electric_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "maxwell_electric_field",
        device_interface,
        precision,
        True,
    )


def magnetic_field(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the magnetic field boundary operator."""
    if domain.identifier != "rwg0":
        raise ValueError("Domain space must be an RWG type function space.")

    if dual_to_range.identifier != "snc0":
        raise ValueError("Dual to range space must be an SNC type function space.")

    return _common.create_operator(
        "maxwell_magnetic_field_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "maxwell_magnetic_field",
        device_interface,
        precision,
        True,
    )


def multitrace_operator(
    grid,
    wavenumber,
    epsilon_r=1,
    mu_r=1,
    target=None,
    space_type="magnetic_dual",
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """
    Simplified version of multitrace operator assembly.

    Parameters
    ----------
    grid : Grid
        Bempp grid object.
    wavenumber : complex
        A real or complex wavenumber
    epsilon_r : float
        Relative permittivity with respect to vacuum.
    mu_r : float
        Relative permeability with respect to vacuum.
    target : Grid
        The grid for the range spaces. If target is None then
        target is set to the input grid (that is the domain
        grid).
    space_type : string
        One of "all_rwg", "all_bc", "magnetic_dual" (default),
        "electric_dual". These lead to the following
        choices of space, range, and dual_to_range:
        default - (RWG, RWG), (BC, BC), (SNC, SNC)
        all_dual - (BC, BC), (RWG, RWG), (RBC, RBC)
        magnetic_dual - (RWG, BC), (RWG, BC), (RBC, SNC)
        electric_dual - (BC, RWG), (BC, RWG), (SNC, RBC)
    parameters : Parameters
        An optional parameters object.
    assembler : string
        The assembler type.
    device_interface : DeviceInterface
        The device interface object to be used.
    precision : string
        Either "single" or "double" for single or
        double precision mode.

    Output
    ------
    The Maxwell multitrace operator of the form
    [[M, E], [-E, M]], where M represens the magnetic
    and E the electric field boundary operators in
    the respective spaces defined through space_type.
    Note that the operators in the first and second
    row have different discretisations depending on
    the type of spaces used.

    """
    import bempp_cl.api

    if space_type == "all_rwg":
        rwg = bempp_cl.api.function_space(grid, "RWG", 0)
        bc = bempp_cl.api.function_space(grid, "BC", 0)
        snc = bempp_cl.api.function_space(grid, "SNC", 0)

        if target is not None:
            bc_target = bempp_cl.api.function_space(target, "BC", 0)
            snc_target = bempp_cl.api.function_space(target, "SNC", 0)
        else:
            bc_target = bc
            snc_target = snc

        domain = [rwg, rwg]
        range_ = [bc_target, bc_target]
        dual_to_range = [snc_target, snc_target]
    elif space_type == "all_bc":
        bc = bempp_cl.api.function_space(grid, "BC", 0)
        rwg = bempp_cl.api.function_space(grid, "RWG", 0)
        rbc = bempp_cl.api.function_space(grid, "RBC", 0)

        if target is not None:
            rwg_target = bempp_cl.api.function_space(target, "RWG", 0)
            rbc_target = bempp_cl.api.function_space(target, "RBC", 0)
        else:
            rwg_target = rwg
            rbc_target = rbc

        domain = [bc, bc]
        range_ = [rwg_target, rwg_target]
        dual_to_range = [rbc_target, rbc_target]

    elif space_type == "electric_dual":
        rwg = bempp_cl.api.function_space(grid, "RWG", 0)
        snc = bempp_cl.api.function_space(grid, "SNC", 0)
        bc = bempp_cl.api.function_space(grid, "BC", 0)
        rbc = bempp_cl.api.function_space(grid, "RBC", 0)

        if target is not None:
            rwg_target = bempp_cl.api.function_space(target, "RWG", 0)
            snc_target = bempp_cl.api.function_space(target, "SNC", 0)
            bc_target = bempp_cl.api.function_space(target, "BC", 0)
            rbc_target = bempp_cl.api.function_space(target, "RBC", 0)
        else:
            rwg_target = rwg
            snc_target = snc
            bc_target = bc
            rbc_target = rbc

        domain = [rwg, bc]
        range_ = [rwg_target, bc_target]
        dual_to_range = [rbc_target, snc_target]

    elif space_type == "magnetic_dual":
        rwg = bempp_cl.api.function_space(grid, "RWG", 0)
        snc = bempp_cl.api.function_space(grid, "SNC", 0)
        bc = bempp_cl.api.function_space(grid, "BC", 0)
        rbc = bempp_cl.api.function_space(grid, "RBC", 0)

        if target is not None:
            rwg_target = bempp_cl.api.function_space(target, "RWG", 0)
            snc_target = bempp_cl.api.function_space(target, "SNC", 0)
            bc_target = bempp_cl.api.function_space(target, "BC", 0)
            rbc_target = bempp_cl.api.function_space(target, "RBC", 0)
        else:
            rwg_target = rwg
            snc_target = snc
            bc_target = bc
            rbc_target = rbc

        domain = [bc, rwg]
        range_ = [bc_target, rwg_target]
        dual_to_range = [snc_target, rbc_target]

    else:
        raise ValueError("space_type must be one of 'all_rwg', 'all_dual', 'electric_dual', 'magnetic_dual'")
    return _multitrace_operator_impl(
        domain,
        range_,
        dual_to_range,
        wavenumber,
        epsilon_r=epsilon_r,
        mu_r=mu_r,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )


def _multitrace_operator_impl(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    epsilon_r=1,
    mu_r=1,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    from bempp_cl.api.assembly.blocked_operator import BlockedOperator

    rho = _np.sqrt(epsilon_r / mu_r)

    magnetic1 = magnetic_field(
        domain[0],
        range_[0],
        dual_to_range[0],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    magnetic2 = magnetic_field(
        domain[1],
        range_[1],
        dual_to_range[1],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    electric1 = electric_field(
        domain[1],
        range_[0],
        dual_to_range[0],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    electric2 = electric_field(
        domain[0],
        range_[1],
        dual_to_range[1],
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)
    blocked[0, 0] = magnetic1
    blocked[0, 1] = (1.0 / rho) * electric1
    blocked[1, 0] = -rho * electric2
    blocked[1, 1] = magnetic2

    return blocked


class _OsrcMtE:
    """Implementation of the OSRC DtN operator."""

    def __init__(
        self, type, domains, ranges, dual_to_ranges, parameters, operator_options, device_interface=None, precision=None
    ):
        from bempp_cl.api.operators import OperatorDescriptor

        self._device_interface = device_interface
        self.type = type
        self._operator_descriptor = OperatorDescriptor(
            "osrc_mte",
            operator_options,
            "vectorial_laplace_beltrami",
            "default_sparse",
            precision,
            True,
            None,
            1,
        )

        self.domains_ = domains
        self.ranges_ = ranges
        self.dual_to_ranges_ = dual_to_ranges
        self.coeff = 0

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _matvec1(self, v):
        import itertools

        res = 0
        rhs = list(itertools.chain(*[v.tolist(), _np.zeros(self.pi[0].shape[0] - v.shape[0]).tolist()]))
        for element in self.pi:
            res -= element.matvec(rhs)[0 : self.domains_[0].global_dof_count]
        res = self.mass * res + self.pade_coeffs[3] * v
        return self.lambda_2_inv * res

    def _matvec2(self, v):
        res = self.coeff * self.mass * v + self.pade_coeffs[3] * v
        return self.lambda_2_inv * res

    def _assemble(self):
        """Assemble the operator."""
        from bempp_cl.api.operators.boundary.sparse import lambda_1
        from bempp_cl.api.operators.boundary.sparse import lambda_2
        from bempp_cl.api.operators.boundary.sparse import mte_operators
        from bempp_cl.api.assembly.discrete_boundary_operator import InverseSparseDiscreteBoundaryOperator
        from scipy.sparse.linalg import LinearOperator

        wavenumber, npade, theta, damped_wavenumber = self.descriptor.options

        if damped_wavenumber is None:
            dk = wavenumber + 1.0j * 0.39 * wavenumber ** (1.0 / 3) * _np.sqrt(2) ** (2.0 / 3)
        else:
            dk = damped_wavenumber

        mte_op = mte_operators(self.domains_, self.ranges_, self.dual_to_ranges_, dk)

        self.lambda_2_inv = InverseSparseDiscreteBoundaryOperator(lambda_2(mte_op).weak_form())

        self.pade_coeffs = _common.pade_coeffs(npade, theta)
        self.mass = mte_op[1].weak_form()
        if self.type == 1:
            self.pi = []
            for i in range(npade):
                self.pi.append(
                    (self.pade_coeffs[1][i] / self.pade_coeffs[2][i]) * lambda_1(mte_op, self.pade_coeffs[2][i], dk)
                )
            return LinearOperator(self.lambda_2_inv.shape, matvec=self._matvec1)
        else:
            for j in range(int(_np.floor(4.0 * npade / 5)), npade):
                self.coeff += self.pade_coeffs[1][j]
            return LinearOperator(self.lambda_2_inv.shape, matvec=self._matvec2)


def osrc_mte(
    domains,
    ranges,
    dual_to_ranges,
    wavenumber,
    npade=2,
    theta=_np.pi / 2.0,
    type=1,
    damped_wavenumber=None,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the OSRC approximation to the NtD operator."""
    if domains[0].identifier != "snc0":
        raise ValueError("Domain space must be a SNC type function space.")

    if domains[1].shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain space must be of type 'p1_discontinuous'.")

    if ranges[0].identifier != "snc0":
        raise ValueError("Domain space must be a SNC type function space.")

    if ranges[1].shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain space must be of type 'p1_discontinuous'.")

    if dual_to_ranges[0].identifier != "snc0":
        raise ValueError("Domain space must be a SNC type function space.")

    if dual_to_ranges[1].shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain space must be of type 'p1_discontinuous'.")

    return _OsrcMtE(
        type,
        domains,
        ranges,
        dual_to_ranges,
        parameters,
        [wavenumber, npade, theta, damped_wavenumber],
        device_interface,
        precision,
    )

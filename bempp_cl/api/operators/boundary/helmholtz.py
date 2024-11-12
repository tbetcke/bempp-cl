"""Interfaces to Helmholtz operators."""

import numpy as _np

from bempp_cl.api.operators.boundary import common as _common
from bempp_cl.api.assembly.boundary_operator import BoundaryOperator as _BoundaryOperator


def single_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz single-layer boundary operator."""
    from .modified_helmholtz import single_layer as _modified_single_layer

    if _np.real(wavenumber) == 0:
        return _modified_single_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_single_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz double-layer boundary operator."""
    from .modified_helmholtz import double_layer as _modified_double_layer

    if _np.real(wavenumber) == 0:
        return _modified_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def adjoint_double_layer(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz adj. double-layer boundary operator."""
    from .modified_helmholtz import (
        adjoint_double_layer as _modified_adjoint_double_layer,
    )

    if _np.real(wavenumber) == 0:
        return _modified_adjoint_double_layer(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_adjoint_double_layer_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_adjoint_double_layer",
        "default_scalar",
        device_interface,
        precision,
        True,
    )


def hypersingular(
    domain,
    range_,
    dual_to_range,
    wavenumber,
    parameters=None,
    assembler="default_nonlocal",
    device_interface=None,
    precision=None,
):
    """Assemble the Helmholtz hypersingular boundary operator."""
    from .modified_helmholtz import hypersingular as _hypersingular

    if domain.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Domain shapeset must be of type 'p1_discontinuous'.")

    if dual_to_range.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Dual to range shapeset must be of type 'p1_discontinuous'.")

    if _np.real(wavenumber) == 0:
        return _hypersingular(
            domain,
            range_,
            dual_to_range,
            _np.imag(wavenumber),
            parameters,
            assembler,
            device_interface,
            precision,
        )

    return _common.create_operator(
        "helmholtz_hypersingular_boundary",
        domain,
        range_,
        dual_to_range,
        parameters,
        assembler,
        [_np.real(wavenumber), _np.imag(wavenumber)],
        "helmholtz_single_layer",
        "helmholtz_hypersingular",
        device_interface,
        precision,
        True,
    )


def multitrace_operator(
    grid,
    wavenumber,
    target=None,
    space_type="p1",
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
    target : Grid
        The grid for the range spaces. If target is None then
        target is set to the input grid (that is the domain
        grid).
    space_type : string
        Controls which discretisation spaces are used.
        Supported values:
        - "p1", which means that all operators are discretised
        with P1 basis functions.
        - "p1-dp0", which means that P1 basis functions are
        used for the first row and column, and DP0 for the second.
        - "p1-dual0", which means that P1 basis functions are
        used for the first row and column, and DUAL0 for the second.
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
    The Helmholtz multitrace operator of the form
    [[-dlp, slp], [hyp, adj_dlp]], where
    dlp : double layer boundary operator
    slp : single layer boundary operator
    hyp : hypersingular boundary operator
    adj_dlp : adjoint double layer boundary operator.

    """
    import bempp_cl.api
    from bempp_cl.api.assembly.blocked_operator import BlockedOperator

    if space_type == "p1":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = space1
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = target_space1
            target_space_dual1 = target_space1
            target_space_dual0 = target_space1
        else:
            target_space1 = space1
            target_space0 = space1
            target_space_dual1 = space1
            target_space_dual0 = space1
    elif space_type == "p1-dp0":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = bempp_cl.api.function_space(grid, "DP", 0)
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = bempp_cl.api.function_space(target, "DP", 0)
            target_space_dual1 = target_space1
            target_space_dual0 = target_space0
    elif space_type == "p1-dual":
        space1 = bempp_cl.api.function_space(grid, "P", 1)
        space0 = bempp_cl.api.function_space(grid, "DUAL", 0)
        if target is not None:
            target_space1 = bempp_cl.api.function_space(target, "P", 1)
            target_space0 = bempp_cl.api.function_space(target, "DUAL", 0)
            target_space_dual1 = target_space1
            target_space_dual0 = target_space0
    else:
        raise ValueError(f"Unknown space type: {space_type}")

    slp = single_layer(
        space0,
        target_space1,
        target_space_dual0,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    dlp = double_layer(
        space1,
        target_space1,
        target_space_dual0,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    hyp = hypersingular(
        space1,
        target_space0,
        target_space_dual1,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    adj_dlp = adjoint_double_layer(
        space0,
        target_space0,
        target_space_dual1,
        wavenumber,
        parameters=parameters,
        assembler=assembler,
        device_interface=device_interface,
        precision=precision,
    )

    blocked = BlockedOperator(2, 2)

    blocked[0, 0] = -dlp
    blocked[0, 1] = slp
    blocked[1, 0] = hyp
    blocked[1, 1] = adj_dlp

    return blocked


def osrc_dtn(
    space,
    wavenumber,
    npade=2,
    theta=_np.pi / 3.0,
    damped_wavenumber=None,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the OSRC approximation to the DtN operator."""
    if space.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Space shapeset must be of type 'p1_discontinuous'.")

    return _OsrcDtN(
        space,
        parameters,
        [wavenumber, npade, theta, damped_wavenumber],
        device_interface,
        precision,
    )


class _OsrcDtN(_BoundaryOperator):
    """Implementation of the OSRC DtN operator."""

    def __init__(self, space, parameters, operator_options, device_interface=None, precision=None):
        from bempp_cl.api.operators import OperatorDescriptor

        super().__init__(space, space, space, parameters)

        self._device_interface = device_interface

        self._operator_descriptor = OperatorDescriptor(
            "osrc_dtn",
            operator_options,
            "laplace_beltrami",
            "default_sparse",
            precision,
            True,
            None,
            1,
        )

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _assemble(self):
        """Assemble the operator."""
        from bempp_cl.api.operators.boundary.sparse import identity
        from bempp_cl.api.operators.boundary.sparse import laplace_beltrami
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        space = self._domain
        wavenumber, npade, theta, damped_wavenumber = self.descriptor.options

        mass = identity(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()
        stiff = laplace_beltrami(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()

        if damped_wavenumber is None:
            bbox = space.grid.bounding_box
            rad = _np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2.0
            dk = wavenumber + 0.4j * wavenumber ** (1.0 / 3.0) * rad ** (-2.0 / 3.0)
        else:
            dk = damped_wavenumber

        c0, alpha, beta, _ = _common.pade_coeffs(npade, theta)

        series = c0 * mass
        for i in range(npade):
            element = (
                alpha[i] / (dk**2) * stiff * InverseSparseDiscreteBoundaryOperator(mass - beta[i] / (dk**2) * stiff)
            )
            series -= element * mass
        operator = 1.0j * wavenumber * series

        return operator


def osrc_ntd(
    space,
    wavenumber,
    npade=2,
    theta=_np.pi / 3.0,
    damped_wavenumber=None,
    parameters=None,
    device_interface=None,
    precision=None,
):
    """Assemble the OSRC approximation to the NtD operator."""
    if space.shapeset.identifier != "p1_discontinuous":
        raise ValueError("Space shapeset must be of type 'p1_discontinuous'.")

    return _OsrcNtD(
        space,
        parameters,
        [wavenumber, npade, theta, damped_wavenumber],
        device_interface,
        precision,
    )


class _OsrcNtD(_BoundaryOperator):
    """Implementation of the OSRC NtD operator."""

    def __init__(self, space, parameters, operator_options, device_interface=None, precision=None):
        from bempp_cl.api.operators import OperatorDescriptor

        super().__init__(space, space, space, parameters)

        self._device_interface = device_interface

        self._operator_descriptor = OperatorDescriptor(
            "osrc_ntd",
            operator_options,
            "laplace_beltrami",
            "default_sparse",
            precision,
            True,
            None,
            1,
        )

    @property
    def descriptor(self):
        """Operator descriptor."""
        return self._operator_descriptor

    def _assemble(self):
        from bempp_cl.api.operators.boundary.sparse import identity
        from bempp_cl.api.operators.boundary.sparse import laplace_beltrami
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            InverseSparseDiscreteBoundaryOperator,
        )

        space = self._domain
        wavenumber, npade, theta, damped_wavenumber = self.descriptor.options

        mass = identity(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()
        stiff = laplace_beltrami(
            space,
            space,
            space,
            self._parameters,
            self._device_interface,
            self.descriptor.precision,
        ).weak_form()

        if damped_wavenumber is None:
            bbox = space.grid.bounding_box
            rad = _np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2.0
            dk = wavenumber + 0.4j * wavenumber ** (1.0 / 3.0) * rad ** (-2.0 / 3.0)
        else:
            dk = damped_wavenumber

        c0, alpha, beta, _ = _common.pade_coeffs(npade, theta)

        series = c0 * mass
        for i in range(npade):
            element = (
                alpha[i] / (dk**2) * stiff * InverseSparseDiscreteBoundaryOperator(mass - beta[i] / (dk**2) * stiff)
            )
            series -= element * mass
        operator = (
            1.0
            / (1.0j * wavenumber)
            * (mass * InverseSparseDiscreteBoundaryOperator(mass - 1.0 / (dk**2) * stiff) * series)
        )

        return operator

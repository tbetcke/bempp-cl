"""Various assemblers to discretize boundary operators."""


def _create_assembler(domain, dual_to_range, identifier, parameters, device_interface=None):
    """Create assembler based on string."""
    from bempp_cl.core.singular_assembler import SingularAssembler
    from bempp_cl.core.dense_assembler import DenseAssembler
    from bempp_cl.core.diagonal_assembler import DiagonalAssembler
    from bempp_cl.api.fmm.fmm_assembler import FmmAssembler
    from bempp_cl.api import check_for_fmm

    # from bempp_cl.core.numba.dense_assembler import DenseAssembler
    from bempp_cl.core.sparse_assembler import SparseAssembler

    # from bempp_cl.core.dense_assembler import DenseAssembler
    # from bempp_cl.core.sparse_assembler import SparseAssembler
    # from bempp_cl.core.dense_evaluator import DenseEvaluatorAssembler
    # from bempp_cl.core.dense_multitrace_evaluator import DenseMultitraceEvaluatorAssembler

    if identifier == "only_singular_part":
        return SingularAssembler(domain, dual_to_range, parameters)
    if identifier == "only_diagonal_part":
        return DiagonalAssembler(domain, dual_to_range, parameters)
    if identifier == "dense":
        return DenseAssembler(domain, dual_to_range, parameters)
    if identifier == "default_nonlocal":
        return DenseAssembler(domain, dual_to_range, parameters)
    if identifier == "sparse":
        return SparseAssembler(domain, dual_to_range, parameters)
    if identifier == "fmm":
        if not check_for_fmm():
            raise ValueError("No compatible FMM library found. Please install Exafmm from github.com/exafmm/exafmm-t.")
        return FmmAssembler(domain, dual_to_range, parameters)
    else:
        raise ValueError("Unknown assembler type.")
    # if identifier == "dense_evaluator":
    # return DenseEvaluatorAssembler(domain, dual_to_range, parameters)
    # if identifier == "multitrace_evaluator":
    # return DenseMultitraceEvaluatorAssembler(domain, dual_to_range, parameters)


class AssemblerInterface(object):
    """Default Assembler interface object."""

    def __init__(
        self,
        domain,
        dual_to_range,
        assembler,
        device_interface,
        precision,
        parameters=None,
    ):
        """Initialize assembler based on assembler_type string."""
        import bempp_cl.api
        import bempp_cl.api as _api

        self._domain = domain
        self._dual_to_range = dual_to_range
        self._parameters = _api.assign_parameters(parameters)
        self._device_interface = device_interface
        self._precision = precision

        if self._device_interface is None:
            self._device_interface = bempp_cl.api.DEFAULT_DEVICE_INTERFACE

        if not isinstance(assembler, str):
            self._implementation = assembler
        else:
            self._implementation = _create_assembler(
                domain, dual_to_range, assembler, self.parameters, device_interface
            )

    @property
    def domain(self):
        """Return domain space."""
        return self._domain

    @property
    def dual_to_range(self):
        """Return dual to range space."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    def assemble(self, operator_descriptor, *args, **kwargs):
        """Assemble the operator."""
        return self._implementation.assemble(
            operator_descriptor,
            self._device_interface,
            self._precision,
            *args,
            **kwargs,
        )


class AssemblerBase(object):
    """Base class for assemblers."""

    def __init__(self, domain, dual_to_range, parameters=None):
        """Instantiate the base class."""
        import bempp_cl.api as api

        self._domain = domain
        self._dual_to_range = dual_to_range
        self._parameters = api.assign_parameters(parameters)

    @property
    def domain(self):
        """Return domain."""
        return self._domain

    @property
    def dual_to_range(self):
        """Return dual to range."""
        return self._dual_to_range

    @property
    def parameters(self):
        """Return parameters."""
        return self._parameters

    def assemble(self, operator_descriptor, *args, **kwargs):
        """Assemble the operator."""
        raise NotImplementedError("Needs to be implemented by derived class.")


class PotentialAssembler(object):
    """Base class for potential assemblers."""

    def __init__(
        self,
        space,
        points,
        operator_descriptor,
        device_interface,
        assembler,
        parameters,
    ):
        """Interface for potential operators."""
        self.space = space
        self.points = points
        self.kernel_dimension = operator_descriptor.kernel_dimension
        self._is_complex = operator_descriptor.is_complex

        self._implementation = select_potential_implementation(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )

    def evaluate(self, x):
        """Evaluate the potential."""
        import numpy as np

        if not self._is_complex:
            if np.iscomplexobj(x):
                return self._implementation.evaluate(np.real(x)) + 1j * self._implementation.evaluate(np.imag(x))
            else:
                return self._implementation.evaluate(x)
        else:
            return self._implementation.evaluate(x)


def select_potential_implementation(space, points, operator_descriptor, device_interface, assembler, parameters):
    """Select a potential operator implementation."""
    import bempp_cl.api

    parameters = bempp_cl.api.assign_parameters(parameters)
    if device_interface is None:
        device_interface = bempp_cl.api.DEFAULT_DEVICE_INTERFACE

    if assembler == "dense":
        from bempp_cl.core.dense_potential_assembler import DensePotentialAssembler

        return DensePotentialAssembler(space, operator_descriptor, points, device_interface, parameters)
    elif assembler == "fmm":
        from bempp_cl.api.fmm.fmm_assembler import FmmPotentialAssembler

        if not bempp_cl.api.check_for_fmm():
            raise ValueError("No compatible FMM library found. Please install Exafmm from github.com/exafmm/exafmm-t.")

        return FmmPotentialAssembler(space, operator_descriptor, points, device_interface, parameters)
    else:
        raise ValueError(f"Unknown potential assembler: {assembler}")

"""Diagonal assembly."""

from bempp_cl.api.assembly import assembler as _assembler


class DiagonalAssembler(_assembler.AssemblerBase):
    """Implementation of a diagonal assembler for integral operators."""

    # pylint: disable=useless-super-delegation
    def __init__(self, domain, dual_to_range, parameters=None):
        """Create a diagonal assembler instance."""
        super().__init__(domain, dual_to_range, parameters)

    def assemble(self, operator_descriptor, device_interface, precision, *args, **kwargs):
        """Diagonal assembly of the integral operator."""
        from bempp_cl.api.assembly.discrete_boundary_operator import DiagonalOperator

        if self.domain.requires_dof_transformation or self.dual_to_range.requires_dof_transformation:
            raise ValueError("Spaces that require dof transformations not supported for diagonal assembly.")

        if self.domain != self.dual_to_range:
            raise ValueError("Only identical spaces currently supported for diagonal assembly.")

        if (
            self.domain.identifier == self.dual_to_range.identifier and self.dual_to_range.identifier == "p1_continuous"
        ) or (
            self.domain.identifier == self.dual_to_range.identifier
            and self.dual_to_range.identifier == "p0_discontinuous"
        ):
            values = operator_descriptor.singular_part.weak_form().to_sparse().diagonal()
        else:
            raise ValueError(
                "Only spaces of type 'p0_discontinuous' or 'p1_continuous' supported for diagonal assembly."
            )

        return DiagonalOperator(
            values,
            shape=(self.dual_to_range.global_dof_count, self.domain.global_dof_count),
        )

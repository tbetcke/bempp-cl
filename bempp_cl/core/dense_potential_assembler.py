"""Implementation of potential operators."""


class DensePotentialAssembler(object):
    """Implementation of a potential assembler."""

    # pylint: disable=useless-super-delegation
    def __init__(
        self,
        space,
        operator_descriptor,
        points,
        device_interface,
        parameters=None,
    ):
        """Create a dense assembler instance."""
        from bempp_cl.core.dispatcher import potential_dispatcher

        implementation = potential_dispatcher(
            device_interface,
            space.localised_space,
            operator_descriptor,
            points,
            parameters,
        )

        self.space = space
        kernel_dimension = operator_descriptor.kernel_dimension

        def potential_evaluator(x):
            """Evaluate the potential."""
            x_transformed = self.space.map_to_full_grid @ (self.space.dof_transformation @ x)
            result = implementation(x_transformed)
            return result.reshape([kernel_dimension, -1], order="F")

        self._evaluator = potential_evaluator

    def evaluate(self, x):
        """Call the potential evaluator."""
        return self._evaluator(x)

"""Dispatch kernel calls to different implementations."""


def singular_assembler_dispatcher(device_interface, *args):
    """Dispatch the singular assembler to the different implementations."""
    interface_type = device_interface.split("_")[0]

    if interface_type == "opencl":
        from bempp_cl.core.opencl_assemblers import singular_assembler

        singular_assembler(device_interface, *args)

    elif interface_type == "numba":
        from bempp_cl.core.numba_assemblers import singular_assembler

        singular_assembler(device_interface, *args)

    else:
        raise ValueError("Device interface must be one of 'numba', 'opencl'.")


def dense_assembler_dispatcher(device_interface, *args):
    """Dispatcher for dense assemblers."""
    interface_type = device_interface.split("_")[0]

    if interface_type == "opencl":
        from bempp_cl.core.opencl_assemblers import dense_assembler

        dense_assembler(device_interface, *args)

    elif interface_type == "numba":
        from bempp_cl.core.numba_assemblers import dense_assembler

        dense_assembler(device_interface, *args)

    else:
        raise ValueError("Device interface must be one of 'numba', 'opencl'.")


def potential_dispatcher(device_interface, *args):
    """Potential assembler dispatcher."""
    interface_type = device_interface.split("_")[0]

    if interface_type == "numba":
        from bempp_cl.core.numba_assemblers import potential_assembler

        return potential_assembler(device_interface, *args)

    elif interface_type == "opencl":
        from bempp_cl.core.opencl_assemblers import potential_assembler

        return potential_assembler(device_interface, *args)

    else:
        raise ValueError("Device interface must be one of 'numba', 'opencl'.")

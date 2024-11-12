"""OpenCL routines."""

import pyopencl as _cl
import os as _os

_CURRENT_PATH = _os.path.dirname(_os.path.realpath(__file__))
_INCLUDE_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/include"))
_KERNEL_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/kernels"))

_DEFAULT_CPU_DEVICE = None
_DEFAULT_CPU_CONTEXT = None

_DEFAULT_GPU_DEVICE = None
_DEFAULT_GPU_CONTEXT = None


def select_cl_kernel(operator_descriptor, mode):
    """Select OpenCL kernel."""
    singular_assemblers = {
        "default_scalar": "evaluate_dense_singular",
        "laplace_hypersingular": "evaluate_dense_laplace_hypersingular_singular",
        "helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_singular",
        "modified_helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_singular",
        "maxwell_electric_field": "evaluate_dense_electric_field_singular",
        "maxwell_magnetic_field": "evaluate_dense_magnetic_field_singular",
    }

    regular_assemblers = {
        "default_scalar": "evaluate_dense_regular",
        "laplace_hypersingular": "evaluate_dense_laplace_hypersingular_regular",
        "helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_regular",
        "modified_helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_regular",
        "maxwell_electric_field": "evaluate_dense_electric_field_regular",
        "maxwell_magnetic_field": "evaluate_dense_magnetic_field_regular",
    }

    potential_assemblers = {
        "default_scalar": "evaluate_scalar_potential",
        "maxwell_electric_field": "evaluate_electric_field_potential",
        "maxwell_magnetic_field": "evaluate_magnetic_field_potential",
        "maxwell_electric_far_field": "evaluate_maxwell_electric_far_field",
        "maxwell_magnetic_far_field": "evaluate_maxwell_magnetic_far_field",
    }

    kernels = {
        "laplace_single_layer": "laplace_single_layer",
        "laplace_double_layer": "laplace_double_layer",
        "laplace_adjoint_double_layer": "laplace_adjoint_double_layer",
        "helmholtz_single_layer": "helmholtz_single_layer",
        "helmholtz_double_layer": "helmholtz_double_layer",
        "helmholtz_far_field_single_layer": "helmholtz_single_layer_far_field",
        "helmholtz_far_field_double_layer": "helmholtz_double_layer_far_field",
        "helmholtz_adjoint_double_layer": "helmholtz_adjoint_double_layer",
        "modified_helmholtz_single_layer": "modified_helmholtz_real_single_layer",
        "modified_helmholtz_double_layer": "modified_helmholtz_real_double_layer",
        "modified_helmholtz_adjoint_double_layer": "modified_helmholtz_real_adjoint_double_layer",
    }

    if mode == "singular":
        return (
            singular_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type],
        )

    elif mode == "regular":
        return (
            regular_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type],
        )
    elif mode == "potential":
        return (
            potential_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type],
        )
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_kernel_compile_options(options, precision):
    """Create compiler options from parameters and precision."""
    import numbers

    if precision == "single":
        literal = "f"
    else:
        literal = ""

    compile_options = []
    for key, value in options.items():
        if value is None:
            compile_options += ["-D", "{0}".format(key)]
        else:
            if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
                value_string = str(value) + literal
            else:
                value_string = str(value)
            compile_options += ["-D", "{0}={1}".format(key, value_string)]

    compile_options += ["-I", _INCLUDE_PATH]

    # Add precision flag

    if precision == "single":
        val = 0
    else:
        val = 1

    compile_options.append("-DPRECISION={0}".format(val))

    return compile_options


def build_program(assembly_function, options, precision, device_type="cpu"):
    """Build the kernel and return it."""
    file_name = assembly_function + ".cl"
    kernel_file = _os.path.join(_KERNEL_PATH, file_name)

    kernel_string = open(kernel_file).read()
    kernel_options = get_kernel_compile_options(options, precision)

    return _cl.Program(default_context(device_type), kernel_string).build(options=kernel_options).kernel_function


def get_kernel_from_operator_descriptor(operator_descriptor, options, mode, force_novec=False, device_type="cpu"):
    """Return compiled kernel from operator descriptor."""
    precision = operator_descriptor.precision
    assembly_function, kernel_name = select_cl_kernel(operator_descriptor, mode=mode)

    vec_length = get_vector_width(precision, device_type)
    vec_string = get_vec_string(precision, device_type)

    if not mode == "singular":
        if force_novec or vec_length == 1:
            assembly_function += "_novec"
        else:
            assembly_function += "_vec"
    options["KERNEL_FUNCTION"] = kernel_name
    options["VEC_LENGTH"] = vec_length
    options["VEC_STRING"] = vec_string
    return build_program(assembly_function, options, precision, device_type)


def get_kernel_from_name(name, options, precision="double", device_type="cpu"):
    """Return compiled kernel from name."""

    vec_length = get_vector_width(precision, device_type)
    vec_string = get_vec_string(precision, device_type)

    options["VEC_LENGTH"] = vec_length
    options["VEC_STRING"] = vec_string
    return build_program(name, options, precision, device_type)


def get_vector_width(precision, device_type="cpu"):
    """Return vector width."""
    import bempp_cl.api

    mode_to_length = {"novec": 1, "vec4": 4, "vec8": 8, "vec16": 16}

    if device_type == "gpu":
        return 1
    if bempp_cl.api.VECTORIZATION_MODE == "auto":
        return get_native_vector_width(default_device(device_type), precision)
    else:
        return mode_to_length[bempp_cl.api.VECTORIZATION_MODE]


def get_vec_string(precision, device_type="cpu"):
    """Return vectorisation string."""
    vec_strings = {1: "novec", 4: "vec4", 8: "vec8", 16: "vec16"}

    return vec_strings[get_vector_width(precision, device_type)]


def default_cpu_device():
    """Return the default CPU device."""
    import bempp_cl.api
    import os

    # pylint: disable=W0603
    global _DEFAULT_CPU_DEVICE
    global _DEFAULT_CPU_CONTEXT

    if "BEMPP_CPU_DRIVER" in os.environ:
        name = os.environ["BEMPP_CPU_DRIVER"]
    else:
        name = None

    if _DEFAULT_CPU_DEVICE is None:
        try:
            ctx, device = find_cpu_driver(name)
        except:  # noqa: E722
            raise RuntimeError("Could not find suitable OpenCL CPU driver.")
        _DEFAULT_CPU_CONTEXT = ctx
        _DEFAULT_CPU_DEVICE = device
        bempp_cl.api.log(f"OpenCL CPU Device set to: {_DEFAULT_CPU_DEVICE.name}")
    return _DEFAULT_CPU_DEVICE


def default_gpu_device():
    """Return the default GPU device."""
    import bempp_cl.api
    import os

    # pylint: disable=W0603
    global _DEFAULT_GPU_DEVICE
    global _DEFAULT_GPU_CONTEXT

    if "BEMPP_GPU_DRIVER" in os.environ:
        name = os.environ["BEMPP_GPU_DRIVER"]
    else:
        name = None

    if _DEFAULT_GPU_DEVICE is None:
        try:
            ctx, device = find_gpu_driver(name)
        except:  # noqa: E722
            raise RuntimeError("Could not find a suitable OpenCL GPU driver.")
        _DEFAULT_GPU_CONTEXT = ctx
        _DEFAULT_GPU_DEVICE = device
        bempp_cl.api.log(f"OpenCL GPU Device set to: {_DEFAULT_GPU_DEVICE.name}")
    return _DEFAULT_GPU_DEVICE


def default_device(device_type="cpu"):
    """Return default device."""

    if device_type == "cpu":
        return default_cpu_device()
    elif device_type == "gpu":
        return default_gpu_device()
    else:
        raise ValueError(f"Unknown value for 'mode' = {device_type}.")


def default_context(device_type="cpu"):
    """Return the default context"""

    if device_type == "cpu":
        return default_cpu_context()
    elif device_type == "gpu":
        return default_gpu_context()
    else:
        raise ValueError(f"Unknown value for 'mode' = {device_type}.")


def default_cpu_context():
    """Return default CPU context."""

    if _DEFAULT_CPU_CONTEXT is None:
        default_cpu_device()

    return _DEFAULT_CPU_CONTEXT


def default_gpu_context():
    """Return default GPU context."""

    if _DEFAULT_GPU_CONTEXT is None:
        default_gpu_device()

    return _DEFAULT_GPU_CONTEXT


def find_cpu_driver(name=None):
    """Find the first available CPU OpenCL driver."""
    found = False
    ctx = None
    device = None
    for platform in _cl.get_platforms():
        if found:
            break
        if name and name not in platform.name:
            continue
        ctx = _cl.Context(
            dev_type=_cl.device_type.ALL,
            properties=[(_cl.context_properties.PLATFORM, platform)],
        )
        for device in ctx.devices:
            if device.type == _cl.device_type.CPU:
                found = True
    if not found:
        raise ValueError(f"Could not find CPU driver containing name {name}.")
    return ctx, device


def find_gpu_driver(name=None):
    """Find the first available GPU OpenCL driver."""

    found = False
    ctx = None
    device = None
    for platform in _cl.get_platforms():
        if found:
            break
        if name and name not in platform.name:
            continue
        ctx = _cl.Context(
            dev_type=_cl.device_type.ALL,
            properties=[(_cl.context_properties.PLATFORM, platform)],
        )
        for device in ctx.devices:
            if device.type == _cl.device_type.GPU:
                found = True
    if not found:
        raise ValueError(f"Could not find GPU driver containing name {name}.")
    return ctx, device


def set_default_cpu_device_by_name(name):
    """
    Set default CPU device by name.

    This method looks for the given string in the available OpenCL
    drivers and picks the first one that contains the given search
    string.

    """
    import bempp_cl.api

    global _DEFAULT_CPU_CONTEXT
    global _DEFAULT_CPU_DEVICE

    try:
        context, device = find_cpu_driver(name)
    except:  # noqa: E722
        raise RuntimeError("No CPU driver with given name found.")

    _DEFAULT_CPU_CONTEXT = context
    _DEFAULT_CPU_DEVICE = device
    vector_width_single = _DEFAULT_CPU_DEVICE.native_vector_width_float
    vector_width_double = _DEFAULT_CPU_DEVICE.native_vector_width_double

    bempp_cl.api.log(
        f"Default CPU device: {_DEFAULT_CPU_DEVICE.name}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


def set_default_cpu_device(platform_index, device_index):
    """Set the default CPU device."""
    import bempp_cl.api

    # pylint: disable=W0603
    global _DEFAULT_CPU_DEVICE
    global _DEFAULT_CPU_CONTEXT

    platform = _cl.get_platforms()[platform_index]
    device = platform.get_devices()[device_index]
    _DEFAULT_CPU_CONTEXT = _cl.Context(devices=[device], properties=[(_cl.context_properties.PLATFORM, platform)])
    _DEFAULT_CPU_DEVICE = _DEFAULT_CPU_CONTEXT.devices[0]

    vector_width_single = _DEFAULT_CPU_DEVICE.native_vector_width_float
    vector_width_double = _DEFAULT_CPU_DEVICE.native_vector_width_double

    bempp_cl.api.log(
        f"Default CPU device: {_DEFAULT_CPU_DEVICE.name}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


def set_default_gpu_device_by_name(name):
    """
    Set default GPU device by name.

    This method looks for the given string in the available OpenCL
    drivers and picks the first one that contains the given search
    string.

    """
    import bempp_cl.api

    global _DEFAULT_GPU_CONTEXT
    global _DEFAULT_GPU_DEVICE

    try:
        pair = find_gpu_driver(name)
        context, device = pair[0], pair[1]
    except:  # noqa: E722
        raise RuntimeError("No GPU driver with given name found.")

    _DEFAULT_GPU_CONTEXT = context
    _DEFAULT_GPU_DEVICE = device
    vector_width_single = _DEFAULT_GPU_DEVICE.native_vector_width_float
    vector_width_double = _DEFAULT_GPU_DEVICE.native_vector_width_double

    bempp_cl.api.log(
        f"Default GPU device: {_DEFAULT_GPU_DEVICE.name}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


def set_default_gpu_device(platform_index, device_index):
    """Set the default GPU device."""
    import bempp_cl.api

    # pylint: disable=W0603
    global _DEFAULT_GPU_DEVICE
    global _DEFAULT_GPU_CONTEXT

    platform = _cl.get_platforms()[platform_index]
    device = platform.get_devices()[device_index]
    _DEFAULT_GPU_CONTEXT = _cl.Context(devices=[device], properties=[(_cl.context_properties.PLATFORM, platform)])
    _DEFAULT_GPU_DEVICE = _DEFAULT_GPU_CONTEXT.devices[0]

    vector_width_single = _DEFAULT_GPU_DEVICE.native_vector_width_float
    vector_width_double = _DEFAULT_GPU_DEVICE.native_vector_width_double

    bempp_cl.api.log(
        f"Default GPU device: {_DEFAULT_GPU_DEVICE.name}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


def show_available_platforms_and_devices():
    """Print available platforms and devices."""
    platforms = _cl.get_platforms()
    for platform_index, platform in enumerate(platforms):
        print(str(platform_index) + ": " + platform.get_info(_cl.platform_info.NAME))
        devices = platform.get_devices()
        for device_index, device in enumerate(devices):
            print(4 * " " + str(device_index) + ": " + device.get_info(_cl.device_info.NAME))


def get_native_vector_width(device, precision):
    """Get default vector width for device."""
    if precision == "single":
        return device.native_vector_width_float
    elif precision == "double":
        return device.native_vector_width_double
    else:
        raise ValueError("precision must be one of 'single', 'double'.")

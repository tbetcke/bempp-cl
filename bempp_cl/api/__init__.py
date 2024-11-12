"""Global initialization for Bempp."""

# # Monkey patch Numba to emit log messages when compiling

# import numba

# oldcompile = numba.core.registry.CPUDispatcher.compile


# def compile_with_log(*args, **kwargs):
# """Numba compilation with log messages."""
# import bempp_cl.api

# fun_name = args[0].py_func.__name__
# bempp_cl.api.log(f"Compiling {fun_name} for signature {args[1]}.", level="debug")
# res = oldcompile(*args, **kwargs)
# bempp_cl.api.log(f"Compilation finished.", level="debug")
# return res


# numba.core.registry.CPUDispatcher.compile = compile_with_log

import os as _os
import tempfile as _tempfile
import logging as _logging
import time as _time
import platform as _platform

from bempp_cl.api.utils import DefaultParameters
from bempp_cl.api.utils.helpers import MemProfiler

from bempp_cl.api.utils.helpers import assign_parameters
from bempp_cl.api.grid.io import import_grid
from bempp_cl.api.grid.io import export
from bempp_cl.api.grid.grid import Grid
from bempp_cl.api.assembly.grid_function import GridFunction
from bempp_cl.api.assembly.grid_function import real_callable
from bempp_cl.api.assembly.grid_function import complex_callable
from bempp_cl.api.assembly.grid_function import callable

from bempp_cl.api.space import function_space

from bempp_cl.api import shapes
from bempp_cl.api import integration
from bempp_cl.api import operators
from bempp_cl.api.linalg.direct_solvers import lu, compute_lu_factors
from bempp_cl.api.linalg.iterative_solvers import gmres, cg
from bempp_cl.api.assembly.discrete_boundary_operator import as_matrix
from bempp_cl.api.assembly.boundary_operator import ZeroBoundaryOperator
from bempp_cl.api.assembly.boundary_operator import MultiplicationOperator
from bempp_cl.api.assembly.blocked_operator import BlockedOperator
from bempp_cl.api.assembly.blocked_operator import GeneralizedBlockedOperator

from bempp_cl.api.fmm.fmm_assembler import clear_fmm_cache

from bempp_cl.api.utils import pool
from bempp_cl.api.utils.pool import create_device_pool

from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
)

import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", message="splu requires CSC matrix format")

# Try importing OpenCL routines

try:
    from bempp_cl.core.opencl_kernels import set_default_cpu_device
    from bempp_cl.core.opencl_kernels import set_default_cpu_device_by_name
    from bempp_cl.core.opencl_kernels import set_default_gpu_device_by_name
    from bempp_cl.core.opencl_kernels import set_default_gpu_device
except ImportError:
    pass


CONSOLE_LOGGING_HANDLER = None
DEFAULT_LOGGING_FORMAT = "%(name)s:%(levelname)s: %(message)s"
DEBUG = _logging.DEBUG
TIMING = 11
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL

LOG_LEVEL = {
    "debug": DEBUG,
    "timing": TIMING,
    "info": INFO,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
}

GLOBAL_PARAMETERS = DefaultParameters()


def _init_logger():
    """Initialize the Bempp logger."""
    _logging.addLevelName(11, "TIMING")
    logger = _logging.getLogger("bempp")
    logger.setLevel(DEBUG)
    logger.addHandler(_logging.NullHandler())
    return logger


def log(message, level="info", flush=True):
    """Log including default flushing for IPython."""
    LOGGER.log(LOG_LEVEL[level], message)
    if flush:
        flush_log()


def flush_log():
    """Flush all handlers. Necessary for Jupyter."""
    for handler in LOGGER.handlers:
        handler.flush()


def enable_console_logging(level="info"):
    """Enable console logging and return the console handler."""
    # pylint: disable=W0603
    global CONSOLE_LOGGING_HANDLER
    if not CONSOLE_LOGGING_HANDLER:
        console_handler = _logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL[level])
        if pool.is_worker():
            console_handler.setFormatter(
                _logging.Formatter(f"%(name)s:PROC{pool._MY_ID}:%(levelname)s: %(message)s", "%H:%M:%S")
            )
        else:
            console_handler.setFormatter(_logging.Formatter("%(name)s:HOST:%(levelname)s: %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(console_handler)
        CONSOLE_LOGGING_HANDLER = console_handler
    return CONSOLE_LOGGING_HANDLER


# def enable_file_logging(file_name, level=DEBUG, logging_format=DEFAULT_LOGGING_FORMAT):
# """Enable logging to a specific file."""

# file_handler = _logging.FileHandler(file_name)
# file_handler.setLevel(level)
# file_handler.setFormatter(_logging.Formatter(logging_format, "%H:%M:%S"))
# LOGGER.addHandler(file_handler)
# return file_handler


def set_logging_level(level):
    """Set the logging level."""
    LOGGER.setLevel(LOG_LEVEL[level])


# pylint: disable=too-few-public-methods
class Timer:
    """Context manager to measure time in Bempp."""

    def __init__(self, enable_log=True, message="", level="timing"):
        """Construct."""
        self.start = 0
        self.end = 0
        self.interval = 0
        self.enable_log = enable_log
        self.level = level
        self.message = message

    def __enter__(self):
        """Enter."""
        if self.enable_log:
            log("Start operation: " + self.message, level=self.level)
        self.start = _time.time()
        return self

    def __exit__(self, *args):
        """Exit."""
        self.end = _time.time()
        self.interval = self.end - self.start
        if self.enable_log:
            log(
                "Finished Operation: " + self.message + f": {self.interval}s",
                level=self.level,
            )


LOGGER = _init_logger()

BEMPP_PATH = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.realpath(__file__)), ".."))

# pylint: disable=W0702
# try:
#    if _os.environ['BEMPP_CONSOLE_LOGGING'] == '1':
#        enable_console_logging()
# except:
#    pass

TMP_PATH = _tempfile.mkdtemp()

# Get the path to Gmsh


def _gmsh_path():
    """Find Gmsh."""
    from bempp_cl.api.utils import which

    if _os.name == "nt":
        gmp = which("gmsh.exe")
        if gmp is None:
            gmp = which("gmsh")
    else:
        gmp = which("gmsh")
    if gmp is None:
        print("Could not find Gmsh." + "Interactive plotting and shapes module not available.")
    return gmp


def check_for_fmm():
    """Return true of compatible FMM found."""
    exafmm_found = False
    try:
        import exafmm
    except ImportError:
        exafmm_found = False
    else:
        exafmm_found = True

    return exafmm_found


def _get_version():
    """Get version string."""
    from bempp_cl import version

    return version.__version__


GMSH_PATH = _gmsh_path()

__version__ = _get_version()

PLOT_BACKEND = "gmsh"
try:
    ipy = get_ipython()
    if ipy.__class__.__name__ == "ZMQInteractiveShell":
        # We are in a jupyter notebook, so change plotting backend
        PLOT_BACKEND = "jupyter_notebook"
except NameError:
    pass

USE_JIT = True

CPU_OPENCL_DRIVER_FOUND = False
GPU_OPENCL_DRIVER_FOUND = False

if _platform.system() == "Darwin":
    DEFAULT_DEVICE_INTERFACE = "numba"
else:
    try:
        from bempp_cl.core.opencl_kernels import find_cpu_driver

        CPU_OPENCL_DRIVER_FOUND = find_cpu_driver()
    except:  # noqa: E722
        pass

    try:
        from bempp_cl.core.opencl_kernels import find_gpu_driver

        GPU_OPENCL_DRIVER_FOUND = find_gpu_driver()
    except:  # noqa: E722
        pass

    if CPU_OPENCL_DRIVER_FOUND:
        DEFAULT_DEVICE_INTERFACE = "opencl"
    else:
        DEFAULT_DEVICE_INTERFACE = "numba"

if DEFAULT_DEVICE_INTERFACE == "numba":
    log("Numba backend activated. For full performance the OpenCL backend with an OpenCL CPU driver is required.")

DEFAULT_PRECISION = "double"
VECTORIZATION_MODE = "auto"

BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"
POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"

ALL = -1  # Useful global identifier

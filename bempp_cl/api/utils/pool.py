"""Routines to administrate a process pool."""

import numpy as _np

# Variables used inside the workers.
_DATA = {}
_MY_ID = None

# Global variables in the host.
_USE_THREADS = None
_POOL = None

# Variables in host and workers
_NWORKERS = None
_IN_WORKER = False
_BUFFER = None


def worker(in_queue, out_queue, worker_id, nworkers, buf, log, log_level):
    """Definition of a worker."""
    import bempp_cl.api
    from bempp_cl.api.utils import pool
    import traceback

    pool._MY_ID = worker_id
    pool._NWORKERS = nworkers
    pool._BUFFER = buf
    pool._IN_WORKER = True

    if log:
        bempp_cl.api.enable_console_logging(log_level)

    get = in_queue.get
    put = out_queue.put

    while True:
        job = get()
        if job is None:
            break
        fun, args, options = job
        try:
            if options == "NOARGS":
                result = fun()
            elif options == "MAP":
                result = fun(args)
            elif options == "STARMAP":
                result = fun(*args)
            else:
                raise ValueError("Unknown options.")
            put(result)
        except Exception:
            traceback.print_exc()

    bempp_cl.api.flush_log()
    put("FINISHED")


def as_array(dtype, offset, shape):
    """
    Return part of the buffer as array.

    Parameters
    ----------
    dtype : Numpy dtype object
        The type of the array
    offset : int
        Start index in buffer
    nitems : int
        Number of items of type dtype

    """
    from bempp_cl.api.utils import pool

    nitems = _np.prod(shape)

    return _np.frombuffer(pool._BUFFER, dtype=dtype, count=nitems, offset=offset).reshape(*shape)


def to_buffer(*args):
    """Save a number of numpy arrays to a buffer."""
    from bempp_cl.api.utils import pool

    offset = 0
    result = []

    for arr in args:
        ar_size = arr.nbytes
        pool._BUFFER[offset : offset + ar_size] = _np.require(arr.flat, requirements="C").view(dtype="uint8")
        offset += ar_size
        result.append((arr.dtype, arr.shape))
    return result


def from_buffer(arrays):
    """
    Retrieve arrays from buffer.

    arrays is a list of tuples
    [(dtype1, shape1), (dtype2, shape2), ...],
    where dtype is the type of the array and shape
    is a shape tuple.

    """
    from bempp_cl.api.utils import pool

    result = []

    offset = 0
    for ar in arrays:
        dtype, shape = ar
        nbytes = _np.dtype(dtype).itemsize * _np.prod(shape)
        result.append(pool.as_array(dtype, offset, shape))
        offset += nbytes
    return result


class Pool(object):
    """
    A simple pool.

    This pool is different from the multiprocessing pool
    in that map operations are guaranteed to execute on
    all processes, and we can directly target individual
    processes.

    """

    def __init__(self, nworkers, buffer_size=100, log=False, log_level="info"):
        """
        Initialise the pool.

        Parameters
        ----------
        nworkers : int
            Number of workers
        buffer_size : int
            Size of the shared memory buffer
            in MB.
        log : Boolean
            Set to True for logging
        log_level : String
            One of 'debug', 'info', 'warning', 'error', 'critical'

        """
        import bempp_cl.api
        from bempp_cl.api.utils import pool
        from bempp_cl.api.utils.pool import worker

        from multiprocessing import get_context

        self._nworkers = nworkers

        ctx = get_context("spawn")

        pool._BUFFER = ctx.RawArray("b", buffer_size * 1024 * 1024)

        self._senders = [ctx.SimpleQueue() for _ in range(nworkers)]
        self._receivers = [ctx.SimpleQueue() for _ in range(nworkers)]

        self._workers = [
            ctx.Process(
                target=worker,
                args=(
                    self._senders[i],
                    self._receivers[i],
                    i,
                    nworkers,
                    pool._BUFFER,
                    log,
                    log_level,
                ),
            )
            for i in range(nworkers)
        ]
        for w in self._workers:
            w.daemon = True
            w.start()
        bempp_cl.api.log(f"Created pool with {nworkers} workers.")

    @property
    def number_of_workers(self):
        """Return number of workers."""
        return self._nworkers

    def _map_impl(self, fun, args, options):
        """Map implementation."""
        for index, arg in zip(range(self._nworkers), args):
            self._senders[index].put((fun, arg, options))
        return [self._receivers[index].get() for index in range(self._nworkers)]

    def map(self, fun, args=None):
        """Map function onto workers."""
        if args is None:
            options = "NOARGS"
            args = self._nworkers * [None]
        else:
            options = "MAP"
        return self._map_impl(fun, args, options)

    def starmap(self, fun, args):
        """Map function onto workers."""
        return self._map_impl(fun, args, "STARMAP")

    def shutdown(self):
        """Shutdown all workers."""
        for index in range(self._nworkers):
            self._senders[index].put(None)
        for worker in self._workers:
            worker.join()


def _raise_if_not_worker(name):
    """Raise exception if not in worker."""
    if not is_worker():
        raise Exception(f"Method {name} can only be called inside a worker.")


def number_of_workers():
    """Return number of workers."""
    from bempp_cl.api.utils import pool

    if not is_initialised():
        raise Exception("Pool is not initialised.")

    return pool._POOL.number_of_workers


def insert_data(key, data):
    """Insert data."""
    from bempp_cl.api.utils import pool

    _raise_if_not_worker("insert_data")
    pool._DATA[key] = data


def is_initialised():
    """Return true if pool is initialised."""
    return _POOL is not None


def clear_data():
    """Clear the data in all workers."""
    from bempp_cl.api.utils import pool

    pool.execute(_clear_data_worker)


def get_data(key):
    """Return data."""
    from bempp_cl.api.utils import pool

    _raise_if_not_worker("get_data")
    return pool._DATA[key]


def get_id():
    """Return my id."""
    return _MY_ID


def remove_key(key):
    """Remove data from pool."""
    execute(_remove_key_worker, key)


def has_key(key):
    """Return if key exists in pool data."""
    from bempp_cl.api.utils import pool

    return key in pool._DATA


def is_worker():
    """Return true if called from worker process."""
    return _IN_WORKER is True


def create_device_pool(
    identifier,
    buffer_size=100,
    log=False,
    log_level="info",
    max_workers=-1,
    precision=None,
):
    """
    Create a pool based on a given platform identifer.

    identifier : string
        A unique identifier that is part of the platform name.
        Used to find the correct platform.
    buffer_size : int
        Shared memory buffer size in MB
    log : Boolean
        Set to True to log workers.
    log_level : String
        Logging level. One of 'debug', 'info', 'warning', 'error', 'critical'
    max_workers : int
        Maximum number of workers. If max_workers=-1 (default)
        the maximum number of workers is identical to the number
        of devices in the pool.
    precision : string or None
        Precision for the devices in the pool. If precision is None use
        single precision for GPU devices and double precision for CPU devices.
        If precision is 'single' or 'double' use the corresponding mode for the pool.

    """
    import bempp_cl.api
    from bempp_cl.core.cl_helpers import get_context_by_name

    ctx, _ = get_context_by_name(identifier)

    bempp_cl.api.log(f"Creating pool for Platform: {ctx.platform_name}")

    if precision not in [None, "single", "double"]:
        raise ValueError(f"'precision' is {precision}. Allowed values are: 'single', 'double'")

    if max_workers > len(ctx.devices):
        raise ValueError(
            f"Maximum number of workers ({max_workers}) " + f"is bigger than number of devices {len(ctx.devices)}"
        )
    if max_workers == -1:
        ndevices = len(ctx.devices)
    else:
        ndevices = max_workers

    create_pool(ndevices, buffer_size, log, log_level)

    execute(_init_device_worker, identifier, precision)


def create_pool(nworkers, buffer_size=100, log=False, log_level="info"):
    """Create a pool."""
    from bempp_cl.api.utils import pool

    pool._POOL = Pool(nworkers, buffer_size=buffer_size, log=log, log_level=log_level)
    pool._NWORKERS = nworkers


def nworkers():
    """Get number of workers."""
    return _NWORKERS


def map(fun, args):
    """Corresponds to multiprocessing map."""
    return _POOL.map(fun, args)


def starmap(fun, args):
    """Corresponds to multiprocessing map."""
    return _POOL.starmap(fun, args)


def execute(fun, *args):
    """Execute function with the same arguments on all workers."""
    if len(args) == 0:
        return _POOL.map(fun)
    else:
        return _POOL.starmap(fun, nworkers() * [args])


def _assign_ids(nworkers):
    """Assign pool ids."""
    raise NotImplementedError()  # _assign_ids_worker is not defined
    # map(_assign_ids_worker, zip(range(nworkers), nworkers * [nworkers]))


def shutdown():
    """Shutdown the pool."""
    from bempp_cl.api.utils import pool

    pool._POOL.shutdown()
    pool._POOL = None
    pool._NWORKERS = False
    pool._USE_THREADS = None


def _execute_function_without_arguments(fun):
    """Execute function without arguments."""
    return fun()


def _remove_key_worker(key):
    from bempp_cl.api.utils import pool

    del pool._DATA[key]


def _init_device_worker(identifier, precision):
    """Worker to initialise device."""
    import bempp_cl.api
    from bempp_cl.api.utils.pool import get_id
    from bempp_cl.core.cl_helpers import get_context_by_name

    ctx, platform_index = get_context_by_name(identifier)
    bempp_cl.api.set_default_device(platform_index, get_id())
    if precision is not None:
        bempp_cl.api.DEVICE_PRECISION_CPU = precision
        bempp_cl.api.DEVICE_PRECISION_GPU = precision


def _clear_data_worker():
    """Clear worker."""
    from bempp_cl.api.utils import pool

    pool._DATA = {}

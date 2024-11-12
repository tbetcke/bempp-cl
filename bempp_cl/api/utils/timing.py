"""Timing functions."""

import time as _time


def timeit(message):
    """Time a method in bempp_cl."""

    def timeit_impl(fun):
        """Return an implementation of timeit."""

        def timed_fun(*args, **kwargs):
            """Time something."""
            from bempp_cl.api import GLOBAL_PARAMETERS
            from bempp_cl.api import log

            if not GLOBAL_PARAMETERS.verbosity.extended_verbosity:
                return fun(*args, **kwargs)

            start_time = _time.time()
            res = fun(*args, **kwargs)
            end_time = _time.time()
            log(message + " : {0:.3e}s".format(end_time - start_time))
            return res

        return timed_fun

    return timeit_impl


# pylint: disable=too-few-public-methods
class Timer:
    """Context manager to measure time in bempp_cl."""

    def __init__(self):
        """Construct."""
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        """Enter."""
        self.start = _time.time()
        return self

    def __exit__(self, *args):
        """Exit."""
        self.end = _time.time()
        self.interval = self.end - self.start

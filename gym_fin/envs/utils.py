from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Stdlib imports
import contextlib
from functools import lru_cache
import random

# Third party imports
from scipy.stats import norm


@contextlib.contextmanager
def temp_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@lru_cache(maxsize=None)
def cached_cdf(int_val, loc, scale):
    return norm.cdf(int_val, loc=loc, scale=scale)


try:
    # Some profiling decorator (for package line_profiler)
    # https://zapier.com/engineering/profiling-python-boss/
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()

            return profiled_func

        return inner


except ImportError:
    print(
        "Could not import line profiler. "
        "Accidentally left in production code?"
    )

    def do_profile(follow=[]):
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)

            return nothing

        return inner

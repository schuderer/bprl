import contextlib
import random
from functools import lru_cache
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


class LazyStr:
    def __init__(self, func, *args, **kwArgs):
        self.func = func
        self.args = args
        self.kwArgs = kwArgs

    def __str__(self):
        return str(self.func(*self.args, **self.kwArgs))



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
    def do_profile(follow=[]):
        'Helpful if you accidentally leave in production!'
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner

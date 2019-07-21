"""Tests for the gym_fin.utils module"""

import pytest


def test_softmax():
    import numpy as np
    from gym_fin.envs import utils

    rng = np.random.RandomState(0)
    X = rng.randn(5)
    exp_X = np.exp(X)
    sum_exp_X = np.sum(exp_X, axis=0)
    result = exp_X / sum_exp_X
    np.testing.assert_array_almost_equal(utils.softmax(X), result)

    # Handle empty list gracefully
    assert utils.softmax([]) == []


def test_temp_seed():
    from gym_fin.envs import utils
    import random

    with utils.temp_seed(7):
        r = random.randint(0, 100)

    with utils.temp_seed(2):
        r2 = random.randint(0, 100)

    with utils.temp_seed(7):
        assert r == random.randint(0, 100)

    with utils.temp_seed(2):
        assert r2 == random.randint(0, 100)


def test_cached_cdf():
    from gym_fin.envs import utils

    # scipy = pytest.importorskip('scipy')

    c1 = utils.cached_cdf(7, 5, 10)
    c2 = utils.cached_cdf(8, 5, 10)
    assert c2 > c1


def test_do_profile():
    from gym_fin.envs import utils

    pytest.importorskip("line_profiler")

    @utils.do_profile()
    def test_func():
        s = ""
        for i in range(100):
            s += str(i)
        return s

    assert test_func() != ""

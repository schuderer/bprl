# import pytest
from gym_fin.envs import utils


def test_temp_seed():
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
    # scipy = pytest.importorskip('scipy')

    c1 = utils.cached_cdf(7, 5, 10)
    c2 = utils.cached_cdf(8, 5, 10)
    assert c2 > c1

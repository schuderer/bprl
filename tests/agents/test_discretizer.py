"""Tests for the agents.discretizer module"""

# Stdlib imports

# Third-party imports
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
import numpy as np
import pytest

# Application imports
from agents.discretizer import create_discretizers
from agents.discretizer import Discretizer


@pytest.fixture
def box():
    return Box(
        low=np.array([1.2, 256, -8, -2]), high=np.array([1.5, 1024, -4, 8])
    )


@pytest.fixture
def disc():
    return Discrete(4)


@pytest.fixture
def obs():
    return np.array([1.333333, 512, -5, 1])


@pytest.fixture
def obs_out():
    return np.array([1.0, 1300, -9, -3])


@pytest.fixture
def obs_edge():
    return np.array([1.2, 1024, -8, 8])


@pytest.fixture
def obs_center(obs):
    return np.array([1.35, 512, -6, -0])


@pytest.fixture
def max_difference_lin():
    def inner_func(space: Box, num_bins: int):
        spread = space.high - space.low
        return abs(spread / num_bins)

    return inner_func


def test_discretizer_lin(box, obs, max_difference_lin):
    bins = 5
    d = Discretizer(box, num_bins=bins, log_bins=False)
    disc = d.discretize(obs)
    undisc_obs = d.undiscretize(disc)
    assert (abs(obs - undisc_obs) <= max_difference_lin(box, bins)).all()


def test_discretizer_lin_outside(box, obs_out, obs_center):
    d = Discretizer(box, num_bins=5, log_bins=False)
    disc = d.discretize(obs_out)
    undisc_obs = d.undiscretize(disc)
    for i, elem in enumerate(undisc_obs):
        if obs_out[i] < obs_center[i]:
            assert elem == pytest.approx(box.low[i])
        else:
            assert elem == pytest.approx(box.high[i])


def test_discretizer_log(box, obs_center):
    d = Discretizer(box, num_bins=5, log_bins=True)
    disc = d.discretize(obs_center)
    undisc_obs = d.undiscretize(disc)
    for i, elem in enumerate(undisc_obs):
        if obs_center[i] == 0:
            assert elem == pytest.approx(0, abs=1.0)
        else:
            assert elem >= box.low[i]
            assert elem <= box.high[i]


def test_discretizer_log_edges(box, obs_center, obs_edge):
    d = Discretizer(box, num_bins=5, log_bins=True)
    disc = d.discretize(obs_edge)
    undisc_obs = d.undiscretize(disc)
    for i, elem in enumerate(undisc_obs):
        if obs_edge[i] < obs_center[i]:
            assert elem == pytest.approx(box.low[i])
        else:
            assert elem == pytest.approx(box.high[i])


def test_discretizer_log_outside(box, obs_center, obs_out):
    d = Discretizer(box, num_bins=5, log_bins=True)
    disc = d.discretize(obs_out)
    undisc_obs = d.undiscretize(disc)
    for i, elem in enumerate(undisc_obs):
        if obs_out[i] < obs_center[i]:
            assert elem == pytest.approx(box.low[i])
        else:
            assert elem == pytest.approx(box.high[i])


def test_discretizer_discrete():
    discrete_space = Discrete(4 * 7)
    d = Discretizer(discrete_space, num_bins=5, log_bins=False)
    obs = np.array([17])
    disc_arr = d.discretize(obs)
    undisc_obs_arr = d.undiscretize(disc_arr)
    assert disc_arr == np.array(obs)
    assert undisc_obs_arr == np.array(obs)

    # Also accept plain numbers:
    obs = 17
    disc_arr = d.discretize(obs)
    undisc_obs_arr = d.undiscretize(disc_arr)
    assert disc_arr == np.array(obs)
    assert undisc_obs_arr == np.array(obs)

    # Undiscretize should also accept plain numbers:
    undisc_obs = d.undiscretize(17)
    assert undisc_obs == np.array(17)


def test_discretizer_undiscretize_accept_plain_numbers(max_difference_lin):
    bins = 5
    box = Box(low=np.array([1.2]), high=np.array([1.5]))
    obs = [1.3333]
    d = Discretizer(box, num_bins=bins, log_bins=False)
    disc = d.discretize(obs)
    undisc_obs = d.undiscretize(int(disc))
    assert (abs(obs - undisc_obs) <= max_difference_lin(box, bins)).all()


@pytest.mark.parametrize(
    "invalid_obs",
    [
        np.array([1.333333, 512, -5, 1, 3.145]),
        np.array([1.333333, 512, -5]),
        np.array([]),
        np.array([[]]),
        np.array([[1.333333, 512, -5, 1]]),
        np.array([[1.333333, 512, -5, 1]]).T,
    ],
)
def test_discretizer_wrong_shape(box, invalid_obs):
    d = Discretizer(box, num_bins=5, log_bins=False)
    with pytest.raises(ValueError, match="shape"):
        _ = d.discretize(invalid_obs)


def test_discretizer_unsupported_space(box, disc):
    dict_space = Dict(my_box=box, my_disc=disc)
    with pytest.raises(TypeError):
        _ = Discretizer(dict_space, num_bins=5, log_bins=False)


def test_create_discretizers(mocker, box, disc):
    env = mocker.Mock()
    env.observation_space = box
    env.action_space = disc
    state_disc, action_disc = create_discretizers(
        env, num_bins=5, log_bins=False
    )

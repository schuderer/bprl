"""Tests for the agents.value_function module"""

# Stdlib imports
from unittest import mock

# Third-party imports
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
import pytest

# Application imports
from agents.value_function import QFunction


env_box_disc = mock.Mock()
env_box_disc.observation_space = Box(
    low=np.array([-100, 0]), high=np.array([100, 1])
)
env_box_disc.action_space = Discrete(4)


env_disc_disc = mock.Mock()
env_disc_disc.observation_space = Discrete(10)
env_disc_disc.action_space = Discrete(2)


@pytest.fixture
def policy(mocker):
    policy = mocker.Mock()
    policy.side_effect = lambda vals, par: (par["which"], vals[par["which"]])
    return policy


@pytest.mark.parametrize(
    ["env", "obs"], [[env_box_disc, np.array([-10, 0.5])], [env_disc_disc, 3]]
)
def test_value_function_select_action(env, obs, policy):
    policy_params = {"which": 1}
    f = QFunction(env, default_value=7)
    a, aval = f.select_action(obs, policy, policy_params)
    assert a == 1
    assert aval == 7


@pytest.mark.parametrize(
    ["env", "obs"],
    [[env_box_disc, np.array([-10, 0.5])], [env_disc_disc, np.array(3)]],
)
def test_value_function_update_value(env, obs, policy):
    policy_params = {"which": 1}
    f = QFunction(env, default_value=0)
    f.update_value(obs, 1, 100)
    a, aval = f.select_action(obs, policy, policy_params)
    assert a == 1
    assert aval == 100


@pytest.mark.parametrize(
    ["env", "obs"], [[env_box_disc, np.array([-10, 0.5])], [env_disc_disc, 3]]
)
def test_value_function_cache(mocker, env, obs, policy):
    policy_params = {"which": 1}
    f = QFunction(env, default_value=0)
    mocker.spy(f, "_stateKeyFor")
    a, aval = f.select_action(obs, policy, policy_params, save="this")
    assert a == 1
    assert aval == 0
    assert f._stateKeyFor.call_count == 1
    f.update_value(obs, 1, 100, load="this")
    assert f._stateKeyFor.call_count == 1
    a, aval = f.select_action(obs, policy, policy_params, load="this")
    assert a == 1
    assert aval == 100
    assert f._stateKeyFor.call_count == 1

    f.update_value(obs, 1, 20, save="another")
    assert f._stateKeyFor.call_count == 2
    a, aval = f.select_action(obs, policy, policy_params, load="another")
    assert a == 1
    assert aval == 20
    assert f._stateKeyFor.call_count == 2


@pytest.mark.skip(reason="currently unused")
def test_value_function_print(mocker):
    f1 = QFunction(env_box_disc)
    f1.print_q()
    f1.print_q_frozenlake()

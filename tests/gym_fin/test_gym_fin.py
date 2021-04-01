"""Tests for the gym_fin.envs package"""

import pytest

import tests.utils as test_utils

# from gym_fin.envs.sim_env import make_step, env_metadata

# NOTE: import modules to test in the test functions, not here!


def test_no_do_profile_in_code():
    """We don't want any @do_profile decorators in production code"""

    with test_utils.do_profile_error():
        test_utils.import_submodules("gym_fin")


@pytest.mark.skip(reason="registration mechanism is bound to change")
def test_register():
    import gym

    # class SomeClass():
    #     @make_step(
    #         observation_space=gym.spaces.Discrete(2),
    #         observation_space_mapping=lambda self: 1,
    #         action_space=gym.spaces.Discrete(2),
    #         action_space_mapping={0: "A", 1: "B"},
    #         reward_mapping=lambda self: 1,
    #     )
    #     def some_choice():
    #         return "A"

    # "gym_fin:FinBase-gym_fin.envs.sim_env.Entity.choose_some_action-v0"

    id_start = "FinBase-"
    env_ids = gym.envs.registry.env_specs.keys()
    fin_base_envs = [id for id in env_ids if id_start in id]
    assert len(fin_base_envs) > 0

    for env_id in env_ids:
        if env_id.startswith(id_start):
            env = gym.make(env_id)
            assert isinstance(env, gym.Env)

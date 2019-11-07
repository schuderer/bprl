"""Tests for the gym_fin.envs package"""

import tests.utils as test_utils

# NOTE: import modules to test in the test functions, not here!


def test_no_do_profile_in_code():
    """We don't want any @do_profile decorators in production code"""

    with test_utils.do_profile_error():
        test_utils.import_submodules("gym_fin")


def test_register():
    # "gym_fin:FinBase-gym_fin.envs.sim_env.Entity.choose_some_action-v0"
    import gym
    import gym_fin

    _ = dir(gym_fin)  # avoid linting error

    id_start = "FinBase-"
    env_ids = gym.envs.registry.env_specs.keys()
    fin_base_envs = [id for id in env_ids if id_start in id]
    assert len(fin_base_envs) > 0

    for env_id in env_ids:
        if env_id.startswith(id_start):
            env = gym.make(env_id)
            assert isinstance(env, gym.Env)

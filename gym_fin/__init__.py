import os
import warnings

__version__ = "0.0.1"

try:
    from gym.envs.registration import register

    register(
        id="Pension-v0",
        entry_point="gym_fin.envs.pension_env:PensionEnv",
        kwargs={}
        # See gym.envs.registration:EnvSpec for possible arguments
    )

    from random import choice
    import string
    import gym_fin.envs
    from gym_fin.envs.sim_env import env_metadata, generate_env
    from gym_fin.envs.simulation_base import FinBaseSimulation

    for step_function in env_metadata["step_views"].keys():
        env_cls = generate_env(FinBaseSimulation(), step_function)
        cls_name = "_" + "".join(
            choice(string.ascii_uppercase) for _ in range(10)
        )
        setattr(gym_fin.envs, cls_name, env_cls)
        # print(dir(gym_fin.envs))
        # print(cls_name)
        # print(env_cls)
        # print("FinBase-{}-v0".format(step_function))
        # print("gym_fin.envs:{}".format(cls_name))
        register(
            id="FinBase-{}-v0".format(step_function),
            entry_point="gym_fin.envs:{}".format(cls_name),
            kwargs={},
        )

except ImportError as e:
    # ModuleNotFoundError not available on Python version < 3.6
    if os.environ.get("RELAX_IMPORTS") == "true":
        warnings.warn(str(e))
    else:
        raise e

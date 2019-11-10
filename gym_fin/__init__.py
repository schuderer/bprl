import logging
import os
import warnings

__version__ = "0.0.1"

logger = logging.getLogger(__name__)

try:
    from gym.envs.registration import register

    register(
        id="Pension-v0",
        entry_point="gym_fin.envs.pension_env:PensionEnv",
        kwargs={}
        # See gym.envs.registration:EnvSpec for possible arguments
    )

    from random import choices
    import string
    import gym_fin.envs
    from gym_fin.envs.sim_env import env_metadata, generate_env
    from gym_fin.envs.fin_base_sim import FinBaseSimulation

    for step_function in env_metadata["step_views"].keys():
        env_cls = generate_env(FinBaseSimulation(), step_function)
        cls_name = "_" + "".join(choices(string.ascii_uppercase, k=10))
        setattr(gym_fin.envs, cls_name, env_cls)
        env_id = "FinBase-{}-v0".format(step_function)
        logger.debug(f"Registering env with id: {env_id}")
        # print(dir(gym_fin.envs))
        # print(cls_name)
        # print(env_cls)
        # print("FinBase-{}-v0".format(step_function))
        # print("gym_fin.envs:{}".format(cls_name))
        register(
            id=env_id,
            entry_point="gym_fin.envs:{}".format(cls_name),
            kwargs={},
        )

except ImportError as e:
    # ModuleNotFoundError not available on Python version < 3.6
    if os.environ.get("RELAX_IMPORTS") == "true":
        warnings.warn(str(e))
    else:
        raise e

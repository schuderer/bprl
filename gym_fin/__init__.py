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
except ModuleNotFoundError as e:
    if os.environ.get("RELAX_IMPORTS") == "true":
        warnings.warn(e)
    else:
        raise e

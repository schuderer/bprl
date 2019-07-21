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
except ImportError as e:
    # ModuleNotFoundError not available on Python version < 3.6
    if os.environ.get("RELAX_IMPORTS") == "true":
        warnings.warn(str(e))
    else:
        raise e

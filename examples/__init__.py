from gym.envs.registration import register

from examples.pension_env_def import get_env_cls

try:
    register(
        id="PensionExample-v0",
        entry_point=get_env_cls(),
    )
except:
    raise

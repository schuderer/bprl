from gym.envs.registration import register

__version__ = '0.0.1'

register(
    id='Pension-v0',
    entry_point='gym_fin.envs:PensionEnv',
    kwargs={}
    # See gym.envs.registration:EnvSpec for possible arguments
)

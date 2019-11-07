import logging

import gym

# from simulation_base import FinBaseSimulation
# from fin_env import generate_env

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# test code
# env = generate_env(Simulation(), "sim_env.Entity.choose_some_action")
env = gym.make("gym_fin:FinBase-gym_fin.envs.sim_env.Entity.choose_some_action-v0")

obs = env.reset()
obs_space = env.observation_space
print(f"obs_space={obs_space}")
action_space = env.action_space
print(f"action_space={action_space}")

done = False
while not done:
    obs, reward, done, info = env.step(action_space.sample())
    print(f"obs {obs}, reward {reward}, done {done}, info {info}")

import logging

import gym

# from simulation_base import FinBaseSimulation
# from fin_env import generate_env

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

#############
# pseudo-plugin code, imagine we imported this as a module

from random import choice
from gym_fin.envs.sim_env import attach_handler

last_active = True


def before_entity_step_handler(self):
    global last_active
    last_active = self.active


def after_entity_step_handler(self, _result=None):
    global last_active
    if last_active and not self.active:
        print("50% chance that grue doesn't like what it has eaten just now...")
        self.active = choice([True, False])
        if self.active:
            print("Entity got spit out and is alive again!")
        else:
            print("Sorry, no dice.")


entity_step_name = "gym_fin.envs.fin_base_sim.Entity.perform_increment"
attach_handler(before_entity_step_handler, entity_step_name, "before")
attach_handler(after_entity_step_handler, entity_step_name, "after")

# end of pseudo-plugin code
#############

# test code
# env = generate_env(Simulation(), "sim_env.Entity.choose_some_action")
env = gym.make("gym_fin:FinBase-gym_fin.envs.fin_base_sim.Entity.choose_some_action-v0")

obs = env.reset()
obs_space = env.observation_space
print(f"obs_space={obs_space}")
action_space = env.action_space
print(f"action_space={action_space}")

done = False
while not done:
    obs, reward, done, info = env.step(action_space.sample())
    print(f"obs {obs}, reward {reward}, done {done}, info {info}")

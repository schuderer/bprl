import logging
import sys

import numpy as np
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.ppo2 import PPO2
from stable_baselines.a2c import A2C

# import gym_fin


logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)

env = gym.make('gym_fin:Pension-v0')
# vectorized environments allow to easily multiprocess training
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# model = PPO2(MlpPolicy, env, verbose=0)
model = A2C(MlpPolicy, env, verbose=0)


def evaluate(model, num_steps=1000):
    """Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


# Random Agent, before training
mean_reward_before_train = evaluate(model, num_steps=10000)

# Train the agent for 10000 steps
model.learn(total_timesteps=5000000)

# Evaluate the trained agent
mean_reward = evaluate(model, num_steps=100000)

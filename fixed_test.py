# Test code (fixed business policy):

from envs.pension_env import PensionEnv
import sys
import numpy as np

env = PensionEnv()
# env.logger = stderr

print("State space", env.observation_space)
if hasattr(env.observation_space, "low"):
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)

print("Action space:", env.action_space)
if hasattr(env.action_space, "low"):
    print("- low:", env.action_space.low)
    print("- high:", env.action_space.high)


env.logger = sys.stdout

EPISODES = 3
MAX_STEPS = 20000

overall = 0
for episode in range(EPISODES):
    cumul_reward = 0
    observation = env.reset()
    for t in range(MAX_STEPS):
        #env.render()
        prev_obs = observation
        action = 0
        age = prev_obs[0]
        if age < 67:
            action = 0  # -1500
        else:
            action = 1  # +12000
        observation, reward, done, info = env.step(action)
        cumul_reward += reward

        # h = info["human"]
        # c = info["company"]
        # print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
        if hasattr(env, "logger") and env.logger:
            print("year", info["year"],
                    "funds", info["company"].funds,
                    "reputation", info["company"].reputation,
                    "humans", len([h for h in env.humans if h.active]),
                    "meanAge", np.mean([h.age for h in env.humans]),
                    "currAge", info["human"].age,
                    "hFunds", info["human"].funds,
                    "hID", info["human"].id,
                    "stateActionkey", "0-0-0-"+str(action), file=env.logger)

        if done:
            print("Episode {} finished after {} timesteps with cumulative reward {}".format(episode, t+1, cumul_reward))
            overall += cumul_reward
            break

print("Overall reward", overall/EPISODES)


#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)

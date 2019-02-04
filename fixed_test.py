# Test code (fixed business policy):

from envs.pension_env import PensionEnv
from math import floor
from sys import stderr

env = PensionEnv()
#env.logger = stderr

print("State space",env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

print("Action space:", env.action_space)
print("- low:", env.action_space.low)
print("- high:", env.action_space.high)

EPISODES = 10
MAX_STEPS = 2000

overall = 0
for episode in range(EPISODES): # just one for now
    cumul_reward = 0
    observation = env.reset()
    for t in range(MAX_STEPS):
        #env.render()
        prev_obs = observation
        action = 0
        age = prev_obs[0]
        if age < 67:
            action = -1500
        else:
            action = +12000
        observation, reward, done, info = env.step(action)
        cumul_reward += reward
        
        h = info["human"]
        c = info["company"]
        print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
        if done:
            print("Episode {} finished after {} timesteps with average reward {}".format(episode, t+1, cumul_reward/(t+1)))
            overall += cumul_reward/(t+1)
            break

print("Overall average reward", overall/EPISODES)


#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)





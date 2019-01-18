# Test code (fixed business policy):

from env.pension_env import PensionEnv
from math import floor

env = PensionEnv()

print("State space",env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

print("Action space:", env.action_space)
print("- low:", env.action_space.low)
print("- high:", env.action_space.high)

EPISODES = 1
MAX_STEPS = 10

for episode in range(EPISODES): # just one for now
    observation = env.reset()
    for t in range(MAX_STEPS):
        #env.render()
        prev_obs = observation
        action = 0
        age = prev_obs[0]
        if age < 67:
            action = 3000
        else:
            action = -1000
        observation, reward, done, info = env.step(action)
        
        h = info["human"]
        c = info["company"]
        print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)





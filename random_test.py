# Test code (random business policy):

from envs.pension_env import PensionEnv

env = PensionEnv()
#env.logger = stderr

print("State space",env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

print("Action space:", env.action_space)
print("- low:", env.action_space.low)
print("- high:", env.action_space.high)

EPISODES = 10

overall = 0
for episode in range(EPISODES):
    cumul_reward = 0
    observation = env.reset()
    for t in range(2000):
        #env.render()
        prev_obs = observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        currentWorld = (prev_obs, action, reward, observation)
        
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




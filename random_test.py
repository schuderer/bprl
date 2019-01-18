# Test code (random business policy):

from env.pension_env import PensionEnv

env = PensionEnv()

experience = []
for episode in range(1): # just one for now
    observation = env.reset()
    for t in range(1000):
        #env.render()
        prev_obs = observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        currentWorld = (prev_obs, action, reward, observation)
        #experience.append(currentWorld)
        h = info["human"]
        c = info["company"]
        print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)




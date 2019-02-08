# Test code (simple q learner (discretized state & action space)):

import sys
import random
import numpy as np
import gym
from envs.pension_env import PensionEnv
from discretize import Discretizer

env = PensionEnv()
# env = gym.make('Pendulum-v0')
# env = gym.make('CartPole-v0')
# env = gym.make('FrozenLake-v0')
# env = gym.make('MountainCar-v0')

num_bins = 12
log_bins = True
print({"num_bins": num_bins, "log_bins": log_bins})

statesDisc = None
print("State space", env.observation_space)
if type(env.observation_space) is gym.spaces.box.Box:
    discreteStates = False
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)
    state_bins = np.repeat([num_bins], len(env.observation_space.low))
    state_discretizer = Discretizer(env.observation_space.low,
                                    env.observation_space.high,
                                    state_bins,
                                    log=log_bins)
    print("state grid", state_discretizer.grid)
    # todo: check for missing dims
    statesDisc = gym.spaces.discrete.Discrete(state_discretizer.grid.size)
elif type(env.observation_space) is gym.spaces.discrete.Discrete:
    discreteStates = True
    statesDisc = env.observation_space
else:
    raise NotImplementedError("Can only work with Discrete or Box type state spaces.")

actionsDisc = None
print("Action space:", env.action_space)
if type(env.action_space) is gym.spaces.box.Box:
    discreteActions = False
    print("Action space - low:", env.action_space.low)
    print("Action space - high:", env.action_space.high)
    action_bins = np.repeat([num_bins], len(env.action_space.low))
    action_discretizer = Discretizer(env.action_space.low,
                                     env.action_space.high,
                                     action_bins,
                                     log=log_bins)
    print("action grid", action_discretizer.grid)
    # todo: check other dims:
    actionsDisc = gym.spaces.discrete.Discrete(action_discretizer.grid.size)
elif type(env.action_space) is gym.spaces.discrete.Discrete:
    discreteActions = True
    actionsDisc = env.action_space
else:
    raise NotImplementedError(
            "Can only work with Discrete or Box type action spaces.")

# disc = discretize([35, 1000, 10000], state_grid)
# print(disc)
# print(undiscretize(disc, state_grid))


def stateKeyFor(discreteObs):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    return "-".join([str(elem) for elem in discreteObs])


def getQTableDefault():
    return 0  # -10000000
    # return random.random()-0.5


def getActions(discreteObs, qTable):
    keyStart = stateKeyFor(discreteObs)
    return [qTable.get(keyStart + "-" + str(aIdx), getQTableDefault())
            for aIdx in range(actionsDisc.n)]


def maxQ(discreteObs, qTable):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    actions = getActions(discreteObs, qTable)
    # bestActionIdx = np.argmax(actions)  # tie breaking: first item
    # tie breaking: random item:
    bestActionIdx = np.random.choice(np.flatnonzero(actions == np.max(actions)))
    # testMaxQ = maxQ_old(discreteObs, qTable)
    # if testMaxQ[0] != bestActionIdx or testMaxQ[1] != actions[bestActionIdx]:
    #     print("WARNING: {} != {} or {} != {}".format(testMaxQ[0], bestActionIdx, testMaxQ[1], actions[bestActionIdx]))
    #     return 1/0
    return bestActionIdx, actions[bestActionIdx]


def maxQ_old(discreteObs, qTable):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    keyStart = stateKeyFor(discreteObs)
    bestActionIdx = 0
    highestVal = qTable.get(keyStart + "-" + str(bestActionIdx), getQTableDefault())
    for aIdx in range(1, actionsDisc.n):  # start at 1 because we already visited 0
        key = keyStart + "-" + str(aIdx)
        if key in qTable:
            val = qTable[key]
            if val > highestVal:
                highestVal = val
                bestActionIdx = aIdx
            # print("maxQ found something", key, aIdx, val)
    # if highestVal == -np.inf:
    #     highestVal = 0.0
    return bestActionIdx, highestVal


def print_q(qTable):
    for s in range(statesDisc.n):
        if discreteStates:
            s = np.array(s)
        else:  # todo
            # indices = np.unravel_index(range(statesDisc.n), state_grid.shape)
            s = state_discretizer.grid[s]
        print([qTable.get(stateKeyFor(s) + "-" + str(a), getQTableDefault()) for a in range(actionsDisc.n)])


def q_learn(env,
            alpha_min=0.01,
            alpha_decay=1,  # default 1 = fixed alpha (alpha_min)
            gamma=0.99,
            epsilon_min=0.1,
            epsilon_decay=1,  # default: 1 = fixed epsilon (epsilon_min)
            episodes=100,
            max_steps=1000,
            q_table={},
            average_rewards=False):
    '''Generic Q-Learning algorithm. Returns the Q-Table.'''
    print(locals())
    # todo make self-contained

    alpha = 1.0  # initial alpha, will decay
    epsilon = 1.0  # initial epsilon, will decay
    overall = 0
    last_100 = np.zeros((100,))
    aggr_text = "average" if average_rewards else "cumulative"
    for episode in range(episodes):
        # decay defore first episode
        # to make fixed (min_)alpha/epsilon possible with *_decay=1
        alpha = max(alpha * (1-alpha_decay), alpha_min)
        epsilon = max(epsilon * (1-epsilon_decay), epsilon_min)

        cumul_reward = 0
        observation = env.reset()
        # print("size of q table:", len(q_table.keys()))
        # lastHumanId = -1
        for t in range(max_steps):
            # env.render()
            prev_obs = np.array(observation) if discreteStates else state_discretizer.discretize(observation)
            # print("Observation:")
            # print(observation)
            # print("becomes discretized into:")
            # print(prev_obs)

            if alpha == 0 and hasattr(env, "render"):
                env.render()

            # Select action according to epsilon-greedy policy
            actionIdx = actionsDisc.sample()  # random
            if random.random() > epsilon:  # greedy/argmax
                actionIdx = maxQ(prev_obs, q_table)[0]
            action = actionIdx if discreteActions else action_discretizer.undiscretize([actionIdx])
            # print("chosen actionIdx {}, action {}".format(actionIdx, action))

            # Take action
            observation, reward, done, info = env.step(action)
            curr_obs = np.array(observation) if discreteStates else state_discretizer.discretize(observation)
            # print(maxQ(curr_obs, q_table))

            # Update the state that we acted on
            stateActionKey = stateKeyFor(prev_obs)+"-"+str(actionIdx)
            # print(stateActionKey)
            if hasattr(env, "logger") and env.logger:
                print("year", info["year"],
                        "funds", info["company"].funds,
                        "humans", len([h for h in env.humans if h.active]),
                        "currAge", info["human"].age,
                        "hFunds", info["human"].funds,
                        "hID", info["human"].id,
                        "stateActionkey", stateActionKey, file=env.logger)

            qValOld = q_table.get(stateActionKey, getQTableDefault())
            qValNew = qValOld + alpha*(reward + gamma*maxQ(curr_obs, q_table)[1] - qValOld)
            # if qValOld != getQTableDefault():
            #     print(stateActionKey, qValOld, "<--", qValNew, "reward:",reward)
            q_table[stateActionKey] = qValNew

            # h = info["human"]
            # if (h.id != lastHumanId):
            #    print("###### lastHuman {} != h {}".format(lastHumanId, h.id))
            # else:
            #    print("######## A-OK")
            # lastHumanId = info["nextHuman"].id
            # c = info["company"]
            # print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
            cumul_reward += reward
            if done:
                aggr_reward = cumul_reward / (t+1) if average_rewards else cumul_reward
                overall += aggr_reward
                last_100[episode % 100] = aggr_reward
                print("Episode {} finished after {} timesteps with {} reward {} (last 100 mean = {})".format(episode, t+1, aggr_text, aggr_reward, last_100.mean()))
                if type(env).__name__ == "PensionEnv":
                    print("year {}, q table size {}, epsilon {}, alpha {}, #humans {}, reputation {}".format(env.year, len(q_table.keys()), epsilon, alpha, len([h for h in env.humans if h.active]), info["company"].reputation))
                else:
                    print("q table size {}, epsilon {}, alpha {}".format(len(q_table.keys()), epsilon, alpha))
                break
    print("Overall {} reward: {}".format(aggr_text, overall/episodes))
    print("Average reward last 100 episodes:", last_100.mean())
    return q_table


# Run Q-Learning

# PensionEnv
#print("Each episode takes 750 years (with one time step per year per human).")

print("\nLEARNING:\n")


# env.logger = sys.stderr

qTable = q_learn(env,
                 alpha_min=0.01,     # temperature/learning rate, was 0.01
                 alpha_decay=1,      # reduction factor per episode, was 0.003
                 gamma=0.99,         # discount factor, was 0.99
                 epsilon_min=0.03,   # minimal epsilon (exploration rate for e-greedy policy), was 0.03
                 epsilon_decay=1,    # reduction per episode, was 0.003
                 episodes=10000,
                 max_steps=20000,    # abort episode after this number of steps
                 q_table={},
                 average_rewards=False)


# test run:


print("\nTESTING:\n")

if type(env).__name__ == "PensionEnv":
    env.logger = sys.stdout

qTable2 = q_learn(env,
                 alpha_min=0,       # temperature/learning rate
                 alpha_decay=1,     # reduction factor per episode, was 0.003
                 gamma=0,           # discount factor, was 0.99
                 epsilon_min=0.0,   # minimal epsilon (exploration rate for e-greedy policy)
                 epsilon_decay=1,   # reduction factor per episode, was 0.003
                 episodes=3,
                 max_steps=20000,   # abort episode after this number of steps
                 q_table=qTable,
                 average_rewards=False)



# qTable = q_learn(env,
#                  alpha_min=0.01,  # temperature/learning rate
#                  gamma=0.95,  # discount factor, was 0.95
#                  epsilon_min=0.03,    # minimal epsilon (exploration rate for e-greedy policy)
#                  decay=0.995,  #  reduction factor per episode, was 0.997
#                  episodes=1000,
#                  max_steps=2000,  # abort episode after this number of steps
#                  q_table={})


print(qTable)

# (year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)

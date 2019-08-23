# Test code (simple q learner (discretized state & action space)):

#from envs.pension_env import PensionEnv
import gym
import random
import numpy as np

# todo: same as just log2-transforming, then splitting and exp2-ing?? Then maybe can handle bi-directional (0-centered) exp-splits...
def createExpSplits(high, bins, inverse):
    result = []
    range = high
    currBin = bins - 1
    currBorder = range / 2.0
    while currBin > 0:
        if (inverse):
            result.append(currBorder)
        else:
            result.insert(0, currBorder)
        currBin -= 1
        currBorder = currBorder / 2.0
    return result

def createExpGrid(highs, bins, inverse=False):
    return np.array([createExpSplits(h, b, inverse) for h,b in zip(highs,bins)])

testGrid = createExpGrid([256, 1024], np.repeat([8],2))
print("testGrid:")
print(testGrid)

def discretize(vals, grid):
    # from https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb
    return list(int(np.digitize(s, g)) for s, g in zip(vals, grid))

def undiscretize(indices, grid):
    #print("indices",indices,"grid",grid)
    #for i, g in zip(indices, grid):
    #    print("bla",i,g,"blu")
    return list(g[int(i)] for i, g in zip(indices, grid))


print(discretize([40, 12], testGrid))
print("end of test")

#env = PensionEnv()
env = gym.make('FrozenLake-v0')

state_grid = None
statesDisc = None
print("State space", env.observation_space)
if type(env.observation_space) is gym.spaces.box.Box:
    discreteStates = False
    print("- low:", env.observation_space.low)
    print("- high:", env.observation_space.high)
    state_bins = np.repeat([10],len(env.observation_space.low))
    state_grid = np.hstack((
    -createExpGrid(abs(env.observation_space.low), state_bins, inverse=True),
    createExpGrid(env.observation_space.high, state_bins)))
    print("state_grid",state_grid)
    statesDisc = gym.spaces.discrete.Discrete(len(state_grid[0])) # todo: check for missing dims
elif type(env.observation_space) is gym.spaces.discrete.Discrete:
    discreteStates = True
    statesDisc = env.observation_space
else:
    raise NotImplementedError("Can only work with Discrete or Box type state spaces.")

action_grid = None
actionsDisc = None
print("Action space:", env.action_space)
if type(env.action_space) is gym.spaces.box.Box:
    discreteActions = False
    print("Action space - low:", env.action_space.low)
    print("Action space - high:", env.action_space.high)
    action_bins = np.repeat([10],len(env.action_space.low))
    action_grid = np.hstack((
    -createExpGrid(abs(env.action_space.low), action_bins, inverse=True),
    createExpGrid(env.action_space.high, action_bins)))
    print("action_grid",action_grid)
    actionsDisc = gym.spaces.discrete.Discrete(len(action_grid[0]))
elif type(env.action_space) is gym.spaces.discrete.Discrete:
    discreteActions = True
    actionsDisc = env.action_space
else:
    raise NotImplementedError("Can only work with Discrete or Box type action spaces.")

#disc = discretize([35, 1000, 10000], state_grid)
#print(disc)
#print(undiscretize(disc, state_grid))

def stateKeyFor(discreteObs):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    return "-".join([str(elem) for elem in discreteObs])

def maxQ(discreteObs, qTable, defaultAction=0):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    highestVal = -np.inf
    bestActionIdx = defaultAction
    keyStart = stateKeyFor(discreteObs)
    for aIdx in range(0, actionsDisc.n):
        key = keyStart + "-" + str(aIdx)
        if key in qTable:
            val = qTable[key]
            if (val > highestVal):
                highestVal = val
                bestActionIdx = aIdx
            #print("maxQ found something", key, aIdx, val)
    if highestVal == -np.inf:
        highestVal = 0.0
    return(bestActionIdx, highestVal)

def print_q(qTable):
    for s in range(statesDisc.n):
        s = np.array(s)
        print([qTable.get(stateKeyFor(s) + "-" + str(a), 0) for a in range(actionsDisc.n)])

def q_learn(env, alpha=0.2, gamma=0.99, epsilon=0.1, epsilon_decay=1.0, epsilon_min=0.0, episodes=100, max_steps=1000, q_table={}):
    '''Generic Q-Learning algorithm. Returns the Q-Table.'''
    # todo make self-contained
    for episode in range(episodes):
        cumReward = 0
        observation = env.reset()
        #print("size of q table:", len(q_table.keys()))
        print("epsilon:", epsilon)
        for t in range(max_steps):
            #env.render()
            prev_obs = np.array(observation) if discreteStates else discretize(observation, state_grid) 
            
            # Select action according to epsilon-greedy policy
            actionIdx = actionsDisc.sample() # random
            if random.random() > epsilon: # greedy/argmax
                actionIdx = maxQ(prev_obs, q_table, defaultAction=0)[0]
            action = actionIdx if discreteActions else undiscretize([actionIdx], action_grid)[0]
            #print("chosen action", actionIdx)
            
            # Take action
            observation, reward, done, info = env.step(action)
            curr_obs = np.array(observation) if discreteStates else discretize(observation, state_grid)
            #print(maxQ(curr_obs, q_table))
            
            # Update the state that we acted on
            stateActionKey = stateKeyFor(prev_obs)+"-"+str(actionIdx)
            #print(stateActionKey)
            
            qValOld = q_table.get(stateActionKey, 0.0)
            qValNew = qValOld + alpha*(reward + gamma*maxQ(curr_obs, q_table)[1] - qValOld)
            #if qValOld != 0.0:
            #    print(stateActionKey, qValOld, "<--", qValNew)
            q_table[stateActionKey] = qValNew
            
            #h = info["human"]
            #c = info["company"]
            #print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
            cumReward += reward
            if done:
                print("Episode {} finished after {} timesteps with cumulative reward {}".format(episode, t+1, cumReward))
                epsilon = max(epsilon * epsilon_decay, epsilon_min)
                break
    return q_table


# Run Q-Learning

qTable = q_learn(env,
                 alpha=0.2,   # temperature/learning rate
                 gamma=0.99,  # discount factor
                 epsilon=1.0, # initial exploration rate for e-greedy policy
                 epsilon_decay=0.995, # epsilon reduction factor per episode
                 epsilon_min=0.01,    # minimal epsilon
                 episodes=1000,
                 max_steps=10000, # abort episode after this number of steps
                 q_table={})

print_q(qTable)

#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)





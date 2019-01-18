# Test code (simple q learner (discretized state & action space)):

#from envs.pension_env import PensionEnv
import gym
import random
import numpy as np

# Some primitive discretization helpers
# Same-size bins:
def discretizeSimple(val, min, max, numBins):
    binWidth = (max-min) / numBins
    return int((val-min) / binWidth)

# Exponential binning helper:
def discretizeExpPositive(val, min, max, bins):
    range = max - min
    valRel = val - min
    currBin = bins - 1
    currBorder = range / 2.0
    while valRel < currBorder and currBin > 0:
      currBin -= 1
      currBorder = currBorder / 2.0
    return currBin

# Exponential binning (furthest bin = furthest half, next closest bin = furthest half of closer half, etc.)
def discretizeExp(val, center, max, bins):
    #if (max < 0 or min < 0 or bins < 1):
    #  raise IllegalArgumentError("min and max must be 0.0 or greater, bins must be 1 or greater")
    relVal = abs(val) - center
    if val >= center:
      return discretizeExpPositive(relVal, 0, max-center, bins)
    else:
      return -discretizeExpPositive(abs(relVal), 0, max-center, bins)


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
#print(testGrid)
#print(discretizeExp(-3, 0, 1000, 10))

def discretize(vals, grid):
    # from https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb
    return list(int(np.digitize(s, g)) for s, g in zip(vals, grid))

def undiscretize(indices, grid):
    #print("indices",indices,"grid",grid)
    #for i, g in zip(indices, grid):
    #    print("bla",i,g,"blu")
    return list(g[int(i)] for i, g in zip(indices, grid))


#print(discretize([40, 12], testGrid))

#env = PensionEnv()
env = gym.make('FrozenLake-v0')

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
    raise NotImplementedError()

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
    raise NotImplementedError()

#disc = discretize([35, 1000, 10000], state_grid)
#print(disc)
#print(undiscretize(disc, state_grid))

def stateKeyFor(discreteObs):
    if discreteObs.shape == ():
        discreteObs = np.reshape(discreteObs, (1,))
    return "-".join([str(elem) for elem in discreteObs])

def maxQ(discreteObs, defaultAction=0):
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

EPISODES = 1000
MAX_STEPS = 100000

STATE_BINS = 10
ACTION_BINS = 10

epsilon = 1.0 # exploration rate for e-greedy policy
epsilon_decay = 0.995
epsilon_min = 0.01
alpha = 0.2 # temperature/learning rate
gamma = 0.99 # discount factor

qTable = {} # using dict instead of table, where key = state-action and value = value

def tmpVisQTable():
    for s in range(statesDisc.n):
        s = np.array(s)
        print([qTable.get(stateKeyFor(s) + "-" + str(a), 0) for a in range(actionsDisc.n)])

for episode in range(EPISODES):
    cumReward = 0
    observation = env.reset()
    #print("size of q table:", len(qTable.keys()))
    print("epsilon:", epsilon)
    for t in range(MAX_STEPS):
        #env.render()
        prev_obs = np.array(observation) if discreteStates else discretize(observation, state_grid) 
        
        # Select action according to epsilon-greedy policy
        actionIdx = actionsDisc.sample()
        if random.random() > epsilon: # greedy/argmax
            actionIdx = maxQ(prev_obs, defaultAction=0)[0]
        action = actionIdx if discreteActions else undiscretize([actionIdx], action_grid)[0]
        #print("chosen action", actionIdx)
        
        # Take action
        observation, reward, done, info = env.step(action)
        curr_obs = np.array(observation) if discreteStates else discretize(observation, state_grid)
        #print(maxQ(curr_obs))
        
        # Update the state that we acted on
        stateActionKey = stateKeyFor(prev_obs)+"-"+str(actionIdx)
        #print(stateActionKey)
        
        qValOld = qTable.get(stateActionKey, 0.0)
        qValNew = qValOld + alpha*(reward + gamma*maxQ(curr_obs, defaultAction=actionsDisc.sample())[1] - qValOld)
        #if qValOld != 0.0:
        #    print(stateActionKey, qValOld, "<--", qValNew)
        qTable[stateActionKey] = qValNew
        
        #h = info["human"]
        #c = info["company"]
        #print(info["year"], "human:", h.id, h.age, h.funds, h.lastTransaction, h.happiness, "reward:", reward, "company:", c.funds, c.reputation)
        cumReward += reward
        if done:
            print("Episode {} finished after {} timesteps with cumulative reward {}".format(episode, t+1, cumReward))
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            #tmpVisQTable()
            break

tmpVisQTable()

#(year, "human:", i, h.age, fundsBefore, h.funds, h.happiness, "reward:", r, "company:", companies[0].funds, companies[0].reputation)





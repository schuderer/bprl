import math
import random
import numpy as np
from utils import LazyStr
# from utils import do_profile
from discretizer import create_discretizers
import logging

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, env,
                 q_table={},
                 gamma=0.99,
                 min_alpha=0.01,
                 min_epsilon=0.1,
                 alpha_decay=1,  # default 1 = fixed alpha (min_alpha)
                 epsilon_decay=1,  # default: 1 = fixed epsilon (min_epsilon)
                 default_value=0,
                 # average_rewards=False,
                 discretize_bins=12,
                 discretize_log=False
                 ):
        """Initialize the reinforcement learning agent.

            Params:
                env: the openai gym compatible environment to use
                q_table: initial q table (optional, for continuing with pre-trained agent,
                         empty by default)
                gamma: discount factor (1 = endless deferred reward, 0 = myopic)
                min_alpha: min learning rate (1 = newest observations is new truth, 0 = no learning),
                           alpha will decay from 1 towards this value over time using alpha_decay
                min_epsilon: min probability of exploration (0 = no exploration, 1 = no exploitation),
                             epsilon will decay from 1 towards this value over time using epsilon_decay
                alpha_decay: (default: 1.0 == instant decay) simulated annealing,
                             decay learning rate alpha towards zero by this factor each episode
                epsilon_decay: (default: 1.0 == instant decay) decay epsilon
                               towards no exploration by this factor each episode
                default_value=0: initialize state-action value (Q) table with this value
                discretize_bins: discretize any non-discrete state dimensions with this number of bins
                discretize_log: use logarithmically varying instead of equal bin sizes
                                (the farther from zero, the coarser the bin size)

        """
        # average_rewards: user average instead of cumulative rewards
        logger.info(locals())

        self.env = env
        self.q_table = q_table
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.alpha = 1
        self.epsilon = 1
        self.default_value = default_value
        # self.average_rewards = average_rewards


        # Todo: move q-table related stuff to external Approximator
        self.state_disc, self.action_disc = \
            create_discretizers(self.env, discretize_bins, discretize_log)

    @staticmethod
    def _stateKeyFor(discreteObs):
        if discreteObs.shape == ():
            d_obs = np.reshape(discreteObs, (1,))
        else:
            d_obs = discreteObs
        # slow using timeit():
        return '-'.join(map(str, d_obs))
        # 100x faster using timeit(), but only a few percent faster in reality:
        # return reduce(lambda a, v: str(a) + '-' + str(v), discreteObs)

    @staticmethod
    def _getMaxRandomTieBreak(actions):
        # twice as fast as
        # np.random.choice(np.flatnonzero(actions == np.max(actions)))
        best = []
        bestActionVal = max(actions)
        for i, aval in enumerate(actions):
            if aval == bestActionVal:
                best.append(i)
        if len(best) == 1:
            return best[0]
        else:
            return np.random.choice(best)


    def _getActions(self, key_start, q_table):
        return [q_table.get(key_start + '-' + str(aIdx), self.default_value)
                for aIdx in range(self.action_disc.space.n)]

    def _maxQ(self, state_key, q_table):
        actions = self._getActions(state_key, q_table)
        logger.debug(f"Actions for state {state_key}: %s", actions)
        # bestActionIdx = np.argmax(actions)  # tie breaking: first item
        # tie breaking: random item:
        bestActionIdx = self._getMaxRandomTieBreak(actions)
        # bestActionIdx2 = np.random.choice(np.flatnonzero(actions == np.max(actions)))
        # if actions[bestActionIdx] != actions[bestActionIdx2]:
        #     raise ValueError("#### ERROR IN getMaxRandomTieBreak: max values are not the same! ####")

        # logger.debug(f"Best action for state {stateKey}: %s", bestActionIdx)
        # testMaxQ = maxQ_old(discreteObs, qTable)
        # if testMaxQ[0] != bestActionIdx or testMaxQ[1] != actions[bestActionIdx]:
        #     print('WARNING: {} != {} or {} != {}'.format(testMaxQ[0], bestActionIdx, testMaxQ[1], actions[bestActionIdx]))
        return bestActionIdx, actions[bestActionIdx]

    def _maxQ_old(self, discreteObs, qTable):
        if discreteObs.shape == ():
            discreteObs = np.reshape(discreteObs, (1,))
        keyStart = self._stateKeyFor(discreteObs)
        bestActionIdx = 0
        highestVal = qTable.get(keyStart + '-' + str(bestActionIdx), self.default_value)
        for aIdx in range(1, self.action_disc.space.n):  # start at 1 because we already visited 0
            key = keyStart + '-' + str(aIdx)
            if key in qTable:
                val = qTable[key]
                if val > highestVal:
                    highestVal = val
                    bestActionIdx = aIdx
                # print('maxQ found something', key, aIdx, val)
        # if highestVal == -np.inf:
        #     highestVal = 0.0
        return bestActionIdx, highestVal

    def print_q(self, qTable):
        for s in range(self.state_disc.space.n):
            if self.state_disc.grid is None:
                s = np.array(s)
            else:  # todo
                # indices = np.unravel_index(range(statesDisc.n), state_grid.shape)
                s = self.state_disc.grid[s]
            logger.info([qTable.get(self._stateKeyFor(s) + '-' + str(a), self.default_value)
                            for a in range(self.action_disc.space.n)])

    def print_q_frozenlake(self, qTable):
        for s in range(self.state_disc.space.n):
            s = np.array(s)
            print([qTable.get(self._stateKeyFor(s) + "-" + str(a), 0)
                    for a in range(self.action_disc.space.n)])

    # @do_profile(follow=[env.step, penv.Client.live_one_year])
    # @do_profile(follow=[maxQ, getActions, getMaxRandomTieBreak])
    def run_episode(self, max_steps=math.inf, exploit=False):
        """Generic Q-Learning algorithm. Returns the Q-Table.

        Params:
            max_steps: (default: infinite) maximum episode length
            exploit: (default: False) no learning, only exploitation

        Returns:
            q_table, cumulative reward of episode, number of steps, last env info
        """

        # apply decay once before episode to make constant alpha/epsilon possible
        # to use fixed alpha/epsilon, use (min_)alpha/epsilon together with *_decay=1
        self.alpha = max(self.alpha * (1-self.alpha_decay), self.min_alpha)
        self.epsilon = max(self.epsilon * (1-self.epsilon_decay), self.min_epsilon)
        alpha = 0 if exploit else self.alpha
        epsilon = 0 if exploit else self.epsilon

        logger.debug('size of q table: %s', len(self.q_table.keys()))

        cumul_reward = 0
        observation = self.env.reset()
        prev_obs = self.state_disc.discretize(observation)
        prev_state_key = self._stateKeyFor(prev_obs)
        prev_best_action, prev_best_value = self._maxQ(prev_state_key, self.q_table)
        t = 0
        while t < max_steps:
            t += 1
            # self.env.render()
            logger.debug('Observation: %s', observation)
            logger.debug('becomes discretized into: %s', prev_obs)

            if exploit and hasattr(self.env, 'render'):
                self.env.render()

            # Select action according to epsilon-greedy policy
            if random.random() > epsilon:  # greedy/argmax
                # Would be faster, but would ignore last update
                # actionIdx = prev_best_action
                actionIdx, _ = self._maxQ(prev_state_key, self.q_table)
                logger.debug("greedy action: %s", actionIdx)
            else:
                actionIdx = self.action_disc.space.sample()  # random
                logger.debug("random action: %s", actionIdx)
            action = self.action_disc.undiscretize(actionIdx)
            logger.debug('chosen actionIdx %s (%s), action %s (%s)',
                         actionIdx, type(actionIdx), action, type(action))

            # Take action
            observation, reward, done, info = self.env.step(action)

            curr_obs = self.state_disc.discretize(observation)
            curr_state_key = self._stateKeyFor(curr_obs)
            curr_best_action, curr_best_value = self._maxQ(curr_state_key, self.q_table)

            logger.debug('maxQ(curr_state_key, q_table): (%s, %s)',
                         curr_best_action, curr_best_value)

            # Update the state that we acted on
            stateActionKey = prev_state_key + '-' + str(actionIdx)
            logger.debug('stateActionKey: %s', stateActionKey)
            if type(self.env).__name__ == 'PensionEnv':
                logger.info('year %s funds %s reputation %s humans %s meanAge %s '
                            + 'currAge %s hFunds %s hID %s stateActionkey %s',
                            info['year'], info['company'].funds, info['company'].reputation,
                            len([h for h in self.env.humans if h.active]),
                            LazyStr(np.mean, [h.age for h in self.env.humans]),
                            info['human'].age, info['human'].funds, info['human'].id,
                            stateActionKey)

            qValOld = self.q_table.get(stateActionKey, self.default_value)
            td_error = (reward + self.gamma*curr_best_value) - qValOld
            qValNew = qValOld + alpha*td_error
            # if qValOld != getQTableDefault():
            #     print(stateActionKey, qValOld, '<--', qValNew, 'reward:',reward)
            logger.debug("%s <-- %s + %s * [(%s + %s * %s) - %s]",
                         qValNew, qValOld, alpha, reward, self.gamma, curr_best_value, qValOld)
            self.q_table[stateActionKey] = qValNew

            (prev_obs,
             prev_state_key,
             prev_best_action,
             prev_best_value) = (curr_obs,
                                 curr_state_key,
                                 curr_best_action,
                                 curr_best_value)

            # h = info['human']
            # if (h.id != lastHumanId):
            #    print('###### lastHuman {} != h {}'.format(lastHumanId, h.id))
            # else:
            #    print('######## A-OK')
            # lastHumanId = info['nextHuman'].id
            # c = info['company']
            # print(info['year'], 'human:', h.id, h.age, h.funds, h.lastTransaction, h.happiness, 'reward:', reward, 'company:', c.funds, c.reputation)
            cumul_reward += reward
            if done:
                break

        return self.q_table, cumul_reward, t+1, info

# Stdlib imports
import logging
import math
import random

# Third party imports
import numpy as np

# Application imports
from .utils import LazyStr

logger = logging.getLogger(__name__)


def greedy(list_of_state_action_values, params):
    # twice as fast as
    # np.random.choice(np.flatnonzero(actions == np.max(actions)))
    best = []
    bestActionVal = max(list_of_state_action_values)
    for i, aval in enumerate(list_of_state_action_values):
        if aval == bestActionVal:
            best.append(i)
    if len(best) == 1:
        bestActionIdx = best[0]
    else:
        bestActionIdx = np.random.choice(best)

    return bestActionIdx, bestActionVal


def epsilon_greedy(list_of_state_action_values, params):
    if random.random() > params["epsilon"]:
        action_idx, action_val = greedy(list_of_state_action_values, params)
        logger.debug("greedy action idx: %s", action_idx)
        return action_idx, action_val
    else:
        action_idx = np.random.choice(range(len(list_of_state_action_values)))
        logger.debug("random action idx: %s", action_idx)
        return action_idx, list_of_state_action_values[action_idx]


class Agent:
    def __init__(
        self,
        env,
        q_function,
        update_policy=greedy,
        exploration_policy=epsilon_greedy,
        gamma=0.99,
        min_alpha=0.01,
        min_epsilon=0.1,
        alpha_decay=1,  # default 1 = fixed alpha (min_alpha)
        epsilon_decay=1,  # default: 1 = fixed epsilon (min_epsilon)
    ):
        """Initialize the reinforcement learning agent.

            Params:
                env: the openai gym compatible environment to use
                q_function: initial q function
                update_policy: the policy (a python function reference) which is used
                               to get the value estimate for updates. Default: greedy
                exploration_policy: the exploration policy (a python function reference),
                                    default: epsilon_greedy
                gamma: discount factor (1 = endless deferred reward, 0 = myopic)
                min_alpha: min learning rate (1 = newest observation is new truth, 0 = no learning),
                           alpha will decay from 1 towards this value over time using alpha_decay
                min_epsilon: min probability of exploration (0 = no exploration, 1 = no exploitation),
                             epsilon will decay from 1 towards this value over time using epsilon_decay
                alpha_decay: (default: 1.0 == instant decay) simulated annealing,
                             decay learning rate alpha towards zero by this factor each episode
                epsilon_decay: (default: 1.0 == instant decay) decay epsilon
                               towards no exploration by this factor each episode

        """
        logger.info(locals())

        self.env = env
        self.q_function = q_function
        self.update_policy = update_policy
        self.exploration_policy = exploration_policy
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.alpha_decay = alpha_decay
        self.epsilon_decay = epsilon_decay
        self.alpha = 1
        self.epsilon = 1

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
        self.alpha = max(self.alpha * (1 - self.alpha_decay), self.min_alpha)
        self.epsilon = max(
            self.epsilon * (1 - self.epsilon_decay), self.min_epsilon
        )
        alpha = 0 if exploit else self.alpha
        epsilon = 0 if exploit else self.epsilon

        cumul_reward = 0
        prev_observation = self.env.reset()
        policy_params = {"epsilon": epsilon}

        # just for caching:
        # True = just some hashable value we can swap between two states
        prev_cache_key = True
        self.q_function.select_action(
            prev_observation,
            self.update_policy,
            policy_params,
            save=prev_cache_key,
        )

        t = 0
        while t < max_steps:
            # self.env.render()
            logger.debug("Observation: %s", prev_observation)

            if exploit and hasattr(self.env, "render"):
                self.env.render()

            action, action_val = self.q_function.select_action(
                prev_observation,
                self.exploration_policy,
                policy_params,
                load=prev_cache_key,
            )
            logger.debug(
                "chosen action %s (%s), value %s (%s)",
                action,
                type(action),
                action_val,
                type(action_val),
            )

            # Take action
            observation, reward, done, info = self.env.step(action)

            # Update the state that we acted on
            if type(self.env).__name__ == "PensionEnv":
                logger.info(
                    "year %s funds %s reputation %s humans %s meanAge %s "
                    + "currAge %s hFunds %s hID %s stateActionkey %s",
                    info["year"],
                    info["company"].funds,
                    info["company"].reputation,
                    len([h for h in self.env.humans if h.active]),
                    LazyStr(np.mean, [h.age for h in self.env.humans]),
                    info["human"].age,
                    info["human"].funds,
                    info["human"].id,
                )

            curr_cache_key = not prev_cache_key
            curr_best_action, curr_best_value = self.q_function.select_action(
                observation,
                self.update_policy,
                policy_params,
                save=curr_cache_key,
            )

            logger.debug(
                "update policy yields action: (%s, %s)",
                curr_best_action,
                curr_best_value,
            )

            # the original value of the action that we took to get here
            qValOld = action_val
            td_error = (reward + self.gamma * curr_best_value) - qValOld
            qValNew = qValOld + alpha * td_error
            logger.debug(
                "%s <-- %s + %s * [(%s + %s * %s) - %s]",
                qValNew,
                qValOld,
                alpha,
                reward,
                self.gamma,
                curr_best_value,
                qValOld,
            )
            self.q_function.update_value(
                prev_observation, action, qValNew, load=prev_cache_key
            )

            prev_observation, prev_cache_key = observation, curr_cache_key

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
            t += 1

        return self.q_function.q_table, cumul_reward, t + 1, info

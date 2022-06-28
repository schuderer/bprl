import math
import logging
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyTsetlinMachine.tm import RegressionTsetlinMachine
from pyTsetlinMachine.tools import Binarizer

from agents.discretizer import create_discretizers

logger = logging.getLogger(__name__)

sys.path.append("../pyTsetlinMachine")  # noqa


# hyper parameters
@dataclass
class TsetlinRegParams:
    T: int = 10000  # 4000  # TODO try fractions of num clauses; As a strategy for problems where the number of clauses is unknown, and for real-world applications where noise plays a significant role, the RTM can be initialized with a much larger T. Then, since the output, yo, is a fraction of the threshold, T, the error decreases.
    s: int = 2.5  # 2  # For instance, if someone [increases s to 4], clauses will start to learn much finer patterns, such as (1 0 .), (1 1 .), and (0 1 .). This significantly increases the number of clauses needed to capture the sub-patterns.
    number_of_clauses: int = 10000  # 4000  # TODO: try reducing to multiples of T
    states: int = 25  # 25  # 100
    max_target: int = 0  # 4000 (0 for Combat-V0, 300 for cartpole)
    min_target: int = -15  # (-15 for Combat-V0, 0 for cartpole)


class TsetlinQFunction:
    """Tsetlin-based Q-Value (State-Action-Value) Approximator"""

    def __init__(
            self, env, discretize_bins: int = 8, discretize_log: bool = False,
            tsetlin_params: Optional[TsetlinRegParams] = None
    ):
        """Initialize a PyTsetlin-based Q-Value (State-Action-Value) Approximator.
        Uses https://github.com/cair/pyTsetlinMachine

        Params:
            env: the openai gym compatible environment to use
            discretize_bins: use pyTsetlinMachine's Binarizer implementation to
                             discretize any non-discrete state dimensions with this
                             number of bins (must be a power of two) - using bins instead of
                             bits in order to remain compatible with our own discretizer.
            discretize_log: not supported (using pyTsetlin's Binarizer implementation).
            tsetlin_params: instance of TsetlinRegParams (optional, parameters to
                            a Tsetlin Regression Machine)
        """
        # q_table: initial q table (optional, for continuing with pre-filled approximator,
        #          empty by default)
        if discretize_log is True:
            raise ValueError("This implementation does not support log-transformed spaces as "
                             "it is using pyTsetlinMachine's Binarizer implementation")
        self.bits = math.log2(discretize_bins)
        if not self.bits.is_integer():
            raise ValueError(f"`discretize_bins` must be a power of two (2, 4, 8, etc.), was {discretize_bins}")

        self.env = env
        self.state_disc, self.action_disc = create_discretizers(
            self.env, discretize_bins, discretize_log
        )
        if self.state_disc.grid is not None:
            self.num_state_features = self.state_disc.grid.size
        else:
            self.num_state_features = self.state_disc.space.n

        if self.action_disc.grid is not None:
            self.num_actions = self.action_disc.grid.size
        else:
            self.num_actions = self.action_disc.space.n
            # raise ValueError("`TsetlinQFunction` currently only supports discretized Box states.")
        print(self.state_disc.grid)
        self.num_bins = self.state_disc.num_bins
        self.num_features = self.num_state_features + self.num_actions

        self.tparams = tsetlin_params or TsetlinRegParams()
        # RegressionTsetlinMachine(number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, s_range=False)
        self.tsetlin_machine = RegressionTsetlinMachine(
            self.tparams.number_of_clauses,
            self.tparams.T,
            self.tparams.s,
            number_of_state_bits=self.tparams.states,
            max_y=self.tparams.max_target,  # optional
            min_y=self.tparams.min_target,  # optional
            number_of_features=self.num_features  # optional
        )

    def select_action(
            self, observation, policy, policy_params, save=None, load=None
    ):
        """Get action/value tuple of selected action.

        Parameters `save` and `load` are ignored here (they
        are meant to speed up hash lookup for tabular learning).
        """
        discrete_state = self.state_disc.discretize(observation)
        action_values = self._get_action_values(discrete_state)
        # print(f"action_values: {action_values}")
        action_index, action_value = policy(action_values, policy_params)
        action = self.action_disc.undiscretize(action_index)
        # print(f"undiscretized {action_index} --> {action}")
        return action, action_value

    def update_value(self, observation, action, value, save=None, load=None):
        """Update value of observation-action

        Parameters `save` and `load` are ignored here (they
        are meant to speed up hash lookup for tabular learning).
        """
        discrete_state = self.state_disc.discretize(observation)
        discrete_action = self.action_disc.discretize(action)
        features = self._tsetlinFeaturesFor(discrete_state, discrete_action)
        # print(f"update: discrete_action {discrete_action} -> value {value}")
        # print(features)
        self.tsetlin_machine.fit(np.array([features]), value, epochs=1, incremental=True)

    @staticmethod
    def _stateKeyFor(discreteObs):
        if discreteObs.shape == ():
            d_obs = np.reshape(discreteObs, (1,))
        else:
            d_obs = discreteObs
        # slow using timeit():
        return "-".join(map(str, d_obs))
        # 100x faster using timeit(), but only a few percent faster in reality:
        # return reduce(lambda a, v: str(a) + '-' + str(v), discreteObs)

    def _tsetlinFeaturesFor(self, discrete_state, discrete_action):
        if not hasattr(discrete_state, "shape") or discrete_state.shape == ():
            discrete_state = np.reshape(discrete_state, (1,))
        if not hasattr(discrete_action, "shape") or discrete_action.shape == ():
            discrete_action = np.reshape(discrete_action, (1,))
        # print(f"discrete_state = {discrete_state}")
        # print(f"discrete_action = {discrete_action}")
        features = np.zeros(self.num_features, dtype=np.int32)
        idx_offsets = np.array(range(len(discrete_state) + len(discrete_action))) * self.num_bins
        indices = idx_offsets + np.append(discrete_state, discrete_action)
        # print(f"indices = {indices}")
        features[indices] = 1
        # print(f"features = {features}")
        return features

    def _get_action_values(self, discrete_state):
        action_values = []
        for discrete_action in range(self.num_actions):
            features = self._tsetlinFeaturesFor(discrete_state, [discrete_action])
            action_value = self.tsetlin_machine.predict(np.array([features]))
            action_values.append(action_value)
        return action_values

    @property
    def number_of_clauses(self):
        """Twice the number of specified clauses
        (the first `number_of_clauses` number of indices refer to the positive ones
        while the second half of the indices refers to the negated ones
        """
        return self.tsetlin_machine.number_of_clauses * 2

    def get_raw_clauses(self):
        return self.tsetlin_machine.get_state()

    def get_clauses(self):
        for i in range(self.number_of_clauses):
            yield self.get_clause(i)

    def get_clause(self, index: int):
        num_clauses = self.tsetlin_machine.number_of_clauses
        negation_idx = index // num_clauses
        clause_idx = index % num_clauses
        # print(f"negation_idx: {negation_idx}")
        # print(f"clause_idx: {clause_idx}")
        return self.tsetlin_machine.ta_state[clause_idx, :, negation_idx]

    def set_clause(self, index: int, clause):
        num_clauses = self.tsetlin_machine.number_of_clauses
        negation_idx = index // num_clauses
        clause_idx = index % num_clauses
        self.tsetlin_machine.ta_state[clause_idx, :, negation_idx] = clause

    def normalized_clause_mean(self, clause):
        return (np.mean(clause) - self.tsetlin_machine.number_of_states) / (2 * self.tsetlin_machine.number_of_states)

    def normalized_clause_means(self):
        clause_means = np.mean(self.tsetlin_machine.ta_state, axis=1)
        clauses_normed = (clause_means - self.tsetlin_machine.number_of_states) / (
                    2 * self.tsetlin_machine.number_of_states)
        clauses_normed_1d = clauses_normed.swapaxes(0,
                                                    1).flatten()  # first half are the non-negated clauses, second half the negated ones (to match the index convention of get_clause/set_clause)
        return clauses_normed_1d


def test():
    from examples.pension_env_def import get_env_cls
    from agents.q_agent import greedy, epsilon_greedy
    from random import randint
    bins = 8
    p_env_cls = get_env_cls(max_individuals=1)
    env = p_env_cls()
    obs = env.reset()
    print(f"obs: {obs}")
    q = TsetlinQFunction(env, discretize_bins=bins)
    a = q.select_action(obs, greedy, {})
    print(f"a: {a}")
    # print(f"discretize 1.0 -> {q.action_disc.discretize([1.0])}")
    # print(f"discretize 0.75 -> {q.action_disc.discretize([0.75])}")
    # print(f"discretize 0.33 -> {q.action_disc.discretize([0.33])}")
    # print(f"discretize 0.0 -> {q.action_disc.discretize([0.0])}")
    # print(f"undiscretize 8 -> {q.action_disc.undiscretize([8])}")
    # print(f"undiscretize 6 -> {q.action_disc.undiscretize([6])}")
    # print(f"undiscretize 3 -> {q.action_disc.undiscretize([3])}")
    # print(f"undiscretize 0 -> {q.action_disc.undiscretize([0])}")
    # exit()
    for i in range(45000):
        # discrete_action = randint(0, q.num_actions - 1)
        # value = 20 if discrete_action == 2 or discrete_action == 3 else 0
        # action = q.action_disc.undiscretize(discrete_action)
        action = randint(0, 1000) / 1000
        value = 20 if 0.5 <= action < 0.75 else 0
        discrete_action = q.action_disc.discretize(action)
        # print(f"action {discrete_action}/{action}, value {value}")
        q.update_value(obs, action, value)
    a2 = q.select_action(obs, greedy, {})
    print(f"a2: {a2}")


if __name__ == "__main__":
    test()

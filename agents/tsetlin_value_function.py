from agents.discretizer import create_discretizers
import logging
import numpy as np
import math
import sys
import pyximport
pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True,
)
# TODO: set preprocessor flags (using .pyxbld file; make_ext...)
# define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],


sys.path.append("../regression-tsetlin-machine")  # noqa

import RegressionTsetlinMachine  # noqa
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# hyper parameters
@dataclass
class TsetlinRegParams:
    T: int = 10000  # 4000
    s: int = 2  # 2
    number_of_clauses: int = 10000  # 4000
    states: int = 25  # 25  # 100
    max_target: int = 300  # 4000  # TODO: Make dynamic
    min_target: int = 0  # TODO: Make dynamic


class TsetlinQFunction:
    """Tsetlin-based Q-Value (State-Action-Value) Approximator"""

    def __init__(
        self, env, discretize_bins: int = 8, discretize_log: bool = False, tsetlin_params: Optional[TsetlinRegParams] = None
    ):
        """Initialize a Tsetlin-based Q-Value (State-Action-Value) Approximator.

        Params:
            env: the openai gym compatible environment to use
            discretize_bins: discretize any non-discrete state dimensions with this
                             number of bins (must be a power of two)
            discretize_log: use exponentially increasing instead of equal bin sizes
                            (the further from zero, the coarser the bin size)
            tsetlin_params: instance of TsetlinRegParams (optional, parameters to
                            a Tsetlin Regression Machine)
        """
        # q_table: initial q table (optional, for continuing with pre-filled approximator,
        #          empty by default)
        # self.bits = math.log2(discretize_bins)
        # if not self.bits.is_integer():
        #     raise ValueError(f"`discretize_bins` must be a power of two (2, 4, 8, etc.), was {discretize_bins}")

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
        self.tsetlin_machine = RegressionTsetlinMachine.TsetlinMachine(
            self.tparams.number_of_clauses,
            self.num_features,
            self.tparams.states,
            self.tparams.s,
            self.tparams.T,
            self.tparams.max_target,
            self.tparams.min_target
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
        self.tsetlin_machine.update(features, value)

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
            action_value = self.tsetlin_machine.predict(features)
            action_values.append(action_value)
        return action_values


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
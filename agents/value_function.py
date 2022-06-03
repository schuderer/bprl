# Stdlib imports
from collections import defaultdict
import logging

# Third party imports
import numpy as np

# Application imports
from .discretizer import create_discretizers

logger = logging.getLogger(__name__)


class QFunction:
    """Tabular Q-Value (State-Action-Value) Approximator"""

    def __init__(
        self, env, default_value=0, discretize_bins=12, discretize_log=False
    ):
        """Initialize a Tabular Q-Value (State-Action-Value) Approximator.

        Params:
            env: the openai gym compatible environment to use
            default_value=0: initialize state-action value (Q) table with this value
            discretize_bins: discretize any non-discrete state dimensions with this
                             number of bins
            discretize_log: use exponentially increasing instead of equal bin sizes
                            (the further from zero, the coarser the bin size)
        """
        # q_table: initial q table (optional, for continuing with pre-filled approximator,
        #          empty by default)

        self.env = env
        self.default_value = default_value
        self.state_disc, self.action_disc = create_discretizers(
            self.env, discretize_bins, discretize_log
        )
        state_value_default = [
            default_value for _ in range(self.action_disc.space.n)
        ]
        self.q_table = defaultdict(lambda: state_value_default.copy(), {})
        self.remembered_state_keys = {}

    def select_action(
        self, observation, policy, policy_params, save=None, load=None
    ):
        """Get action/value tuple of selected action"""
        if load is None or load not in self.remembered_state_keys:
            discrete_state = self.state_disc.discretize(observation)
            state_key = self._stateKeyFor(discrete_state)
            if save is not None:
                self.remembered_state_keys[save] = state_key
        else:
            state_key = self.remembered_state_keys[load]

        action_values = self._get_action_values(state_key, self.q_table)
        action_index, action_value = policy(action_values, policy_params)
        action = self.action_disc.undiscretize(action_index)  # TODO check: was originally wrapped in int(), but float values should be supported
        return action, action_value

    def update_value(self, observation, action, value, save=None, load=None):
        """Update value of observation-action"""
        if load is None or load not in self.remembered_state_keys:
            discrete_state = self.state_disc.discretize(observation)
            state_key = self._stateKeyFor(discrete_state)
            if save is not None:
                self.remembered_state_keys[save] = state_key
        else:
            state_key = self.remembered_state_keys[load]

        discrete_action = self.action_disc.discretize(action)
        self.q_table[state_key][int(discrete_action)] = value

    # def print_q(self, q_table=None):
    #     q_table = q_table or self.q_table
    #     for s in range(self.state_disc.space.n):
    #         if self.state_disc.grid is None:
    #             s = np.array(s)
    #         else:  # todo
    #             # indices = np.unravel_index(range(statesDisc.n), state_grid.shape)
    #             s = self.state_disc.grid[s]
    #         logger.info(
    #             [
    #                 q_table[self._stateKeyFor(s)][a]
    #                 for a in range(self.action_disc.space.n)
    #             ]
    #         )
    #
    # def print_q_frozenlake(self, q_table=None):
    #     q_table = q_table or self.q_table
    #     for s in range(self.state_disc.space.n):
    #         s = np.array(s)
    #         print(
    #             [
    #                 q_table[self._stateKeyFor(s)][a]
    #                 for a in range(self.action_disc.space.n)
    #             ]
    #         )

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

    def _get_action_values(self, key_start, q_table):
        return q_table[key_start]

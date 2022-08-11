from typing import Union, Callable

env_name: Union[str, Callable]
# env_name = "Pendulum-v1"  # "PensionExample-v0"
env_name = "CartPole-v1"
# env_name = "MountainCar-v0"  # Did not converge with 10.000 clauses
# from pettingzoo.butterfly import pistonball_v6
# env_name = pistonball_v6.env
# import ma_gym
# env_name: Union[str, Callable] = "Combat-v0"

# algo_str = "Q-Tsetlin no comm"
# algo = "Q-Tsetlin best agent com + cert sampling"
algo = "tsetlin"
algo_variant = "no comm"

episodes = 4000  # LASTCHANGE: was 5000
num_agents = 1  # was 4; use -1 for native multi-agent environment (which determines its own num of agents)
num_runs = 20  # LASTCHANGE: was 10

parallel_processes = 18

# By providing more than one value in each parameter list, we tell the app to perform a hyperparameter search:
parameter_grid = {
    "tsetlin_number_of_clauses": [6000, 10000, 14000],  # was 10000
    "tsetlin_T": [100000, 5000000, 20000000, 40000000],  # was 20000000
    "tsetlin_s": [2.5],
    "tsetlin_states": [4, 8, 25, 100],  # was 8
    "tsetlin_max_target": [300],
    "tsetlin_min_target": [0],
    "min_epsilon": [0.01, 0.004, 0.001],  # was 0.004
    "epsilon_decay": [0.0004, 0.0008, 0.0016],  # was 0.0008
    "num_bins": [8, 16],  # was 16
    "log_bins": [False],
}

# # Current "good enough" params:
# parameter_grid = {
#     "tsetlin_number_of_clauses": [10000],  # was 10000
#     "tsetlin_T": [20000000],  # was 20000000
#     "tsetlin_s": [2.5],
#     "tsetlin_states": [8],  # was 8
#     "tsetlin_max_target": [300],
#     "tsetlin_min_target": [0],
#     "min_epsilon": [0.004],  # was 0.004
#     "epsilon_decay": [0.0008],  # was 0.0008
#     "num_bins": [16],  # was 16
#     "log_bins": [False],
# }

from random import randint
import sys
import numpy as np
import pyximport
pyximport.install(setup_args={
    "include_dirs": np.get_include()},
    reload_support=True,
)
# TODO: set preprocessor flags (using .pyxbld file; make_ext...)
# define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],


sys.path.append("../regression-tsetlin-machine")  # noqa

import RegressionTsetlinMachine  # noqa

# hyper parameters
# TODO: tune and include in class
T = 10000  # 4000
s = 2  # 2
number_of_clauses = 10000  # 4000
states = 25  # 100

# #hyper parameters from ArtificialDataDemo
# T = 3
# s = 2
# number_of_clauses = 3
# states = 100

max_target = 100
min_target = 0

one_hot_len = 9
epochs = 1000

reg = RegressionTsetlinMachine.TsetlinMachine(
    number_of_clauses,
    36 + one_hot_len,  # number of features
    states,
    s,
    T,  # threshold
    max_target,
    min_target
)

padding = np.zeros(36, dtype=np.int32)
padding[[3, 15, 22]] = 1
all_possible_one_hot_features = np.eye(one_hot_len, dtype=np.int32)

for i in range(epochs * one_hot_len):
    which = randint(0, one_hot_len - 1)
    one_hot_features = all_possible_one_hot_features[which]
    features = np.append(padding, one_hot_features)
    target_value = 20 if which == 2 or which == 3 else 0
    # print(f"{which}: {features} ({len(features)}) -> {target_value}")
    reg.update(features, target_value)

preds = [
    reg.predict(np.append(padding, all_possible_one_hot_features[which]))
    for which in range(one_hot_len)
]
print(f"targets: {all_possible_one_hot_features[2:3]*20.0}")
print(f"actual:  {preds}")

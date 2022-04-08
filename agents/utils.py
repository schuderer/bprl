import numpy as np

class LazyStr:
    def __init__(self, func, *args, **kwArgs):
        self.func = func
        self.args = args
        self.kwArgs = kwArgs

    def __str__(self):
        return str(self.func(*self.args, **self.kwArgs))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

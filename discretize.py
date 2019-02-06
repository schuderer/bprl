import numpy as np
from math import sqrt

class Discretizer:
    # adapted from https://github.com/udacity/deep-reinforcement-learning/blob/
    #                       master/discretization/Discretization_Solution.ipynb

    def __init__(self, lows, highs, bins, log=False):
        self.grid = self.__create_grid(lows, highs, bins, log)

    def discretize(self, vals):
        return np.array(list(int(np.digitize(s, g, right=True))
                        for s, g in zip(vals, self.grid)))

    def undiscretize(self, indices):
        return np.array(list(g[min([int(i), len(g)-1])] for i, g in zip(indices, self.grid)))

    @staticmethod
    def __create_grid(low, high, bins, log=False):
        space_func = np.linspace if not log else np.geomspace
        if log:
            print("creating logarithmic grid")
            low_signs = np.sign(low)
            high_signs = np.sign(high)
            same_sign = (low_signs == high_signs) \
                        | (low_signs == 0) | (high_signs == 0)
            grid = np.zeros((len(low), bins[0]+1))
            # print(grid)
            for dim in range(len(same_sign)):
                low_mag = 0 if low[dim] == 0 else int(np.log10(abs(low[dim])))
                high_mag = 0 if high[dim] == 0 else int(np.log10(abs(high[dim])))
                print("low_mag", low_mag)
                print("high_mag", high_mag)
                if same_sign[dim]:
                    mag = max(low_mag, high_mag)
                    print("mag", mag, "bins", bins[dim], "sqrt(bins)", sqrt(bins[dim])/2, "result",(mag-sqrt(bins[dim])/2))
                    low[dim] = low[dim] if low[dim] != 0 else 10**float(min((mag-sqrt(bins[dim])/2), 0))
                    high[dim] = high[dim] if high[dim] != 0 else 10**float(min((mag-sqrt(bins[dim])/2), 0))
                    print(low[dim], high[dim], print(10**float(min(mag-sqrt(bins[dim])/2, 0)))
)
                    grid[dim] = space_func(low[dim], high[dim], bins[dim] + 1)
                else:
                    print("dimension {} has different signs".format(dim))
                    l1 = low[dim]
                    h2 = high[dim]
                    lh_range = h2 - l1
                    bin1_ratio = abs(l1) / lh_range
                    b1 = float(int(bins[dim] * bin1_ratio))
                    b2 = float(bins[dim] - b1)
                    print("b1", b1, "b2", b2, "mag1", 10**min(low_mag-sqrt(b1)/2, 0), "mag2", 10**min(high_mag-sqrt(b2)/2, 0))
                    h1 = -1 * 10**min((low_mag-sqrt(b1)/2), 0)  # 0 would lead to nan
                    l2 = 1 * 10**min((high_mag-sqrt(b2)/2), 0)  # 0 would lead to nan
                    grid_below_0 = space_func(l1, h1, b1 + 1)[0:-1]
                    # print("below", grid_below_0)
                    grid_above_0 = space_func(l2, h2, b2 + 1)
                    # print("above", grid_above_0)
                    grid[dim] = np.hstack((grid_below_0, grid_above_0))
        else:
            grid = np.array([space_func(low[dim], high[dim], bins[dim] + 1)
                             for dim in range(len(bins))])
        print("Grid: [<low>, <high>] / <bins> => <splits>")
        for l, h, b, splits in zip(low, high, bins, grid):
            print("    [{}, {}] / {} => {}".format(l, h, b, splits))
        return grid


if __name__ == "__main__":
    print("linear:")
    d = Discretizer([2, 2], [256, 1024], [8, 8])
    print("testGrid 1:")
    print(d.grid)
    print("discretize [40, 12]:")
    testDisc = d.discretize([40, 12])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("end of test 1")
    d = Discretizer([-128, -512], [128, 512], [8, 8])
    print("testGrid 2:")
    print(d.grid)
    print("discretize min values [-128, -512]:")
    testDisc = d.discretize([-128, -512])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize max values [128, 512]:")
    testDisc = d.discretize([128, 512])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize below min values [-1000, -10000]:")
    testDisc = d.discretize([-1000, -10000])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize above max values [10000, 1000]:")
    testDisc = d.discretize([10000, 1000])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("end of test 2")
    print("\n\nlog:")
    d = Discretizer([2, 2], [256, 1024], [8, 8], log=True)
    print("testGrid 1:")
    print(d.grid)
    print("discretize [40, 12]:")
    testDisc = d.discretize([40, 12])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("end of test 1")
    d = Discretizer([-128, -512], [128, 512], [8, 8], log=True)
    print("testGrid 2:")
    print(d.grid)
    print("discretize min values [-128, -512]:")
    testDisc = d.discretize([-128, -512])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize max values [128, 512]:")
    testDisc = d.discretize([128, 512])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize below min values [-1000, -10000]:")
    testDisc = d.discretize([-1000, -10000])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("discretize above max values [10000, 1000]:")
    testDisc = d.discretize([10000, 1000])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))
    print("end of test 2\n")
    d = Discretizer([-1], [1], [8], log=True)
    print("testGrid 3:")
    print(d.grid)
    print("discretize values [-0.2]:")
    testDisc = d.discretize([-0.2])
    print(testDisc)
    print("undiscretize:")
    print(d.undiscretize(testDisc))

from glob import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


episodePattern = re.compile(r'^Episode ([0-9.-]+) finished after ([0-9.-]+) timesteps with cumulative reward ([0-9.-]+)')
episodeInfoPattern = re.compile(r'^(?:year ([0-9.-]+), )?q table size ([0-9.-]+), epsilon ([0-9.-]+), alpha ([0-9.-]+), #humans ([0-9.-]+), reputation ([0-9.-]+)$')
stopEpPattern = re.compile(r'^Average reward last 100 episodes')
individualRunPattern = re.compile(r'^year ([0-9.-]+) funds ([0-9.-]+) reputation ([0-9.-]+) humans ([0-9.-]+) meanAge ([0-9.-]+) currAge ([0-9.-]+) hFunds ([0-9.-]+) hID ([0-9.-]+) stateActionkey')


def processEpisodeLogs(filePattern):
    results = []
    for fileName in glob(filePattern):
        with open(fileName) as f:
            print("\nProcessing file", fileName)
            episodeData = []
            epTuple = None
            for line in f:
                if stopEpPattern.match(line):
                    print("Reached end of training data")
                    break
                episodeMatch = episodePattern.match(line)
                if episodeMatch:
                    epTuple = episodeMatch.groups()
                else:
                    episodeInfoMatch = episodeInfoPattern.match(line)
                    if episodeInfoMatch:
                        if epTuple is None:
                            print("WARNING: Found an out-of-place line. Skipping rest of file.")
                            break
                        row = epTuple + episodeInfoMatch.groups()
                        # print(row)
                        # numericRow = [r for r in row]
                        # print(numericRow)
                        episodeData.append(row)
                        epTuple = None
            epiCols = ["episode", "steps", "reward", "year",
                       "q_table_size", "epsilon", "alpha", "humans",
                       "reputation"]
            epiDf = pd.DataFrame(episodeData, columns=epiCols)
            epiDf = epiDf.apply(pd.to_numeric, errors='coerce')
            # print(epiDf.head())
            # print(" ...")
            # print(epiDf.tail())
            results.append((epiDf, fileName))
    return results


def processRunLog(fileName, noTraining=False):
    with open(fileName) as f:
        print("\nProcessing file", fileName)
        reachedIndividualRun = noTraining
        individualRunData = []
        for line in f:
            if not reachedIndividualRun and stopEpPattern.match(line):
                print("Reached end of training data")
                reachedIndividualRun = True
                continue
            if reachedIndividualRun:
                if episodePattern.match(line):
                    print("Reached end of individual run data")
                    break
                individualMatch = individualRunPattern.match(line)
                if individualMatch:
                    row = individualMatch.groups()
                    individualRunData.append(row)
        columns = ["year", "funds", "reputation", "humans", "meanAge",
                   "curr_age", "human_funds", "human_id"]
        df = pd.DataFrame(individualRunData, columns=columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        # print(df.head())
        # print(" ...")
        # print(df.tail())
        return df


# todo: glob()-combine logs of several runs with same parameters
# according to https://github.com/jbmouret/matplotlib_for_papers
# plotting the median of all runs of same episode as line and the
# inter-quartile range as surface.


def saveAsCsv(df, fileName):
    outName = os.path.splitext(fileName)[0] + ".csv"
    print("Writing csv file", outName)
    df.to_csv(outName, index=False)


def plotTrainingRewards(dfs, fileName):
    df = pd.concat(dfs, axis=1)[['reward']]
    df['median'] = df.median(axis=1)
    df['rolling_mean'] = df['median'].rolling(window=100).mean()
    df['perc25'] = df[['reward']].quantile(q=0.25, axis=1)
    df['perc75'] = df[['reward']].quantile(q=0.75, axis=1)

    # Print confidence interval
    lastMedians = df.tail(1)[['reward']]
    endMean = lastMedians.mean(axis=1)
    stdErr = lastMedians.sem(axis=1)
    print("mean {}, confLow {}, confHigh {}"
          .format(endMean, endMean-stdErr*1.96, endMean+stdErr+1.96))

    mpl.rcdefaults()
    # params = {
    #    'axes.labelsize': 8,
    #    'font.size': 8,
    #    'legend.fontsize': 10,
    #    'xtick.labelsize': 10,
    #    'ytick.labelsize': 10,
    #    'text.usetex': False,
    #    'figure.figsize': [4.5, 4.5]
    #    }
    # rcParams.update(params)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)

    # ax1.set_title("Plot title")
    ax1.set_xlabel('training episode')
    # ax1.set_ylabel('y label')

    ax1.plot(df["median"], label='median cumulative reward per episode')
    ax1.plot(df["rolling_mean"], label='rolling mean over 100 episodes')
    ax1.fill_between(range(df["median"].count()), df['perc25'], df['perc75'], alpha=0.25, linewidth=0)  # , color='#B22400')
    for s in ['top', 'right', 'left']: # ax.spines.values():
        ax1.spines[s].set_visible(False)
    plt.setp(ax1.spines.values(), color="0.9")
    plt.setp([ax1.get_xticklines(), ax1.get_yticklines()], color="0.9")
    ax1.legend(loc='upper left')
    ax1.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)

    outName = os.path.splitext(fileName)[0] + ".pdf"
    print("Writing chart to file", outName)
    fig.savefig(outName, bbox_inches='tight')

    # todo: integrate in complete plot using a grid like 3,2 and this plot as a slice along the complete left (3-row) column, and the other combined charts along the right column
    # see e.g. https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

    # todo: remove show
    # plt.show()
    return fig


def plotRun(df, fileName):
    df = df.groupby('year').mean()

    mpl.rcdefaults()
    # params = {
    #    'axes.labelsize': 8,
    #    'font.size': 8,
    #    'legend.fontsize': 10,
    #    'xtick.labelsize': 10,
    #    'ytick.labelsize': 10,
    #    'text.usetex': False,
    #    'figure.figsize': [4.5, 4.5]
    #    }
    # rcParams.update(params)

    def style(ax, bottom=False):
        for s in ['top', 'right', 'left']: # ax.spines.values():
            ax.spines[s].set_visible(False)
        plt.setp(ax.spines.values(), color="0.9")
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color="0.9")
        ax.legend(loc='lower center')
        ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
        if bottom:
            ax.set_xlabel('year')
        else:
            ax.set_xticklabels([])
            # ax.set_xticks([])
            ax.tick_params(axis='x', colors='white')

    # fig = plt.figure()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    # fig.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, wspace=0.1, hspace=0.1)
    fig.subplots_adjust(hspace=0.1)

    # ax1 = fig.add_subplot(2, 1, 1)
    # ax1.set_title("Plot title")
    # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    ax1.plot(df["funds"], color="cornflowerblue", label='company funds')
    # ax1.set_xlabel('year')
    style(ax1)

    ax2.plot(df["reputation"], color="dodgerblue", label='company reputation')
    style(ax2)

    # ax2 = fig.add_subplot(2, 1, 2)
    ax3.plot(df["humans"], color="orange", label='number of clients')
    style(ax3)

    ax4.plot(df["meanAge"], color="darkgoldenrod", label='mean age of clients')
    style(ax4, bottom=True)


    outName = os.path.splitext(fileName)[0] + "_run.pdf"
    print("Writing chart to file", outName)
    fig.savefig(outName, bbox_inches='tight')
    # todo: remove show
    # plt.show()

    return fig


filePattern = "runs/longevity_19/*.txt"
dfTuples = processEpisodeLogs(filePattern)

dfs = [tup[0] for tup in dfTuples]

df = dfs[0]

df['reward'].count()

rewPlot = plotTrainingRewards(dfs, "figs/long_19_train")



runFile = "output/fixed_test_pension_2.txt"
dfRun = processRunLog(runFile, noTraining=True)

fixedRunPlot = plotRun(dfRun, "figs/fixed")


long_19_run = "output/q_test_pension_longevity_19_examplerun_1.txt"
dfRun2 = processRunLog(long_19_run, noTraining=True)

long_19_1_plot = plotRun(dfRun2, "figs/long_19_1")


long_19_run2 = "output/q_test_pension_longevity_19_examplerun_2.txt"
dfRun3 = processRunLog(long_19_run2, noTraining=True)

long_19_2_plot = plotRun(dfRun3, "figs/long_19_2")


def plotQTable(qTable, stateGrid, name="", xState=0, yState=1):
    nBins = len(stateGrid[0])+1
    debit = 0
    credit = 1
    qMap = np.zeros((nBins, nBins))
    for x in range(nBins):  # x = age
        for y in range(nBins):  # y = funds
            val = qTable.get("{}-{}-{}".format(x, y, credit), 0) - qTable.get("{}-{}-{}".format(x, y, debit), 0)
            qMap[nBins-1-y, x] = val

    sns.set(context="paper")
    ax = sns.heatmap(qMap, linewidth=0.5, cmap="coolwarm", cbar=False, square=True, vmin=-1, vmax=1) #  center=0)
    ageGrid = stateGrid[xState]
    fundsGrid = stateGrid[yState]
    xTickLabels = [""]
    xTickLabels.extend(["{:.0f}".format(ageGrid[i]) for i in range(nBins-1)])
    yTickLabels = [""]
    yTickLabels.extend(["{:.0f}".format(fundsGrid[i]) for i in range(nBins-2, -1, -1)])
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nBins))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nBins))
    ax.set_xticklabels(xTickLabels, rotation=0)
    ax.set_yticklabels(yTickLabels, rotation=0)
    ax.set_xlabel("customer age")
    ax.set_ylabel("company funds")
    if name != "":
        fig = ax.get_figure()
        fig.savefig(name, bbox_inches='tight')
    # ax.xlabel("age")
    # ax.ylabel("funds")
    # ax.legend()
    return ax


#  model == expect to behave like that; simulation = actual implementation, run
#  environment that models financial services / business processes

stateGrid_long_19 = [[  1.00000000e+00,   1.46779927e+00,   2.15443469e+00,   3.16227766e+00, 4.64158883e+00,   6.81292069e+00,   1.00000000e+01,   1.46779927e+01, 2.15443469e+01,   3.16227766e+01,   4.64158883e+01,   6.81292069e+01, 1.00000000e+02], [ -1.00000000e+06,  -1.00000000e+05,  -1.00000000e+04,  -1.00000000e+03,
   -1.00000000e+02,  -1.00000000e+01,   1.00000000e+00,   1.00000000e+01,
    1.00000000e+02,   1.00000000e+03,   1.00000000e+04,   1.00000000e+05,
    1.00000000e+06]]

qTable_long_19_2 = {'8-11-0': 94.85750359596473, '9-11-1': 93.40323065017274, '8-10-0': 54.05487303099865, '9-10-0': 69.4452590372413, '9-6-0': 28.350606637340828, '10-10-1': 0.0, '8-11-1': 88.69965983586648, '9-10-1': 0.0, '9-9-0': 43.48613366033944, '9-11-0': 95.0580167838315, '8-6-0': 2.9663781106297673, '8-10-1': 0.0, '0-10-1': 15.319100097519744, '10-9-1': 0.0, '8-9-1': 0.0, '10-11-1': 95.5552419744198, '0-11-0': 74.40527194908847, '9-6-1': 0.0, '10-10-0': 77.7631753592817, '9-9-1': 0.0, '8-9-0': 10.466543791570732, '10-11-0': 96.16730642224013, '10-9-0': 52.50022631368486, '10-6-1': 0.0, '10-6-0': 47.267848204387754, '8-12-1': 96.89479613623092, '9-12-0': 97.69003889839313, '10-12-1': 97.72497266156432, '8-12-0': 98.29330602203332, '9-12-1': 97.66739364403784, '11-11-0': 97.39174531384064, '11-10-0': 90.39064853287226, '11-12-0': 98.44059070537676, '11-11-1': 96.537239260211, '12-12-1': 98.20931866611063, '12-11-1': 95.9104943675021, '11-9-1': 0.0, '10-12-0': 98.30945416910251, '11-10-1': 0.0, '12-10-1': 0.0, '11-12-1': 98.08502073963916, '12-11-0': 96.45089471340923, '11-6-0': 25.099394364670093, '12-10-0': 91.85700280656347, '12-9-0': 49.73730421259588, '12-6-0': 24.967896842942707, '12-12-0': 97.48316451537686, '11-9-0': 49.55524985337189, '12-9-1': 0.0, '0-11-1': 74.09507814228127, '0-6-1': 0.0, '0-9-1': 0.3472440423522913, '0-9-0': 1.0870036388169635, '0-10-0': 13.289073071107142, '0-6-0': 0.17096349741069655, '0-12-1': 81.01054815466709, '8-6-1': 0.0, '10-13-1': 62.000330888305314, '9-13-1': 65.37576691088769, '8-13-1': 21.784954532010495, '11-13-0': 0.7057965640065662, '10-13-0': 0.9732023924806424, '0-12-0': 19.786256147889635, '11-13-1': 34.36227034336228, '9-13-0': 0.43486906879583603, '8-13-0': 0.01, '12-13-1': 0.9807091303179641}


plotQTable(qTable_long_19_2, stateGrid_long_19, name="figs/qtable_long_19_2.pdf")

qTable_long_19_1 = {'8-11-0': 97.7608832910341, '9-11-0': 97.86056856479647, '9-11-1': 95.18174556301601, '10-11-0': 97.85113652791519, '10-11-1': 96.11866236926647, '11-11-1': 81.55740846551616, '10-9-0': 24.520976827244688, '10-10-1': 0.0, '9-12-0': 99.98816738559135, '8-12-0': 99.98806691055029, '10-12-0': 99.98831253087715, '11-12-0': 99.98825668347541, '11-12-1': 99.98353306257485, '9-12-1': 99.97936123840172, '10-12-1': 99.98206246590023, '12-12-1': 99.88681478562623, '12-12-0': 99.98773124376882, '8-12-1': 99.96665661232119, '11-6-1': 0.0, '9-10-1': 0.0, '11-11-0': 50.70707648054032, '12-11-0': 77.45663849592351, '11-10-0': 91.39247931446965, '10-10-0': 95.83944341729408, '10-13-0': 99.94671713612937, '9-13-1': 99.98749001910494, '11-9-1': 0.0, '9-10-0': 93.85127544060744, '8-10-0': 89.58942694297086, '8-11-1': 94.51165778983463, '11-13-0': 99.92468246348758, '10-13-1': 99.98773152369704, '8-13-0': 99.49356878307913, '12-13-1': 99.98415462631984, '11-10-1': 0.0, '12-10-0': 21.27637867283867, '9-13-0': 99.92472677313215, '11-13-1': 99.98745136791563, '0-11-1': 86.59750171856211, '11-6-0': 22.254040343525595, '11-9-0': 41.32242098320778, '12-11-1': 9.890765028745728, '9-9-1': 0.0, '12-13-0': 99.96987987988949, '9-6-0': 18.211942705557174, '10-6-0': 12.09663716735362, '0-12-0': 99.95534947470554, '0-12-1': 99.98792177608104, '0-10-1': 28.31154770612593, '9-9-0': 31.23644770015639, '0-11-0': 78.87565204566951, '0-6-1': 0.0, '8-10-1': 0.0, '0-9-1': 0.20424647057218262, '8-13-1': 99.98465222536989, '0-6-0': 0.06703068854126591, '0-9-0': 0.0, '8-9-0': 6.144653646037612, '8-6-0': 1.6943950163076584, '0-13-1': 37.72116961823961, '9-6-1': 0.0, '0-10-0': 2.4473686028369053, '8-6-1': 0.0}

plotQTable(qTable_long_19_1, stateGrid_long_19, name="figs/qtable_long_19_1.pdf")


# if __name__ == "__main__":
#     filePattern = "output/q_test_pension_longevity_14.txt"
#     processEpisodeLogs(filePattern, [
#         saveAsCsv,
#         plotTrainingRewards
#     ])

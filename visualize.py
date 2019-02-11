from glob import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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

    # Todo print confidence interval

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


# if __name__ == "__main__":
#     filePattern = "output/q_test_pension_longevity_14.txt"
#     processEpisodeLogs(filePattern, [
#         saveAsCsv,
#         plotTrainingRewards
#     ])

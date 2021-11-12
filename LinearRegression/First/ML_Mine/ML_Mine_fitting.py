import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing


def formatF(x):
    return np.format_float_scientific(x, precision = 3, exp_digits=2)


def plotML_V1(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")

    x1 = rpcImonData["inst_lumi"]
    x7 = (rpcImonData["Imon_change_date"] - rpcImonData["Imon_change_date"][0]).astype(int) / 10**9

    X = pd.concat([x1, x7], ignore_index=True, axis=1)
    y = rpcImonData["Imon"]

    XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)

    lineFitter = LinearRegression()
    lineFitter.fit(XTrain, yTrain)

    yPredict = lineFitter.predict(XValid)

    yDiff = yValid - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, lineFitter.predict(X), '.', label="predicted")
    ax1.set_xlabel("(Predict VS Measured), axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()

    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, lineFitter.predict(X), '.', label="predicted")
    ax2.set_xlabel("(Predict VS Measured) axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon Test)")
    ax3.annotate(f"Mean: {formatF(np.mean(yDiff))} \n Mean(abs): {formatF(np.mean(np.abs(yDiff)))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: ML_3Variable, (Train: 80, Test: 20) \nC0: {formatF(lineFitter.intercept_)}, C1: {formatF(lineFitter.coef_[0])}, C7: {formatF(lineFitter.coef_[1])}")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()     


if __name__ == "__main__":
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/"
    plotPath = "/Users/mainroot/RPC_graph/ML_Mine/ML_Mine_fitting/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepPaths, plotSepPaths = [], []
    for folder in rpcImonSepFolders:
        rpcImonSepPaths.append(rpcImonPath + folder + "/")
        plotSepPaths.append(plotPath + folder + "/")


    for i in range(len(rpcImonSepPaths)):
        dpidNames = os.listdir(rpcImonSepPaths[i])
        pool = multiprocessing.Pool(6)
        m = multiprocessing.Manager()
        pool.starmap(plotML_V1, [(rpcImonSepPaths[i], plotSepPaths[i], dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()

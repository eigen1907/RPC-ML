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

    columnList = ["C0", "C1", "C7"]

    C0Part = np.full((len(rpcImonData),), lineFitter.intercept_)
    C1Part = lineFitter.coef_[0] * x1
    C7Part = lineFitter.coef_[1] * x7

    partList = [C0Part, C1Part, C7Part]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), tight_layout=True)
    axes[0].plot(rpcImonData.Imon_change_date, partList[0], '.', label="C0 * X0")
    axes[0].plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured Imon")
    axes[0].legend(loc="best")
    axes[0].set_xlabel("C0 * X0")

    axes[1].plot(rpcImonData.Imon_change_date, partList[1], '.', label="C1 * X1")
    axes[1].plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured Imon")
    axes[1].legend(loc="best")
    axes[1].set_xlabel("C1 * X1")

    axes[2].plot(rpcImonData.Imon_change_date, partList[2], '.', label="C7 * X7")
    axes[2].plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured Imon")
    axes[2].legend(loc="best")
    axes[2].set_xlabel("C7 * X7")

    plt.title("After ML_3Variable, each partition of C_n * X_n")
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()



if __name__ == "__main__":    
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/"
    plotPath = "/Users/mainroot/RPC_graph/ML_Mine/ML_Mine_partitionVSmeasured/"
    
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
    
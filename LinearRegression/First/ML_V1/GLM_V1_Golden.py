import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing


def concatStr(str_list):
    return ''.join(str_list)


def formatF(x):
    return np.format_float_scientific(x, precision = 3, exp_digits=2)


def normMinMax(x):
    return (x - x.min())/(x.max() - x.min())


def plot_GLM_Golden(rpcImonPath, plotPath, dpidName):
    
    #if os.path.isfile(plotPath + dpidName[0:-4] + ".png"):
        #print(f"Already has plot {plotPath + dpidName[0:-4]}.png")
        #return
    
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")

    initialTime = rpcImonData["Imon_change_date"][0]


    trainData, testData = train_test_split(rpcImonData, test_size=0.4, shuffle=True, random_state=34)

    train_x1 = trainData["inst_lumi"]
    train_x2 = trainData["Vmon"]
    train_x3 = trainData["temp"]
    train_x4 = trainData["inst_lumi"] * np.exp(trainData["Vmon"] / trainData["press"])
    train_x5 = trainData["relative_humodity"]
    train_x6 = trainData["press"]
    train_x7 = (trainData["Imon_change_date"] - initialTime).astype(int) / 10**9

    train_X = pd.concat([train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7], ignore_index=True, axis=1)
    train_y = trainData["Imon"]


    test_x1 = testData["inst_lumi"]
    test_x2 = testData["Vmon"]
    test_x3 = testData["temp"]
    test_x4 = testData["inst_lumi"] * np.exp(testData["Vmon"] / testData["press"])
    test_x5 = testData["relative_humodity"]
    test_x6 = testData["press"]
    test_x7 = (testData["Imon_change_date"] - initialTime).astype(int) / 10**9

    test_X = pd.concat([test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7], ignore_index=True, axis=1)
    test_y = testData["Imon"]

    lineFitter = LinearRegression()
    if len(train_X) == 0:
        return

    lineFitter.fit(train_X, train_y)


    predict_y = lineFitter.predict(test_X)
    yDiff = test_y - predict_y
    
    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(221)
    ax1.plot(testData.Imon_change_date, testData.Imon, '.', label="measured")
    ax1.plot(testData.Imon_change_date, predict_y, '.', label="predicted")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Imon")
    ax1.legend()

    ax2 = plt.subplot(223)
    ax2.plot(testData.inst_lumi, testData.Imon, '.', label="measured")
    ax2.plot(testData.inst_lumi, predict_y, '.', label="predicted")
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100, log=True)
    ax3.set_xlabel(f"(Measured - Predicted)'s Histogram")
    ax3.annotate(f"Mean(abs): {formatF(np.mean(np.abs(yDiff)))}\n RMS: {formatF(np.sqrt(np.mean(yDiff**2)))}", xy=(1, 1), xycoords='axes fraction', fontsize=16, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber: {dpidName[0:-9]} \n GLM_V1 (Train: 60, Test: 40), Golden, Separated Data")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()
    print(f"Finish plotting {plotPath + dpidName[0:-4]}.png")


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/ML_V1/GLM_V1_Golden/"

    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        pool = multiprocessing.Pool(50)
        m = multiprocessing.Manager()
        pool.starmap(plot_GLM_Golden, [(rpcImonSepPath, plotSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()






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


def plot_GLM_Original(rpcImonPath, plotPath, dpidName):
    rpcImonData2016 = pd.read_csv(rpcImonPath + "2016/" + dpidName + "_2016.csv", low_memory=False)
    rpcImonData2017 = pd.read_csv(rpcImonPath + "2017/" + dpidName + "_2017.csv", low_memory=False)
    rpcImonData2018 = pd.read_csv(rpcImonPath + "2018/" + dpidName + "_2018.csv", low_memory=False)
    
    
    rpcImonData = pd.concat([rpcImonData2016, rpcImonData2017, rpcImonData2018], ignore_index=True, axis=0)


    
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
    ax1.plot(testData.Imon_change_date, testData.Imon, '.', label="Measured")
    ax1.plot(testData.Imon_change_date, predict_y, '.', label="Predicted")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Imon")
    ax1.legend()

    ax2 = plt.subplot(223)
    ax2.plot(testData.inst_lumi, testData.Imon, '.', label="Measured")
    ax2.plot(testData.inst_lumi, predict_y, '.', label="Predicted")
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100, log=True)
    ax3.set_xlabel(f"Measured - Predicted")
    ax3.annotate(f"Mean(abs): {formatF(np.mean(np.abs(yDiff)))}\n RMS: {formatF(np.sqrt(np.mean(yDiff**2)))}", xy=(1, 1), xycoords='axes fraction', fontsize=16, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber: {dpidName} \n GLM_V1 (Train: 60, Test: 40), Original Data")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName + ".png")
    plt.close()     



if __name__ == "__main__":
    ### 원래 데이터 (Not Golden) 2016, 2017, 2018 하나로 합치기 (SharedDpids에 대해서)    ## dpid_316 같은 형식으로 나옴
    DATAPATH = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/Original/"
    dpidList2016 = os.listdir(DATAPATH + "2016/")
    dpidList2017 = os.listdir(DATAPATH + "2017/")
    dpidList2018 = os.listdir(DATAPATH + "2018/")

    for i in range(len(dpidList2016)):
        dpidList2016[i] = dpidList2016[i][0:-9]

    for i in range(len(dpidList2017)):
        dpidList2017[i] = dpidList2017[i][0:-9]

    for i in range(len(dpidList2018)):
        dpidList2018[i] = dpidList2018[i][0:-9]


    sharedDpidList = list(set(dpidList2016) & set(dpidList2017) & set(dpidList2018))

    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/Original/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/ML_V1/GLM_V1_Origin/"
    pool = multiprocessing.Pool(50)
    m = multiprocessing.Manager()
    pool.starmap(plot_GLM_Original, [(rpcImonPath, plotPath, sharedDpid) for sharedDpid in sharedDpidList])
    pool.close()
    pool.join()


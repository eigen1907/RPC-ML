import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.cluster import DBSCAN
import gc


def concatStr(str_list):
    return ''.join(str_list)


def formatF(x):
    return np.format_float_scientific(x, precision = 3, exp_digits=2)


def normMinMax(x):
    return (x - x.min())/(x.max() - x.min())


def plot_SLR_AfterDBSCAN(rpcImonPath, plotPath, dpidName):
    print("#"*80)
    print(f"Start draw {plotPath[72:-1]}, {dpidName[0:-9]}")
    print("#"*80)
    rpcImonData = pd.read_csv(concatStr([rpcImonPath, dpidName]), low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")

    #### delete outlier as DBSCAN
    Imon = rpcImonData["Imon"]

    Lumi = rpcImonData["inst_lumi"]

    X = pd.concat([normMinMax(Imon), normMinMax(Lumi)], ignore_index=True, axis=1)

    model = DBSCAN(eps=0.1, min_samples=round(len(rpcImonData)/100))
    modelLabels = model.fit_predict(X)


    #### label != -1 => normal, label = -1 => outlier
    rpcImonData["label"] = modelLabels


    trainData, testData = train_test_split(rpcImonData, test_size=0.4, shuffle=True, random_state=34)

    trainData = trainData[trainData["label"]!=-1]


    train_x = trainData["inst_lumi"].values.reshape(-1, 1)
    train_y = trainData["Imon"]


    test_x = testData["inst_lumi"].values.reshape(-1, 1)
    test_y = testData["Imon"]

    lineFitter = LinearRegression()

    if len(train_x) == 0:
        print("train_data is not exist")
        return
        
    lineFitter.fit(train_x, train_y)


    predict_y = lineFitter.predict(test_x)
    yDiff = test_y - predict_y

    c = ["b", "g", "m", "y", "k"]
    
    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(221)
    for i in range(len(testData.label.unique())-1):
        ax1.plot(testData[testData["label"]==i].Imon_change_date, testData[testData["label"]==i].Imon, ".", c=c[i%5], label=f"measured_cluster{i}")
    ax1.plot(testData[testData["label"]==-1].Imon_change_date, testData[testData["label"]==-1].Imon, ".", c="r", label="measured_outlier")
    ax1.plot(testData.Imon_change_date, predict_y, '.', c='orange', label="predicted")
    ax1.set_xlabel("(Predict VS Measured), axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()

    ax2 = plt.subplot(223)
    for i in range(len(testData.label.unique())-1):
        ax2.plot(testData[testData["label"]==i].inst_lumi, testData[testData["label"]==i].Imon, ".", c=c[i%5], label=f"measured_cluster{i}")
    ax2.plot(testData[testData["label"]==-1].inst_lumi, testData[testData["label"]==-1].Imon, ".", c="r", label="measured_outlier")
    ax2.plot(testData.inst_lumi, predict_y, '.', c="orange", label="predicted")
    ax2.set_xlabel("(Predict VS Measured) axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict)'s Histogram")
    ax3.annotate(f"Mean(abs): {formatF(np.mean(np.abs(yDiff)))}\n RMS: {formatF(np.sqrt(np.mean(yDiff**2)))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber: {dpidName[0:-9]} \n SLR After DBSCAN (Train: 60, Test: 40)")
    plt.tight_layout()
    plt.savefig(concatStr([plotPath, dpidName[0:-4], ".png"]))
    plt.close("all")     
    print(f"Finish draw {plotPath[72:-1]}, {dpidName[0:-9]}")

    gc.collect()

    return


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/SLR/I-L_fitting_AfterDBSCAN/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepFolders.remove("2018_normal")
    """
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        #for dpidName in dpidNames:
            #plot_GLM_AfterDBSCAN(rpcImonSepPath, plotSepPath, dpidName)
        pool = multiprocessing.Pool(8)
        m = multiprocessing.Manager()
        pool.starmap(plot_SLR_AfterDBSCAN, [(rpcImonSepPath, plotSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()
    """
    
    rpc2018NormalPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/2018_normal/"
    plot2018NormalPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/SLR/I-L_fitting_AfterDBSCAN/2018_normal/"

    
    dpidNames = os.listdir(rpc2018NormalPath)
    for dpidName in dpidNames:
        plot_SLR_AfterDBSCAN(rpc2018NormalPath, plot2018NormalPath, dpidName)
        
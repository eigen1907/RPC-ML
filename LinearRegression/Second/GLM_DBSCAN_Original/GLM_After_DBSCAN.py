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


def plot_GLM_AfterDBSCAN(rpcImonPath, plotPath, dpidName):
    print("#"*80)
    print(f"Start draw {plotPath[72:-1]}, {dpidName[0:-9]}")
    print("#"*80)
    rpcImonData = pd.read_csv(concatStr([rpcImonPath, dpidName]), low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")

    initialTime = rpcImonData["Imon_change_date"][0]

    #### delete outlier as DBSCAN
    Imon = rpcImonData["Imon"]

    Lumi = rpcImonData["inst_lumi"]

    X = pd.concat([normMinMax(Imon), normMinMax(Lumi)], ignore_index=True, axis=1)

    model = DBSCAN(eps=0.1, min_samples=round(len(rpcImonData)/100))
    modelLabels = model.fit_predict(X)


    #### label != -1 => normal, label = -1 => outlier
    rpcImonData["label"] = modelLabels


    trainData, testData = train_test_split(rpcImonData, test_size=0.4, shuffle=True, random_state=34)

    ### traindata isn't including outlier
    trainData = trainData[trainData["label"]!=-1]


    train_x1 = trainData["inst_lumi"]
    train_x2 = trainData["Vmon"]
    train_x3 = trainData["temp"]
    train_x4 = trainData["inst_lumi"] * np.exp(trainData["Vmon"] / trainData["press"])
    train_x5 = trainData["relative_humodity"]
    train_x6 = trainData["press"]
    train_x7 = (trainData["Imon_change_date"] - initialTime).astype(int) / 10**9

    train_X = pd.concat([train_x1, train_x2, train_x3, train_x4, train_x5, train_x6, train_x7], ignore_index=True, axis=1)
    train_y = trainData["Imon"]


    ### testset include outlier
    test_x1 = testData["inst_lumi"]
    test_x2 = testData["Vmon"]
    test_x3 = testData["temp"]
    test_x4 = testData["inst_lumi"] * np.exp(testData["Vmon"] / testData["press"])
    test_x5 = testData["relative_humodity"]
    test_x6 = testData["press"]
    test_x7 = (testData["Imon_change_date"] - initialTime).astype(int) / 10**9    


    test_X = pd.concat([test_x1, test_x2, test_x3, test_x4, test_x5, test_x6, test_x7], ignore_index=True, axis=1)
    test_y = testData["Imon"]

    ### testset except outlier
    testDataNoOutlier = testData[testData["label"]!=-1]

    test_x1_no_outlier = testDataNoOutlier["inst_lumi"]
    test_x2_no_outlier = testDataNoOutlier["Vmon"]
    test_x3_no_outlier = testDataNoOutlier["temp"]
    test_x4_no_outlier = testDataNoOutlier["inst_lumi"] * np.exp(testDataNoOutlier["Vmon"] / testDataNoOutlier["press"])
    test_x5_no_outlier = testDataNoOutlier["relative_humodity"]
    test_x6_no_outlier = testDataNoOutlier["press"]
    test_x7_no_outlier = (testDataNoOutlier["Imon_change_date"] - initialTime).astype(int) / 10**9    


    test_X_no_outlier = pd.concat([test_x1_no_outlier, test_x2_no_outlier, test_x3_no_outlier, test_x4_no_outlier, test_x5_no_outlier, test_x6_no_outlier, test_x7_no_outlier], ignore_index=True, axis=1)
    test_y_no_outlier = testDataNoOutlier["Imon"]
    

    ### testset only outlier
    testDataOutlier = testData[testData["label"]==-1]

    test_x1_outlier = testDataOutlier["inst_lumi"]
    test_x2_outlier = testDataOutlier["Vmon"]
    test_x3_outlier = testDataOutlier["temp"]
    test_x4_outlier = testDataOutlier["inst_lumi"] * np.exp(testDataOutlier["Vmon"] / testDataOutlier["press"])
    test_x5_outlier = testDataOutlier["relative_humodity"]
    test_x6_outlier = testDataOutlier["press"]
    test_x7_outlier = (testDataOutlier["Imon_change_date"] - initialTime).astype(int) / 10**9    


    test_X_outlier = pd.concat([test_x1_outlier, test_x2_outlier, test_x3_outlier, test_x4_outlier, test_x5_outlier, test_x6_outlier, test_x7_outlier], ignore_index=True, axis=1)
    test_y_outlier = testDataOutlier["Imon"]


    ### linear regression with train data
    lineFitter = LinearRegression()
    if len(train_X) == 0:
        print("train_X is not exist")
        return
    if len(test_X) == 0:
        print("test_X is not exist")
        return
    if len(test_X_no_outlier) == 0:
        print("test_X_no_outlier is not exist")
        return
    if len(test_X_outlier) == 0:
        print("test_X_outlier is not exist")
        return

    lineFitter.fit(train_X, train_y)

    predict_y = lineFitter.predict(test_X)
    predict_y_no_outlier = lineFitter.predict(test_X_no_outlier)
    predict_y_outlier = lineFitter.predict(test_X_outlier)
    


    yDiff = test_y - predict_y
    yDiffNoOutlier = test_y_no_outlier - predict_y_no_outlier
    yDiffOutlier = test_y_outlier - predict_y_outlier
    
    c = ["b", "g", "m", "y", "k"]

    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(221)
    for i in range(len(testData.label.unique())-1):
        ax1.plot(testData[testData["label"]==i].Imon_change_date, testData[testData["label"]==i].Imon, ".", c=c[i%5], label=f"measured_cluster{i}")
    ax1.plot(testData[testData["label"]==-1].Imon_change_date, testData[testData["label"]==-1].Imon, ".", c="r", label="measured_outlier")
    ax1.plot(testData.Imon_change_date, predict_y, '.', c='orange', label="predicted")
    ax1.set_xlabel("Imon_change_date")
    ax1.set_ylabel("Imon")
    ax1.legend()

    ax2 = plt.subplot(223)
    for i in range(len(testData.label.unique())-1):
        ax2.plot(testData[testData["label"]==i].inst_lumi, testData[testData["label"]==i].Imon, ".", c=c[i%5], label=f"measured_cluster{i}")
    ax2.plot(testData[testData["label"]==-1].inst_lumi, testData[testData["label"]==-1].Imon, ".", c="r", label="measured_outlier")
    ax2.plot(testData.inst_lumi, predict_y, '.', c="orange", label="predicted")
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(yDiffNoOutlier, bins=80, label="Cluster", log=True, color="b" ,alpha=0.7)
    ax3.hist(yDiffOutlier, bins=80, label="Outlier", log=True, color="r", alpha=0.7)
    ax3.set_xlabel(f"(Measured - Predicted)'s Histogram, Log scale")
    ax3.annotate(
        f"Mean(abs): {formatF(np.mean(np.abs(yDiff)))}\n RMS: {formatF(np.sqrt(np.mean(yDiff**2)))}\n RMS(Except Outlier): {formatF(np.sqrt(np.mean(yDiffNoOutlier**2)))}",
        xy=(1, 1),
        xycoords="axes fraction",
        fontsize=16,
        horizontalalignment="right",
        verticalalignment="bottom"
    )
    ax3.legend()

    plt.suptitle(f"Chamber: {dpidName[0:-9]} \n GLM After DBSCAN (Train: 60, Test: 40)")
    plt.tight_layout()
    plt.savefig(concatStr([plotPath, dpidName[0:-4], ".png"]))
    plt.close()
         
    print(f"Finish draw {plotPath[72:-1]}, {dpidName[0:-9]}")
    gc.collect()
    
    return


if __name__ == "__main__":
    
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/ML_V1/ML_V1_AfterDBSCAN/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepFolders.remove("2018_normal")
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        #for dpidName in dpidNames:
            #plot_GLM_AfterDBSCAN(rpcImonSepPath, plotSepPath, dpidName)
        pool = multiprocessing.Pool(20)
        m = multiprocessing.Manager()
        pool.starmap(plot_GLM_AfterDBSCAN, [(rpcImonSepPath, plotSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()
    
    rpc2018NormalPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/2018_normal/"
    plot2018NormalPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/ML_V1/ML_V1_AfterDBSCAN/2018_normal/"

    
    dpidNames = os.listdir(rpc2018NormalPath)
    for dpidName in dpidNames:
        plot_GLM_AfterDBSCAN(rpc2018NormalPath, plot2018NormalPath, dpidName) 

        

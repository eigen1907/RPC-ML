import os
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.cluster import DBSCAN
import gc
from tqdm import tqdm


def concatStr(str_list):
    return ''.join(str_list)


def formatF(x):
    return np.format_float_scientific(x, precision = 3, exp_digits=2)


def normMinMax(x):
    return (x - x.min())/(x.max() - x.min())


def GLM_AfterDBSCAN_score(rpcImonPath, dpidName):
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
        print("train_data is not exist")
        return

    lineFitter.fit(train_X, train_y)

    predict_y = lineFitter.predict(test_X)
    yDiff = test_y - predict_y

    with open('/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_RMS_score.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([rpcImonPath, dpidName, formatF(np.sqrt(np.mean(yDiff**2)))])
        f.close()

    gc.collect()
    return


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepFolders.remove("2018_normal")
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)

        pool = multiprocessing.Pool(8)
        m = multiprocessing.Manager()
        pool.starmap(GLM_AfterDBSCAN_score, [(rpcImonSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()

    rpc2018NormalPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/2018_normal/"
    dpidNames = os.listdir(rpc2018NormalPath)
    for dpidName in tqdm(dpidNames):
        GLM_AfterDBSCAN_score(rpc2018NormalPath, dpidName) 

    print("Done!!")

        
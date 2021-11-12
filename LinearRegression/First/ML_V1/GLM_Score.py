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


def GLM_score(rpcImonPath, dpidName):
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

    with open('/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_score.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([rpcImonPath[92:-1], dpidName[0:-4], formatF(np.sqrt(np.mean(yDiff**2)))])
        f.close()

    gc.collect()
    return


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)

        pool = multiprocessing.Pool(30)
        m = multiprocessing.Manager()
        pool.starmap(GLM_score, [(rpcImonSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()

    print("Done!!")
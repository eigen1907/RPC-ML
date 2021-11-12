import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.cluster import DBSCAN
import gc


def addDBSCAN(inputDataPath, outputDataPath, dpidName):
    data = pd.read_csv(inputDataPath + dpidName, low_memory=False)

    data["Imon_change_date"] = pd.to_datetime(data["Imon_change_date"])


    x1 = data["Imon"]

    x2 = data["inst_lumi"]

    X = pd.concat([(x1 - x1.min())/(x1.max() - x1.min()), (x2 - x2.min())/(x2.max() - x2.min())], ignore_index=True, axis=1)
    model = DBSCAN(eps=0.1, min_samples=round(len(data)/100))
    model_labels = model.fit_predict(X)
    data["DBSCAN_label"] = model_labels

    data.to_csv(outputDataPath + dpidName, index=False)

    gc.collect()


if __name__ == "__main__":
    inputDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCSep/"
    outputDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCSepDBSCAN/"

    inputDataFolders = os.listdir(inputDataPath)
    
    inputDataPaths, outputDataPaths = [], []
    for folder in inputDataFolders:
        inputDataPaths.append(inputDataPath + folder + "/")
        outputDataPaths.append(outputDataPath + folder + "/")


    for i in range(len(inputDataPaths)):
        dpidNames = os.listdir(inputDataPaths[i])
        pool = multiprocessing.Pool(3)
        m = multiprocessing.Manager()
        pool.starmap(addDBSCAN, [(inputDataPaths[i], outputDataPaths[i], dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()
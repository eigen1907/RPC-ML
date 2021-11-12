import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.cluster import DBSCAN
import gc


def figure(rpcImonPath, plotPath, dpidName):
    print("#"*80)
    print(f"Start draw {plotPath[72:-1]}, {dpidName[0:-9]}")
    print("#"*80)
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)

    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    x1 = rpcImonData["Imon"]

    x2 = rpcImonData["inst_lumi"]

    X = pd.concat([(x1 - x1.min())/(x1.max() - x1.min()), (x2 - x2.min())/(x2.max() - x2.min())], ignore_index=True, axis=1)
    model = DBSCAN(eps=0.1, min_samples=round(len(rpcImonData)/100))
    model_labels = model.fit_predict(X)
    rpcImonData["label"] = model_labels


    c = ["b", "g", "m", "y", "k"]
    plt.figure(figsize=(14, 8))
    for i in range(len(rpcImonData.label.unique())-1):
        plt.plot(rpcImonData[rpcImonData["label"]==i].inst_lumi, rpcImonData[rpcImonData["label"]==i].Imon, ".", c=c[i%5], label=f"cluster{i}")
    plt.plot(rpcImonData[rpcImonData["label"]==-1].inst_lumi, rpcImonData[rpcImonData["label"]==-1].Imon, ".", c="r", label="outlier")
    plt.xlabel("axis-: inst_lumi, axis-y: Imon")
    plt.title(f"Result of DBSCAN Clustering, {plotPath[72:-1]}, {dpidName[0:-9]}, Number of Cluster: {len(rpcImonData.label.unique()) - 1}")
    plt.legend()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()
    print(f"Finish draw {plotPath[72:-1]}, {dpidName[0:-9]}")

    gc.collect()


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/Preprocessing/DBSCAN/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepFolders.remove("2018_normal")
    
    rpcImonSepPaths, plotSepPaths = [], []
    for folder in rpcImonSepFolders:
        rpcImonSepPaths.append(rpcImonPath + folder + "/")
        plotSepPaths.append(plotPath + folder + "/")


    for i in range(len(rpcImonSepPaths)):
        dpidNames = os.listdir(rpcImonSepPaths[i])
        pool = multiprocessing.Pool(6)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcImonSepPaths[i], plotSepPaths[i], dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()
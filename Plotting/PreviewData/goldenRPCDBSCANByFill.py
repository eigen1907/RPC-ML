import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def HV_Formula(rpcData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = rpcData.press
    temp = rpcData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/(temp + temp0)))


def figure(rpcDataPath, rpcPlotPath):
    rpcData = pd.read_csv(rpcDataPath, low_memory=False)

    rpcData["Imon_change_date"] = pd.to_datetime(rpcData["Imon_change_date"])
    
    rpcData = rpcData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    date = rpcData.Imon_change_date
    Imon = rpcData.Imon
    inst_lumi = rpcData.inst_lumi
    Vmon = rpcData.Vmon

    c = ["b", "g", "m", "y", "k"]
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    if len(rpcData.DBSCAN_label.unique()) == 1:
        ax1.plot(date, Imon, '.', c="b", label="No Cluster")
        ax1.set_xlabel("Imon_change_date")
        ax1.set_ylabel("Imon")
        ax1.legend()

        ax2.plot(inst_lumi, Imon, '.', c="b", label="No Cluster")
        ax2.set_xlabel("inst_lumi")
        ax2.set_ylabel("Imon")
        ax2.legend()

        ax3.plot(date, Vmon, '.', c="b", label="No Cluster")
        ax3.set_xlabel("Imon_change_date")
        ax3.set_ylabel("Vapp")
        ax3.legend()

        ax4.plot(HV_Formula(rpcData), Vmon, '.', c="b", label="No Cluster")
        ax4.set_xlabel("Veff")
        ax4.set_ylabel("Vapp")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(rpcPlotPath)
        plt.close()

    else:
        for i in range(len(rpcData.DBSCAN_label.unique())-1):
            ax1.plot(date[rpcData["DBSCAN_label"]==i], rpcData[rpcData["DBSCAN_label"]==i].Imon, '.', c=c[i%5], label=f"cluster{i}")
            ax2.plot(inst_lumi[rpcData["DBSCAN_label"]==i], Imon[rpcData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")
            ax3.plot(date[rpcData["DBSCAN_label"]==i], Vmon[rpcData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")
            ax4.plot(HV_Formula(rpcData[rpcData["DBSCAN_label"]==i]), Vmon[rpcData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")

        ax1.plot(date[rpcData["DBSCAN_label"]==-1], rpcData[rpcData["DBSCAN_label"]==-1].Imon, '.', c="r", label="outlier")
        ax1.set_xlabel("Imon_change_date")
        ax1.set_ylabel("Imon")
        ax1.legend()

        ax2.plot(inst_lumi[rpcData["DBSCAN_label"]==-1], Imon[rpcData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
        ax2.set_xlabel("inst_lumi")
        ax2.set_ylabel("Imon")
        ax2.legend()

        ax3.plot(date[rpcData["DBSCAN_label"]==-1], Vmon[rpcData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
        ax3.set_xlabel("Imon_change_date")
        ax3.set_ylabel("Vapp")
        ax3.legend()

        ax4.plot(HV_Formula(rpcData[rpcData["DBSCAN_label"]==-1]), Vmon[rpcData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
        ax4.set_xlabel("Veff")
        ax4.set_ylabel("Vapp")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(rpcPlotPath)
        plt.close()



if __name__ == "__main__":
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCDBSCANByFill/"
    rpcPlotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/DataPreview/GoldenRPCDBSCANByFill/"
    dpidNames = os.listdir(rpcDataPath)

    for dpidName in dpidNames:
        createFolder(rpcPlotPath + dpidName)
        fillNumbers = os.listdir(rpcDataPath + dpidName + "/")
        pool = multiprocessing.Pool(200)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcDataPath + dpidName + "/" + fillNumber, rpcPlotPath + dpidName + "/" + fillNumber[0:-4] + ".png") for fillNumber in fillNumbers])    
        pool.close()
        pool.join()

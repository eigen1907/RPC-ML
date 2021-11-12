import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing



def concatStr(str_list):
    return ''.join(str_list)


def HV_Formula(rpcImonData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = rpcImonData.press
    temp = rpcImonData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/(temp + temp0)))


def figure(rpcImonPath, rpcImonFile, figurePath):
    print(f"Start to plot {rpcImonFile[0:-4]}")

    rpcImonData = pd.read_csv(rpcImonPath + rpcImonFile, low_memory=False)

    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    date = rpcImonData.Imon_change_date
    Imon = rpcImonData.Imon
    inst_lumi = rpcImonData.inst_lumi
    Vmon = rpcImonData.Vmon

    c = ["b", "g", "m", "y", "k"]
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    for i in range(len(rpcImonData.DBSCAN_label.unique())-1):
        ax1.plot(date[rpcImonData["DBSCAN_label"]==i], rpcImonData[rpcImonData["DBSCAN_label"]==i].Imon, '.', c=c[i%5], label=f"cluster{i}")
        ax2.plot(inst_lumi[rpcImonData["DBSCAN_label"]==i], Imon[rpcImonData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")
        ax3.plot(date[rpcImonData["DBSCAN_label"]==i], Vmon[rpcImonData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")
        ax4.plot(HV_Formula(rpcImonData[rpcImonData["DBSCAN_label"]==i]), Vmon[rpcImonData["DBSCAN_label"]==i], '.', c=c[i%5], label=f"cluster{i}")
    
    ax1.plot(date[rpcImonData["DBSCAN_label"]==-1], rpcImonData[rpcImonData["DBSCAN_label"]==-1].Imon, '.', c="r", label="outlier")
    ax1.set_xlabel("Imon_change_date")
    ax1.set_ylabel("Imon")
    ax1.legend()

    ax2.plot(inst_lumi[rpcImonData["DBSCAN_label"]==-1], Imon[rpcImonData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")
    ax2.legend()

    ax3.plot(date[rpcImonData["DBSCAN_label"]==-1], Vmon[rpcImonData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
    ax3.set_xlabel("Imon_change_date")
    ax3.set_ylabel("Vapp")
    ax3.legend()
    
    ax4.plot(HV_Formula(rpcImonData[rpcImonData["DBSCAN_label"]==-1]), Vmon[rpcImonData["DBSCAN_label"]==-1], '.', c="r", label="outlier")
    ax4.set_xlabel("Veff")
    ax4.set_ylabel("Vapp")
    ax4.legend()

    plt.suptitle(f"{rpcImonFile[0:-4]}")
    plt.tight_layout()
    plt.savefig(figurePath + rpcImonFile[0:-4] + ".png")
    plt.close()


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCSepDBSCAN/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/DataPreview/GoldenRPCSepDBSCAN/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        pool = multiprocessing.Pool(100)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcImonSepPath, dpidName, plotSepPath) for dpidName in dpidNames])
        pool.close()
        pool.join()
    

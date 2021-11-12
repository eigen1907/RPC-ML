import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing


def concatStr(str_list):
    return ''.join(str_list)


def figure(rpcImonPath, rpcImonFile, figurePath):
    rpcImonData = pd.read_csv(rpcImonPath + rpcImonFile, low_memory=False)

    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.')
    ax1.set_xlabel("Imon_change_date")
    ax1.set_ylabel("Imon")

    ax2 = plt.subplot(222)
    ax2.plot(rpcImonData.Imon_change_date, rpcImonData.inst_lumi, '.')
    ax2.set_xlabel("Imon_change_date")
    ax2.set_ylabel("inst_lumi")

    ax3 = plt.subplot(223)
    ax3.plot(rpcImonData.Imon_change_date, rpcImonData.Vmon/rpcImonData.press, '.')
    ax3.set_xlabel("Imon_change_date")
    ax3.set_ylabel("Vmon / press")

    ax4 = plt.subplot(224)
    ax4.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.')
    ax4.set_xlabel("inst_lumi")
    ax4.set_ylabel("Imon")

    plt.suptitle(f"{rpcImonFile[0:-4]}")
    plt.tight_layout()
    plt.savefig(figurePath + rpcImonFile[0:-4] + ".png")
    plt.close()


if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/Imon-preprocessed_data/GoldenRPCSeparate/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/Preprocessing/AfterSep/"

    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        pool = multiprocessing.Pool(150)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcImonSepPath, dpidName, plotSepPath) for dpidName in dpidNames])
        pool.close()
        pool.join()






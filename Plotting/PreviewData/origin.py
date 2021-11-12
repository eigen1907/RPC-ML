import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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



    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(date, rpcImonData.Imon, '.')
    ax1.set_xlabel("Imon_change_date")
    ax1.set_ylabel("Imon")

    ax2 = plt.subplot(222)
    ax2.plot(inst_lumi, Imon, '.')
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")

    ax3 = plt.subplot(223)
    ax3.plot(date, Vmon, '.')
    ax3.set_xlabel("Imon_change_date")
    ax3.set_ylabel("Vapp")

    ax4 = plt.subplot(224)
    ax4.plot(HV_Formula(rpcImonData), Vmon, '.')
    ax4.set_xlabel("Veff")
    ax4.set_ylabel("Vapp")

    plt.suptitle(f"{rpcImonFile[0:-4]}")
    plt.tight_layout()
    plt.savefig(figurePath + rpcImonFile[0:-4] + ".png")
    plt.close()

if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/DataPreview/Origin/"

    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        pool = multiprocessing.Pool(128)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcImonSepPath, dpidName, plotSepPath) for dpidName in dpidNames])
        pool.close()
        pool.join()
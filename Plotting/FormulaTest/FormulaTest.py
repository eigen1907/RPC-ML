import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing


def HVFormula(rpcImonData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = rpcImonData.press
    temp = rpcImonData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/(temp + temp0)))


def concatStr(str_list):
    return ''.join(str_list)


def plotFormularVSIoverL(rpcImonPath, plotPath, dpidName):    
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")

    date = rpcImonData.Imon_change_date

    I = rpcImonData.Imon
    L = rpcImonData.inst_lumi
    V = rpcImonData.Vmon
    formula = HVFormula(rpcImonData)

    """
    plt.figure(figsize=(16, 8))
    
    ax1 = plt.subplot(221)
    ax1.plot(L, I, '.')
    ax1.set_xlabel("inst_lumi")
    ax1.set_ylabel("Imon")

    ax2 = plt.subplot(222)
    ax2.plot(formula, V, '.')
    ax2.set_xlabel("Formula")
    ax2.set_ylabel("Vmon")

    ax3 = plt.subplot(223)
    ax3.plot(date, I/L, '.', label="Imon/inst_lumi")
    ax3.plot(date, V/formula, '.', label="Vmon/Formula (=Veff)")
    ax3.set_xlabel("Date")
    ax3.legend()

    ax4 = plt.subplot(224)
    ax4.plot(V/formula, I/L, ".")
    ax4.set_xlabel("Vmon/Formula")
    ax4.set_ylabel("Imon/inst_lumi")
    try:
        ax4.set_ylim(0, np.mean(I/L)*5)
    except:
        print("Value Error, inst_lumi is zero")


    plt.tight_layout()
    """
    plt.figure(figsize=(16, 8))
    plt.plot(formula, V, ".")
    plt.xlabel("Formula")
    plt.ylabel("Vmon(Vapp)")
    plt.title(f"{dpidName[0:-4]}\nVmon(V_app) / Formula = V_eff")

    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()
    print(f"Finish plotting {plotPath + dpidName[0:-4]}.png")

if __name__ == "__main__":
    rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPC/"
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/Preprocessing/FormulaTest/"

    rpcImonSepFolders = os.listdir(rpcImonPath)
    for folder in rpcImonSepFolders:
        rpcImonSepPath = concatStr([rpcImonPath, folder, "/"])
        plotSepPath = concatStr([plotPath, folder, "/"])
        dpidNames = os.listdir(rpcImonSepPath)
        pool = multiprocessing.Pool(8)
        m = multiprocessing.Manager()
        pool.starmap(plotFormularVSIoverL, [(rpcImonSepPath, plotSepPath, dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()

    
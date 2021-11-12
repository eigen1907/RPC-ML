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


def HVFormula(rpcImonData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = rpcImonData.press
    temp = rpcImonData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/(temp + temp0)))


def figure(rpcDataPath, rpcPlotPath):
    rpcData = pd.read_csv(rpcDataPath, low_memory=False)

    rpcData["Imon_change_date"] = pd.to_datetime(rpcData["Imon_change_date"])
    
    rpcData = rpcData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    runNumber = rpcData["run_number"].unique()

    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    for i in range(len(runNumber)):
        data = rpcData[rpcData["run_number"] == runNumber[i]]
        label = str(data.run_number.unique()[0])
        ax1.plot(data.Imon_change_date, data.Imon, ".", label=label)
        ax2.plot(data.inst_lumi, data.Imon, ".", label=label)
        ax3.plot(data.Imon_change_date, data.Vmon, ".", label=label)
        ax4.plot(HVFormula(data), data.Vmon, ".", label=label)

    ax1.legend(title="RunNumber")
    ax2.legend(title="RunNumber")
    ax3.legend(title="RunNumber")
    ax4.legend(title="RunNumber")

    ax1.set_xlabel("Imon_change_date")
    ax2.set_xlabel("inst_lumi")
    ax3.set_xlabel("Imon_change_date")
    ax4.set_xlabel("Veff")

    ax1.set_ylabel("Imon")
    ax2.set_ylabel("Imon")
    ax3.set_ylabel("Vmon(Vapp)")
    ax4.set_ylabel("Vmon(Vapp)")

    plt.tight_layout()
    plt.savefig(rpcPlotPath)
    plt.close()
    print(f"{rpcDataPath} is Done!")



if __name__ == "__main__":
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalByFill/"
    rpcPlotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/DataPreview/OriginByFill/"
    dpidNames = os.listdir(rpcDataPath)

    for dpidName in dpidNames:
        createFolder(rpcPlotPath + dpidName)
        fillNumbers = os.listdir(rpcDataPath + dpidName + "/")
        pool = multiprocessing.Pool(200)
        m = multiprocessing.Manager()
        pool.starmap(figure, [(rpcDataPath + dpidName + "/" + fillNumber, rpcPlotPath + dpidName + "/" + fillNumber[0:-4] + ".png") for fillNumber in fillNumbers])    
        pool.close()
        pool.join()

    
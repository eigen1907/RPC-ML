import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing


def concatStr(str_list):
    return ''.join(str_list)


def HV_Formula(rpcData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = rpcData.press
    temp = rpcData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/(temp + temp0)))


def figure(sharedDpid, dataPath, plotPath):
    sectionList = ["2016_former", "2016_latter", "2017", "2018_dropping", "2018_normal"]
    dataList = []
    for i in range(len(sectionList)):
        dataList.append(pd.read_csv(f"{dataPath}{sectionList[i]}/{sharedDpid}_{sectionList[i][0:4]}.csv", low_memory=False))

    c = ["r", "g", "m", "y", "b"]

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    for i in range(len(dataList)):
        dataList[i]["Imon_change_date"] = pd.to_datetime(dataList[i]["Imon_change_date"])   
        dataList[i] = dataList[i].drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

        date = dataList[i].Imon_change_date
        Imon = dataList[i].Imon
        inst_lumi = dataList[i].inst_lumi
        Vmon = dataList[i].Vmon

        ax1.plot(date, Imon, '.', c=c[i], label=f"{sectionList[i]}", alpha=0.3)
        ax2.plot(inst_lumi, Imon, '.', c=c[i], label=f"{sectionList[i]}", alpha=0.3)
        ax3.plot(date, Vmon, '.', c=c[i], label=f"{sectionList[i]}", alpha=0.3)
        ax4.plot(HV_Formula(dataList[i]), Vmon, '.', c=c[i], label=f"{sectionList[i]}", alpha=0.3)

    ax1.set_xlabel("Imon_change_date")
    ax1.set_ylabel("Imon")
    ax1.legend()
    
    ax2.set_xlabel("inst_lumi")
    ax2.set_ylabel("Imon")
    ax2.legend()

    ax3.set_xlabel("Imon_change_date")
    ax3.set_ylabel("Vapp")
    ax3.legend()

    ax4.set_xlabel("Veff")
    ax4.set_ylabel("Vapp")
    ax4.legend()

    plt.suptitle(f"{sharedDpid}")
    plt.tight_layout()
    plt.savefig(plotPath + sharedDpid + ".png")
    plt.close()
    

if __name__ == "__main__":
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSep/"
    rpcPlotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/DataPreview/OriginConcat/"

    rpc2016former = os.listdir(f"{rpcDataPath}2016_former/")
    rpc2016latter = os.listdir(f"{rpcDataPath}2016_latter/")
    rpc2017 = os.listdir(f"{rpcDataPath}2017/")
    rpc2018dropping = os.listdir(f"{rpcDataPath}2018_dropping/")
    rpc2018normal = os.listdir(f"{rpcDataPath}2018_normal/")

    for i in range(len(rpc2016former)):
        rpc2016former[i] = rpc2016former[i][0:-9]

    for i in range(len(rpc2016latter)):
        rpc2016latter[i] = rpc2016latter[i][0:-9]

    for i in range(len(rpc2017)):
        rpc2017[i] = rpc2017[i][0:-9]

    for i in range(len(rpc2018dropping)):
        rpc2018dropping[i] = rpc2018dropping[i][0:-9]

    for i in range(len(rpc2018normal)):
        rpc2018normal[i] = rpc2018normal[i][0:-9]

    sharedDpids = list(set(rpc2016former) & set(rpc2016latter) & set(rpc2017) & set(rpc2018dropping) & set(rpc2018normal))

    pool = multiprocessing.Pool(128)
    m = multiprocessing.Manager()
    pool.starmap(figure, [(sharedDpid, rpcDataPath, rpcPlotPath) for sharedDpid in sharedDpids])
    pool.close()
    pool.join()
    
    
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing


def figure(rpcImonPath, rpcImonFile, figurePath):
    rpcImonData = pd.read_csv(rpcImonPath + rpcImonFile, low_memory=False)

    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])

    x1 = rpcImonData["Imon"]
    x1Max = max(x1)

    x2 = rpcImonData["inst_lumi"]
    x2Max = max(x2)

    x1Norm = x1 / x1Max
    x2Norm = x2 / x2Max

    X = pd.concat([x1Norm, x2Norm], ignore_index=True, axis=1)
    model = DBSCAN(eps=0.08, min_samples=50)
    model_labels = model.fit_predict(X)
    rpcImonData["label"] = model_labels


    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData[rpcImonData["label"]==0].Imon_change_date, rpcImonData[rpcImonData["label"]==0].Imon, '.')
    ax1.set_xlabel("axis-x: Imon_change_date, axis-y: Imon")

    ax2 = plt.subplot(222)
    ax2.plot(rpcImonData[rpcImonData["label"]==0].Imon_change_date, rpcImonData[rpcImonData["label"]==0].inst_lumi, '.')
    ax2.set_xlabel("axis-x: Imon_change_date, axis-y: inst_lumi")

    ax3 = plt.subplot(223)
    ax3.plot(rpcImonData[rpcImonData["label"]==0].Imon_change_date, rpcImonData[rpcImonData["label"]==0].Vmon/rpcImonData[rpcImonData["label"]==0].press, '.')
    ax3.set_xlabel("axis-x: Imon_change_date, axis-y: Vmon/press")

    ax4 = plt.subplot(224)
    ax4.plot(rpcImonData[rpcImonData["label"]==0].inst_lumi, rpcImonData[rpcImonData["label"]==0].Imon, '.')
    ax4.set_xlabel("axis-x: inst_lumi, axis-y: Imon")

    plt.suptitle(f"{rpcImonFile[0:-4]}")
    plt.tight_layout()
    plt.savefig(figurePath + rpcImonFile[0:-4] + ".png")
    plt.close()


if __name__ == "__main__":
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2018_dropping/"
    figurePath = "/Users/mainroot/RPC_graph/Preprocessing/AfterSepAfterDBSCAN/2018_dropping/"
    rpcImonFiles = os.listdir(rpcImonPath)

    pool = multiprocessing.Pool(3)
    m = multiprocessing.Manager()
    pool.starmap(figure, [(rpcImonPath, rpcImonFile, figurePath) for rpcImonFile in rpcImonFiles])
    pool.close()
    pool.join()
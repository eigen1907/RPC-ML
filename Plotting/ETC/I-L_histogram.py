import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def is_outlier(points, thresh=3.5):
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


"""
rpcImonFormerPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_former/"
rpcImonLatterPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_latter/"

dpidName = "dpid_392_2016.csv"

rpcImonFormerData = pd.read_csv(rpcImonFormerPath + dpidName, low_memory=False)
rpcImonLatterData = pd.read_csv(rpcImonLatterPath + dpidName, low_memory=False)

rpcImonFormerData.Imon_change_date = pd.to_datetime(rpcImonFormerData.Imon_change_date)
rpcImonLatterData.Imon_change_date = pd.to_datetime(rpcImonLatterData.Imon_change_date)


ratioFormerData = rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi
ratioLatterData = rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi

fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221)
ax1.plot(rpcImonFormerData.Imon_change_date, rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi, '.', label="Former", alpha=0.5, color="b")
ax1.plot(rpcImonLatterData.Imon_change_date, rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi, '.', label="Latter", alpha=0.5, color="r")
ax1.set_xlabel("(Former VS Latter), axis-x: Imon_change_date, axis-y: Imon / inst_lumi")
ax1.legend()

ax2 = plt.subplot(223)
ax2.plot(rpcImonFormerData.inst_lumi, rpcImonFormerData.Imon, '.', label="Former", alpha=0.5, color="b")
ax2.plot(rpcImonLatterData.inst_lumi, rpcImonLatterData.Imon, '.', label="Latter", alpha=0.5, color="r")
ax2.set_xlabel("(Former VS Latter) axis-x: inst_lumi, axis-y: Imon")
ax2.legend()

ax3 = plt.subplot(122)
ax3.hist(ratioFormerData[~is_outlier(ratioFormerData)], bins=100, label="Former", alpha=0.5, color="b", edgecolor="black")
ax3.hist(ratioLatterData[~is_outlier(ratioLatterData)], bins=100, label="Latter", alpha=0.5, color="r", edgecolor="black")
ax3.set_xlabel(f"(Former VS Latter) Histogram(Imon / inst_lumi)")
ax3.annotate(f"Former Mean: {np.mean(rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi)} \nLatter Mean: {np.mean(rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi)}", \
    xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')
ax3.legend()

plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \n Histogram of Imon/inst_lumi")
plt.tight_layout()
plt.show()
"""

def I_L_histogram(rpcImonFormerPath, rpcImonLatterPath, dpidName, plotPath):
    try:
        rpcImonFormerData = pd.read_csv(rpcImonFormerPath + dpidName, low_memory=False)
        rpcImonLatterData = pd.read_csv(rpcImonLatterPath + dpidName, low_memory=False)
    except:
        print(f"Maybe Either Former and Latter is empty Dpid : {dpidName[0:-4]}")
        return

    rpcImonFormerData.Imon_change_date = pd.to_datetime(rpcImonFormerData.Imon_change_date)
    rpcImonLatterData.Imon_change_date = pd.to_datetime(rpcImonLatterData.Imon_change_date)


    ratioFormerData = rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi
    ratioLatterData = rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonFormerData.Imon_change_date, rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi, '.', label="Former", alpha=0.5, color="b")
    ax1.plot(rpcImonLatterData.Imon_change_date, rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi, '.', label="Latter", alpha=0.5, color="r")
    ax1.set_xlabel("(Former VS Latter), axis-x: Imon_change_date, axis-y: Imon / inst_lumi")
    ax1.legend()

    ax2 = plt.subplot(223)
    ax2.plot(rpcImonFormerData.inst_lumi, rpcImonFormerData.Imon, '.', label="Former", alpha=0.5, color="b")
    ax2.plot(rpcImonLatterData.inst_lumi, rpcImonLatterData.Imon, '.', label="Latter", alpha=0.5, color="r")
    ax2.set_xlabel("(Former VS Latter) axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()

    ax3 = plt.subplot(122)
    ax3.hist(ratioFormerData[~is_outlier(ratioFormerData)], bins=100, label="Former", alpha=0.5, color="b", edgecolor="black")
    ax3.hist(ratioLatterData[~is_outlier(ratioLatterData)], bins=100, label="Latter", alpha=0.5, color="r", edgecolor="black")
    ax3.set_xlabel(f"(Former VS Latter) Histogram(Imon / inst_lumi)")
    ax3.annotate(f"Former Mean: {np.mean(rpcImonFormerData.Imon / rpcImonFormerData.inst_lumi)} \nLatter Mean: {np.mean(rpcImonLatterData.Imon / rpcImonLatterData.inst_lumi)}", \
        xy=(1, 1), xycoords='axes fraction', fontsize=10, \
        horizontalalignment='right', verticalalignment='bottom')
    ax3.legend()

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \n Histogram of Imon/inst_lumi")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()

if __name__ == "__main__":
    rpcImonFormerPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_former/"
    rpcImonLatterPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_latter/"
    plotPath = "/Users/mainroot/RPC_graph/I-L_histogram/2016/"

    dpidNames = os.listdir(rpcImonFormerPath)

    pool = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    pool.starmap(I_L_histogram, [(rpcImonFormerPath, rpcImonLatterPath, dpidName, plotPath) for dpidName in dpidNames])
    pool.close()
    pool.join()


#dpid_6455_2016
#2016-10-18 19:56:49,20.5,9515.0,-1.0,
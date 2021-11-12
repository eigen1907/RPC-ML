import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.cluster import DBSCAN
import gc


scoreGLM = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM+DBSCAN_RMS_score.csv", low_memory=False)
scoreSLR = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/SLR+DBSCAN_RMS_score.csv", low_memory=False)
scoreGLM2 = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_score.csv", low_memory=False)

print(scoreGLM.head())
print(scoreSLR.head())

print(len(scoreGLM))
print(len(scoreSLR))
print(len(scoreGLM2))

scoreGLM["info"] = scoreGLM["DataPath"] + "/" + scoreGLM["DpidName"]
scoreSLR["info"] = scoreSLR["DataPath"] + "/" + scoreSLR["DpidName"]
scoreGLM2["info"] = scoreGLM2["DataPath"] + "/" + scoreGLM2["DpidName"]

scoreGLM = scoreGLM.drop(columns=['DataPath', 'DpidName'])
scoreSLR = scoreSLR.drop(columns=['DataPath', 'DpidName'])
scoreGLM2 = scoreGLM2.drop(columns=['DataPath', 'DpidName'])

scoreAll = pd.merge(scoreGLM, scoreGLM2, how='outer', on='info')
scoreAll = pd.merge(scoreAll, scoreSLR, how='outer', on='info')


print(scoreAll.head())

"""
plt.plot(scoreAll.GLM2_RMS_Score, scoreAll.GLM_RMS_Score, ".")
plt.xlabel("GLM RMS")
plt.ylabel("GLM After DBSCAN RMS")
plt.xlim(0, 16)
plt.ylim(0, 16)
plt.title("Compare BeforeAfter DBSCAN")
plt.savefig("/store/hep/users/eigen1907/CMS-RPC_store/etc/CompareScatter2.png")
plt.close()

plt.hist(scoreAll.GLM2_RMS_Score, bins=100, histtype="step", label="Before DBSCAN")
plt.hist(scoreAll.GLM_RMS_Score, bins=100, histtype="step", label="After DBSCAN")
plt.xlabel("RMS Score")
plt.title("RMS Histogram (All Dpid)")
plt.legend()
plt.savefig("/store/hep/users/eigen1907/CMS-RPC_store/etc/compRMSHist.png")
plt.close()



plt.hist(scoreAll.SLR_RMS_Score, bins=100, histtype="step")
plt.xlabel(f"RMS (Mean: {np.mean(scoreAll.SLR_RMS_Score)})")
plt.title("SLR After DBSCAN's RMS Histogram (All Dpid)")
plt.savefig("/store/hep/users/eigen1907/CMS-RPC_store/etc/SLR_ScoreHist.png")
plt.close()

plt.hist(scoreAll.GLM_RMS_Score, bins=100, histtype="step")
plt.xlabel(f"RMS (Mean: {np.mean(scoreAll.GLM_RMS_Score)})")
plt.title("ML_V1 After DBSCAN's RMS Histogram (All Dpid)")
plt.savefig("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_ScoreHist.png")
plt.close()
"""

scoreAll.to_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/RMS_score_all.csv", index=False)




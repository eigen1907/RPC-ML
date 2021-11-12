import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.cluster import DBSCAN
import gc

scoreGLM = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_RMS_score.csv", low_memory=False)
scoreSLR = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/SLR_RMS_score.csv", low_memory=False)

#print(scoreGLM.head())
#print(scoreSLR.head())

#print(scoreGLM["DataPath"][0])
#print(scoreSLR["DataPath"][0])

for i in range(len(scoreGLM)):
    scoreGLM["DataPath"][i] = scoreGLM["DataPath"][i][92:-1]
    scoreGLM["DpidName"][i] = scoreGLM["DpidName"][i][0:-4]

for i in range(len(scoreSLR)):
    scoreSLR["DataPath"][i] = scoreSLR["DataPath"][i][92:-1]
    scoreSLR["DpidName"][i] = scoreSLR["DpidName"][i][0:-4]

print(scoreGLM.head())
print(scoreSLR.head())


scoreGLM.to_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_RMS_score.csv", index=False)
scoreSLR.to_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/SLR_RMS_score.csv", index=False)
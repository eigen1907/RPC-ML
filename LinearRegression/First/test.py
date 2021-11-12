import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing





rpcImonData = pd.read_csv("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_former/dpid_323_2016.csv", low_memory=False)
rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format='%Y-%m-%d %H:%M:%S', errors="raise")
x1 = rpcImonData["inst_lumi"]
x2 = rpcImonData["Vmon"]
x3 = rpcImonData["temp"]
x4 = rpcImonData["inst_lumi"] * np.exp(rpcImonData["Vmon"] / rpcImonData["press"])
x5 = rpcImonData["relative_humodity"]
x6 = rpcImonData["press"]
x7 = (rpcImonData["Imon_change_date"] - rpcImonData["Imon_change_date"][0]).astype(int) / 10**9
X = pd.concat([x1, x2, x3, x4, x5, x6, x7], ignore_index=True, axis=1)
y = rpcImonData["Imon"]
XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)
lineFitter = LinearRegression()
lineFitter.fit(XTrain, yTrain)
yPredict = lineFitter.predict(XValid)
yDiff = yValid - yPredict
columnList = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
total = rpcImonData.Imon
C0Part = np.full((len(rpcImonData),), lineFitter.intercept_)
C1Part = lineFitter.coef_[0] * x1
C2Part = lineFitter.coef_[1] * x2
C3Part = lineFitter.coef_[2] * x3
C4Part = lineFitter.coef_[3] * x4    
C5Part = lineFitter.coef_[4] * x5
C6Part = lineFitter.coef_[5] * x6
C7Part = lineFitter.coef_[6] * x7

partList = [C0Part, C1Part, C2Part, C3Part, C4Part, C5Part, C6Part, C7Part]



"""
fig, axes = plt.subplots(2, 4, figsize=(20, 8), tight_layout=True)

for i in range(8):
    axes[i//4, i%4].plot(rpcImonData.Imon_change_date, lineFitter.predict(X) + partList[i], '.', label=f"C{i} * X{i}")
    #axes[i//4, i%4].plot(rpcImonData.Imon_change_date, lineFitter.predict(X), '.', label="predicted Imon")
    axes[i//4, i%4].legend(loc="best")
    axes[i//4, i%4].set_xlabel(f"C{i} * X{i}")
"""
plt.plot(rpcImonData.Imon_change_date, lineFitter.predict(X) + partList[4], ".")


plt.title("After ML_V1, each partition of C_n * Value")
plt.show()


import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_former/"
rpcImonFile = "dpid_392_2016.csv"

rpcImonData = pd.read_csv(rpcImonPath + rpcImonFile, low_memory=False)

rpcImonData = rpcImonData.sort_values(by="inst_lumi", ascending=False, kind="quicksort")

rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)


print(rpcImonData)
x = rpcImonData.inst_lumi.values[0:int(len(rpcImonData)*0.5)].reshape(-1, 1)
y = rpcImonData.Imon[0:int(len(rpcImonData)*0.5)]
print(len(x))
print(len(y))

x = pd.concat([rpcImonData.inst_lumi[0:int(len(rpcImonData)*0.25)], rpcImonData.inst_lumi[int(len(rpcImonData)*0.75):]], ignore_index=True).values.reshape(-1, 1)
y = pd.concat([rpcImonData.Imon[0:int(len(rpcImonData)*0.25)], rpcImonData.Imon[int(len(rpcImonData)*0.75):]], ignore_index=True)
print(len(x))
print(len(y))

"""
#lineFitter = LinearRegression(fit_intercept=False)
lineFitter = LinearRegression()
lineFitter.fit(x, y)

yPredict = lineFitter.predict(rpcImonData.inst_lumi.values.reshape(-1, 1))

yDiff = rpcImonData.Imon - yPredict

fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221)
ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
ax1.legend()

ax2 = plt.subplot(223)
ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
ax2.legend()

ax3 = plt.subplot(122)
ax3.hist(yDiff, bins=100)
ax3.set_xlabel(f"Measured - Predict Histogram(Imon), Diff's Mean: {np.mean(yDiff)}")

plt.suptitle(f"Chamber, Year: {rpcImonFile[0:-4]}")
plt.tight_layout()
plt.show()

print(np.mean(np.abs(yDiff)))
print(np.sum(np.abs(yDiff)))

#plt.plot(yDiff, ".")
#plt.show()
"""
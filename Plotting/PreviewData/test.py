import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


def concatStr(str_list):
    return ''.join(str_list)

#print(len(rpcImonData[rpcImonData["inst_lumi"]==0]))


def applingHighVoltageFormula(rpcImonData):
    alpha = 0.8
    press0 = 965
    temp0 = 293
    press = press0 + rpcImonData.press
    temp = temp0 + rpcImonData.temp
    return (1 - alpha + alpha*(press/press0)*(temp0/temp))



rpcImonPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCSeparate/2018_normal/dpid_326_2018.csv"

rpcImonData = pd.read_csv(rpcImonPath, low_memory=False)
rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])


fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221)
ax1.plot(rpcImonData.Imon_change_date, applingHighVoltageFormula(rpcImonData), '.')
ax1.set_xlabel("Time")
ax1.set_ylabel("Formula")

ax2 = plt.subplot(222)
ax2.plot(rpcImonData.Imon_change_date, rpcImonData.Vmon, '.')
ax2.set_xlabel("Time")
ax2.set_ylabel("Vmon")

ax3 = plt.subplot(223)
ax3.plot(rpcImonData.Imon_change_date, rpcImonData.Vmon / applingHighVoltageFormula(rpcImonData), '.')
ax3.set_xlabel("Time")
ax3.set_ylabel("Vmon / Formula")

ax4 = plt.subplot(224)
ax4.plot(rpcImonData.Imon_change_date, rpcImonData.Vmon / applingHighVoltageFormula(rpcImonData)**2, '.')
ax4.set_xlabel("Time")
ax4.set_ylabel("Vmon / Formula^2")



plt.tight_layout()
plt.savefig("/users/eigen1907/CMS-RPC/Plotting/Second/test.png")
plt.close()



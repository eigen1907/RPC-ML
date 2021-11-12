import pandas as pd
import matplotlib.pyplot as plt

### origin vs golden 
originPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/Original/2017/"
goldenPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPC/2017/"

dpidName = "dpid_324_2017.csv"

originData = pd.read_csv(originPath + dpidName)
goldenData = pd.read_csv(goldenPath + dpidName)



for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
    originData[colName] = pd.to_datetime(originData[colName])
    goldenData[colName] = pd.to_datetime(goldenData[colName])



### In plot, golden data draw over origin
fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221)
ax1.plot(originData.Imon_change_date, originData.Imon, '.', label="Original")
ax1.plot(goldenData.Imon_change_date, goldenData.Imon, '.', label="Golden")
ax1.set_xlabel("axis-x: Imon_change_date, axis-y: Imon")
ax1.legend()

ax2 = plt.subplot(222)
ax2.plot(originData.Imon_change_date, originData.Vmon, '.', label="Original")
ax2.plot(goldenData.Imon_change_date, goldenData.Vmon, '.', label="Golden")
ax2.set_xlabel("axis-x: Imon_change_date, axis-y: Vmon")
ax2.legend()

ax3 = plt.subplot(223)
ax3.plot(originData.Imon_change_date, originData.Vmon/originData.press, '.', label="Original")
ax3.plot(goldenData.Imon_change_date, goldenData.Vmon/goldenData.press, '.', label="Golden")
ax3.set_xlabel("axis-x: Imon_change_date, axis-y: Vmon/press")
ax3.legend()

ax4 = plt.subplot(224)
#ax4.plot(originData.inst_lumi, originData.Imon, '.', label="Original")
ax4.plot(goldenData["inst_lumi"], goldenData["Imon"], ".", label="Golden")
#ax4.plot(goldenData.inst_lumi, goldenData.Imon, '.', label="Golden")
ax4.set_xlabel("axis-x: inst_lumi, axis-y: Imon")
ax4.legend()


plt.suptitle("golden data draw over origin")
plt.tight_layout()
plt.show()


### In plot, origin data draw over golden (never show golden data)
fig = plt.figure(figsize=(14, 8))
ax1 = plt.subplot(221)
ax1.plot(goldenData.Imon_change_date, goldenData.Imon, '.', label="Golden")
ax1.plot(originData.Imon_change_date, originData.Imon, '.', label="Original")
ax1.set_xlabel("axis-x: Imon_change_date, axis-y: Imon")
ax1.legend()


ax2 = plt.subplot(222)
ax2.plot(goldenData.Imon_change_date, goldenData.Vmon, '.', label="Golden")
ax2.plot(originData.Imon_change_date, originData.Vmon, '.', label="Original")
ax2.set_xlabel("axis-x: Imon_change_date, axis-y: Vmon")
ax2.legend()

ax3 = plt.subplot(223)
ax3.plot(goldenData.Imon_change_date, goldenData.Vmon/goldenData.press, '.', label="Golden")
ax3.plot(originData.Imon_change_date, originData.Vmon/originData.press, '.', label="Original")
ax3.set_xlabel("axis-x: Imon_change_date, axis-y: Vmon/press")
ax3.legend()

ax4 = plt.subplot(224)
ax4.plot(goldenData.inst_lumi, goldenData.Imon, '.', label="Golden")
ax4.plot(originData.inst_lumi, originData.Imon, '.', label="Original")
ax4.set_xlabel("axis-x: inst_lumi, axis-y: Imon")
ax4.legend()

plt.suptitle("origin data draw over golden (never show golden data)")
plt.tight_layout()
plt.show()

"""
plt.plot(goldenData["inst_lumi"], goldenData["Imon"], ".", label="Golden")
plt.plot(originData["inst_lumi"], originData["Imon"], ".", label="Origin")
plt.xlabel("origin data draw over golden (never show golden data)")
plt.title(f"axis-x: inst_lumi, axis-y: Imon, Data: {dpidName[0:-4]}")
plt.show()

plt.plot(originData["inst_lumi"], originData["Imon"], ".", label="Origin")
plt.plot(goldenData["inst_lumi"], goldenData["Imon"], ".", label="Golden")
plt.xlabel("golden data draw over origin")
plt.title(f"axis-x: inst_lumi, axis-y: Imon, Data: {dpidName[0:-4]}")
plt.show()
"""

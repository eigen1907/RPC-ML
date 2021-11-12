import json
import pandas as pd

rpcIMONFileName = "dpid_315_2016.csv"
runInfoFileName = "collisions2016.csv"
certFileName = "2016_271036-284044.json"
timeOfLS = 23.3 ## 23.3seconds = 2^18 orbits, 1 orbit = 26.7km/(speed of light)

rpcIMONs = pd.read_csv(rpcIMONFileName)

rpcIMONs = rpcIMONs.drop(columns=['Imon_change_date2'])
for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
    rpcIMONs[colName] = pd.to_datetime(rpcIMONs[colName])

runsInfo = pd.read_csv(runInfoFileName)
for colName in ['start_time','end_time']:
    runsInfo[colName] = pd.to_datetime(runsInfo[colName])

runsInfo = runsInfo[runsInfo['l1_hlt_mode_stripped'].str.match('collisions.*')]

selIMONs = []
certJSON = json.load(open(certFileName))
for runNumber, lumis in certJSON.items():
    runNumber = int(runNumber)
    runInfo = runsInfo[runsInfo.run_number == runNumber]
    if len(runInfo) == 0: continue
    if len(runInfo) != 1: print("Warning: multiple runs in JSON, should not happen!!!")
    runInfo = runInfo.iloc[0]
    fillNumber = runInfo.fill_number

    t0 = runInfo['start_time']
    for l1, l2 in lumis:
        t1 = t0 + pd.Timedelta(timeOfLS*(l1-1), unit='sec')
        t2 = t0 + pd.Timedelta(timeOfLS*l2, unit='sec')

        selIMON = rpcIMONs[(rpcIMONs.Imon_change_date >= t1) & (rpcIMONs.Imon_change_date <= t2)]
        nSel = len(selIMON)
        if nSel == 0: continue
        selIMON = selIMON.assign(run_number=runNumber)
        selIMON = selIMON.assign(fill_number=fillNumber)
        selIMONs.append(selIMON)

dfAll = pd.concat(selIMONs, ignore_index=True)

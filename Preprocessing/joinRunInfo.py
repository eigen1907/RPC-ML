#!/usr/bin/env python
import json
import pandas as pd

rpcIMONFileName = "dpid_315_2016.csv"
runInfoFileName = "collisions2016.csv"
certFileName = "2016_271036-284044.json"
timeOfLS = 23.3 ## 23.3seconds = 2^18 orbits, 1 orbit = 26.7km/(speed of light)

## Load the IMON data
print('='*80)
print("Loading RPC IMON info, '%s'" % rpcIMONFileName)
rpcIMONs = pd.read_csv(rpcIMONFileName)

#### convert dates stored in str formats to the native datetime format
rpcIMONs = rpcIMONs.drop(columns=['Imon_change_date2'])
for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
    rpcIMONs[colName] = pd.to_datetime(rpcIMONs[colName])

print("  Number of total entries =", len(rpcIMONs))
#print("  Invalid IMON change date =", len(rpcIMONs[rpcIMONs['Imon_change_date'] != rpcIMONs['Imon_change_date']]))
print("  IMON change before lumi starts =", len(rpcIMONs[rpcIMONs['lumi_start_date'] > rpcIMONs['Imon_change_date']]))
print("  IMON change after lumi ends    =", len(rpcIMONs[rpcIMONs['lumi_end_date'] < rpcIMONs['Imon_change_date']]))

## Load the run number information taken from the OMS
print('-'*80)
print("Loading run info, '%s'" % runInfoFileName)
runsInfo = pd.read_csv(runInfoFileName)
for colName in ['start_time','end_time']:
    runsInfo[colName] = pd.to_datetime(runsInfo[colName])

print("  Number of total entries =", len(runsInfo))
#runsInfo = runsInfo[runInfo['hlt_key'].str.match('.*cdaq/physics/.*')] ## Select physics runs
#print(runsInfo[~runsInfo['hlt_key'].str.match('.*cdaq/physics/.*')]['hlt_key'])
runsInfo = runsInfo[runsInfo['l1_hlt_mode_stripped'].str.match('collisions.*')] ## Select physics runs
print("  Entries after collision run selection =", len(runsInfo))
print("  Number of unique fill numbers =", len(runsInfo['fill_number'].unique()))
print("  Number of unique run numbers  =", len(runsInfo['run_number'].unique()))
print('='*80)

## Next step is to filter out IMON data points which are not certified as good LS.
##   Note: we don't use pandas here since the format is not that trivial for the pandas
## Load the CMS JSON file for the data certification and loop over the good LS
## and collect matching runInfo & IMONs within the good LS ranges.
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

## Join the runInfo & IMON dataframes to build the 'master' dataframe
dfAll = pd.concat(selIMONs, ignore_index=True)
print(dfAll)

## We are ready to analyze the data
import matplotlib.pyplot as plt
### To list up runs:
for fillNumber in dfAll.fill_number.unique():
    df = dfAll[dfAll.fill_number == fillNumber]
    runNumbers = df.run_number.unique()
    imonValues = df.Imon
    lumiValues = df.inst_lumi

    print('Fill=%d nRun=%d nIMON=%d' % (fillNumber, len(runNumbers), len(imonValues)))
    print(lumiValues)

    fig = plt.figure(figsize=(10,5))
    #plt.plot(df.inst_lumi, df.Imon, '.')
    for runNumber in runNumbers:
        dfRun = df[df.run_number == runNumber]
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(223)
        ax3 = plt.subplot(122)
        ax2.set_xlabel('time')
        ax1.set_ylabel('Inst. lumi (nb^-1)')
        ax2.set_ylabel('IMON current (uA)')
        ax3.set_xlabel('Inst. lumi (nb^-1)')
        ax3.set_ylabel('IMON current (uA)')
        ax1.plot(dfRun.Imon_change_date, dfRun.inst_lumi, '.-')
        ax2.plot(dfRun.Imon_change_date, dfRun.Imon, '.-')
        ax3.plot(dfRun.inst_lumi, dfRun.Imon, '.')
    plt.tight_layout()
    plt.show()

### You can draw values collapsing everything
#runNumbers = dfAll.run_number.unique()
#fillNumbers = dfAll.fill_number.unique()
#df = dfAll
#plt.plot(df.inst_lumi, df.Imon, '.')
#plt.show()

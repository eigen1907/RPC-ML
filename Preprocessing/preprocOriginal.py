import json
import pandas as pd
import numpy as np
import os
import multiprocessing


def preprocOriginal(rpcImonFile, runInfoFile, outputFile):
    columnList = ['Imon_change_date', 'Imon', 'Vmon', 'inst_lumi', 'lumi_start_date', 'lumi_end_date', 
    'Imon_change_date2', 'uxc_change_date', 'temp', 'press', 'relative_humodity', 'dew_point']
    dateDtypeList = ['Imon_change_date', 'lumi_start_date', 'lumi_end_date', 'uxc_change_date']
    floatDtypeList = ['Imon', 'Vmon', 'inst_lumi', 'temp', 'press', 'relative_humodity', 'dew_point']
    try:
        rpcImonData = pd.read_csv(
            rpcImonFile,
            names=columnList,
            low_memory=False
        )
    except:
        print("="*100)
        print("read_csv error")
        print(rpcImonFile)
    
    rpcImonData = rpcImonData.drop(columns=['Imon_change_date2'])
    rpcImonData = rpcImonData.dropna()

    for colName in dateDtypeList:
        rpcImonData[colName] = pd.to_datetime(rpcImonData[colName])

    for colName in floatDtypeList:
        rpcImonData[colName] = pd.to_numeric(rpcImonData[colName], errors='coerce', downcast='float')

    rpcImonData = rpcImonData[rpcImonData.inst_lumi >= 0]
    
    rpcImonData = rpcImonData.dropna()

    #### runNumberFile
    runInfoData = pd.read_csv(runInfoFile)

    runInfoData = runInfoData.dropna()

    for colName in ['start_time','end_time']:
        runInfoData[colName] = pd.to_datetime(runInfoData[colName])


    selImons = []
    for index, row in runInfoData.iterrows():
        start = row.start_time
        end = row.end_time
        runNumber = row.run_number
        fillNumber = row.fill_number

        selImon = rpcImonData[(rpcImonData.Imon_change_date >= start) & (rpcImonData.Imon_change_date < end)]

        selImon = selImon.assign(run_number=int(runNumber))
        selImon = selImon.assign(fill_number=int(fillNumber))
        selImons.append(selImon)
        
    try:
        rpcImonCertData = pd.concat(selImons, ignore_index=True)
        rpcImonCertData = rpcImonCertData[rpcImonCertData.inst_lumi >= 0]
        rpcImonCertData.dropna()
        rpcImonCertData.to_csv(outputFile, index=False)
    except:
        print("="*100)
        print("concat error")
        print(f"rpcImonFile: {rpcImonFile}")
        print(f"selImons: {selImons}")


if __name__ == "__main__":
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonOrigin/"

    newDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original2/"

    runInfoPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/RunNumberCSV/run_number_"

    years = ["2016", "2017", "2018"]

    for year in years:
        rpcDataFolder = rpcDataPath + year + "/"
        newDataFolder = newDataPath + year + "/"
        runInfoFile = runInfoPath + year + ".csv"

        dpidList = os.listdir(rpcDataFolder)

        pool = multiprocessing.Pool(200)
        m = multiprocessing.Manager()
        pool.starmap(preprocOriginal, [(rpcDataFolder + dpid, runInfoFile, newDataFolder + dpid) for dpid in dpidList])
        pool.close()
        pool.join()
    


            

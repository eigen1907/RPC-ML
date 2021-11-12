import json
import pandas as pd
import numpy as np
import os
import multiprocessing


def selectGoldenInterval(rpcImonFile, runInfoFile, runCertFile, outputFile):
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
    
    rpcImonData = rpcImonData.dropna()

    #### runNumberFile (RPC only)
    runInfoData = pd.read_csv(runInfoFile)

    runInfoData = runInfoData.dropna()

    for colName in ['start_time','end_time']:
        runInfoData[colName] = pd.to_datetime(runInfoData[colName])
    runInfoData = runInfoData[runInfoData['l1_hlt_mode_stripped'].str.match('collisions.*')]


    #### 2016 runCertFile, JSON format, golden data
    runCertData = json.load(open(runCertFile))


    #### select certification interval, condition is only golden data
    timeOfLS = 23.3 ## 23.3seconds = 2^18 orbits, 1 orbit = 26.7km/(speed of light)
    selImons = []
    for runNumber, lumis in runCertData.items():
        runNumber = int(runNumber)
        runInfo = runInfoData[runInfoData.run_number == runNumber]
        if len(runInfo) == 0:
            continue
        if len(runInfo) != 1:
            print("Warning: multiple runs in JSON, should not happen!!!")
        runInfo = runInfo.iloc[0]
        fillNumber = int(runInfo.fill_number)

        t0 = runInfo['start_time']
        for l1, l2 in lumis:
            t1 = t0 + pd.Timedelta(timeOfLS*(l1-1), unit='sec')
            t2 = t0 + pd.Timedelta(timeOfLS*l2, unit='sec')

            selImon = rpcImonData[(rpcImonData.Imon_change_date >= t1) & (rpcImonData.Imon_change_date <= t2)]
            nSel = len(selImon)
            if nSel == 0: continue
            selImon = selImon.assign(run_number=runNumber)
            selImon = selImon.assign(fill_number=fillNumber)
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
    IN_PATH_2016 = "/Users/mainroot/RPC_origin_data/2016/"
    IN_PATH_2017 = "/Users/mainroot/RPC_origin_data/2017/"
    IN_PATH_2018 = "/Users/mainroot/RPC_origin_data/2018/"

    RUN_INFO_FILE_2016 = "/Users/mainroot/RPC_origin_data/RunNumber/run_number_2016_RPC.csv"
    RUN_INFO_FILE_2017 = "/Users/mainroot/RPC_origin_data/RunNumber/run_number_2017_RPC.csv"
    RUN_INFO_FILE_2018 = "/Users/mainroot/RPC_origin_data/RunNumber/run_number_2018_RPC.csv"

    RUN_CERT_FILE_2016 = "/Users/mainroot/RPC_origin_data/RunCert/run_cert_2016.json"
    RUN_CERT_FILE_2017 = "/Users/mainroot/RPC_origin_data/RunCert/run_cert_2017.json"
    RUN_CERT_FILE_2018 = "/Users/mainroot/RPC_origin_data/RunCert/run_cert_2018.json"

    OUT_PATH_2016 = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPC/2016/"
    OUT_PATH_2017 = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPC/2017/"
    OUT_PATH_2018 = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPC/2018/"

    pool = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    fileList = os.listdir(IN_PATH_2016)
    pool.starmap(selectGoldenInterval, [(IN_PATH_2016 + fileName, RUN_INFO_FILE_2016, RUN_CERT_FILE_2016, OUT_PATH_2016 + fileName) for fileName in fileList])
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    fileList= os.listdir(IN_PATH_2017)
    pool.starmap(selectGoldenInterval, [(IN_PATH_2017 + fileName, RUN_INFO_FILE_2017, RUN_CERT_FILE_2017, OUT_PATH_2017 + fileName) for fileName in fileList])
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    fileList= os.listdir(IN_PATH_2018)
    pool.starmap(selectGoldenInterval, [(IN_PATH_2018 + fileName, RUN_INFO_FILE_2018, RUN_CERT_FILE_2018, OUT_PATH_2018 + fileName) for fileName in fileList])
    pool.close()
    pool.join()


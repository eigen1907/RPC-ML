import pandas as pd
import multiprocessing
import os


def separateOriginal2016(rpcImonFile, outputFile1, outputFile2):
    rpcImonData = pd.read_csv(rpcImonFile, low_memory=False)

    for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
        rpcImonData[colName] = pd.to_datetime(rpcImonData[colName])

    ### cutting testing data 2016-05-08 16:00:00 ~ 2016-05-08 20:00:00
    #rpcImonData = rpcImonData[(rpcImonData.Imon_change_date < pd.to_datetime("2016-05-08 16:00:00")) | (rpcImonData.Imon_change_date > pd.to_datetime("2016-05-08 20:00:00"))]

    ### Data 2016's has changing of apply voltage fomular in 2016-09-23   
    ### rpcImonData1 : Start ~ 2016-09-23 17:36:46
    rpcImonData1 = rpcImonData[rpcImonData.Imon_change_date < pd.to_datetime("2016-09-23 17:36:46")]
    ### rpcImonData2 : 2016-09-23 17:36:46 ~ End
    rpcImonData2 = rpcImonData[rpcImonData.Imon_change_date >= pd.to_datetime("2016-09-23 17:36:46")]

    if len(rpcImonData1) != 0:
        rpcImonData1.to_csv(outputFile1, index=False)
    if len(rpcImonData2) != 0:
        rpcImonData2.to_csv(outputFile2, index=False)



def separateOriginal2017(rpcImonFile, outputFile):
    rpcImonData = pd.read_csv(rpcImonFile, low_memory=False)

    for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
        rpcImonData[colName] = pd.to_datetime(rpcImonData[colName])

    ### cutting testing data 2017-09-26 11:00:00 ~ 16:00:00
    #rpcImonData = rpcImonData[(rpcImonData.Imon_change_date < pd.to_datetime("2017-09-26 11:00:00")) | (rpcImonData.Imon_change_date > pd.to_datetime("2017-09-26 16:00:00"))]

    if len(rpcImonData) != 0:
        rpcImonData.to_csv(outputFile, index=False)


def separateOriginal2018(rpcImonFile, outputFile1, outputFile2):
    rpcImonData = pd.read_csv(rpcImonFile, low_memory=False)

    for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
        rpcImonData[colName] = pd.to_datetime(rpcImonData[colName])

    ### cutting testing data 2018-06-07 06:00:00 ~ 2018-06-09 06:00:00
    #rpcImonData = rpcImonData[(rpcImonData.Imon_change_date < pd.to_datetime("2018-06-07 06:00:00")) | (rpcImonData.Imon_change_date > pd.to_datetime("2018-06-09 06:00:00"))]
    
    ### Data 2018 has voltage dropping section
    ### rpcImonData1: 2018-07-27 ~ 2018-08-19 19:00:00 (Dropping section)
    rpcImonData1 = rpcImonData[(rpcImonData.Imon_change_date > pd.to_datetime("2018-07-27")) & (rpcImonData.Imon_change_date < pd.to_datetime("2018-08-19 19:00:00"))]
    ### rpcImonData2: Start ~ 2018-07-27 | 2018-08-19 19:00:00 ~ End (Normal Section)
    rpcImonData2 = rpcImonData[(rpcImonData.Imon_change_date < pd.to_datetime("2018-07-27")) | (rpcImonData.Imon_change_date > pd.to_datetime("2018-08-19 19:00:00"))]
    
    if len(rpcImonData1) != 0:
        rpcImonData1.to_csv(outputFile1, index=False)
    if len(rpcImonData2) != 0:
        rpcImonData2.to_csv(outputFile2, index=False)
    

if __name__ == "__main__":
    IN_PATH_2016 = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original/2016/"
    IN_PATH_2017 = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original/2017/"
    IN_PATH_2018 = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original/2018/"


    OUT_PATH_2016_FORMER = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSeparate/2016_former/"
    OUT_PATH_2016_LATTER = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSeparate/2016_latter/"
    OUT_PATH_2017 = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSeparate/2017/"
    OUT_PATH_2018_DROPPING = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSeparate/2018_dropping/"
    OUT_PATH_2018_NORMAL = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginalSeparate/2018_normal/"


    pool = multiprocessing.Pool(100)
    m = multiprocessing.Manager()
    fileList = os.listdir(IN_PATH_2016)
    pool.starmap(separateOriginal2016, [(IN_PATH_2016 + fileName, OUT_PATH_2016_FORMER + fileName, OUT_PATH_2016_LATTER + fileName) for fileName in fileList])
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(100)
    m = multiprocessing.Manager()
    fileList = os.listdir(IN_PATH_2017)
    pool.starmap(separateOriginal2017, [(IN_PATH_2017 + fileName, OUT_PATH_2017 + fileName) for fileName in fileList])
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(100)
    m = multiprocessing.Manager()
    fileList = os.listdir(IN_PATH_2018)
    pool.starmap(separateOriginal2018, [(IN_PATH_2018 + fileName, OUT_PATH_2018_DROPPING + fileName, OUT_PATH_2018_NORMAL + fileName) for fileName in fileList])
    pool.close()
    pool.join()


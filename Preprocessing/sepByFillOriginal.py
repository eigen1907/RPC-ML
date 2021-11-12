import pandas as pd
import numpy as np
import os
import multiprocessing

 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)






if __name__ == "__main__":
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/Original/"

    rpc2016 = os.listdir(f"{rpcDataPath}2016/")
    rpc2017 = os.listdir(f"{rpcDataPath}2017/")
    rpc2018 = os.listdir(f"{rpcDataPath}2018/")



    for i in range(len(rpc2016)):
        rpc2016[i] = rpc2016[i][0:-9]

    for i in range(len(rpc2017)):
        rpc2017[i] = rpc2017[i][0:-9]

    for i in range(len(rpc2018)):
        rpc2018[i] = rpc2018[i][0:-9]


    sharedDpids = list(set(rpc2016) & set(rpc2017) & set(rpc2018))

    newDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/OriginByFill/"

    
    for sharedDpid in sharedDpids:
        createFolder(newDataPath + sharedDpid)
    
        
    rpcDataSepFolders = os.listdir(rpcDataPath)
    for rpcDataSepFolder in rpcDataSepFolders:
        rpcDataSepPath = rpcDataPath + rpcDataSepFolder
        for sharedDpid in sharedDpids:
            rpcData = pd.read_csv(rpcDataSepPath + "/" + sharedDpid + "_" + rpcDataSepFolder[0:4] + ".csv")
            fillNumberList = rpcData["fill_number"].unique()
            for fillNumber in fillNumberList:
                rpcData[rpcData["fill_number"] == fillNumber].to_csv(newDataPath + str(sharedDpid) + "/" + str(fillNumber) + ".csv", index=False)
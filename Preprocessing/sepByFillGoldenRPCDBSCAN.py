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
    rpcDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCSepDBSCAN/"

    rpc2016former = os.listdir(f"{rpcDataPath}2016_former/")
    rpc2016latter = os.listdir(f"{rpcDataPath}2016_latter/")
    rpc2017 = os.listdir(f"{rpcDataPath}2017/")
    rpc2018dropping = os.listdir(f"{rpcDataPath}2018_dropping/")
    rpc2018normal = os.listdir(f"{rpcDataPath}2018_normal/")


    for i in range(len(rpc2016former)):
        rpc2016former[i] = rpc2016former[i][0:-9]

    for i in range(len(rpc2016latter)):
        rpc2016latter[i] = rpc2016latter[i][0:-9]

    for i in range(len(rpc2017)):
        rpc2017[i] = rpc2017[i][0:-9]

    for i in range(len(rpc2018dropping)):
        rpc2018dropping[i] = rpc2018dropping[i][0:-9]

    for i in range(len(rpc2018normal)):
        rpc2018normal[i] = rpc2018normal[i][0:-9]

    sharedDpids = list(set(rpc2016former) & set(rpc2016latter) & set(rpc2017) & set(rpc2018dropping) & set(rpc2018normal))

    newDataPath = "/store/hep/users/eigen1907/CMS-RPC_store/DataStore/ImonPreproc/GoldenRPCDBSCANByFill/"

    
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


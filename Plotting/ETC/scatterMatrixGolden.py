import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing



def scatterMatrix(rpcImonPath, rpcImonFile, scatterMatrixPath):
    rpcImonData = pd.read_csv(rpcImonPath + rpcImonFile, low_memory=False)

    for colName in ['Imon_change_date','lumi_start_date','lumi_end_date','uxc_change_date']:
        rpcImonData[colName] = pd.to_datetime(rpcImonData[colName])
    
    rpcImonData = rpcImonData.drop(columns=['Imon_change_date', 'lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'fill_number', 'dew_point', 'relative_humodity'])
    
    sns.pairplot(
        rpcImonData,
        diag_kws=dict(fill=False),
    )

    plt.savefig(scatterMatrixPath + rpcImonFile[0:-4] + ".png")
    plt.close()

if __name__ == "__main__":
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPC/2016/"
    scatterMatrixPath = "/Users/mainroot/RPC_graph/ScatterMatrixGoldenRPC/2016/"
    rpcImonFiles = os.listdir(rpcImonPath)

    pool = multiprocessing.Pool(6)
    m = multiprocessing.Manager()
    pool.starmap(scatterMatrix, [(rpcImonPath, rpcImonFile, scatterMatrixPath) for rpcImonFile in rpcImonFiles])
    pool.close()
    pool.join()






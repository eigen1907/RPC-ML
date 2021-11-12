import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing
from itertools import compress



if __name__ == "__main__":
    ### 5개 구간에서 모두 존재하는 검출기 찾기 => SharedDpids    ## dpid_316 같은 형식으로 나옴
    rpc2016former = os.listdir("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_former/")
    rpc2016latter = os.listdir("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2016_latter/")
    rpc2017 = os.listdir("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2017/")
    rpc2018dropping = os.listdir("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2018_dropping/")
    rpc2018normal = os.listdir("/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/2018_normal/")
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
    ### 여기부터 그래프 그리기

    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/"
    plotPath = "/Users/mainroot/RPC_graph/ML_V1_partition_AllTerm/"


    for dpidName in sharedDpids:
        data2016Former = pd.read_csv(rpcImonPath + "2016_former/" + dpidName + "_2016.csv", low_memory=False)
        data2016Latter = pd.read_csv(rpcImonPath + "2016_latter/" + dpidName + "_2016.csv", low_memory=False)
        data2017 = pd.read_csv(rpcImonPath + "2017/" + dpidName + "_2017.csv", low_memory=False)
        data2018Normal = pd.read_csv(rpcImonPath + "2018_normal/" + dpidName + "_2018.csv", low_memory=False)
        data2018Dropping = pd.read_csv(rpcImonPath + "2018_dropping/" + dpidName + "_2018.csv", low_memory=False)

        rpcImonDataList = [data2016Former, data2016Latter, data2017, data2018Normal, data2018Dropping]

        dataInteg = pd.concat(rpcImonDataList, ignore_index=True)
        

        
        #fig, axes = plt.subplots(2, 4, figsize=(20, 8), tight_layout=True)
        
        rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
        rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        x1 = rpcImonData["inst_lumi"]
        x2 = rpcImonData["Vmon"]
        x3 = rpcImonData["temp"]
        x4 = rpcImonData["inst_lumi"] * np.exp(rpcImonData["Vmon"] / rpcImonData["press"])
        x5 = rpcImonData["relative_humodity"]
        x6 = rpcImonData["press"]
        x7 = (rpcImonData["Imon_change_date"] - rpcImonData["Imon_change_date"][0]).astype(int) / 10**9
        X = pd.concat([x1, x2, x3, x4, x5, x6, x7], ignore_index=True, axis=1)
        y = rpcImonData["Imon"]
        XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)
        lineFitter = LinearRegression()
        lineFitter.fit(XTrain, yTrain)
        yPredict = lineFitter.predict(XValid)
        yDiff = yValid - yPredict
        columnList = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        C0Part = np.full((len(rpcImonData),), lineFitter.intercept_)
        C1Part = lineFitter.coef_[0] * x1
        C2Part = lineFitter.coef_[1] * x2
        C3Part = lineFitter.coef_[2] * x3
        C4Part = lineFitter.coef_[3] * x4    
        C5Part = lineFitter.coef_[4] * x5
        C6Part = lineFitter.coef_[5] * x6
        C7Part = lineFitter.coef_[6] * x7
        partList = [C0Part, C1Part, C2Part, C3Part, C4Part, C5Part, C6Part, C7Part]
        for i in range(8):
            axes[i//4, i%4].plot(rpcImonData.Imon_change_date, partList[i], '.')
            axes[i//4, i%4].set_xlabel(f"C{i} * X{i}")


        plt.suptitle(f"After ML_V1, each partition of C_n * X_n, All Term, Dpid: {dpidName}")
        plt.savefig(plotPath + dpidName + ".png")
        plt.close()  
        
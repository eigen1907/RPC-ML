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
    plotPath = "/Users/mainroot/RPC_graph/ML_Mine/ML_Mine_partition_AllTerm/"


    for dpidName in sharedDpids:
        data2016Former = pd.read_csv(rpcImonPath + "2016_former/" + dpidName + "_2016.csv", low_memory=False)
        data2016Latter = pd.read_csv(rpcImonPath + "2016_latter/" + dpidName + "_2016.csv", low_memory=False)
        data2017 = pd.read_csv(rpcImonPath + "2017/" + dpidName + "_2017.csv", low_memory=False)
        data2018Normal = pd.read_csv(rpcImonPath + "2018_normal/" + dpidName + "_2018.csv", low_memory=False)
        data2018Dropping = pd.read_csv(rpcImonPath + "2018_dropping/" + dpidName + "_2018.csv", low_memory=False)

        rpcImonDataList = [data2016Former, data2016Latter, data2017, data2018Normal, data2018Dropping]

        fig, axes = plt.subplots(1, 3, figsize=(16, 8), tight_layout=True)

        for rpcImonData in rpcImonDataList:
            rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point'])
            rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"], format="%Y-%m-%d %H:%M:%S", errors="raise")

            x1 = rpcImonData["inst_lumi"]
            x7 = (rpcImonData["Imon_change_date"] - rpcImonData["Imon_change_date"][0]).astype(int) / 10**9

            X = pd.concat([x1, x7], ignore_index=True, axis=1)
            y = rpcImonData["Imon"]

            XTrain, XValid, yTrain, yValid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=34)

            lineFitter = LinearRegression()
            lineFitter.fit(XTrain, yTrain)

            yPredict = lineFitter.predict(XValid)

            yDiff = yValid - yPredict

            columnList = ["C0", "C1", "C7"]

            C0Part = np.full((len(rpcImonData),), lineFitter.intercept_)
            C1Part = lineFitter.coef_[0] * x1
            C7Part = lineFitter.coef_[1] * x7

            partList = [C0Part, C1Part, C7Part]
            
            axes[0].plot(rpcImonData.Imon_change_date, partList[0], '.')
            axes[0].set_xlabel("C0 * X0")

            axes[1].plot(rpcImonData.Imon_change_date, partList[1], '.')
            axes[1].set_xlabel("C1 * X1")

            axes[2].plot(rpcImonData.Imon_change_date, partList[2], '.')
            axes[2].set_xlabel("C7 * X7")


        plt.suptitle(f"After ML_3Variable, each partition of C_n * X_n, All Term, Dpid: {dpidName}")
        plt.savefig(plotPath + dpidName + ".png")
        plt.close()
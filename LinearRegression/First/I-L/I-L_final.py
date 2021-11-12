import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing


### Regression Condition: inst_lumi => All Data, intercept => True
def plotFunction1(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = rpcImonData.inst_lumi.values.reshape(-1, 1)
    y = rpcImonData.Imon
    
    lineFitter = LinearRegression()
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(x)
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()


### Regression Condition: inst_lumi => All Data, intercept => False
def plotFunction2(rpcImonPath, plotPath, dpidName):

    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = rpcImonData.inst_lumi.values.reshape(-1, 1)
    y = rpcImonData.Imon
    
    lineFitter = LinearRegression(fit_intercept=False)
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(x)
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()


### Regression Condition: inst_lumi => Top 50%, intercept => True
def plotFunction3(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = rpcImonData.inst_lumi.values[0:int(len(rpcImonData)*0.5)].reshape(-1, 1)
    y = rpcImonData.Imon[0:int(len(rpcImonData)*0.5)]
    
    lineFitter = LinearRegression()
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(rpcImonData.inst_lumi.values.reshape(-1, 1))
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()


### Regression Condition: inst_lumi => Top 50%, intercept => False
def plotFunction4(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = rpcImonData.inst_lumi.values[0:int(len(rpcImonData)*0.5)].reshape(-1, 1)
    y = rpcImonData.Imon[0:int(len(rpcImonData)*0.5)]
    
    lineFitter = LinearRegression(fit_intercept=False)
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(rpcImonData.inst_lumi.values.reshape(-1, 1))
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()    


### Regression Condition: inst_lumi => Top 25% Bottom 25%, intercept => True
def plotFunction5(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = pd.concat([rpcImonData.inst_lumi[0:int(len(rpcImonData)*0.25)], rpcImonData.inst_lumi[int(len(rpcImonData)*0.75):]], ignore_index=True).values.reshape(-1, 1)
    y = pd.concat([rpcImonData.Imon[0:int(len(rpcImonData)*0.25)], rpcImonData.Imon[int(len(rpcImonData)*0.75):]], ignore_index=True)
    
    lineFitter = LinearRegression()
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(rpcImonData.inst_lumi.values.reshape(-1, 1))
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()     


### Regression Condition: inst_lumi => Top 25% Bottom 25%, intercept => True
def plotFunction6(rpcImonPath, plotPath, dpidName):
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    rpcImonData = rpcImonData.drop(columns=['lumi_start_date', 'lumi_end_date', 'uxc_change_date', 'dew_point', 'relative_humodity'])
    rpcImonData["Imon_change_date"] = pd.to_datetime(rpcImonData["Imon_change_date"])
    rpcImonData = pd.read_csv(rpcImonPath + dpidName, low_memory=False)
    
    rpcImonData.Imon_change_date = pd.to_datetime(rpcImonData.Imon_change_date)
    
    x = pd.concat([rpcImonData.inst_lumi[0:int(len(rpcImonData)*0.25)], rpcImonData.inst_lumi[int(len(rpcImonData)*0.75):]], ignore_index=True).values.reshape(-1, 1)
    y = pd.concat([rpcImonData.Imon[0:int(len(rpcImonData)*0.25)], rpcImonData.Imon[int(len(rpcImonData)*0.75):]], ignore_index=True)
    
    lineFitter = LinearRegression(fit_intercept=False)
    lineFitter.fit(x, y)
    
    yPredict = lineFitter.predict(rpcImonData.inst_lumi.values.reshape(-1, 1))
    
    yDiff = rpcImonData.Imon - yPredict
    
    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax1.plot(rpcImonData.Imon_change_date, rpcImonData.Imon, '.', label="measured")
    ax1.plot(rpcImonData.Imon_change_date, yPredict, '.', label="predicted")
    ax1.set_xlabel("Predict VS Measured, axis-x: Imon_change_date, axis-y: Imon")
    ax1.legend()
    
    ax2 = plt.subplot(223)
    ax2.plot(rpcImonData.inst_lumi, rpcImonData.Imon, '.', label="measured")
    ax2.plot(rpcImonData.inst_lumi, yPredict, '.', label="predicted")
    ax2.set_xlabel("Predict VS Measured axis-x: inst_lumi, axis-y: Imon")
    ax2.legend()
    
    ax3 = plt.subplot(122)
    ax3.hist(yDiff, bins=100)
    ax3.set_xlabel(f"(Measured - Predict) Histogram(Imon), Bins=100")
    ax3.annotate(f"Mean: {np.mean(yDiff)} \n Mean(abs): {np.mean(np.abs(yDiff))}", xy=(1, 1), xycoords='axes fraction', fontsize=10, \
    horizontalalignment='right', verticalalignment='bottom')

    plt.suptitle(f"Chamber, Year: {dpidName[0:-4]} \nRegression Condition: \n(inst_lumi: All Data, Intercept: True)")
    plt.tight_layout()
    plt.savefig(plotPath + dpidName[0:-4] + ".png")
    plt.close()     


if __name__ == "__main__":
    rpcImonPath = "/Users/mainroot/RPC_modified_data/SecondaryArrangement/GoldenRPCSeparate/"
    plotPath = "/Users/mainroot/RPC_graph/I-L_fitting/InterceptFalseTop25Bottom25/"
    rpcImonSepFolders = os.listdir(rpcImonPath)
    rpcImonSepPaths, plotSepPaths = [], []
    for folder in rpcImonSepFolders:
        rpcImonSepPaths.append(rpcImonPath + folder + "/")
        plotSepPaths.append(plotPath + folder + "/")

    for i in range(len(rpcImonSepPaths)):
        dpidNames = os.listdir(rpcImonSepPaths[i])
        pool = multiprocessing.Pool(6)
        m = multiprocessing.Manager()
        pool.starmap(plotFunction6, [(rpcImonSepPaths[i], plotSepPaths[i], dpidName) for dpidName in dpidNames])
        pool.close()
        pool.join()

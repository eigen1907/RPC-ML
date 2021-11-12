import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image




GLM_ScoreData = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/GLM_RMS_score.csv", low_memory=False)
SLR_ScoreData = pd.read_csv("/store/hep/users/eigen1907/CMS-RPC_store/etc/SLR_RMS_score.csv", low_memory=False)



while True:
    i = int(input(f"Which RMS score graph do you want to see? (0 ~ {len(GLM_ScoreData)}): "))
    if i < 0 or i > len(GLM_ScoreData) - 1: print("You entered wrong number")
    plotPath = "/store/hep/users/eigen1907/CMS-RPC_store/PlotStore/ML_V1/ML_V1_AfterDBSCAN/"
    targetStr = GLM_ScoreData["DataPath"][i] + "/" + GLM_ScoreData["DpidName"][i] + ".png"
    plotPath += targetStr
    f = Image.open(plotPath).show()
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 

from torchvision import transforms

import numpy as np
import pandas as pd

trn = pd.read_csv("./dnn_data/trn.tsv", sep='\t')
val = pd.read_csv("./dnn_data/val.tsv", sep='\t')

X_features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6", "feature_7", "feature_8"]
y_feature = ["y"]

trn_X_pd, trn_y_pd = trn[X_features], trn[y_feature]
val_X_pd, val_y_pd = val[X_features], val[y_feature]

trn_X = torch.from_numpy(trn_X_pd.astype(float).as_matrix())
trn_y = torch.from_numpy(trn_y_pd.astype(float).as_matrix())

val_X = torch.from_numpy(val_X_pd.astype(float).as_matrix())
val_y = torch.from_numpy(val_y_pd.astype(float).as_matrix())


trn = data_utils.TensorDataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size = 64, shuffle=True)

val = data_utils.TensorDataset(val_X, val_y)
val_loader = data_utils.DataLoader(val, batch_size = 64, shuffle=False)

class MLPRegressor(nn.Module):
    
    def __init__(self, X_features):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(len(X_features), 50)
        h2 = nn.Linear(50, 35)
        h3 = nn.Linear(35, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.Tanh(),
            h2,
            nn.Tanh(),
            h3,
        )
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
        
    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)
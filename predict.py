from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd

df = pd.read_csv('result_data.csv', index_col = 0)
# get the dataset
def get_dataset():
    X = df[["LAeq", "LA5-95","Loudness", "Green", "Sky", "Grey", ]]
    return torch.FloatTensor(X.values)

class MultOutRegressor(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim=32,seed=1234):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.target_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

X = get_dataset()
model = MultOutRegressor(6, 5)
model.load_state_dict(torch.load('./model_state_dict.pt'))
res = model(X[0])
sum_res = res[0]+res[1]+res[2]+res[3]+res[4]+1.568
m = nn.Sigmoid()
print(m(sum_res).item())
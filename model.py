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
def update(input , target , model, criterion , optimizer,max_norm=5) :
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output , target.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return loss 

def one_epoch(dataloader , model, criterion , optimizer ) :
    result = torch.FloatTensor([0])
    for idx , (input , target) in enumerate(dataloader) :
        loss = update(input , target , model, criterion , optimizer)
        result = torch.add(result , loss)
    else :
        result /= idx+1
        return result.detach().cpu().numpy()

def visualize(result) :
    display.clear_output(wait=True)
    plt.plot(result)
    plt.show()

def train(n_epochs , dataloader , model, criterion , optimizer , log_interval=10) :
    epoch_loss = []
    for epoch in range(n_epochs) :
        loss = one_epoch(dataloader , model, criterion , optimizer )
        if epoch > 0 :
            epoch_loss.append(loss)
        if epoch % log_interval == 0 :
            # visualize(epoch_loss)
            print(epoch)
    else :
        return np.min(epoch_loss)

df = pd.read_csv('result_data.csv', index_col = 0)
# get the dataset
def get_dataset():
    X = df[["LAeq", "LA5-95","Loudness", "Green", "Sky", "Grey", ]]
    y = df[["Fascination","Being_away_from","Negative", "Valence", "Arousal"]]
    #y = df[["Fascination"]]

    return X.values, y.values
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data, self.y_data = get_dataset()

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

class TabularDataSet(Dataset) :
    def __init__(self, X , Y) :
        self._X = np.float32(X)
        self._Y = Y

    def __len__(self,) :
        return len(self._Y)

    def __getitem__(self,idx) :
        return self._X[idx], self._Y[idx]

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

class MultOutChainedRegressor(nn.Module):
    def __init__(self, input_dim, target_dim,order, hidden_dim=32,seed=1234 ):
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self._order = order 
        assert len(self._order) == self.target_dim
        assert min(self._order) == 0 

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_seq = []
        self.nested_dim = self.hidden_dim
        self.output_rank = {}
        for idx , order in enumerate(self._order) : 
            self.output_seq.append(nn.Linear(self.nested_dim, 1))
            self.nested_dim += 1 
            self.output_rank[idx] = order 
        else :
            self.linears = nn.ModuleList(self.output_seq)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        _x = x
        last_vector = torch.zeros(size=(x.size()[0],self.target_dim))
        for idx , rank in self.output_rank.items() :
            y_hat = self.output_seq[idx](_x)
            last_vector[:,[rank]] += y_hat
            _x = torch.cat([_x,y_hat],axis=1)
        return last_vector
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

dataset = CustomDataset()
train_dl = DataLoader(dataset, batch_size=2, shuffle=True)

scaler=  StandardScaler()

#model = MultOutChainedRegressor(6 , 5 , order=[5,1,0])
model = MultOutRegressor(6 , 5)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
train(2000, train_dl ,model, criterion , optimizer,log_interval=50)
torch.save(model.state_dict(), "./model_state_dict.pt")
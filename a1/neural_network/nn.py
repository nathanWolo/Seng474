import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append("..")
from utils import read_data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

EPOCHS = 20
train, test = read_data.partition_data(read_data.read_data())

scaler = StandardScaler()

train, test = scaler.fit_transform(train), scaler.transform(test)

x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

for i in range(len(y_train)):
    if y_train[i] < 0:
        y_train[i] = 0
    else:
        y_train[i] = 1

for i in range(len(y_test)):
    if y_test[i] < 0:
        y_test[i] = 0
    else:
        y_test[i] = 1

## train data
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
## test data    
class TestData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

test_data = TestData(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
train_data = TrainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))


train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=10, drop_last=True)


''' 

Citation: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
provides the general structure for the definition of the BinaryClassification class

'''


class BinaryClassification(nn.Module):
    def __init__(self, hdim=64):
        super(BinaryClassification, self).__init__()
        #input dim is 104 features
        self.layer_1 = nn.Linear(104, hdim) 
        self.layer_2 = nn.Linear(hdim, hdim)
        self.layer_out = nn.Linear(hdim, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hdim)
        self.batchnorm2 = nn.BatchNorm1d(hdim)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BinaryClassification(hdim=128)
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
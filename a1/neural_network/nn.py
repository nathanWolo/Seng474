import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append("..")
from utils import read_data
import itertools
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

''' 

Citation: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
provides the general structure for the definition of the BinaryClassification class

'''


class BinaryClassification(nn.Module):
    def __init__(self, hdim=64, dropout=0.1):
        super(BinaryClassification, self).__init__()
        #input dim is 104 features
        self.layer_1 = nn.Linear(104, hdim) 
        self.layer_2 = nn.Linear(hdim, hdim)
        self.layer_out = nn.Linear(hdim, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
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


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def train_model(hdim=64, lr=1e-4, epochs=10, batch_size=10, dropout=0.1):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, drop_last=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BinaryClassification(hdim=hdim, dropout=dropout)
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for e in range(1, epochs+1):
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
        

        #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    final_train_acc = epoch_acc/len(train_loader)
    income_classification_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            income_test_pred = model(X_batch)
            income_test_pred = torch.sigmoid(income_test_pred)
            income_pred_tag = torch.round(income_test_pred)
            income_classification_list.append(income_pred_tag.cpu().numpy())

    income_classification_list = [a.tolist() for a in income_classification_list]
    income_classification_list = list(itertools.chain(*income_classification_list))
    #print(income_classification_list)
    #print(classification_report(y_test[:-6], income_classification_list))
    print(len(y_test))
    print(batch_size)
    print(len(y_test[:-(len(y_test) % batch_size)]))
    if len(y_test[:-(len(y_test) % batch_size)]) != 0:
        final_test_acc = binary_acc(torch.FloatTensor(income_classification_list), torch.FloatTensor(y_test[:-(len(y_test) % batch_size)])).item()/len(y_test[:-(len(y_test) % batch_size)])
    else: 
        final_test_acc = binary_acc(torch.FloatTensor(income_classification_list), torch.FloatTensor(y_test)).item()/len(y_test)
    print("Final train accuracy: ", final_train_acc)
    print("Final test accuracy: ", final_test_acc)
    return final_train_acc, final_test_acc
#train_model(epochs=10, hdim=128, batch_size=20)


'''function to test and plot various batch sizes, with default parameters otherwise'''

def batch_size_test():
    batch_size_list = [2,3,4,5,7,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 800]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_batch_size = 0
    for batch_size in batch_size_list:
        final_train_acc, final_test_acc = train_model(epochs=2, batch_size=batch_size)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_batch_size = batch_size
    plt.plot(batch_size_list, train_acc_list, label="train accuracy")
    plt.plot(batch_size_list, test_acc_list, label="test accuracy")
    plt.plot(best_batch_size, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at batch size: " + str(best_batch_size))
    plt.xlabel("batch size")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_batch_size_test.png")
    #plt.show()
#batch_size_test()

'''Function to test performance on varying number of epochs
 '''

def epoch_test():
    epoch_list = [1,2,3,4,5,6,7,8,9,10]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_epoch = 0
    for epoch in epoch_list:
        final_train_acc, final_test_acc = train_model(epochs=epoch, batch_size=4)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_epoch = epoch
    plt.plot(epoch_list, train_acc_list, label="train accuracy")
    plt.plot(epoch_list, test_acc_list, label="test accuracy")
    plt.plot(best_epoch, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at epoch: " + str(best_epoch))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_epoch_test.png")
    #plt.show()


'''Function to test performance on varying number of hidden dimensions'''

def hdim_test():
    hdim_list = [2,4,8,16,32,64,128,256,512,1024]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_hdim = 0
    for hdim in hdim_list:
        final_train_acc, final_test_acc = train_model(epochs=1, hdim=hdim, batch_size=4)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_hdim = hdim
    plt.plot(hdim_list, train_acc_list, label="train accuracy")
    plt.plot(hdim_list, test_acc_list, label="test accuracy")
    plt.plot(best_hdim, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at hidden dimension: " + str(best_hdim))
    plt.xlabel("hidden dimension")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_hdim_test.png")
    #plt.show()


'''function to test and plot various learning rates, with default parameters otherwise'''

def lr_test():
    lr_list = [0.0001, 0.001, 0.01, 0.1, 1]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_lr = 0
    for lr in lr_list:
        final_train_acc, final_test_acc = train_model(epochs=2, lr=lr, batch_size=4)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_lr = lr
    plt.plot(lr_list, train_acc_list, label="train accuracy")
    plt.plot(lr_list, test_acc_list, label="test accuracy")
    plt.plot(best_lr, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at learning rate: " + str(best_lr))
    plt.xlabel("learning rate")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_lr_test.png")
    #plt.show()

'''function to test and plot various activation functions, with default parameters otherwise'''

def activation_test():
    activation_list = ["relu", "sigmoid", "tanh"]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_activation = 0
    for activation in activation_list:
        final_train_acc, final_test_acc = train_model(epochs=2, activation=activation, batch_size=4)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_activation = activation
    plt.plot(activation_list, train_acc_list, label="train accuracy")
    plt.plot(activation_list, test_acc_list, label="test accuracy")
    plt.plot(best_activation, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at activation: " + str(best_activation))
    plt.xlabel("activation")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_activation_test.png")
    #plt.show()

'''function to test different dropout rates, with default parameters otherwise'''

def dropout_test():
    dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_dropout = 0
    for dropout in dropout_list:
        final_train_acc, final_test_acc = train_model(epochs=2, dropout=dropout, batch_size=4)
        train_acc_list.append(final_train_acc)
        test_acc_list.append(final_test_acc)
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_dropout = dropout
    plt.plot(dropout_list, train_acc_list, label="train accuracy")
    plt.plot(dropout_list, test_acc_list, label="test accuracy")
    plt.plot(best_dropout, best_test_acc, 'ro', label="best test accuracy: " + 
            "{:.4f}".format(best_test_acc) + " at dropout: " + str(best_dropout))
    plt.xlabel("dropout")
    plt.ylabel("accuracy")
    plt.legend(loc="best")
    plt.savefig("nn_dropout_test.png")
    #plt.show()

#dropout_test()
#lr_test()
#hdim_test()
#epoch_test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 21:02:59 2020

@author: williamstanford
"""

import numpy as np
import os 
import torch
from src.models import tGAE as tgae
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

path = '/Users/williamstanford/CS790-TopoAE_Classification/GAE_experiments/topoGAE_results_d2/Citeseer/'

class Net(nn.Module):
    def __init__(self, latent_input_size, num_classes):
        super(Net, self).__init__()
        
        self.classifier = nn.Sequential(
                        nn.Linear(latent_input_size, 250),
                        nn.Linear(250, 100),
                        nn.ReLU(True),
                        nn.BatchNorm1d(100),
                        nn.Linear(100, num_classes),
                        nn.Dropout(0.2),
                        nn.LogSoftmax()
                        )
        
    def forward(self, x):
        return self.classifier(x)

ds = 'Citeseer'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds, ds, "public", T.NormalizeFeatures())
model_path = path+'model_state_dict.pth' 
data = dataset[0]
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev) 
model = tgae.TopoEncoder(dataset.num_features, 2)
model.load_state_dict(torch.load(model_path))
z = model(x, train_pos_edge_index)
labels = data.y

latent = z.clone().detach().numpy()
labels = labels.clone().detach().numpy()

#net = Net(nn.Parameter(model.conv3.weight).shape[1], dataset.num_classes)
net = Net(dataset.num_features, dataset.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%

dataset.num_features
#%%
full_data  = data.x.clone().detach().numpy()
#%%
train_x, val_x, train_y, val_y = train_test_split(full_data, labels, test_size = 0.2)

#%%
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
net.to(device)

def full_gd(net, criterion, optimizer, X_train, y_train, n_epochs=2000):
  train_losses = np.zeros(n_epochs)
  test_losses = np.zeros(n_epochs)

  for it in range(n_epochs): 
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_test = net(test_X)
    loss_test = criterion(outputs_test, test_Y)

    train_losses[it] = loss.item()
    test_losses[it] = loss_test.item()

    if (it + 1) % 50 == 0:
      print(f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

  return train_losses, test_losses


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)

train_X = torch.from_numpy(train_x)
test_X = torch.from_numpy(val_x)
train_Y = torch.from_numpy(train_y)
test_Y = torch.from_numpy(val_y)

train_losses, test_losses = full_gd(net, criterion, optimizer, train_X, train_Y)

#%%
plt.plot(train_losses, label = 'train loss')
plt.plot(test_losses, label = 'test loss')
plt.legend()
plt.show()

#%%
def test_classification(net):
    correct = 0
    total = 0    
    with torch.no_grad():
        net_out = net(test_X)
        
        for i in range(test_X.shape[0]):
            
            real_class = test_Y[i]
            predicted_class = torch.argmax(net_out[i,:])
            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy:", round(correct/total,3))
    
test_classification(net)
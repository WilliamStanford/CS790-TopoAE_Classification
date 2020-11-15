#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:04:33 2020

@author: williamstanford
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions import Normal

import os

#%%
os.chdir('/Users/williamstanford/CS790-TopoAE_Classification/topologicalautoencoders')

from src.models.approx_based import TopologicallyRegularizedAutoencoder
#from src.models.submodules import DeepAE_pshare_deL1

#%%
mean_channels = (0.131,)
std_channels = (0.308,)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the training and test datasets
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms)

#Prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)

   
#%%      
model = TopologicallyRegularizedAutoencoder(autoencoder_model='DeepAE')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
model.to(device)

reconstruction_loss = nn.MSELoss()
label_loss = nn.CrossEntropyLoss()

#%%
n_epochs = 3

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data in train_loader:
        images, _ = data
        images = images.to(device)
        
        optimizer.zero_grad()
        reconstruction, predicted_label = model(images)
        
        loss_re = reconstruction_loss(images, reconstruction)
        loss_re.backward()
                
        optimizer.step()
        train_loss += loss_re.item()*images.size(0)
          
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


#%%
parameter_sharing = True

if parameter_sharing:
    model.param_share()

for param in model.encoder.parameters():
    param.requires_grad = False
    
for param in model.decoder.parameters():
    param.requires_grad = False

i = 0
for param in model.classifier.parameters():
    if i == 0:
        param.requires_grad = False    
    i = i + 1
    
    
n_epochs = 5

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data in train_loader:
        images, label = data
        images = images.to(device)
        
        optimizer.zero_grad()
        reconstruction, predicted_label = model(images)
        
        loss_lp = label_loss(predicted_label, label)
        loss_lp.backward()
        
        optimizer.step()
        train_loss += loss_lp.item()*images.size(0)
          
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    

#%%

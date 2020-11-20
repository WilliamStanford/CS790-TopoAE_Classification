#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 01:14:42 2020

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

path = '/Users/williamstanford/CS790-TopoAE_Classification/GAE_experiments/GAE_results_d2/Cora/'


ds = 'Cora'
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

#%%
def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.forward(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

test(data.val_pos_edge_index, data.val_neg_edge_index)

#%%
data
#%%
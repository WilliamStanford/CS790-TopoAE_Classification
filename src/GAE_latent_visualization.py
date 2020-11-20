#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:07:07 2020

@author: williamstanford
"""
import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges

import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

from src.models import tGAE as tgae

#%%
def plot_latents(ax, path, ds, channels, model_name):
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid("/tmp/"+ds, ds, "public", T.NormalizeFeatures())
    model_path = path+'model_state_dict.pth'    
    model = tgae.TopoEncoder(dataset.num_features, channels)
    model.load_state_dict(torch.load(model_path))
    data = dataset[0]
    labels = data.y
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)
    x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)    
    z = model(x, train_pos_edge_index)
    
    color_list = ['lightblue', 'lightgreen', 'tomato', 
              'plum', 'teal', 'sandybrown', 'forestgreen']
    
    colors = [color_list[y] for y in labels]    
    latent = z.cpu().detach().numpy()
    
    ax.scatter(latent[:,0], latent[:,1], color=colors, s=2)        
   # else:
#    xs, ys = zip(*TSNE(random_state=42).fit_transform(latent))
#    ax.scatter(xs, ys, color=colors, s=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.text(.1,.1, model_name,  horizontalalignment='center')

#%%

def g_paths(ds_name, i):
    
    results_path = '/Users/williamstanford/CS790-TopoAE_Classification/GAE_experiments/'
    save_path_topo_d2 = results_path + 'topoGAE_results_d2/' + ds_name + '/'
    save_path_topo = results_path + 'topoGAE_results/' + ds_name + '/'
    save_path_vanilla_d2 = results_path + 'GAE_results_d2/' + ds_name + '/'
    save_path_vanilla = results_path + 'GAE_results/' + ds_name + '/'

    save_paths = [save_path_topo_d2, save_path_topo, save_path_vanilla_d2, save_path_vanilla]

    return save_paths[i]

hidden_channels = [2, 16]
        
#%%
fig = plt.figure(figsize=(4,8))
plt.subplots_adjust(wspace=0, hspace=0) 
plt.tight_layout()

ax1 = fig.add_subplot(4,2, 1)
ax2 = fig.add_subplot(4,2, 2)
ax3 = fig.add_subplot(4,2, 3)
ax4 = fig.add_subplot(4,2, 4)

ax5 = fig.add_subplot(4,2, 5)
ax6 = fig.add_subplot(4,2, 6)
ax7 = fig.add_subplot(4,2, 7)
ax8 = fig.add_subplot(4,2, 8)

datasets =['Cora', 'Citeseer', 'PubMed']


plot_latents(ax1, g_paths('Cora', 3), datasets[0], hidden_channels[1], 'GAE 16-d')
plot_latents(ax2, g_paths('Citeseer', 3), datasets[1], hidden_channels[1], 'GAE 16-d')


plot_latents(ax3, g_paths('Cora', 1), datasets[0], hidden_channels[1], 'TopoGAE 16-d')
plot_latents(ax4, g_paths('Citeseer', 1), datasets[1], hidden_channels[1], 'TopoGAE 16-d')

plot_latents(ax5, g_paths('Cora', 2), datasets[0], hidden_channels[0], 'GAE 16-d')
plot_latents(ax6, g_paths('Citeseer', 2), datasets[1], hidden_channels[0], 'GAE 16-d')


plot_latents(ax7, g_paths('Cora', 0), datasets[0], hidden_channels[0], 'TopoGAE 16-d')
plot_latents(ax8, g_paths('Citeseer', 0), datasets[1], hidden_channels[0], 'TopoGAE 16-d')

ax1.set_ylabel('GAE')
ax3.set_ylabel('TopoGAE')
ax1.set_xlabel('Cora')
ax1.xaxis.set_label_position('top') 
ax2.set_xlabel('Citeseer')
ax2.xaxis.set_label_position('top') 
ax5.set_ylabel('GAE-2d')
ax7.set_ylabel('TopoGAE-2d')
#%%
fig = plt.figure(figsize=(4,4))
plt.subplots_adjust(wspace=0, hspace=0) 
plt.tight_layout()

ax1 = fig.add_subplot(2,2, 1)
ax2 = fig.add_subplot(2,2, 2)

ax3 = fig.add_subplot(2,2, 3)
ax4 = fig.add_subplot(2,2, 4)


datasets =['Cora', 'Citeseer', 'PubMed']


plot_latents(ax1, g_paths('Cora', 2), datasets[0], hidden_channels[0], 'GAE 16-d')
plot_latents(ax2, g_paths('Citeseer', 2), datasets[1], hidden_channels[0], 'GAE 16-d')
#plot_latents(ax3, g_paths('PubMed', 3), datasets[2], hidden_channels[1], 'GAE 16-d')


plot_latents(ax3, g_paths('Cora', 0), datasets[0], hidden_channels[0], 'TopoGAE 16-d')
plot_latents(ax4, g_paths('Citeseer', 0), datasets[1], hidden_channels[0], 'TopoGAE 16-d')
#plot_latents(ax12, g_paths('PubMed', 1), datasets[2], hidden_channels[1], 'TopoGAE 16-d')

ax1.set_ylabel('GAE-2d')
ax3.set_ylabel('TopoGAE-2d')

 
    
    
    
    
    
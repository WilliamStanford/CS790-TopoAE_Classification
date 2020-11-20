#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:34:36 2020

@author: williamstanford
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.utils import train_test_split_edges

import time
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from sklearn.metrics import roc_auc_score, average_precision_score

os.chdir('/Users/williamstanford/CS790-TopoAE_Classification/')

from src import topology
import sys
sys.setrecursionlimit(4000)

EPS = 1e-15
MAX_LOGSTD = 10

#%%
class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges='symmetric'):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = topology.PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components
    
#%%
class TopoEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, lam=.1):
        super(TopoEncoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 64, cached=True)
        self.conv2 = pyg_nn.GCNConv(64), 32, cached=True)
        self.conv3 = pyg_nn.GCNConv(32, out_channels, cached=True)

        self.latent_norm = torch.nn.Parameter(data=torch.ones(1),
                                      requires_grad=True)
        
        self.topo_sig = TopologicalSignatureDistance()
        self.decoder = InnerProductDecoder()
        self.lam = lam

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss, {'reconstruction_error': pos_loss + neg_loss}
    
        
    @staticmethod
    def _compute_distance(x):
        x_flat = x.view(x.size(0), -1)
        diff = np.zeros((x_flat.size(0), x_flat.size(0)))
        
        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                if i < j:
                
                    cur_distance = torch.dist(x_flat[i,:], x_flat[j,:])
                    diff[i,j] = cur_distance
                    diff[j,i] = cur_distance
                

                    
        return  torch.from_numpy(diff)
           
        
    def rec_and_top_loss(self, x, latent, edge_index, x_distances):

        # Precomputed x_distance for graph the remains constant every epoch
        # x_distances = self._compute_distance_alternative2(x.clone())     
        dimensions = x.size()
        x_distances = x_distances // x_distances.max()
        
        latent_distances = self._compute_distance(latent.clone().detach())
        latent_distances = latent_distances / self.latent_norm
        
        # Use reconstruction loss of autoencoder
        ae_rec_loss, ae_loss_comp = self.recon_loss(latent, edge_index)        
        
        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)        

        # normalize topo_error according to batch_size
        batch_size = dimensions[0]
        topo_error = topo_error / float(batch_size) 
        loss = ae_rec_loss + self.lam * topo_error
        loss_components = {
            'loss.autoencoder': ae_rec_loss,
            'loss.topo_error': topo_error
        }
        loss_components.update(topo_error_components)
        loss_components.update(ae_loss_comp)
        return (
            loss,
            loss_components
        )
       

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        latent = self.conv3(x, edge_index)

        return latent


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    
#%%
def train(epoch, distance_x, stats, latent_topology, top_loss=True):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index)
    ae_loss, ae_loss_components = model.recon_loss(z, train_pos_edge_index)
    
    if top_loss:
        loss, loss_components = model.rec_and_top_loss(x, z, train_pos_edge_index,
                                                        distance_x)                 
    else:
        loss = ae_loss
        stats['ae_loss'].append(loss.item())
            
    loss.backward()
    optimizer.step()
    stats['epoch_loss'].append(float(loss.item())) 
    latent_topology['output'].append(z.clone().detach().numpy())
    print('total_loss')
    print(loss.item())
    print('top_loss')
    print(loss_components['loss.topo_error'].item())
    if top_loss:
        stats['ae_loss'].append(float(loss_components['loss.autoencoder'].item()))
        stats['top_loss'].append(float(loss_components['loss.topo_error'].item()))
    
def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.forward(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


#%%
def build_train_model(ds_name, hidden_channels, distance_x, top_loss=True):    
    start_time = time.time()
    
    stats = {
            'epoch': [],
            'epoch_time': [],
            'top_loss': [],
            'ae_loss': [],
            'epoch_loss': [],
            'total_time': [],
            'AUC': [],
            'AP': []
            } 
    
    latent_topology = {
            'output':[]
            }
                      
    for epoch in range(1, 100):
        epoch_start = time.time()
        train(epoch, distance_x, stats, latent_topology, top_loss)
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        if epoch % 5 == 0:
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
            print(np.round((time.time() - start_time), 4))
            
        stats['epoch'].append(epoch)
        stats['epoch_time'].append(np.round((time.time() - epoch_start), 4))
        stats['AUC'].append(auc)
        stats['AP'].append(ap)
        
    stats['total_time'].append(np.round((time.time() - start_time), 4))
    
  
    results_path = '/Users/williamstanford/CS790-TopoAE_Classification/GAE_experiments/'

    if top_loss:
        if hidden_channels == 2:           
            save_path = results_path + '/topoGAE_results_d2/' + ds_name + '/'
            torch.save(model.state_dict(), save_path+'model_state_dict.pth')
            np.save(save_path+'run_stats.npy', stats)
            np.save(save_path+'latent_topology.npy', latent_topology) 
        else:
            save_path = results_path + '/topoGAE_results/' + ds_name + '/'
            torch.save(model.state_dict(), save_path+'model_state_dict.pth')
            np.save(save_path+'run_stats.npy', stats)    
            np.save(save_path+'latent_topology.npy', latent_topology) 
    else:
        if hidden_channels == 2:
            save_path = results_path + 'GAE_results_d2/' + ds_name + '/'
            torch.save(model.state_dict(), save_path+'model_state_dict.pth')
            np.save(save_path+'run_stats.npy', stats)
            np.save(save_path+'latent_topology.npy', latent_topology) 
        
        else:
            save_path = results_path + 'GAE_results/' + ds_name + '/'
            torch.save(model.state_dict(), save_path+'model_state_dict.pth')
            np.save(save_path+'run_stats.npy', stats)
            np.save(save_path+'latent_topology.npy', latent_topology)

#%%
distance_x = model._compute_distance(x.clone())
  
#%%
ds_name='Cora'
hidden_channels = 2
t_loss = True

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds_name, ds_name, "public", T.NormalizeFeatures())

model = TopoEncoder(dataset.num_features, hidden_channels).to(dev)

data = dataset[0]
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
#distance_x = model._compute_distance(x.clone())
  
build_train_model(ds_name, hidden_channels, distance_x, t_loss) 


#%%
ds_name='Pubmed'
hidden_channels = 16
t_loss = True

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds_name, ds_name, "public", T.NormalizeFeatures())

model = TopoEncoder(dataset.num_features, hidden_channels).to(dev)

data = dataset[0]
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
#distance_x = model._compute_distance(x.clone())
  
build_train_model(ds_name, hidden_channels, distance_x, t_loss)  

#%%
#%%
#%%
#%%
ds_name='Citeseer'
hidden_channels = 2
t_loss = False

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds_name, ds_name, "public", T.NormalizeFeatures())

model = TopoEncoder(dataset.num_features, hidden_channels).to(dev)

data = dataset[0]
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
#distance_x = model._compute_distance(x.clone())
  
build_train_model(ds_name, hidden_channels, distance_x, t_loss)

#%%
ds_name='Cora'
hidden_channels = 2
t_loss = False

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds_name, ds_name, "public", T.NormalizeFeatures())

model = TopoEncoder(dataset.num_features, hidden_channels).to(dev)

data = dataset[0]
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
#distance_x = model._compute_distance(x.clone())
  
build_train_model(ds_name, hidden_channels, distance_x, t_loss)  
 
#%%
dataset.num_features
    
#%%
ds_name='PubMed'
hidden_channels = 2
t_loss = False

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid("/tmp/"+ds_name, ds_name, "public", T.NormalizeFeatures())

model = TopoEncoder(dataset.num_features, hidden_channels).to(dev)

data = dataset[0]
labels = data.y
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
#distance_x = model._compute_distance(x.clone())
  
build_train_model(ds_name, hidden_channels, distance_x, t_loss)  













    #%%
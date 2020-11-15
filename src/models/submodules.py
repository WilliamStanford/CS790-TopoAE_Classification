#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:29:05 2020

@author: williamstanford
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from .base import AutoencoderModel

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class DeepAE(AutoencoderModel):
    """1000-500-250-2-250-500-1000."""
    def __init__(self, num_classes, param_share=True, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.param_share = param_share
        
        n_input_dims = np.prod(input_dims)
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)),
            nn.Linear(n_input_dims, 1000),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.ReLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 250),
            nn.ReLU(True),
            nn.BatchNorm1d(250),
            nn.Linear(250, 500),
            nn.ReLU(True),
            nn.BatchNorm1d(500),
            nn.Linear(500, 1000),
            nn.ReLU(True),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, n_input_dims),
            View((-1,) + tuple(input_dims)),
            nn.Tanh()
        )
        
        self.reconst_error = nn.MSELoss()
        self.pred_error = nn.CrossEntropyLoss()

           
        if param_share == False:
            self.classifier = nn.Sequential(
                nn.Linear(2, self.num_classes),
                nn.Dropout(0.2),
                nn.LogSoftmax()
                )
            
        if param_share == True:
            self.classifier = nn.Sequential(
                nn.Linear(2, 250),
                nn.Linear(250, 100),
                nn.ReLU(True),
                nn.BatchNorm1d(100),
                nn.Linear(100, self.num_classes),
                nn.Dropout(0.2),
                nn.LogSoftmax()
                )
            
    def param_share(self):
        self.classifier[0].weight = nn.Parameter(self.encoder[-1].weight.transpose(0,1))

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        label_predictions = self.classifier(latent)          
        label_prediction_loss = self.pred_error(label_predictions)
            
        return reconst_error, label_prediction_loss, {'reconstruction_error': reconst_error}
        
    
class LinearAE(AutoencoderModel):
    """input dim - 2 - input dim."""
    def __init__(self, num_classes, param_share=True, input_dims=(1, 28, 28)):
        super().__init__()
        self.input_dims = input_dims
        n_input_dims = np.prod(input_dims)
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            View((-1, n_input_dims)),
            nn.Linear(n_input_dims, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, n_input_dims),
            View((-1,) + tuple(input_dims)),
        )
       
        self.reconst_error = nn.MSELoss()
        self.pred_error = nn.CrossEntropyLoss()
        
        if param_share == False:
            self.classifier = nn.Sequential(
                nn.Linear(2, self.num_classes),
                nn.Dropout(0.2),
                nn.LogSoftmax()
                )
            
        if param_share == True:
            self.classifier = nn.Sequential(
                nn.Linear(2, 250),
                nn.Linear(250, 100),
                nn.ReLU(True),
                nn.BatchNorm1d(100),
                nn.Linear(100, self.num_classes),
                nn.Dropout(0.2),
                nn.LogSoftmax()
                )
            
    def param_share(self):
        self.classifier[0].weight = nn.Parameter(self.encoder[-1].weight.transpose(0,1))

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        label_predictions = self.classifier(latent)          
        label_prediction_loss = self.pred_error(label_predictions)
            
        return reconst_error, label_prediction_loss, {'reconstruction_error': reconst_error}
    

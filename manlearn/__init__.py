#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:22:49 2023

@author: fmry
"""

#%% Sources

#%% Modules

from .VAE_SVHN import VAE_SVHN
from .VAE_DTU import VAE_DTU
from .VAE_CelebA import VAE_CELEBA
from .VAE_MNIST import VAE_MNIST
from .VAE_3D import VAE_3D
from .data_loader import load_celeba, load_svhn, load_mnist, load_dtu, load_surface
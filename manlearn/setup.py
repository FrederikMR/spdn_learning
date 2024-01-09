#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:19:54 2023

@author: fmry
"""

#%% Sources

#%% Modules

#PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import grad

#Scipy
from scipy.integrate import solve_bvp, solve_ivp

#Sympy
import sympy as sym

#numpy
import numpy as np

#pandas
import pandas as pd

#typing
from typing import List, Any
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:43:10 2023

@author: fmry
"""

#%% Sources

#%% Modules

from .setup import *

#%% Load CelebA

def load_celeba(path:str, 
                batch_size:int=100,
                img_size:str=64, 
                frac:float=.8,
                workers:int=0):

    dataset = dset.ImageFolder(root=path,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataset_subset = torch.utils.data.Subset(dataset, range(int(len(dataset)*frac)))
    
    trainloader = DataLoader(dataset = dataset_subset, batch_size= batch_size,
                                 shuffle = True, pin_memory = True, num_workers = workers)
    
    return dataset, dataset_subset, trainloader

#%% Load SVHN

def load_svhn(path:str,
              batch_size:int=100,
              img_size:str=64,
              frac:float=0.8,
              workers:int=0,
              train_type='train'):
    
    dataset = dset.SVHN(root=path,
                        split = train_type,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
    
    dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), int(frac*len(dataset)), replace=False))

    trainloader = DataLoader(dataset = dataset_subset, batch_size= batch_size,
                                 shuffle = True, pin_memory = True, num_workers = workers)
    
    return dataset, dataset_subset, trainloader

#%% Load DTU

def load_dtu(path:str, 
                batch_size:int=100,
                img_size:str=64, 
                frac:float=.8,
                workers:int=0):

    dataset = dset.ImageFolder(root=path,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataset_subset = torch.utils.data.Subset(dataset, range(int(len(dataset)*frac)))
    
    trainloader = DataLoader(dataset = dataset_subset, batch_size= batch_size,
                                 shuffle = True, pin_memory = True, num_workers = workers)
    
    return dataset, dataset_subset, trainloader

#%% Load MNIST

def load_mnist(path:str, 
                batch_size:int=100,
                img_size:str=64, 
                frac:float=.8,
                workers:int=0):

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = dset.MNIST(root=path, train=True, download=True, transform=transform)
   
    dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), int(frac*len(dataset)), replace=False))

    trainloader = DataLoader(dataset = dataset_subset, batch_size= batch_size,
                                 shuffle = True, pin_memory = True, num_workers = workers)
    
    return dataset, dataset_subset, trainloader

#%% Load surface data

def load_surface(path:str,
                 batch_size:int=100,
                 workers:int=0,
                 device='cpu'):
    
    df = pd.read_csv(path)
    DATA = torch.Tensor(df.values).to(device) #DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)

    if device == 'cpu':
        trainloader = DataLoader(dataset = DATA, batch_size= batch_size,
                                 shuffle = True, pin_memory=True, num_workers = workers)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= batch_size,
                                 shuffle = True)
        
    return DATA, trainloader




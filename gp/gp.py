#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:15:04 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch import FloatTensor

from typing import Tuple, Callable

#%% Code

class GP(object):
    
    def __init__(self,
                 mu_fun = Callable[[FloatTensor], FloatTensor],
                 k_fun = Callable[[FloatTensor, FloatTensor], FloatTensor],
                 sigman2:float=1.0,
                 )->None:
        
        self.mu_fun = mu_fun
        self.k_fun = k_fun
        self.sigman2 = sigman2
        
        return
    
    def __str__(self)->str:
        
        return "Euclidean Gaussian Process Class"
    
    def simulate(self, mu:FloatTensor, cov:FloatTensor, N_sim:int=10)->FloatTensor:
        
        return
    
    def kmat(self, x1:FloatTensor, x2:FloatTensor)->FloatTensor:
        
        if len(x1.shape) == 1:
            return self.k_fun(x1,x2)
        else:   
            return torch.vmap(lambda x: torch.vmap(lambda y: self.k_fun(x,y))(x2))(x1)
    
    def prior(self, x:FloatTensor)->Tuple[FloatTensor,FloatTensor]:
        
        return self.mu_fun(x), self.kmat(x,x)
    
    def fit(self, x_training:FloatTensor, y_training:FloatTensor)->None:
        
        self.x_training = x_training
        self.y_training = y_training
        
        self.K11 = self.kmat(x_training,x_training)+self.sigman2*torch.eye(len(x_training))
        self.mu_training = torch.vmap(self.mu_fun)(x_training)
        self.K11_solved = torch.linalg.solve(self.K11, y_training-self.mu_training)
        
        return
    
    def posterior(self, x_test:FloatTensor, y_test:FloatTensor)->Tuple[FloatTensor, FloatTensor]:
        
        if len(y_test) == 1:
            mu_test = self.mu_fun(y_test)
            K12 = torch.vmap(lambda x: self.k_fun(x_test,x))(self.x_training)
            K21 = torch.vmap(lambda x: self.k_fun(x,x_test))(self.x_training)
            K22 = self.k_fun(x_test, x_test)
        else:
            mu_test = torch.vmap(self.mu_fun)(y_test)
            K12 = self.kmat(x_test, self.x_training)
            K21 = self.kmat(self.x_training, x_test)
            K22 = self.kmat(x_test, x_test)
            
        mubar = mu_test+torch.einsum('...i,i->...')
        solved = torch.linalg.solve(self.K11, K21)
        cov = K22-torch.mm(K12,solved)
        
        return mubar, cov
    
class WGP(GP):
    
    def __init__(self,
                 Exp_fun:Callable[[FloatTensor,FloatTensor], FloatTensor],
                 Log_fun:Callable[[FloatTensor, FloatTensor], FloatTensor],
                 mu_fun = Callable[[FloatTensor], FloatTensor],
                 k_fun = Callable[[FloatTensor, FloatTensor], FloatTensor],
                 sigman2:float=1.0,
                 )->None:
         super().__init__(lambda x: 0, k_fun, sigman2)
         
         self.Exp_fun = Exp_fun
         self.Log_fun = Log_fun
         self.mu_fun = mu_fun
         self.k_fun = k_fun
         
         return
   
    def __str__(self)->str:
        
        return "Wrapped Gaussian Process on Riemannian Manifolds"
    
    def simulate(self, mu:FloatTensor, cov:FloatTensor, N_sim:int=10)->FloatTensor:
        
        return

    def fit(self, x_training:FloatTensor, y_training:FloatTensor)->None:

        mu_training = torch.vmap(x_training)
        log_data = torch.vmap(self.Log_fun)(mu_training, y_training)
        
        (super(GP, self)).fit(x_training, log_data)
        
        return
    
    def posterior(self, x_test:FloatTensor, y_test:FloatTensor)->Tuple[FloatTensor,FloatTensor]:
        
        mu_test = torch.vmap(x_test)
        log_data = torch.vmap(self.Log_fun)(mu_test, y_test)
        
        mu, cov = (super(GP, self)).posterior(mu_test, log_data)
        
        return mu, cov
    
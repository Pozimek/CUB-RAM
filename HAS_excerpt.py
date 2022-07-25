#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:35:29 2022

HAS code excerpt for dissertation figure.

@author: piotr
"""
from utils import showTensor

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gennorm


#STEP 1: Compute distance matrix
support_size = (9, 9) #size of the HAS matrix
x, y = nn.Parameter(torch.tensor([4.0,4.0])) #spotlight fixation coordinates
B = torch.arange(support_size[0]).repeat(support_size[1],1) # coordinate grid
xd = B - x
yd = B.T - y
distance_matrix = torch.sqrt((xd**2 + yd**2).clamp(min=1e-6))


#STEP 2: Use distance matrix to compute support Gaussian
def modGaussian(tensor, a, f, c):
        return a * torch.exp(-(tensor**f)/(2*c**2))

amplitude = 0.98 # amplitude of the spotlight
spotlight = modGaussian(distance_matrix, 1 - amplitude, 2, 10)

#spotlight2 = torch.tensor(gennorm.pdf(distance_matrix.detach(), 0.25))

#STEP 3: Add spotlight values
def int_round(n):
    """Round to nearest integer and cast to int."""
    if n - np.floor(np.abs(n)) < 0.5:
        return int(np.floor(n))
    return int(np.ceil(n))

#Compute spotlight coordinates
width = 3 # pixel width of the spotlight area
from_y, to_y = int_round(y.item()) - width//2, \
                int_round(y.item()) + 1 + width//2
from_x, to_x = int_round(x.item()) - width//2, \
                int_round(x.item()) + 1 + width//2

#add spotlight to matrix    
spotlight[from_y:to_y, from_x:to_x] += amplitude

#showTensor(spotlight2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 00:42:00 2021

@author: piotr
"""

import torch
import torch.nn as nn
from utils import showTensor

torch.pi = torch.acos(torch.zeros(1)).item() * 2

h = 1000
w = 1000

bases = []

for u in range(1,5):
    for v in range(1,5):
        for Fy in [torch.cos, torch.sin]:
            for Fx in [torch.cos, torch.sin]:
                X = Fx(torch.tensor([torch.pi*u*i/h for i in range(h)], device='cuda'))
                Y = Fy(torch.tensor([torch.pi*v*j/w for j in range(w)], device='cuda'))
                bases.append(torch.ger(X,Y))
    
for basis in bases:
    showTensor(basis, size=(3,3))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:35:29 2022

HAS code excerpt for dissertation figure.

@author: piotr
"""

import torch
import torch.nn as nn

from utils import showTensor, validate_values, print_patch, ir

def generalizedGaussian(self, tensor, a, f, c):
        return a * torch.exp(-(tensor**f)/(2*c**2))

width = 10 # pixel size of the spotlight distribution
amplitude = 0.98 # amplitude of the spotlight distribution

#coordinate scaling factor for controlling spotlight width
div = torch.sqrt(torch.tensor(2))*(width/2 - 0.5)  

out_shape = (10, 10) #size of the HAS support (= size of peripheral patch)

# coordinate grid
B = torch.arange(out_shape[0]).repeat(out_shape[1],1)

# Spotlight fixation coordinates
coords = nn.Parameter(torch.tensor([4.0,4.0]))

xd = (B - coords[0])/div
yd = (B - coords[1])/div
xd.retain_grad()
d = torch.sqrt((xd**2 + yd.T**2).clamp(min=1e-6))
d.retain_grad()
#        spotlight = self.modGauss(d, self.a, self.f, self.c)
support = generalizedGaussian(d, 1 - amplitude, 2, 10)

# add spotlight onto support tensor
from_y, to_y = ir(coords[1].item()) - width//2, \
                ir(coords[1].item()) + 1 + width//2
from_x, to_x = ir(coords[0].item()) - width//2, \
                ir(coords[0].item()) + 1 + width//2
#        support[from_y:to_y, from_x:to_x] += self.modGauss(d[from_y:to_y, from_x:to_x].detach(), self.a, self.f, self.c)
support[from_y:to_y, from_x:to_x] += amplitude


class HAS(nn.Module):
    def __init__(self, out_shape, width = 3, amplitude = 0.98):
        super(HAS, self).__init__()
        self.amplitude = amplitude #amplitude of the spotlight distribution
        self.width = width #pixel size of the spotlight distribution
        
        # coordinate scaling factor for controlling spotlight width
        self.div = torch.sqrt(torch.tensor(2))*(width/2 - 0.5)
        
        # coordinate grid to be used later
        self.B = torch.arange(out_shape[0]).repeat(out_shape[1],1)
        
    def modGauss(self, tensor, a, f, c):
        """Generalized Gaussian"""
        return a * torch.exp(-(tensor**f)/(2*c**2))
        
    def forward(self, coords):
        # clamps necessary to prevent NaN gradients
        xd = (self.B - coords[0])/self.div
        yd = (self.B - coords[1])/self.div
        xd.retain_grad()
        d = torch.sqrt((xd**2 + yd.T**2).clamp(min=1e-6))
        d.retain_grad()
#        spotlight = self.modGauss(d, self.a, self.f, self.c)
        support = self.modGauss(d, 1-self.amplitude, 2, 10)
        
        # add spotlight
        from_y, to_y = ir(coords[1].item())-self.width//2, ir(coords[1].item()) + 1 + self.width//2
        from_x, to_x = ir(coords[0].item())-self.width//2, ir(coords[0].item()) + 1 + self.width//2
#        support[from_y:to_y, from_x:to_x] += self.modGauss(d[from_y:to_y, from_x:to_x].detach(), self.a, self.f, self.c)
        support[from_y:to_y, from_x:to_x] += self.amplitude
        return support

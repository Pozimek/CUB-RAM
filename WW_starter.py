#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:31:43 2022

WW Investigations starter script.

@author: piotr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def out_shape(module, in_shape):
    """
    A convenience function that computes an nn.Module's output shape given an 
    input shape. Skips batch dimension. 
    
    Useful for automating architecture builds.
    """
    device = next(module.parameters()).device
    T = torch.zeros((2,)+in_shape).to(device)
    with torch.no_grad():
        O = module(T)
    return O.shape[1:]

class ResNetEncoder(nn.Module):
    """ 
    A wrapper for building ResNet18 while easily modifying its architectural
    aspects. Strips FC and avgpool layers by default. Comment out lines 
    appropriately to control strides. 
    
    Args:
    in_shape: size of input image as torch.size (without batch dim)
    blocks: number of residual blocks to use. Less than 4 will strip blocks
            from the end.
    pretrained: whether to preload a Imagenet pretrained resnet
    maxpool: whether to get keep one maxpooling layer for a larger output size
    stride: whether to keep the 2,2 strides or reduce them down to 1,1
    """
    def __init__(self, in_shape, blocks = 4, pretrained = True, maxpool = True,
                 stride = True):
        super(ResNetEncoder, self).__init__()
        resnet = list(resnet18(pretrained = pretrained).children())
        if not maxpool: del resnet[3]
        if not stride:
#            resnet[0].stride = (1,1)
#            resnet[-5].conv1.stride = (1,1) #layer2
#            resnet[-5].downsample[0].stride = (1,1)
#            resnet[-4].conv1.stride = (1,1) #layer3
#            resnet[-4].downsample[0].stride = (1,1)
            resnet[-3][0].conv1.stride = (1,1) #layer4
            resnet[-3][0].downsample[0].stride = (1,1)
        trim = 2 + 4 - blocks
        self.resnet = nn.Sequential(*resnet[:-trim])
        self.out_shape = out_shape(self.resnet, in_shape)
        
    def forward(self, x):
        return self.resnet(x)
 
#### WhatWhere stuff ####
        
class WW_module(nn.Module):
    """
    Channel-wise global weight sharing operation. Expects input to be (B,C,h,w).
    
    Args
    - in_shape: the spatial dimensions of the input feature maps
    - out_channels: the size of 'where' dim, ie the number of distributions
    """
    def __init__(self, in_shape, out_channels, k=1):
        super(WW_module, self).__init__()
        self.spatial = in_shape[1:]
        self.W = nn.Parameter(
                torch.FloatTensor(torch.Size((1,1,)+self.spatial+(out_channels,))))
        torch.nn.init.xavier_uniform_(self.W)
        
        self.out_shape = torch.Size((k*in_shape[0],out_channels))
    
    def forward(self, x):
        x = x[..., None] * self.W
        return x.sum(dim=(2,3)) #(B, 'what', 'where')

class WhatMix(nn.Module):
    def __init__(self, in_shape, out_shape, act=nn.Identity, bias=True):
        super(WhatMix, self).__init__()
        self.conv = nn.Conv1d(in_shape[0], out_shape[0], 1, bias=bias)
        self.out_shape = torch.Size(out_shape)
        self.activation = act()
        
    def forward(self, x):
        return self.activation(self.conv(x))

class WhereMix(nn.Module):
    def __init__(self, in_shape, out_shape, act=nn.Identity, bias=True):
        super(WhereMix, self).__init__()
        self.conv = nn.Conv1d(in_shape[1], out_shape[1], 1, bias=bias)
        self.out_shape = torch.Size(out_shape)
        self.activation = act()#(B, 'what', 'where')
        
    def forward(self, x):
        x = self.activation(self.conv(x.transpose(1,2)))
        return x.transpose(1,2)
 
class WWMix(nn.Module):
    """A WhatMix followed by a WhereMix."""
    def __init__(self, in_shape, out_shape, activation=nn.Identity, bias=True):
        super(WWMix, self).__init__()
        self.what = WhatMix(in_shape, out_shape, activation, bias)
        self.where = WhereMix(in_shape, out_shape, activation, bias)
        self.out_shape = torch.Size(out_shape)
    
    def forward(self, x):
        return self.where(self.what(x))
    
class Dconv(nn.Module):
    """
    A convolution that shares weights across both channel and spatial dims.
    Not an efficient implementation.
    """
    def __init__(self, num_convs, kernel_size=3, stride=(1,1), 
                 bias=True):
        super(Dconv, self).__init__()
        self.unfold = nn.Unfold((kernel_size, kernel_size), stride=stride)
        
        weight_dims = (1,num_convs, kernel_size, kernel_size)
        self.num_convs = num_convs
        self.stride = stride
        self.W = nn.Parameter(torch.FloatTensor(torch.Size(weight_dims)))
        self.B = nn.Parameter(torch.FloatTensor((num_convs)))
        
        with torch.no_grad():
            self.B.zero_()
            torch.nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        B, C, H, W = x.shape
        out = []
        for dconv in range(self.num_convs):
            kernel = self.W[:,dconv,:,:].unsqueeze(1)
            bias = self.B[dconv].unsqueeze(0)
            for c in range(C):
                c_in = x[:,c,:,:].unsqueeze(1)
                out.append(F.conv2d(c_in, kernel, bias = bias,
                                    stride=self.stride))
        return torch.cat(out, 1)
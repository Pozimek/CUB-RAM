#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:12:15 2021

Foveal autoencoder.

@author: piotr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import out_shape

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, 
                               bias=False)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out) + self.skip(x))
        return out
        
class Decoder(nn.Module):
    def __init__(self, in_shape):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, 128, 5)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5)
        
        self.out_shape = out_shape(nn.Sequential(
                self.deconv1, self.deconv2, self.deconv3, self.deconv4), in_shape)
        
    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction
        
class Encoder(nn.Module):
    def __init__(self, in_shape):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 32, 7, stride=3, 
                               padding=3, bias=False)
        self.layer1 = ResBlock(32, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        
        self.out_shape = out_shape(nn.Sequential(
                self.conv1, self.layer1, self.layer2, self.layer3), in_shape)
        
    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.layer1(c1))
        c3 = F.relu(self.layer2(c2))
        c4 = F.relu(self.layer3(c3))
        return c4
      
in_shape = (3,37,37)
E = Encoder(in_shape)

x = torch.zeros((1,)+in_shape)
y = E(x)

D = Decoder(E.out_shape)
recon = D(y)
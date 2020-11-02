#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:38:01 2020

CUB-RAM architecture, baseline script.

Either the model script will take the form of a constructor, or each variant
will be built manually from modules. As such this is only a 'baseline' script.

@author: piotr
"""
from torchvision.models import resnet18
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import conv_outshape, ir, out_shape

from modules import crude_retina, sensor_resnet18, ResNet18_short
from modules import ConvLSTM, ActiveConvLSTM
from modules import classification_network, location_network, baseline_network

class CUBRAM_baseline(nn.Module):
    def __init__(self, name, g, k, s, std, gpu):
        """
        Args
        ----
        - g: size of the square patches extracted by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - std: standard deviation of the Gaussian policy.
        """
        super(CUBRAM_baseline, self).__init__()
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 7
        
        # Sensor
        self.retina = crude_retina(g, k, s)
        #outputs a list of g_t maps (k, (sensor.conv_out))
        #each map has 32 channels
        self.sensor = sensor_resnet18(self.retina)
        
        # Memory
        rnn_input = [self.sensor.out_shape[0]] * k
        hidden_channels = [128] * k
        self.rnn = ConvLSTM(rnn_input, hidden_channels, 3)
        
        # Auxiliary Modules
        h_t_shape = 128*self.sensor.out_shape[1:].numel()
        
        self.sensor.out_shape.numel()
        fc_size = 512
        self.locator = location_network(h_t_shape, fc_size, std)
        self.classifier = classification_network(k*h_t_shape, fc_size, 200)
        self.baseliner = baseline_network(k*h_t_shape, fc_size)
        
    def reset(self, B=1):
        """
        Initialize the hidden state and the location vectors. 
        Called every time a new batch is introduced.
        """
        self.rnn.reset()
        dtype = (torch.cuda.FloatTensor if self.gpu else torch.FloatTensor)

        l_t = torch.Tensor(B,2).uniform_(-1, 1) #start at random location
        l_t = Variable(l_t).type(dtype)

        return l_t
    
    def set_timesteps(self, n):
        print("\nWas using {} timesteps, now using {} timesteps.").format(
                self.timesteps, n)
        self.timesteps = n
        
    def forward(self, x):
        """
        Process a minibatch of input images.
        """
        # initialize
        locs, log_pis, baselines = [], [], []
        l_t_prev = self.reset(B=x.shape[0])
        
        # process minibatch
        for t in range(self.timesteps):
            g_t = self.sensor(x, l_t_prev)
            h_t = self.rnn(g_t)
            log_pi, l_t = self.locator(h_t[-1]) #peripheral hidden state only
            h_t_flat = torch.cat(
                    [h.unsqueeze(1) for h in h_t], 1).flatten(1, -1)
#            h_t_flat = torch.cat([
#                    self.pool(h).flatten(1) for h in h_t], 1)
            b_t = self.baseliner(h_t_flat).squeeze(0) #input all hidden states
            
            l_t_prev = l_t
            # store tensors
            locs.append(l_t)
            baselines.append(b_t)
            log_pis.append(log_pi)
        
        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        
        # classify
        log_probas = self.classifier(h_t_flat) #input all hidden states
        
        return log_probas, locs, log_pis, baselines
    

#TODO refactor all code to make ACONVLSTM a clean drop-in modular replacement.
#TODO figure out if the above TODO is even humanely possible
class CUBRAM_ACONVLSTM(nn.Module):
    def __init__(self, name, g, k, s, std, gpu):
        """Active Convolutional LSTM variant of the CUB-RAM. Ad-hoc model code
        until refactoring, so it's really messy."""
        super(CUBRAM_ACONVLSTM, self).__init__()
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 7
        
        # Sensor
        self.retina = crude_retina(g, k, s)
        self.sensor = sensor_resnet18(self.retina)
        
        # Memory, [[512,16,16], [512,6,6]]
        mem_shapes = [out_shape(self.sensor.resnet18,(3, 500, 500)),
                      out_shape(self.sensor.resnet18,(3, 500//s, 500//s))]
        
        #disparity of downscaling due to discretisation. Ouch.
        scaling = [(500. * 500. / mem_shapes[0][-2:].numel()) / 
                   (g * g / self.sensor.out_shape[-2:].numel()),
                   (500. * 500. / mem_shapes[1][-2:].numel()) / 
                   (g*s * g*s / self.sensor.out_shape[-2:].numel())
                ]
        
        #convert to arrays, reduce channel dim
        mem_shapes[0] = torch.tensor(mem_shapes[0]).numpy()
        mem_shapes[1] = torch.tensor(mem_shapes[1]).numpy()
        mem_shapes[0][0] /= 4
        mem_shapes[1][0] /= 4
        
        rnn_input = [self.sensor.out_shape] * k
        self.rnn = ActiveConvLSTM(rnn_input, mem_shapes, 3, scaling)
        
        fc_size = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #AVGPOOL before downstream
        self.locator = location_network(128, fc_size, std)
        self.baseliner = baseline_network(128*2, fc_size)
        self.classifier = classification_network(128*2,fc_size, 200)
        
    def reset(self, B=1):
        self.rnn.reset()
        dtype = (torch.cuda.FloatTensor if self.gpu else torch.FloatTensor)

        l_t = torch.Tensor(B,2).uniform_(-1, 1) #start at random location
        l_t = Variable(l_t).type(dtype)

        return l_t
    
    def set_timesteps(self, n):
        print("\nWas using {} timesteps, now using {} timesteps.").format(
                self.timesteps, n)
        self.timesteps = n
        
    def forward(self, x):
        # initialize
        locs, log_pis, baselines = [], [], []
        l_t_prev = self.reset(B=x.shape[0])
        
        # process minibatch
        for t in range(self.timesteps):
            g_t = self.sensor(x, l_t_prev)
            h_t = self.rnn(g_t, l_t_prev)
            log_pi, l_t = self.locator(self.pool(h_t[-1])) #peripheral hidden state only
            h_t_flat = torch.cat([
                    self.pool(h).flatten(1) for h in h_t], 1)
            b_t = self.baseliner(h_t_flat).squeeze(0) #input all hidden states
            
            l_t_prev = l_t
            # store tensors
            locs.append(l_t)
            baselines.append(b_t)
            log_pis.append(log_pi)
        
        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        
        # classify
        log_probas = self.classifier(h_t_flat) #input all hidden states
        
        return log_probas, locs, log_pis, baselines
    

class ff_r18(nn.Module):
    def __init__(self):
        super(ff_r18, self).__init__()
        self.name = 'ff_r18'
        self.resnet18 = ResNet18_short()
        self.resnet18.load_state_dict(resnet18(pretrained=True).state_dict())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 200)
        
    def forward(self, x):
        x = self.resnet18(x)
        x = self.pool(x)
        x = F.relu(self.fc1(x.flatten(1)))
        pred = F.log_softmax(self.fc2(x), dim=1)
        
        return pred, 0, 0, 0
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
from utils import conv_outshape, ir

from modules import baseline_network, crude_retina, glimpse_network
from modules import classification_network, location_network, ConvLSTM

class CUBRAM_baseline(nn.Module):
    def __init__(self, name, g, k, s, std, gpu):
        """
        Args
        ----
        - g: size of the square patches extracted by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - vis_size: size of visual vector.
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
        self.sensor = glimpse_network(self.retina)
        
        # Memory
        rnn_input = hidden_channels = [32] * k
        self.rnn = ConvLSTM(rnn_input, hidden_channels, 3)
        
        # Auxiliary Modules
        h_t_shape = self.sensor.conv_out
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
    
    
class debug_model(nn.Module):
    def __init__(self):
        """
        remove rnn and location net, keep retina (for input resolution 
        invariance) and glimpse+classifier in some form.
        """
        super(debug_model, self).__init__()
        self.name = "debug"
        self.std = 0.05
        self.gpu = True
        self.timesteps = 3
        self.V = 0
        
        # Sensor
        self.retina = crude_retina(150, 2, 2)
        #OUT: 2x[3x150x150]
        
#        ##
#        self.res18 = resnet18(pretrained=True) #XXX you are here
#        self.res18.fc = nn.Linear(self.res18.fc.in_features, 200)
#        ##
        
        # Convs. Use the peripheral tensor only
        self.conv1 = nn.Conv2d(3, 64, (5,5), stride=2)  
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride=1)
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1)
        self.conv4 = nn.Conv2d(64, 32, (1,1), stride=1)
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
#        
#        
#        # Compute conv output shape
#        cc_shape = conv_outshape((150,150), self.conv1)
#        cc_shape = conv_outshape(cc_shape, self.conv2)
#        cc_shape = np.array(cc_shape)//2
#        cc_shape = conv_outshape(cc_shape, self.conv3)
#        cc_shape = conv_outshape(cc_shape, self.conv4)
#        cc_shape = np.array(cc_shape)//2
##        self.fc_in = cc_shape[0] * cc_shape[1] * 32
        self.fc_in = 8192
#        
#        # Classify
#        fc_size = 512
#        self.classifier = classification_network(self.fc_in, fc_size, 200)
        
        k = 1
        h_t_shape = self.fc_in
        # Memory
        rnn_input = hidden_channels = [32] * k
        self.rnn = ConvLSTM(rnn_input, hidden_channels, 3)
        
        # Auxiliary Modules
        fc_size = 512
        self.classifier = classification_network(k*h_t_shape, fc_size, 200)
        
    def set_vis(self, v):
        """
        Sets the desired visualisation level for the next forward pass.
        """
        self.V = v
        
    def forward(self, x):
        # fixate on the center of the image
        fixation = [[0., 0.]]*x.shape[0]
        fixation = Variable(torch.Tensor(fixation)).type(torch.cuda.FloatTensor)
        
        # Subsample, peripheral only
#        x = self.retina.foveate(x, fixation, self.V)[:,-1]
#        x = F.log_softmax(self.res18(x), dim=1)

#        # Convs. Use the peripheral tensor only
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = self.maxpool1(x)
#        x = F.relu(self.conv3(x))
#        x = F.relu(self.conv4(x))
#        x = self.maxpool2(x)
#        
#        # Classify
#        x = self.classifier(x.flatten(1,-1))
#        
        # Return empty tensors with appropriate shape
        log_pis = [[0]]
        log_pis = Variable(torch.Tensor(log_pis)).type(torch.cuda.FloatTensor)
        baselines = [[0]]
        baselines = Variable(torch.Tensor(baselines)).type(torch.cuda.FloatTensor)
#        
#        self.V = 0 #reset visualisation level
#        return x, None, log_pis, baselines
        
        # initialize
        log_pis, baselines = [], []
        
        # process minibatch
        self.rnn.reset()
        for t in range(self.timesteps):
            g_t = self.retina.foveate(x, fixation, self.V)[:,-1]
            
            g_t = F.relu(self.conv1(g_t))
            g_t = F.relu(self.conv2(g_t))
            g_t = self.maxpool1(g_t)
            g_t = F.relu(self.conv3(g_t))
            g_t = F.relu(self.conv4(g_t))
            g_t = self.maxpool2(g_t).unsqueeze(0)
            
            h_t = self.rnn(g_t)
            h_t_flat = torch.cat(h_t).flatten(1, -1)
        
        # classify
        log_probas = self.classifier(h_t_flat) #input all hidden states
        
        return log_probas, None, log_pis, baselines
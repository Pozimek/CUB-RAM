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
from modules import ConvLSTM, ActiveConvLSTM, FC_RNN
from modules import classification_network, location_network, baseline_network
from modules import classification_network_short

class CUBRAM_baseline(nn.Module):
    """
    Old baseline net with ConvLSTM.
    """
    def __init__(self, name, std, retina, gpu):
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
        self.timesteps = 1
        
        # Sensor
        self.retina = retina
        k = self.retina.k
        #outputs a list of g_t maps (k, (sensor.conv_out))
        #each map has 32 channels
        self.sensor = sensor_resnet18(self.retina)
        
        # Memory
        rnn_input = [self.sensor.out_shape[0]] * k
        hidden_channels = [128] * k
        self.rnn = ConvLSTM(rnn_input, hidden_channels, 3)
        
        # Auxiliary Modules
        h_t_shape = 128*self.sensor.out_shape[1:].numel()
        
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
    
    def loss(self, *args):
        """
        Computes the hybrid loss (with intermediate classification loss)
        """
        log_probas, log_pi, baselines, y = args
        # extract prediction
        prediction = torch.max(log_probas[:,-1], 1)[1].detach()
        
        # compute reward
        #XXX TODO: cleanup baselines shape, starting from modules.py
        baselines = baselines.squeeze()
        R = (prediction == y).float() 
        R = R.unsqueeze(1).repeat(1, self.timesteps) 
        adjusted_R = R - baselines.detach()
    
        # intermediate classification supervision
        loss_classify = F.nll_loss(log_probas[:,0,:], y)
        for i in range(1, self.timesteps):
            loss_classify += F.nll_loss(log_probas[:,i,:], y)
        
        loss_classify = loss_classify/self.timesteps #average
        
        loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
        loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
        loss_baseline = F.mse_loss(baselines, R)
        return loss_classify, loss_reinforce, loss_baseline
        
    
    def set_timesteps(self, n):
        print("\nWas using {} timesteps, now using {} timesteps.".format(
                self.timesteps, n))
        self.timesteps = n
        
    def forward(self, x):
        # initialize
        locs, log_pis, baselines, log_probas = [], [], [], []
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
            y_p = self.classifier(h_t_flat) #input all hidden states
            l_t_prev = l_t
            
            # store tensors
            locs.append(l_t)
            baselines.append(b_t)
            log_pis.append(log_pi)
            log_probas.append(y_p)
        
        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        log_probas = torch.stack(log_probas).transpose(1, 0)
        
        return log_probas, locs, log_pis, baselines

class RAM_baseline(nn.Module):
    """ Vanilla RAM baseline with FC-RNN"""
    def __init__(self, name, std, retina, gpu):
        """
        Args
        ----
        - g: size of the square patches extracted by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - std: standard deviation of the Gaussian policy.
        """
        super(RAM_baseline, self).__init__()
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 1
        
        # Sensor
        self.retina = retina
        k = self.retina.k
        self.sensor = sensor_resnet18(self.retina)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Memory
        rnn_input = self.sensor.out_shape[0]*k
        rnn_hidden = 1024
        self.rnn = FC_RNN(rnn_input, rnn_hidden)
        
        fc_size = 512
        self.locator = location_network(rnn_hidden, rnn_hidden//2, std)
        self.classifier = classification_network_short(rnn_hidden, 200)
        self.baseliner = baseline_network(rnn_hidden, fc_size)
        
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
    
    def loss(self, *args):
        """
        Computes the hybrid loss (with intermediate classification loss)
        """
        log_probas, log_pi, baselines, y = args
        # extract prediction
        prediction = torch.max(log_probas[:,-1], 1)[1].detach()
        
        # compute reward
        #XXX TODO: cleanup baselines shape, starting from modules.py
        baselines = baselines.squeeze()
        R = (prediction == y).float() 
        R = R.unsqueeze(1).repeat(1, self.timesteps) 
        adjusted_R = R - baselines.detach()
    
        # intermediate classification supervision
        loss_classify = F.nll_loss(log_probas[:,0,:], y)
        for i in range(1, self.timesteps):
            loss_classify += F.nll_loss(log_probas[:,i,:], y)
        
        loss_classify = loss_classify/self.timesteps #average
        
        loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
        loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
        loss_baseline = F.mse_loss(baselines, R)
        return loss_classify, loss_reinforce, loss_baseline
        
    
    def set_timesteps(self, n):
        print("\nWas using {} timesteps, now using {} timesteps.".format(
                self.timesteps, n))
        self.timesteps = n
        
    def forward(self, x):
        # initialize
        locs, log_pis, baselines, log_probas = [], [], [], []
        l_t_prev = self.reset(B=x.shape[0])
        
        # process minibatch
        for t in range(self.timesteps):
            g_t = self.sensor(x, l_t_prev)
            g_t_flat = torch.cat(
                    [self.avgpool(g) for g in g_t], 1).squeeze()
            h_t = self.rnn(g_t_flat)
            
            log_pi, l_t = self.locator(h_t)
            h_t_flat = h_t.flatten(1,-1)
            b_t = self.baseliner(h_t_flat).squeeze(0) 
            y_p = self.classifier(h_t_flat)
            l_t_prev = l_t
            
            # store tensors
            locs.append(l_t)
            baselines.append(b_t)
            log_pis.append(log_pi)
            log_probas.append(y_p)
        
        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        log_probas = torch.stack(log_probas).transpose(1, 0)
        
        return log_probas, locs, log_pis, baselines
    

#TODO refactor all code to make ACONVLSTM a clean drop-in modular replacement.
#TODO figure out if the above TODO is even humanely possible
#TODO retina as a parameter to constructor
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
    def __init__(self, retina, pretrained=True):
        super(ff_r18, self).__init__()
        self.require_locs = True #assumes x consists of (image, locations)
        self.retina = retina
        self.n_patches = retina.k
        self.name = 'ff_r18'
        self.resnet18 = ResNet18_short()
        if pretrained: self.resnet18.load_state_dict(resnet18(pretrained=True).state_dict())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 200)
        
    def forward(self, x):
        # Compute bird centroids
        y_locs = x[1]
        l_t_prev = torch.zeros((len(x[0]), 2)).cuda()
        for i, sample in enumerate(y_locs):
            locs = sample[torch.where(sample[:,2] == 1)]
            centroid = torch.mean(locs,0)[:2].int()
            l_t_prev[i] = centroid
        
        #foveate, process patches individually
        phi = self.retina.foveate(x[0], l_t_prev)
        out = []
        for i in range(self.n_patches):
            x = phi[:,i,:,:,:] #select patch
            x = self.resnet18(x)
            x = self.pool(x)
            out.append(x)
        
        #average patches and predict
        x = torch.mean(torch.stack(out), 0)
        x = F.relu(self.fc1(x.flatten(1)))
        pred = F.log_softmax(self.fc2(x), dim=1)
        pred = pred.unsqueeze(1) #timestep dimension

        return pred, 0, 0, 0
    
    def loss(self, *args):
        log_probas, log_pi, baselines, y = args
        loss_classify = F.nll_loss(log_probas.squeeze(1), y)
        return loss_classify
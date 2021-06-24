#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:38:01 2020

CUB-RAM architectures.

@author: piotr
"""
from torchvision.models import resnet18
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import conv_outshape, ir, out_shape
from random import shuffle

from modules import crude_retina, glimpse_module
from modules import ConvLSTM, ActiveConvLSTM, FC_RNN
from modules import classification_network, location_network, baseline_network
from modules import classification_network_short, WhereMix_classifier, accumulative_classifier
from modules import RAM_sensor, bnlstm, laRNN
from modules import PositionalEmbeddingNet
from modules import PredictiveModule, APCModule
from modules import Bottleneck_module
from modules import WW_module, WW_LSTM, WW_LSTM_stack, INFWhereMix, WhereMix, WhatMix, INFWWMix, WWMix
from modules import Tclassifier

def guide_fn(y_locs, timestep, stoch=True):
    """
    Given anatomical part locations and the timestep, returns the absolute 
    fixation location for the network to look at based on a prioritized list.
    Necessary due to occlusion: if desired part is not visible, the fn returns
    the next most relevant location. Motivated by trying to preserve a
    canonical order of exposure.
    """
    # A prioritized list of lists for hardcoded fixations. Canonical order.
    #Roughly follows: head, legs, torso front, beak, tail/wings.
    ALL_PRIORITIES = [
            [ 6, 10,  5,  4, 14,  9,  1,  3,  2,  0,  8, 12,  7, 11, 13], #1
            [ 7, 11,  2,  3, 14,  8, 12,  0,  9,  4,  5,  6, 10, 13,  1], #2
            [ 3,  2, 14,  7, 11,  0,  1,  5,  6, 10,  4,  9, 12,  8, 13], #3
            [ 1, 14,  5, 10,  6,  4,  9,  3,  2,  0,  7, 11,  8, 12, 13], #4
            [13,  8, 12,  0,  9,  4, 14,  2,  3,  5,  1, 11,  7, 10,  6], #5
            [0 ,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
            ]

    P = ALL_PRIORITIES[timestep]
    fix = torch.zeros((y_locs.shape[0],2))
    for priority in range(len(P)):
        #choose y_locs to assign based on priorities but only insert them where one hasnt been assigned
        ind = np.where((y_locs[:,P[priority],-1]==1) & (fix[:,0]==0) & (fix[:,1]==0))
        fix[ind] = y_locs[ind, P[priority],:2]
        if stoch:
            noise = torch.zeros_like(fix[ind])
            noise.data.normal_(std=4)
            fix[ind] = fix[ind] + torch.round(noise)
    
    return fix.cuda()

class Guide():
    """
    Returns fixation locations in either canonical or custom order.
    Always shuffles fixation order during training.
    """
    def __init__(self, T, training, Vmode, train_all = False):
        self.T = T
        self.mode = Vmode
        if self.mode == "canonical": self.order = [i for i in range(T)]
        elif self.mode == "MHL3": self.order = [3,0,4] 
        elif self.mode == "MHL5": self.order = [3,2,0,4,1]
        elif self.mode == "inverted": self.order = [4,3,2,1,0]
        elif type(self.mode) is list: self.order = self.mode
        else: raise Exception("No such guide mode")
        if training: 
            if train_all: #make all fixations available during training via shuffle
                shuffle(self.order)
            else: #shuffle only those fixations available at T
                Tset = self.order[:T+1] 
                shuffle(Tset)
                self.order[:T+1] = Tset
    
    def __call__(self, y_locs, timestep, stoch=True):
        if timestep < self.T:
            return guide_fn(y_locs, self.order[timestep], stoch)
        else:
            return guide_fn(y_locs, 5, stoch) #return dummy value
    
class RAM_baseline(nn.Module):
    """ Vanilla RAM baseline with drop-in modules, most up-to-date version 
    as of may2021
    
    current config: fovper"""
    def __init__(self, name, std, retina, feature_extractor, rnn, dropout, gpu,
                 fixation_set = "MHL3"):
        """
        Args
        - name: namestring of the model. #TODO: refactor
        - std: standard deviation of the gaussian location sampling policy
        - retina: RAMsensor, visual pre-proessing sensor.
        - feature extractor: a convnet extracting features (resnet atm)
        - rnn: a constructor for the memory module
        """
        super(RAM_baseline, self).__init__()
        self.hardcoded_locs = True #XXX refactor into main or config
        self.require_locs = self.hardcoded_locs
        self.fixation_set = fixation_set #at val time. shuffle, canonical, MHL3, MHL5
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 1
        self.T = torch.tensor((500,500), device='cuda') #image shape
        
        # Sensor
        self.retina = retina
        self.k = self.retina.k #num of patches
#        self.k = 1 #peronly
        self.glimpse_module = glimpse_module(retina, feature_extractor, avgpool = True)
        
        bottle_in = self.glimpse_module.out_shape
        self.bottleneck = Bottleneck_module(bottle_in, self.k, layers = [512,512])
        self.loc_bottleneck = nn.Sequential(nn.Linear(2,16), nn.ReLU())
        
        # Memory and positional embeddings
#        rnn_input = self.glimpse_module.out_shape.numel() #+ 2*4 #appended loc history
        self.dropout = nn.Dropout(p=dropout)
        self.rnn_input = self.bottleneck.out_shape.numel()
        self.rnn_hidden = 1024
        if rnn is laRNN: self.rnn_hidden = self.rnn_input
#        self.posembed = PositionalEmbeddingNet(self.glimpse_module.out_shape)
#        rnn_input += 18
        self.rnn = rnn(self.rnn_input, self.rnn_hidden)
        
        # Predictive module
#        self.lookahead = PredictiveModule(self.rnn_hidden, 4)
#        self.ph = None
#        self.h = None
        
        # Predictive Active Coding
#        self.APC = APCModule(self.rnn_hidden, 4, self.rnn_input)
#        self.g = None
#        self.pg = None
        
        fc_size = 512
        self.locator = location_network(self.rnn_hidden, self.rnn_hidden//2, std)
#        self.classifier = classification_network(self.rnn_hidden, 512, 200)
        self.classifier = classification_network_short(self.rnn_hidden, 200)
                            
        self.baseliner = baseline_network(self.rnn_hidden, fc_size)
        
    def reset(self, B=1):
        """Initialize the hidden state and the location vectors at new batch."""
        self.rnn.reset(B)
        self.glimpse_module.reset()
        dtype = (torch.cuda.FloatTensor if self.gpu else torch.FloatTensor)
        l_t = torch.Tensor(B,2).uniform_(-1, 1) #start at random location
        l_t = Variable(l_t).type(dtype)

        return l_t
    
    def set_timesteps(self, n):
        self.timesteps = n
        
    def forward(self, x):
        # initialize
        if self.hardcoded_locs:
            y_locs = x[1]
            x = x[0]
            _ = self.reset(B=x.shape[0])
            guide = Guide(self.timesteps, self.training, self.fixation_set)
            l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        else:
            l_t_prev = self.reset(B=x.shape[0])
            
        locs, log_pis, baselines, log_probas = [l_t_prev], [], [], []
#        self.ph = torch.zeros((x.shape[0], self.timesteps, self.rnn_hidden)).cuda()
#        self.h = torch.zeros((x.shape[0], self.timesteps, self.rnn_hidden)).cuda()
        
        self.pg = torch.zeros((x.shape[0], self.timesteps, self.rnn_input)).cuda()
        self.g = torch.zeros((x.shape[0], self.timesteps, self.rnn_input)).cuda()
        
        # position relative to first fixation
#        L = torch.zeros((x.shape[0], 2 * 4)).cuda()
        pos_t = torch.zeros_like(l_t_prev).cuda()
#        pg_t = torch.zeros((x.shape[0], self.rnn_input)).cuda()
        
        # process minibatch
        for t in range(self.timesteps):
            # Update position
            if t!= 0: pos_t = pos_t + l_t_prev
            
            # Extract features, compute glimpse vector
            g_t = self.glimpse_module(x, l_t_prev)
            if t==0: l_t_prev = torch.zeros_like(l_t_prev, device='cuda') #mask out absolute coordinates at t=0
            lg_t = self.loc_bottleneck(l_t_prev)
            g_t = [T.squeeze() for T in g_t]
#            g_t = [g_t[1]] #peronly
            g_t = torch.cat(g_t + [lg_t], 1)
            g_t = self.bottleneck(g_t)
            
            self.g[:,t,:] = g_t.detach() # ground truth for APC
#            g_t = g_t - pg_t.detach() # apply APC
            
            # memory
            if type(self.rnn) is bnlstm:
                h_t = self.rnn(g_t, t) #bnlstm requires timestep
            else:
                h_t = self.rnn(g_t)
            h_t_flat = h_t.flatten(1,-1)
            
            log_pi, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t_flat).squeeze(0) 
            y_p = self.classifier(self.dropout(h_t_flat))
            
            if self.hardcoded_locs:
                l_t_prev = self.retina.to_egocentric_action(
                        guide(y_locs, t+1) - self.retina.fixation) 
            else: 
                l_t_prev = l_t
            
            # next timestep APC
#            pg_t = self.APC(torch.cat((h_t_flat.detach(), l_t_prev, pos_t), 1))
#            self.pg[:,t,:] = pg_t
            
            # predictive module
#            self.h[:,t,:] = h_t_flat.detach()
#            self.ph[:,t,:] = self.lookahead(torch.cat((h_t_flat, l_t_prev, pos_t), 1))
            
            # store tensors
            locs.append(l_t_prev)
            baselines.append(b_t)
            log_pis.append(log_pi)
            log_probas.append(y_p)
        
        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        log_probas = torch.stack(log_probas).transpose(1, 0)
        locs = torch.stack(locs).transpose(1, 0)
        
        return log_probas, locs, log_pis, baselines
    
    def loss(self, *args):
        """Computes the hybrid loss"""
        log_probas, log_pi, baselines, y, locs, y_locs = args
        
        # extract prediction
        prediction = torch.max(log_probas[:,-1], 1)[1].detach()
        
        # compute reward
        #TODO: cleanup baselines shape, starting from modules.py
        baselines = baselines.squeeze()
        R = (prediction == y).float() 
        R = R.unsqueeze(1).repeat(1, self.timesteps) 
        adjusted_R = R - baselines.detach()
    
        ## intermediate classification supervision
        loss_classify = F.nll_loss(log_probas[:,-1,:], y)
#        for i in range(1, self.timesteps):
#            loss_classify += F.nll_loss(log_probas[:,i,:], y)
#        loss_classify = loss_classify/self.timesteps #average
        
#        # predictive module loss
#        cos = nn.CosineSimilarity(dim=2)
##        loss_lookahead = torch.tensor(0)
#        loss_lookahead = 1.5 * torch.sum(1 - cos(self.h[:,1:,:], self.ph[:,:-1,:]), dim=1)
#        loss_lookahead = torch.mean(loss_lookahead, dim=0) #avg batch
        
        
        if self.hardcoded_locs: 
            return loss_classify, torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
#        ## Key feature distance loss. Experimental.
#        S = RAM_sensor() #recycle code
#        S.width = self.retina.width
##        dist = torch.zeros_like(locs[:,:-1,0]) #final loc is never used
#        targets = torch.zeros_like(locs[:,:-1]) #final loc is never used
#        abs_locs = torch.zeros_like(locs[:,:-1])
#        
#        #convert locs to absolute coordinates, compute distance to closest feature
#        y_locs[np.where(y_locs[...,-1]==0)] = torch.tensor([2000.0,2000.0,0.0]) #hack away invisible features
#        y_locs = y_locs[:,:,:2].cuda()
#        b_dim = range(len(y_locs))
#        for timestep in range(abs_locs.shape[1]):
#            if timestep==0: abs_locs[:,timestep,:] = S.to_exocentric(self.T, locs[:,0,:])
#            else:
#                S.fixation = abs_locs[:,timestep-1,:]
#                abs_locs[:,timestep,:] = S.to_egocentric(self.T, locs[:,timestep,:])
##            dist[:,timestep] = torch.min(torch.cdist(abs_locs[:,timestep,:].unsqueeze(1), y_locs), 2)[0].squeeze() #one liners are fun
#            closest_ind = torch.min(torch.cdist(abs_locs[:,timestep,:].unsqueeze(1), y_locs), 2)[1].squeeze()
#            #compute optimal action
#            targets[:, timestep] = S.to_egocentric_action(y_locs[b_dim, closest_ind,:] - abs_locs[:,timestep,:])
        
#        #compute optimal action divergence loss
#        loss_dist = F.mse_loss(locs[:,:-1], targets)
        
        # reinforce loss
        loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
        loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
        loss_baseline = F.mse_loss(baselines, R)
        return loss_classify, loss_reinforce, loss_baseline, torch.tensor(0)

class WW_RAM(nn.Module):
    """WhereWhat_RAM model, june 2021"""
    def __init__(self, name, std, retina, feature_extractor, gpu, 
                 fixation_set = "MHL3"):
        super(WW_RAM, self).__init__()
        self.require_locs = True
        self.fixation_set = fixation_set
        self.full_set_training = False #whether to train w/ all fixations from set
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 1
        self.T = torch.tensor((500,500), device='cuda') #image shape
        
        # Sensor
        self.retina = retina
        self.k = self.retina.k #num of patches
#        self.k = 1 #peronly hack
        self.glimpse_module = glimpse_module(retina, feature_extractor, avgpool = False)
        
        # WW module
        WW_in_shape = self.glimpse_module.out_shape
#        self.WW_where = 10
        self.WW_where = 1
        self.WW_module = WW_module(WW_in_shape, self.WW_where, self.k)
        
        # Positional WhereMix
#        self.posINF = INFWhereMix(2, self.WW_module.out_shape, 10)
        self.posINF = INFWhereMix(7, self.WW_module.out_shape, self.WW_where)
        
        # Memory
        self.mem_in = self.WW_module.out_shape
        self.mem_what = 512
#        self.mem_where = 10
        self.mem_where = 1
        self.mem_hidden = self.mem_what * self.mem_where
        self.memory = WW_LSTM(self.mem_in, self.mem_what, self.mem_where, gate_op=WhereMix, in_op=WhereMix)
        
#        fc_size = 512
#        self.locator = location_network(self.mem_hidden, self.mem_hidden//2, std)
        self.classifier = classification_network_short(self.mem_hidden, 200)
#        self.classifier = Tclassifier(self.mem_hidden, 200) #XXX tclassifier
#        self.baseliner = baseline_network(self.mem_hidden, fc_size)
        
    def reset(self, B=1):
        """Initialize the hidden state and the location vectors at new batch."""
        self.memory.reset(B)
        self.glimpse_module.reset()
        dtype = (torch.cuda.FloatTensor if self.gpu else torch.FloatTensor)
        l_t = torch.Tensor(B,2).uniform_(-1, 1) #start at random location
        l_t = Variable(l_t).type(dtype)

        return l_t
    
    def set_timesteps(self, n):
        self.timesteps = n
        
    def forward(self, x):
        # Initial fixation
        if self.require_locs:
            y_locs = x[1]
            x = x[0]
            _ = self.reset(B=x.shape[0])
            guide = Guide(self.timesteps, self.training, self.fixation_set, 
                          train_all = self.full_set_training)
            l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        else:
            l_t_prev = self.retina.to_exocentric_action(self.T, self.reset(B=x.shape[0]))
            
        locs, log_pis, baselines, log_probas = [], [], [], []
        
        # add initial fixation location (enable recovering absolute coords)
        locs.append(l_t_prev)
        
        # position relative to initial fixation
        pos_t = torch.zeros_like(l_t_prev, device='cuda')
        
        # classification bins
#        a_t = Variable(torch.zeros((x.shape[0], 200), device='cuda'))
        
        # process minibatch
        for t in range(self.timesteps):
            # Maintain position
            if t!= 0: pos_t = pos_t + l_t_prev
            
            # Extract features, compute WW matrix
            g_t = self.glimpse_module(x, l_t_prev)
            if t==0: l_t_prev = torch.zeros_like(l_t_prev, device='cuda') #mask out absolute coordinates at t=0
            g_t = [T.squeeze() for T in g_t]
            WWfov = self.WW_module(g_t[0])
#            WWper = self.WW_module(g_t[1])
#            WW = torch.cat((WWfov,WWper),dim=1) #along 'what' dim
#            WW = torch.cat((WWfov,WWper),dim=2) #along 'where' dim
            WW = WWfov
            
            # Apply PE
            time = torch.zeros((x.shape[0], 5), device='cuda')
            time[:,t] = 1.0
            PE_in = torch.cat((pos_t, time), dim=1)
            WW = self.posINF(WW, PE_in)
#            WW = self.posINF(WW, pos_t)
            
            # memory
            h_t = self.memory(WW)
            h_t_flat = h_t.flatten(1,-1)
#            
#            b_t = self.baseliner(h_t_flat).squeeze(0)
            b_t = torch.Tensor([0.]).cuda() #DUMMY 
            log_pi = torch.Tensor([0., 0.]).cuda() #DUMMY 
            y_p = self.classifier(h_t_flat)
#            a_t = self.classifier(h_t_flat, a_t)

            # Decide where to look next            
            if self.require_locs:
                l_t_prev = self.retina.to_egocentric_action(
                        guide(y_locs, t+1) - self.retina.fixation) 
            else: 
                log_pi, l_t_prev = self.locator(h_t)
                
            # store tensors
            locs.append(l_t_prev)
            baselines.append(b_t)
            log_pis.append(log_pi)
            log_probas.append(y_p)
        
        # apply log_softmax to accumulative classifier output
#        log_probas[-1] = F.log_softmax(log_probas[-1], dim=1)

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pis = torch.stack(log_pis).transpose(1, 0)
        log_probas = torch.stack(log_probas).transpose(1, 0)
        locs = torch.stack(locs).transpose(1, 0)
        
        return log_probas, locs, log_pis, baselines
    
    def loss(self, *args):
        """Computes the hybrid loss"""
        log_probas, log_pi, baselines, y, locs, y_locs = args
        prediction = torch.max(log_probas[:,-1], 1)[1].detach()
        
        # classification supervision
        loss_classify = F.nll_loss(log_probas[:,-1,:], y)
#        b = 0.05
#        loss_classify = (loss_classify - b).abs() + b
        
        # Intermediate classification supervision
#        loss_classify = sum(F.nll_loss(log_probas[:,t,:], y) for t in range(self.timesteps))
        
        if self.require_locs: 
            return loss_classify, torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        # compute reward
        baselines = baselines.squeeze()
        R = (prediction == y).float() 
        R = R.unsqueeze(1).repeat(1, self.timesteps) 
        adjusted_R = R - baselines.detach()

        # other losses
        loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
        loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
        loss_baseline = F.mse_loss(baselines, R)
        return loss_classify, loss_reinforce, loss_baseline, torch.tensor(0)

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
    

class FF_GlimpseModel(nn.Module):
    """ A feedforward model using a feature extractor (eg ResNet18) to classify
    a number of image observations processed by a RAM_sensor. The different
    observations are aggregated using different strategies:
        1. Imagespace aggregation using masking: occlude all of the input image
        and gradually uncover it with each observation. Preserves all relevant
        information.
        2. Imagespace aggregation using spatial concatenation of observed image
        patches. Relative spatial information is lost. Q: shuffle order?
        3. Featurespace aggregation using averaging of feature vectors after 
        passing image patches sequentially. Relative spatial information is lost
        alongside imagespace patch differentiation. Order-invariant. Ensemble.
        4. Output aggregation using averaging of softmax outputs after passing
        image patches sequentially. Relative confidences and spatial information
        is lost, order invariant. Ensemble.
        5. Output aggregation using averaging of classifier pre-softmax outputs
        after passing image patches sequentially. Spatial information is lost,
        order invariant. Ensemble.
    """
    def __init__(self, name, retina, feature_extractor, strategy=4, gpu=True, 
                 fixation_set = "MHL3"):
        super(FF_GlimpseModel, self).__init__()
        self.hardcoded_locs = True
        self.require_locs = self.hardcoded_locs
        self.fixation_set = fixation_set
        self.name = name
        self.T = torch.tensor((500,500)).cuda() #image shape
        self.gpu = gpu
        self.timesteps = 1
        self.zero = torch.tensor([0], device='cuda')
        
        # Which of the 5 aggregation strategies to employ
        self.strategy = strategy 
        
        self.retina = retina
        self.glimpse_module = glimpse_module(retina, feature_extractor, avgpool=True)
        self.fc = nn.Linear(self.glimpse_module.out_shape.numel(), 200)

    def set_timesteps(self, n):
        self.timesteps = n
        
    def forward(self, x):
        self.glimpse_module.reset()
        if self.retina.k != 1 and (self.strategy == 1 or self.strategy == 2):
                raise Exception("Imagespace aggregation currently does not \
                                support multi-patch retina.")
        if self.strategy == 1:
            return self.mask_aggregation(x[0], x[1])
        elif self.strategy == 2:
            return self.spatial_concatenation(x[0], x[1])
        elif self.strategy == 3:
            return self.feature_averaging(x[0], x[1])
        elif self.strategy == 4:
            return self.softmax_averaging(x[0], x[1])
        elif self.strategy == 5:
            return self.presoftmax_averaging(x[0], x[1])
      
    def loss(self, *args):
        """Computes negative log likelihood classification loss"""
        log_probas, _, _, y, _, _ = args
        return F.nll_loss(log_probas[:,-1,:], y)
    
    def mask_aggregation(self, x, y_locs):
        """Strategy 1"""
        # First fixation
        if self.gpu: x = x.cuda()
        guide = Guide(self.timesteps, self.training, self.fixation_set)
        l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        locs = [l_t_prev]
        
        R = self.glimpse_module.retina
        size = R.g
        masks = torch.zeros_like(x)
        B, C, H, W = x.shape
        
        # process image to compute masks
        for t in range(self.timesteps):
            if R.fixation is None: #first fixation has to be exocentric
                coords = R.to_exocentric(self.T, l_t_prev)
            else: coords = R.to_egocentric(self.T, l_t_prev)
            R.fixation = coords
            
            # Patch corners
            from_x, from_y = coords[:, 1] - (size // 2), coords[:, 0] - (size // 2)
            to_x, to_y = from_x + size, from_y + size
            # Clamp
            from_x = torch.max(self.zero, from_x)
            from_y = torch.max(self.zero, from_y)
            to_y = torch.min(self.T[0], to_y)
            to_x = torch.min(self.T[1], to_x)
            
            for b in range(B):
                masks[b, :, from_y[b]:to_y[b], from_x[b]:to_x[b]] = 1
            
            # Get and store next fixation
            l_t_prev = self.retina.to_egocentric_action(
                    guide(y_locs, t+1) - self.retina.fixation)
            locs.append(l_t_prev)
        
        # Apply masks
        x = x * masks #XXX would be good to visualize here
        log_probas = self.imagespace_aggregate(x)
        
        # convert list to tensors and reshape
        locs = torch.stack(locs).transpose(1, 0)
        
        return log_probas, locs, torch.Tensor(0), torch.Tensor(0)
    
    def spatial_concatenation(self, x, y_locs):
        """Strategy 2"""
        # First fixation
        guide = Guide(self.timesteps, self.training, self.fixation_set)
        l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        locs = [l_t_prev]
        
        patches = []
        
        # process image to extract patches
        for t in range(self.timesteps):
            # Extract patch
            phi = self.glimpse_module.retina.foveate_ego(x, l_t_prev)
            patches.append(phi)

            # Get and store next fixation
            l_t_prev = self.retina.to_egocentric_action(
                    guide(y_locs, t+1) - self.retina.fixation) 
            locs.append(l_t_prev)
        
        # Produce and classify concatenation
        concat = torch.cat(patches, dim=4).squeeze() #XXX would be good to visualize here
        log_probas = self.imagespace_aggregate(concat)
        
        # convert list to tensors and reshape
        locs = torch.stack(locs).transpose(1, 0)
        
        return log_probas, locs, torch.Tensor(0), torch.Tensor(0)
    
    def imagespace_aggregate(self, x):
        """Feedforward pass shared between imagespace aggregation strategies
        (1 and 2)"""
        G = self.glimpse_module
        features = G.feature_extractor(x)
        
        #avgpooling sparse feature matrices can produce a very low magnitude
        #feature vector, so I'm amplifying them beforehand. 
        if G.avgpool: 
#            if self.strategy == 1: features = 100*features
            features = G.avgpool(features)
        
        log_probas = [F.log_softmax(self.fc(features.squeeze()), dim=1)]
        log_probas = torch.stack(log_probas).transpose(1, 0)
        
        return log_probas
    
    def feature_averaging(self, x, y_locs):
        """Strategy 3"""
        guide = Guide(self.timesteps, self.training, self.fixation_set)
        l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        locs, g = [l_t_prev], []
        
        # process image
        for t in range(self.timesteps):
            # Extract features
            g_t = self.glimpse_module(x, l_t_prev)
            
            # Combine patches
            g_t = torch.mean(torch.cat(g_t, dim=2),dim=2).squeeze()
            
            # Get next fixation
            l_t_prev = self.retina.to_egocentric_action(
                    guide(y_locs, t+1) - self.retina.fixation)
                
            # Store tensors
            locs.append(l_t_prev)
            g.append(g_t)
        
        # Average feature tensors
        g = torch.mean(torch.stack(g, dim=1),dim=1)
        
        # Classify
        log_probas = [F.log_softmax(self.fc(g), dim=1)]
        
        # convert list to tensors and reshape
        log_probas = torch.stack(log_probas).transpose(1, 0)
        locs = torch.stack(locs).transpose(1, 0)
        
        return log_probas, locs, torch.Tensor(0), torch.Tensor(0)
    
    def output_aggregate(self, x, y_locs):
        """Feedforward pass shared between output aggregation strategies 
        (4 and 5)"""
        guide = Guide(self.timesteps, self.training, self.fixation_set)
        l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        locs, log_probas = [l_t_prev], []
        
        # process image
        for t in range(self.timesteps):
            # Extract features
            g_t = self.glimpse_module(x, l_t_prev)
            
            # Combine patches
            g_t = torch.mean(torch.cat(g_t, dim=2),dim=2).squeeze()
            
            # Classify
            y_p = F.log_softmax(self.fc(g_t), dim=1)

            # Get next fixation
            l_t_prev = self.retina.to_egocentric_action(
                    guide(y_locs, t+1) - self.retina.fixation)
                
            # store tensors
            locs.append(l_t_prev)
            log_probas.append(y_p)
        
        # convert list to tensors and reshape
        log_probas = torch.stack(log_probas).transpose(1, 0)
        locs = torch.stack(locs).transpose(1, 0)
        return log_probas, locs
    
    def softmax_averaging(self, x, y_locs):
        """Strategy 4"""
        log_probas, locs = self.output_aggregate(x, y_locs)
        
        # Compute final classification output
        averaged_softmax = torch.mean(log_probas, dim=1)
        log_probas[:,-1,:] = averaged_softmax
        
        return log_probas, locs, torch.Tensor(0), torch.Tensor(0)
    
    def presoftmax_averaging(self, x, y_locs):
        """Strategy 5"""
        log_probas, locs = self.output_aggregate(x, y_locs)
        
        # Compute final classification output
        averaged_presoftmax = torch.mean(log_probas, dim=1)
        log_probas[:,-1,:] = F.log_softmax(averaged_presoftmax, dim=1)
#        for t in range(self.timesteps):
#            log_probas[:,t,:] = F.log_softmax(log_probas[:,t,:], dim=1)
        
        return log_probas, locs, torch.Tensor(0), torch.Tensor(0)
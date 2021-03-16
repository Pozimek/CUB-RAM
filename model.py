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

from modules import crude_retina, sensor_resnet18, ResNet18_short, glimpse_module
from modules import ConvLSTM, ActiveConvLSTM, FC_RNN
from modules import classification_network, location_network, baseline_network
from modules import classification_network_short
from modules import RAM_sensor

def guide(y_locs, timestep):
    """
    Given anatomical part locations and the timestep, returns the absolute 
    fixation location for the network to look at based on a prioritized list.
    Necessary due to occlusion: if desired part is not visible, the fn returns
    the next most relevant location. Motivated by trying to preserve a
    canonical order of exposure.
    """
    # A prioritized list for hardcoded fixations. 
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
    
    return fix.cuda()

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
    """ Vanilla RAM baseline with drop-in modules, most up-to-date version 
    as of mar2021"""
    def __init__(self, name, std, retina, feature_extractor, rnn, gpu):
        """
        Args
        ----
        - name: namestring of the model. #TODO: refactor
        - std: standard deviation of the gaussian location sampling policy
        - retina: RAMsensor, visual pre-proessing sensor.
        - feature extractor: a convnet extracting features (resnet atm)
        - rnn: a constructor for the memory module
        """
        super(RAM_baseline, self).__init__()
        self.hardcoded_locs = True
        self.require_locs = self.hardcoded_locs
        self.name = name
        self.std = std
        self.gpu = gpu
        self.timesteps = 1
        self.T = torch.tensor((500,500)).cuda() #image shape
        
        # Sensor
        self.retina = retina
        self.k = self.retina.k #num of patches
        self.glimpse_module = glimpse_module(retina, feature_extractor)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Memory
        rnn_input = self.glimpse_module.out_shape.numel() + 2*4 #appended loc history
        rnn_hidden = 1024
        self.rnn = rnn(rnn_input, rnn_hidden)
        
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
        self.glimpse_module.reset()
        dtype = (torch.cuda.FloatTensor if self.gpu else torch.FloatTensor)

        l_t = torch.Tensor(B,2).uniform_(-1, 1) #start at random location
        l_t = Variable(l_t).type(dtype)

        return l_t
    
    def set_timesteps(self, n):
        print("\nWas using {} timesteps, now using {} timesteps.".format(
                self.timesteps, n))
        self.timesteps = n
        
    def forward(self, x):
        # initialize
        locs, log_pis, baselines, log_probas = [], [], [], []
        if self.hardcoded_locs: 
            y_locs = x[1]
            x = x[0]       
            _ = self.reset(B=x.shape[0])
            l_t_prev = self.retina.to_exocentric_action(self.T, guide(y_locs, 0))
        else:
            l_t_prev = self.reset(B=x.shape[0])
        
        # add initial fixation location (enable recovering absolute coords)
        locs.append(l_t_prev)
        
        # loc history tensor to be concatenated with g_t
        L = torch.zeros((x.shape[0], 2 * 4)).cuda()
        
        # process minibatch
        for t in range(self.timesteps):
            g_t = self.glimpse_module(x, l_t_prev)
            
#            g_t = self.avgpool(torch.cat(g_t,dim=-1)).squeeze() #averaging fov and per
#            g_t = self.avgpool(g_t[0]).squeeze() #foveal classification only for a test 21feb
#            g_t = self.avgpool(sum(g_t)).squeeze() #summing fov and per
            g_t = sum(g_t) #no/delayed avgpool
            
            #append fixation history
            if t!=0: L[:, (t-1)*2:(t-1)*2+2] = l_t_prev.detach()
            g_t = torch.cat([g_t.flatten(1),L], dim=1)
            
            h_t = self.rnn(g_t)
            h_t_flat = h_t.flatten(1,-1)
            log_pi, l_t = self.locator(h_t)
            b_t = self.baseliner(h_t_flat).squeeze(0) 
            y_p = self.classifier(h_t_flat)
            if self.hardcoded_locs:
                l_t_prev = self.retina.to_egocentric_action(guide(y_locs, t+1) - self.retina.fixation)
            else: 
                l_t_prev = l_t
            
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
        """
        Computes the hybrid loss (with intermediate classification loss)
        """
        log_probas, log_pi, baselines, y, locs, y_locs = args
        # extract prediction
        prediction = torch.max(log_probas[:,-1], 1)[1].detach()
        
        # compute reward
        #TODO: cleanup baselines shape, starting from modules.py
        baselines = baselines.squeeze()
        R = (prediction == y).float() 
        R = R.unsqueeze(1).repeat(1, self.timesteps) 
        adjusted_R = R - baselines.detach()
    
        # intermediate classification supervision
        loss_classify = F.nll_loss(log_probas[:,-1,:], y)
#        for i in range(1, self.timesteps):
#            loss_classify += F.nll_loss(log_probas[:,i,:], y)
        
        loss_classify = loss_classify/self.timesteps #average
        if self.hardcoded_locs: 
            return loss_classify, torch.tensor(0), torch.tensor(0), torch.tensor(0)
        
        ## Key feature distance loss. Experimental.
        S = RAM_sensor() #recycle code
        S.width = self.retina.width
#        dist = torch.zeros_like(locs[:,:-1,0]) #final loc is never used
        targets = torch.zeros_like(locs[:,:-1]) #final loc is never used
        abs_locs = torch.zeros_like(locs[:,:-1])
        
        #convert locs to absolute coordinates, compute distance to closest feature
        y_locs[np.where(y_locs[...,-1]==0)] = torch.tensor([2000.0,2000.0,0.0]) #hack away invisible features
        y_locs = y_locs[:,:,:2].cuda()
        b_dim = range(len(y_locs))
        for timestep in range(abs_locs.shape[1]):
            if timestep==0: abs_locs[:,timestep,:] = S.to_exocentric(self.T, locs[:,0,:])
            else:
                S.fixation = abs_locs[:,timestep-1,:]
                abs_locs[:,timestep,:] = S.to_egocentric(self.T, locs[:,timestep,:])
#            dist[:,timestep] = torch.min(torch.cdist(abs_locs[:,timestep,:].unsqueeze(1), y_locs), 2)[0].squeeze() #one liners are fun
            closest_ind = torch.min(torch.cdist(abs_locs[:,timestep,:].unsqueeze(1), y_locs), 2)[1].squeeze()
            #compute optimal action
            targets[:, timestep] = S.to_egocentric_action(y_locs[b_dim, closest_ind,:] - abs_locs[:,timestep,:])
            
        #compute optimal action divergence loss
        loss_dist = F.mse_loss(locs[:,:-1], targets)
            
#        #compute distance loss
#        a = 10
#        b = 50
#        loss_dist = torch.mean(torch.sum(F.relu((dist-a)/b),dim=1),dim=0)
        
        # reinforce loss
        loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
        loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
        loss_baseline = F.mse_loss(baselines, R)
        return loss_classify, loss_reinforce, loss_baseline, loss_dist 

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
    def __init__(self, retina=None, pretrained=True):
        super(ff_r18, self).__init__()
        self.require_locs = True #assumes x consists of (image, locations)
        self.retina = retina
        self.n_patches = retina.k if retina is not None else 1
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
            centroid = (centroid - 250)/250 #normalize
            l_t_prev[i] = centroid
        
        #foveate, process patches individually, average features
        if self.retina is not None:
            phi = self.retina.foveate_exo(x[0], l_t_prev)
            out = []
            for i in range(self.n_patches):
                x = phi[:,i,:,:,:] #select patch
                x = self.resnet18(x)
                x = self.pool(x)
                out.append(x)
            #torch.mean might have gradient problems, use pool instead
            x = self.pool(torch.cat(out, dim=-1))
        else: 
            x = x[0].cuda()
            x = self.resnet18(x)
            x = self.pool(x)
            
        #average patches and predict
        x = F.relu(self.fc1(x.flatten(1)))
        pred = F.log_softmax(self.fc2(x), dim=1)
        pred = pred.unsqueeze(1) #timestep dimension

        return pred, 0, 0, 0
    
    def loss(self, *args):
        log_probas, log_pi, baselines, y, locs, y_locs = args
        loss_classify = F.nll_loss(log_probas.squeeze(1), y)
        return loss_classify
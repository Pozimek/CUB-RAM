#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:29:27 2021

A Tester class for evaluating the performance and behaviours of RAM models.

@author: piotr
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from torch.utils.tensorboard import SummaryWriter
from random import randint
from utils import showArray
from modules import RAM_sensor
from model import guide


def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False)
    return rect


class Tester(object):
    def __init__(self, C, data_loader, model):
        self.C = C
        
        # data loader params
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.num_train = len(self.data_loader.dataset)
        self.num_valid = len(self.data_loader.dataset.test_data)
        self.batch_size = self.C.training.batch_size
        
        # model
        self.model = model
        if C.gpu: self.model.cuda()
        
        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        
        #TODO: consider using Tensorboard for multiple image vis
    
    def get_random_batch(self, eval_set):
        # Prepare model and choose data split
        self.model.eval()
        if eval_set: 
            self.data_loader.dataset.test()
            num = self.num_valid
        else: 
            self.data_loader.dataset.train()
            num = self.num_train
        
        # Sample a random batch
        i = randint(1, num//self.batch_size)
        it = iter(self.data_loader)
        for _ in range(i):
            x, y, y_locs = it.next()
        y = y.cuda()
        
        return x, y, y_locs
    
    def visualize_sensor(self, eval_set = False):
        """
        Quick vis for sensor outputs
        """
        x, y, y_locs = self.get_random_batch(eval_set)
        sensor = self.model.retina
        sensor.reset()
        T = torch.tensor((500,500)).cuda()
        G = guide(y_locs, 0)
        locs = sensor.to_exocentric_action(T, G)
        sensor_out = sensor.foveate_exo(x, locs).cpu().numpy()
        B = 0
        
        fov = sensor_out[B,0,:,:,:]
        per = sensor_out[B,1,:,:,:]
        
        for im in [x.cpu().numpy()[0], fov, per]:
            im = np.moveaxis(im, 0, -1) - im.min()
            im = im/im.max()
            plt.figure(figsize=(5,5))
            plt.axis('off')
            plt.imshow(im)
            plt.show()
            print(locs[0])         #y, x
    
    def occlusion_check(self, eval_set = False):
        """
        Tests for occluded parts in the dataset.
        """
        # Choose data split
        if eval_set: 
            self.data_loader.dataset.test()
            num = self.num_valid
        else: 
            self.data_loader.dataset.train()
            num = self.num_train
            
        # Iterate
        O = False
        it = iter(self.data_loader)
        for x, y, y_locs in it:
            if O is False: O = y_locs[:,:,-1]
            else: O = torch.cat((O,y_locs[:,:,-1]))
        
        return num-torch.sum(O,dim=0)
    
    def visualize_gaze(self, eval_set = False):
        """
        Visualizes model's gaze path on a random batch of training images.
        """
        # Sample a batch
        x, y, y_locs = self.get_random_batch(eval_set)
        if self.model.require_locs: x = (x, y_locs)
        
        # Feed through model
        log_probas, locs, log_pi, baselines = self.model(x)
        correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
        
        # Compute absolute gaze locations
        S = RAM_sensor()
        S.width = self.model.retina.width
        T = torch.tensor((500,500)).cuda()
        abs_locs = torch.zeros_like(locs[:,:-1])
        
        for timestep in range(locs.shape[1]-1):
            if timestep==0: abs_locs[:,timestep,:] = S.to_exocentric(T, locs[:,0,:])
            else:
                S.fixation = abs_locs[:,timestep-1,:]
                abs_locs[:,timestep,:] = S.to_egocentric(T, locs[:,timestep,:])
        
        # Retrieve to CPU
        if self.model.require_locs: x = x[0].cpu().numpy()
        else: x = x[0].cpu().numpy()
        abs_locs = abs_locs.detach().cpu().numpy()
        
        for ind, im in enumerate(x):
            # Normalize
            im = np.moveaxis(im, 0, -1) - im.min()
            im = im/im.max()
            
            # Visualize sensor fov
            if ind==0:
                fov_rect = bounding_box(250, 250, self.C.RAM.foveal_size)
                per_rect = bounding_box(250, 250, self.C.RAM.foveal_size * self.C.RAM.scaling)
                fig, ax = plt.subplots(1, figsize=(8,8))
                ax.axis('off')
                ax.imshow(im)
                ax.add_patch(fov_rect)
                ax.add_patch(per_rect)
                plt.show()
                
            im_locs = abs_locs[ind]
            print(locs[ind])
            print(im_locs)
            
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.imshow(im)
            
            # Plot gaze path on top of images
            plt.plot(im_locs[:,1], im_locs[:,0], c='g')
            plt.scatter(im_locs[0,1], im_locs[0,0], c='r', s=60)
            plt.scatter(x=im_locs[1:,1], y=im_locs[1:,0], s=60)
            
            # Classification accuracy indicator
            colour = 'r' if not correct[ind] else 'g'
            marker = 'X' if not correct[ind] else 'P'
            plt.scatter(475, 25, s = 200, c = colour, marker = marker)
            plt.show()
            
    def visualize_ylocs(self, eval_set = False):
        """
        
        """
        # Sample a batch
        x, y, y_locs = self.get_random_batch(eval_set)
        if self.model.require_locs: x = (x, y_locs)
        
        # Feed through model
        log_probas, locs, log_pi, baselines = self.model(x)
#        correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
        
        # Compute absolute gaze locations
        S = RAM_sensor()
        S.width = self.model.retina.width
        T = torch.tensor((500,500)).cuda()
#        abs_locs = torch.zeros_like(locs[:,:-1])
        
        locs2 = []
        l_t_prev = S.to_exocentric_action(T, guide(y_locs, 0))
        S.fixation = S.to_exocentric(T, l_t_prev)
        locs2.append(l_t_prev)
        
        for t in range(5):
            l_t_prev = S.to_egocentric_action(guide(y_locs, t+1) - S.fixation)
            S.fixation = S.to_egocentric(T, l_t_prev)
            locs2.append(l_t_prev)
        
        locs2 = torch.stack(locs2).transpose(1, 0)
        
        print(locs2.shape, locs.shape)
#        for i in range(len(locs2)):
        i = 0
        print("hand: ", locs2[i])
        print("model: ", locs[i])
        print("y_locs: ", y_locs[i])
        abs_locs = torch.zeros_like(locs)
        abs_locs[:,0,:] = S.to_exocentric(T, locs[:,0,:])
        print("abs_locs: ", abs_locs[i])
        
#        fix_locs = torch.zeros_like(locs)
#        G = guide(y_locs, 0)
#        
#        fix_locs[:,0,:] = S.to_exocentric_action(T,)
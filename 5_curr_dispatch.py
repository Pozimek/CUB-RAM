#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 23:49:39 2022

Chapter 5 (active vision attention, ch4 in latex) experiment dispatch script.

Experiment focus: evaluating curriculum learning for attention
WW_LSTM vs SpaCat
Regular vs curriculum

Curriculum is implemented by varying sensor parameters, i.e. gradually 
decreasing foveal FOV and peripheral resolution.

@author: piotr
"""

import torch
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
 
from model import RAM_ch5, SpaCat
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import WW_LSTM, baseline_network, location_network
from modules import WW_module, WhereMix
from ch5_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed

def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    
    # Build name based on architecture variant for current experiment
    config.name = "ch5curr"
    config.name += "-s{}".format(config.seed)
    config.name += "mem{}".format(config.vars.mem)
    config.name += "curr{}".format(config.vars.curr)
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(config.seed) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    # Sensor (and curriculum params for easy reference)
    stage = 0 if config.vars.curr else 3
    fov_sizes = [37*4, 37*3, 37*2, 37]
    scaling = [370/size for size in fov_sizes]
    retina = crude_retina(fov_sizes[stage], config.retina.n_patches, 
                          scaling[stage], config.gpu, clamp = True)
    
    # Build model
    timesteps = 5
    if int(config.vars.mem) == 0: #WW_LSTM
        FE = ResNetEncoder(retina.out_shape)
        where_dim = 5
        WWfov = WW_module(FE.out_shape, where_dim)
        WWper = WW_module(FE.out_shape, where_dim)
        mem_in = WWfov.out_shape[:-1] + (WWfov.out_shape[-1]*2,)
        downstream_in = mem_in.numel()
        
        # Memory
        memory = WW_LSTM(mem_in, FE.out_shape[0], where_dim*2, gate_op=WhereMix, 
                         in_op=WhereMix)
        
        # Classifier
        classifier = classification_network_short(downstream_in, 200)
        
        # Location and baseliner (lots of params!!!)
        loc_net = location_network(downstream_in, 512, config.retina.std)
        baseliner = baseline_network(downstream_in, 512) 
        
        model = RAM_ch5(config.name, retina, FE, WWfov, WWper, memory, 
                        classifier, loc_net, baseliner, gpu=True)
    
    elif config.vars.mem == 1: #SpaCat
        FE_in_shape = (retina.out_shape[0], retina.out_shape[1]*2, 
                       retina.out_shape[2]*timesteps)
        FE = ResNetEncoder(FE_in_shape)
        where_dim = 10
        WW = WW_module(FE.out_shape, where_dim)
        downstream_in = WW.out_shape.numel()
        
        baseliner = baseline_network(downstream_in, 512) 
        loc_net = location_network(downstream_in, 512, config.retina.std)
        classifier = classification_network_short(downstream_in, 200)
        model = SpaCat(config.name, retina, FE, baseliner, loc_net, WW,
                       classifier)
    
    model.set_timesteps(timesteps) #more timesteps for attention eval
    trainer = Trainer(config, loader, model)
    trainer.train(curriculum = config.vars.curr)
    
if __name__ == '__main__':
    for seed in [1,9,919]:
        for mem in [0,1]: # 0:WW_LSTM, 1:SpaCat
            for curr in [0,1]:
                config = get_ymlconfig('./5_curr_dispatch.yml')
                config.seed = seed
                
                config.vars.mem = mem 
                config.vars.curr = curr
                
    #            config.training.resume = True
                main(config)
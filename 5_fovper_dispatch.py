#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 19:21:56 2022

Chapter 5 (active vision attention, ch4 in latex) experiment dispatch script.

Experiment focus: evaluating foveal+peripheral vs foveal-only classification
on hardcoded policies.
WW_LSTM vs SpaCat
fovper vs fov

@author: piotr
"""
import torch
import torch.nn as nn

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
 
from model import RAM_ch5, SpaCat
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import WW_LSTM, baseline_network, location_network
from modules import WW_module, WhereMix
from ch3_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed


def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    
    # Build name based on architecture variant for current experiment
    config.name = "ch5fovper"
    config.name += "-s{}".format(config.seed)
    config.name += "mem{}".format(config.vars.mem)
    config.name += "fovper{}".format(config.vars.fovper)
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(config.seed) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    # Sensor (and curriculum params for easy reference)
    config.retina.n_patches = 2 if config.vars.fovper else 1
    retina = crude_retina(config.retina.foveal_size, config.retina.n_patches, 
                          config.retina.scaling, config.gpu, clamp = False)
    
    # Build model
    timesteps = 3
    if int(config.vars.mem) == 0: #WW_LSTM
        FE = ResNetEncoder(retina.out_shape)
        where_dim = 5
        WWfov = WW_module(FE.out_shape, where_dim)
        WWper = WW_module(FE.out_shape, where_dim) if config.vars.fovper \
                else nn.Identity()
                
        mem_in = WWfov.out_shape[:-1] + (WWfov.out_shape[-1] * 
                                            config.retina.n_patches,)
        downstream_in = mem_in.numel()
        
        # Memory
        memory = WW_LSTM(mem_in, FE.out_shape[0], where_dim *
                             config.retina.n_patches, gate_op=WhereMix, 
                             in_op=WhereMix)
        
        # Classifier
        classifier = classification_network_short(downstream_in, 200)
        
        # Location and baseliner
        loc_net = location_network(downstream_in, 512, config.retina.std)
        baseliner = baseline_network(downstream_in, 512) 
        
        model = RAM_ch5(config.name, retina, FE, WWfov, WWper, memory, 
                        classifier, loc_net, baseliner, gpu=True, 
                        hardcoded = True)
    
    elif config.vars.mem == 1: #SpaCat
        FE_in_shape = (retina.out_shape[0], 
                       retina.out_shape[1]*config.retina.n_patches, 
                       retina.out_shape[2]*timesteps)
        FE = ResNetEncoder(FE_in_shape)
        where_dim = 5 * config.retina.n_patches
        WW = WW_module(FE.out_shape, where_dim)
        downstream_in = WW.out_shape.numel()
        
        baseliner = baseline_network(downstream_in, 512) 
        loc_net = location_network(downstream_in, 512, config.retina.std)
        classifier = classification_network_short(downstream_in, 200)
        model = SpaCat(config.name, retina, FE, baseliner, loc_net, WW,
                       classifier, hardcoded = True)
    
    model.set_timesteps(timesteps) #only 3 timesteps for fovper vs fovonly
    trainer = Trainer(config, loader, model)
    trainer.train()
    
if __name__ == '__main__':
    for seed in [1,9,919]:
        for mem in [0,1]: # 0:WW_LSTM, 1:SpaCat
            for fovper in [0,1]: # 0:fovonly, 1:fovper
                config = get_ymlconfig('./5_fovper_dispatch.yml')
                config.seed = seed
                
                config.vars.mem = mem 
                config.vars.fovper = fovper 
                
    #            config.training.resume = True
                main(config)
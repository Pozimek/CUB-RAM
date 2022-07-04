#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 03:35:13 2022

Chapter 4 (active vision memory, ch3 in latex) experiment dispatch script.

Experiment focus: evaluating different recurrent memory variants.
WW_LSTM vs WW_RNN vs LSTM vs RNN

Using the multiple instances MHL method, ie trains a separate instance for 
every timestep.

@author: piotr
"""

import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
 
from model import RAM_ch4
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import lstm_, WW_LSTM, vanilla_rnn, WW_rnn
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
    config.name = "ch4mem"
    config.name += "-s{}".format(config.seed)
    config.name += "VAR{}".format(int(config.vars.variant))
    config.name += "T{}".format(int(config.vars.timesteps))
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(config.seed) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    # Sensor and FE
    retina = crude_retina(config.retina.foveal_size, config.retina.n_patches, 
                          config.retina.scaling, config.gpu, clamp = False)
    FE = ResNetEncoder(retina.out_shape)
    
    # WW stuff
    where_dim = 10
    if config.vars.variant <= 1: #WW variants
        WWfov = WW_module(FE.out_shape, where_dim)
        WWper = WW_module(FE.out_shape, where_dim)
        mem_in = WWfov.out_shape[:-1] + (WWfov.out_shape[-1],)
        mem_shape = mem_in
        classifier_in = mem_shape.numel()
    else: #non-WW variants
        WWfov = nn.AdaptiveAvgPool2d((1,1))
        WWper = nn.AdaptiveAvgPool2d((1,1))
        mem_in = FE.out_shape[0]
        mem_shape = FE.out_shape[0] #small, for closer param # to WW variants
        classifier_in = mem_shape
    
    # Memory
    mem_variants = [partial(WW_LSTM, mem_in, FE.out_shape[0], where_dim, 
                            gate_op=WhereMix, in_op=WhereMix),
                    partial(WW_rnn, mem_in, mem_shape),
                    partial(lstm_, mem_in, mem_shape),
                    partial(vanilla_rnn, mem_in, mem_shape)]
    
    memory = mem_variants[config.vars.variant]()
    
    # Classifier
    classifier = classification_network_short(classifier_in, 200)
    
    model = RAM_ch4(config.name, retina, FE, WWfov, WWper, memory, classifier,
                    gpu=True)
    model.set_timesteps(config.vars.timesteps)
    
    trainer = Trainer(config, loader, model)
    trainer.train()
    
if __name__ == '__main__':
    for seed in [1,9,919]:
        for variant in range(4):
            for timesteps in [1,2,3]:
                config = get_ymlconfig('./4_mem_dispatch.yml')
                config.seed = seed
                # 0:WW_LSTM, 1:WW_RNN, 2:LSTM 3:RNN
                config.vars.variant = variant
                config.vars.timesteps = timesteps
                
    #            config.training.resume = True
                main(config)
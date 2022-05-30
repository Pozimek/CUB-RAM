#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:48:39 2022

Chapter 4 (active vision memory, ch3 in latex) experiment dispatch script.

Experiment 2: evaluating different recurrent memory variants.
WW_LSTM vs WW_RNN vs LSTM vs RNN

Notes:
- These would be done more justice if you varied the hidden state size.
- More justice: different fixloc sets to exemplify problem scenarios
- More justice: MHL3 with and without randoms inbetween, exemplify L fixlocs
- More justice: warying where_dim
- Old WW stuff catted fovper along what_dim, now you're doing where_dim

@author: piotr
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.autograd import Variable
 
from model import RAM_ch4
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import lstm_, WW_LSTM, vanilla_rnn, WW_rnn
from modules import WW_module
from ch3_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed

def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    
    # Build name based on architecture variant for current experiment
    config.name = "ch4ii"
    config.name += "-s{}".format(config.seed)
    config.name += "VAR{}".format(int(config.vars.strategy))
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(303) # separate seed for more control
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
    if config.vars.variant <= 1:
        WWfov = WW_module(FE.out_shape, where_dim)
        WWper = WW_module(FE.out_shape, where_dim)
    else:
        WWfov = nn.AdaptiveAvgPool2d((1,1))
        WWper = nn.AdaptiveAvgPool2d((1,1))
    
    # Memory
    mem_variants = [WW_LSTM(mem_in, mem_what, mem_where, gate_op=WhereMix, in_op=WhereMix),
                    WW_rnn(),
                    lstm_(),
                    vanilla_rnn()]
    memory = mem_variants[config.vars.variant]
    
    # Classifier
    classifier = classification_network_short(memory.out_shape, 200)
    
    
    model = RAM_ch4(config.name, retina, FE, WWfov, WWper, memory, classifier,
                    gpu=True)
    
    trainer = Trainer(config, loader, model)
    trainer.train()
    
if __name__ == '__main__':
    for seed in [1,9,919]:
        for variant in range(4):
            config = get_ymlconfig('./4iii_dispatch.yml')
            config.seed = seed
            # 0:WW_LSTM, 1:WW_RNN, 2:LSTM 3:RNN
            config.vars.variant = variant
            
#            config.training.resume = True
            main(config)
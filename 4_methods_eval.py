#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 19:28:37 2022

Chapter 4 (activevision memory, ch3 in altex) evaluation script.

Experiment focus: evaluating different memory training methods.

- Baseline model: vanilla lstm.
- Training methods tested:
"Train T3" vs 
"Train T123" (shares results with Train T3) vs
"Train T123 + curriculum preloading" vs
"Train T3 + Intermediate/greedy classification" vs
BN-LSTM with the first method.

- The first two methods share results.

0[3], 1[3], 2, 3
26.3, 52.7, 55.6
21.2, 48.9, 50.4
37.8, 54.2, 51.9
30.8, 51.5, 54.2

@author: piotr
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
 
from model import RAM_ch4
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import lstm_, bnlstm
from ch3_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed

def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    
    # Build name based on architecture variant for current experiment
    config.name = "ch4methods"
    config.name += "-s{}".format(config.seed)
    config.name += "VAR{}".format(int(config.vars.variant))
    
    #name wasn't set correctly, currT3 logs&ckpts are appended w/ 'T1T2T3'
    if config.vars.variant == 1:
        config.name += "T1T2T3"
    else:
        config.name += "T{}".format(config.vars.timesteps)
    
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
    
    # WW stuff for RAM_ch4 compatibility
    WWfov = nn.AdaptiveAvgPool2d((1,1))
    WWper = nn.AdaptiveAvgPool2d((1,1))
    mem_in = FE.out_shape[0]#*2  fovonly edit
    mem_shape = FE.out_shape[0] #small, for closer param # to WW variants
    classifier_in = mem_shape
    
    # Memory
    memory = bnlstm(mem_in, mem_shape) if config.vars.variant == 3 \
                else lstm_(mem_in, mem_shape)
    
    # Classifier
    classifier = classification_network_short(classifier_in, 200)
    
    model = RAM_ch4(config.name, retina, FE, WWfov, WWper, memory, classifier,
                    gpu=True)
    
    # change config name to avoid overwriting old logs
    load_name = config.name
    for t in [1,2,3]:
        model.set_timesteps(t)
        config.name = load_name + "evalT{}".format(t)
        trainer = Trainer(config, loader, model)
        trainer.load_checkpoint(name = load_name, best=True)
        trainer.validate(0)
    
if __name__ == '__main__':
    for seed in [919]:
        for variant in range(4):
            for timesteps in [3]:
                config = get_ymlconfig('./4_methods_dispatch.yml')
                config.seed = seed
                # 0:T123, 1:T123+curr, 2:T3+intermediate 3:T3+BN-LSTM
                config.vars.variant = variant
                config.vars.timesteps = timesteps
                
    #            config.training.resume = True
                main(config)
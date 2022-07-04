#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 02:05:22 2022

Chapter 4 (active vision memory, ch3 in latex) experiment dispatch script.

Experiment focus: evaluating different memory training methods.

- Baseline model: vanilla lstm.
- Training methods tested:
"Train T3" vs 
"Train T123" (shares results with Train T3) vs
"Train T123 + curriculum preloading" vs
"Train T3 + Intermediate/greedy classification" vs
BN-LSTM with the first method.

All methods evaluated at T123, with the 2nd having a separate model trained and
evaluated for each timestep.

Goal: find the training method most suitable for evaluating the network with
the COV method.

The network should ideally be resilient to which timestep it classifies at in a
way that doesn't require loading different model weights, so that it can handle
a less restricted variety of scenarios. Another work in the lit achieved this 
by having a different classifier FC for each timestep (DT-RAM IIRC). But even 
if different model weights have to be loaded the networks' perf at each 
timestep can still be evaluated.

NOTES:
- Unlike previous experiments this one is ran only on one seed. The reason is
time limitations. Note this in dissertation.
- The first two methods share results, so the first one is not having a 
dedicated run.
- 0[T3], 2 and 3 will require an eval script to test at T123.

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
    base_name = config.name
    
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
    
    intermediate = config.vars.variant == 2
    model = RAM_ch4(config.name, retina, FE, WWfov, WWper, memory, classifier,
                    gpu=True, intermediate_loss = intermediate)
    
    # Conditional curriculum etc.
    T = [[config.vars.timestep], [1,2,3], [3], [3]]
    for timestep in T[config.vars.variant]:
        model.set_timesteps(timestep)
        #name wasn't set previously, currT3 logs&ckpts are appended w/ 'T1T2T3'
        config.name = base_name + "T{}".format(timestep) 
        trainer = Trainer(config, loader, model)
        trainer.train()
    
if __name__ == '__main__':
    for seed in [919]:
        for variant in range(4):
            T = [1,2,3] if variant == 0 else [-1]
            for timestep in T:
                config = get_ymlconfig('./4_methods_dispatch.yml')
                config.seed = seed
                # 0:T123, 1:T123+curr, 2:T3+intermediate 3:T3+BN-LSTM
                config.vars.variant = variant
                config.vars.timestep = timestep
                
    #            config.training.resume = True
                main(config)
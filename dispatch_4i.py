#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 01:31:24 2022

Chapter 4 (active vision memory, ch3 in latex) experiment dispatch script.

Experiment 0: evaluating 5 selected fixation locations.

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
 
from model import ff_aggregator_ch4
from modules import crude_retina, classification_network_short, ResNetEncoder 
from modules import proprioceptive_FE, mult_merge, lstm_, location_network
from modules import baseline_network, retinocortical_sensor
from ch3_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed

def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    
    # Build name based on architecture variant for current experiment
    config.name = "ch4i"
    config.name += "-s{}".format(config.seed)
    config.name += "FIXLOC{}".format(int(config.vars.fixloc))
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(303) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    retina = crude_retina(config.retina.foveal_size, config.retina.n_patches, 
                          config.retina.scaling, config.gpu, clamp = False)
    FE = ResNetEncoder(retina.out_shape)
    model = ff_aggregator_ch4(config.name, retina, FE, avgpool = True,
                              strategy=int(config.vars.strategy), 
                              fixation_set = [config.vars.fixloc])
    
    trainer = Trainer(config, loader, model)
    trainer.train()
    
if __name__ == '__main__':
    for seed in [1,9,919]:
        for fixation in range(5):
            #left eye, left leg, breast, beak, tail
            config = get_ymlconfig('./4i_dispatch.yml')
            config.seed = seed
            config.vars.fixloc = fixation
        
#            config.training.resume = True
            main(config)
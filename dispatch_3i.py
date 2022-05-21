#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:08:30 2022

Chapter 3 (Lit review and framework. ch2 in latex) experiment dispatch script.

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
 
from model import RAM_ch3
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
    config.name = "ch3"
    config.name += "-s{}".format(config.seed)
    config.name += "EGO{}".format(int(config.vars.egocentric))
    config.name += "PROP{}".format(int(config.vars.proprioception))
    config.name += "HC{}".format(int(config.vars.hardcoded))
    config.name += "NRW{}".format(int(config.vars.narrow_fov))
    config.name += "RET{}".format(int(config.vars.sw_retina))
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(303) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    # fov control
    if not config.vars.narrow_fov:
        config.retina.foveal_size = 224
        config.retina.n_patches = 1
    
    # sensor
    clamp = config.vars.egocentric and not config.vars.hardcoded #seed=1 didn't run this check and was always set to False
    if config.vars.sw_retina:
        retina = retinocortical_sensor(clamp = clamp)
    else:
        retina = crude_retina(config.retina.foveal_size, config.retina.n_patches, 
                              config.retina.scaling, config.gpu, clamp = clamp)
    
    FE = ResNetEncoder(retina.out_shape)
    FE.out_shape = torch.Size((FE.out_shape[0]*retina.k,) + FE.out_shape[1:])
    dummy = not config.vars.proprioception # whether to use placeholder modules
    prop_FE = proprioceptive_FE(2, FE.out_shape, dummy)
    FE_merge = mult_merge(FE.out_shape, prop_FE.out_shape, dummy)
    
    mem_size = 1024
    memory = lstm_(FE_merge.out_shape[0], mem_size)
    
    classifier = classification_network_short(mem_size, 200)
    loc_net = location_network(mem_size, 512, config.retina.std)
    baseliner = baseline_network(mem_size, 512)
    
    model = RAM_ch3(config.name, retina, FE, prop_FE, FE_merge, memory, loc_net, 
                    classifier, baseliner, config.vars.egocentric,
                    config.vars.hardcoded, config.gpu)
    
    trainer = Trainer(config, loader, model)
    trainer.train()
    
if __name__ == '__main__':
    for seed in [1,9,919]:
                        #ego,  prop,   hc, narrow, sw
        for variant in [#[True, True, False, True, False]#,   #1a. ego prop RL
                        #[False, True, False, True, False],  #1b. exo prop RL
                        #[True, True, True, True, False],    #1c. ego prop HC (testing ego prop_FE w/ good fixations)
                        #[False, True, True, True, False],   #1d. exo prop HC (testing exo prop_FE w/ good fixations)
                        #[True, False, False, False, False], #2a. ego wide RL
                        #[True, False, False, True, False],  #2b. ego narrow RL
                        #[True, False, True, False, False],  #2c. ego wide HC
                        #[True, False, True, True, False]#,   #2d. ego narrow HC
                        [True, False, True, True, True]     #3. retina HC
                        ]: #run 2 more seeds after this is done
            config = get_ymlconfig('./3i_dispatch.yml')
            config.seed = seed
            config.vars.egocentric = variant[0]
            config.vars.proprioception = variant[1]
            config.vars.hardcoded = variant[2]
            config.vars.narrow_fov = variant[3]
            config.vars.sw_retina = variant[4]
        
#            config.training.resume = True
            main(config)
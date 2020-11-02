#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:32:13 2020

Curriculum learning test.

@author: piotr
"""

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
from utils import get_ymlconfig, showTensor
from trainer import Trainer
from CUB_loader import CUBDataset, collate_pad
from model import CUBRAM_baseline, ff_r18
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler


def main(config):
    torch.manual_seed(config.seed)
    kwargs = {}
    if config.gpu:
        torch.cuda.manual_seed(config.seed)
        kwargs = {'num_workers': config.training.num_workers, 
                  'pin_memory': True}
    transform = Compose([ToTensor(), 
                         Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])])
    
    dataset = CUBDataset(transform = transform)
    
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
    

    # Part 1 of curriculum training
    model = ff_r18()
    trainer = Trainer(config, loader, model)
    trainer.train()
    
    # Extract new resnet18 weights
    weights = model.resnet18.state_dict()
    
    # Delete old training objects
    
    del(trainer)
    del(loader)
    del(dataset)
    
    # Part 2 of curriculum training
    dataset2 = CUBDataset(transform = transform)
    loader2 = torch.utils.data.DataLoader(
            dataset2, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset2), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
    
    model2 = CUBRAM_baseline(config.name, config.RAM.foveal_size, 
                            config.RAM.n_patches, config.RAM.scaling, 
                            config.RAM.std, config.gpu)
    model2.sensor.resnet18.load_state_dict(weights)
    del(model)
    
    trainer2 = Trainer(config, loader2, model2)
    trainer2.train()
    

if __name__ == '__main__':
    config = get_ymlconfig('./config.yml')
    main(config)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:49:04 2021

Feedforward model training script.

@author: piotr
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler

from utils import get_ymlconfig, showTensor
import numpy as np
from CUB_loader import CUBDataset, collate_pad
from trainer import Trainer
from model import ff_r18
from modules import retinocortical_sensor, crude_retina, ResNet18_module

def main(config):
    config.seed = 419 #reseed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
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
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                          config.RAM.scaling, config.gpu)
    namestring = "retina-RESNET-reseedada{}"
    
    for steps in range(1,6):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
        feature_extractor = ResNet18_module(blocks=4, maxpool=False)
        model = ff_r18(retina, feature_extractor, [1024], steps-1)
        print("Now training timestep {} ResNet.\n".format(steps))
        config.name = namestring.format(steps)
        trainer = Trainer(config, loader, model)
        trainer.train()

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
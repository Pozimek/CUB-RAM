#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:07:46 2020

Training dispatch script, CUB-RAM.
@author: piotr
"""
#TODO: Figure out how to structure trainer/tester objects wrt research scripts
# ANS: If models will have different outputs or different loss handling then 
# the functionality will be implemented in the model class - the Trainer will call
# those functions. Or if the different training regimes will be tested then the 
# functionality will be passed as a param to the trainer.
#TODO: flatten and clean up config file structure

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
from utils import get_ymlconfig, showTensor
from trainer import Trainer
from CUB_loader import CUBDataset, collate_pad
from model import CUBRAM_baseline
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
    
    model = CUBRAM_baseline(config.name, config.RAM.foveal_size, 
                            config.RAM.n_patches, config.RAM.scaling, 
                            config.RAM.std, config.gpu)
    trainer = Trainer(config, loader, model)
    trainer.train()

if __name__ == '__main__':
    config = get_ymlconfig('./config.yml')
    main(config)
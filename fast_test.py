#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:17:40 2020

A script for testing already trained models

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
from CUB_loader import CUBDataset, collate_pad
from trainer import Trainer
from model import CUBRAM_baseline, ff_r18, RAM_baseline
from modules import retinocortical_sensor, crude_retina


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
    
    config.tensorboard = False
    
#    retina = retinocortical_sensor()
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                          config.RAM.scaling, config.gpu)
    
    config.name = "curriculum-r18-crude"
    
    w_path = os.path.join(config.ckpt_dir, config.name+"_best.pth.tar")
    r18 = ff_r18(retina=retina, pretrained=False)
    r18.load_state_dict(torch.load(w_path)['model_state'])
    r18_trainer = Trainer(config, loader, r18)
    
    r18_trainer.validate(0)
    
if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
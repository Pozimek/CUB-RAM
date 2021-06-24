#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:40:09 2020

Retina training script.

@author: piotr
"""
import os
import matplotlib as mpl
import numpy as np
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
import random
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler

from utils import get_ymlconfig, showTensor
from CUB_loader import CUBDataset, collate_pad
from trainer import Trainer
from tester import Tester
from model import RAM_baseline
from modules import retinocortical_sensor, crude_retina, ResNet18_module, DenseNet_module
from modules import FC_RNN, lstm, bnlstm, lstm_, GRU, laRNN

def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    kwargs = {}
    if config.gpu: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    # 0 - Prepare modules
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                          config.RAM.scaling, config.gpu, clamp = False)
    memory = lstm_
   
    namestring = "retina-RAMfix-fovpercanonical-{}"
    config.name = namestring.format(0)
    
    # 2 - Train RAM, iteratively increasing number of timesteps
    for steps in range(1,6):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
        feature_extractor = ResNet18_module(blocks=4, maxpool = False)
#        feature_extractor = DenseNet_module(layers = (58,), pretrained = False,
#                                            maxpool=False, Tavgpool=False)
        model = RAM_baseline(config.name, config.RAM.std, retina, 
                             feature_extractor, memory, 0, config.gpu, 
                             fixation_set = "canonical")
        print("Now training {}-step RAM.\n".format(steps))
        model.set_timesteps(steps)
        config.name = namestring.format(steps)
        trainer = Trainer(config, loader, model)
        trainer.train()
    #TODO config file restructure
    #TODO config class seems to triplicate dicts

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
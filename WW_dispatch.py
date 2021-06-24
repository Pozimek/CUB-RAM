#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 20:05:56 2021

@author: piotr
"""
import os
import numpy as np
import torch

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

from model import WW_RAM
from modules import crude_retina, ResNet18_module, DenseNet_module
from trainer import Trainer
from CUB_loader import CUBDataset, collate_pad
from utils import get_ymlconfig

def main(config):
#    config.seed = 42 #XXX reseed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
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
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                              config.RAM.scaling, config.gpu, clamp = False)
    namestring = "retina-WWRAMfix-onewhere-{}"
    config.name = namestring.format(0)
    
    for steps in range(1,4):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
#        feature_extractor = DenseNet_module(pretrained = True, maxpool=True, Tavgpool=False)
        feature_extractor = ResNet18_module(blocks=4, maxpool=False, stride=True)
        model = WW_RAM(config.name, config.RAM.std, retina, feature_extractor, 
                    config.gpu, fixation_set = "MHL3")
        
        print("Now training {}-step WW_RAM.\n".format(steps))
        model.set_timesteps(steps)
        config.name = namestring.format(steps)
        trainer = Trainer(config, loader, model)
        trainer.train()

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
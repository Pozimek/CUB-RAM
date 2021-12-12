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
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.autograd import Variable
 
from model import WW_RAM
from modules import crude_retina, ResNet18_module, DenseNet_module
from trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig

def main(config):
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
    dataset = CUBDataset(transform = transform, shuffle=True)
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                              config.RAM.scaling, config.gpu, clamp = False)
    namestring = "retina-WWRAMfix-locattentionv35-{}s{}"
    config.name = namestring.format(0, config.seed)
    
    for steps in range(5,6):
        generator = torch.Generator()
        generator.manual_seed(303)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset, generator = generator), 
            collate_fn = collate_pad, num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'], worker_init_fn = seed_worker)
#        feature_extractor = DenseNet_module(pretrained = True, maxpool=True, Tavgpool=False)
        feature_extractor = ResNet18_module(blocks=4, pretrained = True, maxpool=False, stride=False)
        model = WW_RAM(config.name, config.RAM.std, retina, feature_extractor, 
                    config.gpu, fixation_set = "MHL3")
        
        print("Now training {}-step WW_RAM.\n".format(steps))
        model.set_timesteps(steps)
        config.name = namestring.format(steps, config.seed)
        trainer = Trainer(config, loader, model)
        trainer.train()

if __name__ == '__main__':
    for seed in [9001]:
        config = get_ymlconfig('./dispatch.yml')
        config.seed = seed
        main(config)
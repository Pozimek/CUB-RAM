#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:49:04 2021

@author: piotr
"""
import os
import numpy as np
import torch

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

from model import FF_GlimpseModel
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
#    namestring = "retina-FFglimpseFIX-strat{}-{}"
#    accs = {}
#    
#    for strategy in range(1,6):
#        for steps in range(1,4):
#            print("Now training {}-step FF_GlimpseModel, strategy {}.\n".format(steps, strategy))
#            config.name = namestring.format(strategy, steps)
#            loader = torch.utils.data.DataLoader(
#                dataset, batch_size=config.training.batch_size, 
#                sampler=RandomSampler(dataset), collate_fn = collate_pad,
#                num_workers=config.training.num_workers, 
#                pin_memory=kwargs['pin_memory'],)
#            feature_extractor = ResNet18_module(blocks=4, maxpool=True, stride=True)
#            model = FF_GlimpseModel(config.name, retina, feature_extractor, strategy,
#                        config.gpu, fixation_set = "MHL3")
#            model.set_timesteps(steps)
#            trainer = Trainer(config, loader, model)
#            acc = trainer.train()
#            accs["S" + str(strategy) + "T" + str(steps)] = acc
#    
#    print("Top results: ")
#    print(accs)
    
    namestring = "retina-FFsinglefix-{}"
    accs = {}
    fixations = [[0],[1],[2],[3],[4]]
    
    for fix in fixations:
        strategy = 4
        print("Now training on fixation {}\n".format(fix[0]))
        config.name = namestring.format(fix[0])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=RandomSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=kwargs['pin_memory'],)
        feature_extractor = ResNet18_module(blocks=4, maxpool=True, stride=True)
        model = FF_GlimpseModel(config.name, retina, feature_extractor, strategy,
                    config.gpu, fixation_set = fix)
        model.set_timesteps(1)
        trainer = Trainer(config, loader, model)
        acc = trainer.train()
        accs["Fix" + str(fix[0])] = acc
    
    print("Top results: ")
    print(accs)

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
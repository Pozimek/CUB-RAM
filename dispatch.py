#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:44:16 2020

A new training dispatch script taking into account the DT-RAM paper. Implements
curriculum learning.

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
from model import CUBRAM_baseline, ff_r18, RAM_baseline
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
    
#    # 1 - Pretrain feature extractor
#    r18 = ff_r18()
#    config.name = "curriculum-r18"
#    r18_trainer = Trainer(config, loader, r18)
#    r18_trainer.train()
#    r18_weights = r18.resnet18.state_dict()
    
    # 1 - Load in pretrained feature extractor
    w_path = os.path.join(config.ckpt_dir, "curriculum-r18_best.pth.tar")
    r18 = ff_r18()
    r18.load_state_dict(torch.load(w_path)['model_state'])
    r18_weights = r18.resnet18.state_dict()
#    fc1_weights = r18.fc1.state_dict()
#    fc2_weights = r18.fc2.state_dict()
    
    # 2 - Train RAM, iteratively increasing number of timesteps
    config.name = "curriculum-vanilla-RAM2-1"
#    model = CUBRAM_baseline(config.name, config.RAM.foveal_size, 
#                            config.RAM.n_patches, config.RAM.scaling, 
#                            config.RAM.std, config.gpu)
    #TODO: pass retina object, remove deprecated params
    model = RAM_baseline(config.name, config.RAM.foveal_size, 
                            config.RAM.n_patches, config.RAM.scaling, 
                            config.RAM.std, config.gpu)
    
    #transfer pretrained weights
    model.sensor.resnet18.load_state_dict(r18_weights)
#    model.rnn.fc_in.load_state_dict(fc1_weights)
#    model.classifier.fc.load_state_dict(fc2_weights)
    
    #cleanup
    del(r18)
    
    #train
    for steps in range(2,6):
        print("Now training {}-step RAM.\n".format(steps))
        model.set_timesteps(steps)
        config.name = "curriculum-vanilla-RAM2-{}".format(steps)
        trainer = Trainer(config, loader, model)
        trainer.train()
    #TODO config file restructure
    #TODO config class seems to triplicate dicts
    
if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
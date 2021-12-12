#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 02:21:31 2021

@author: piotr
"""
import torch

from torchvision.transforms import Compose, ToTensor
from torch.utils.data.sampler import RandomSampler
 
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, get_mean_and_std

def main(config):
    transform = Compose([ToTensor()]) 
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    
    generator = torch.Generator()
    generator.manual_seed(303)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.training.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.training.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    mean, std = get_mean_and_std(loader)
    print("Mean: ", mean)
    print("Std: ", std)

if __name__ == '__main__':
    config = get_ymlconfig('./PAM_dispatch.yml')
    main(config)
#                             Normalize(mean=[0.3489, 0.3591, 0.3110], 
#                                       std=[0.2942, 0.2966, 0.2984])])
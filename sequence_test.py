#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 7 01:14:35 2021

A script for estimating the total information extracted out of a fixation 
sequence. Currently implements a lazy approach: test 5 resnets, each trained on
a different fixation, and 'OR' their per-sample classification accuracies to 
estimate an upper bound on total information.

@author: piotr
"""
import os
import numpy as np
import torch

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable

from modules import retinocortical_sensor, crude_retina, ResNet18_module
from CUB_loader import CUBDataset, collate_pad
from utils import get_ymlconfig
from model import ff_r18

config = get_ymlconfig('./dispatch.yml')
torch.manual_seed(config.seed)
np.random.seed(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
kwargs = {}
if config.gpu:
    torch.cuda.manual_seed(config.seed)

transform = Compose([ToTensor(), 
                         Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])])
retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                          config.RAM.scaling, config.gpu)
modelstring = "retina-RESNET-ada{}"

sequence = []

for steps in range(1,6):
    correct = torch.tensor([]).bool().cuda()
    dataset = CUBDataset(transform = transform)
    loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.training.batch_size, 
            sampler=SequentialSampler(dataset), collate_fn = collate_pad,
            num_workers=config.training.num_workers, 
            pin_memory=True,)
    
    feature_extractor = ResNet18_module(blocks=4, maxpool=False)
    model = ff_r18(retina, feature_extractor, [1024], steps-1).cuda()
    
    filename = modelstring.format(steps) + '_best.pth.tar'
    path = os.path.join(config.ckpt_dir, filename)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state'])
    
    model.eval()
    loader.dataset.test()
    num = len(loader.dataset.test_data)
    it = iter(loader)
    for i, (x, y, y_locs) in enumerate(loader):
        with torch.no_grad():
            y = y.cuda()
            x, y = Variable(x), Variable(y)
            if model.require_locs: x = (x, y_locs)
            
            log_probas, locs, log_pi, baselines = model(x)
            batch_acc = torch.max(log_probas[:,-1], 1)[1].detach() == y
            correct = torch.cat((correct, batch_acc.cuda()))
            
    sequence.append(correct)
    
total = sum(sequence)
total_acc = 100 * (total.bool().sum().item() / num)
print(total_acc)

twos = total > 1
twos_acc = 100 * (twos.sum().item() / num)
print(twos_acc)

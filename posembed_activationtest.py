#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:26:49 2021

@author: piotr
"""
import os
import numpy as np
import torch

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
from random import randint

from modules import retinocortical_sensor, crude_retina, ResNet18_module, DenseNet_module
from model import RAM_baseline
from modules import bnlstm
from CUB_loader import CUBDataset, collate_pad
from utils import get_ymlconfig

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
dataset = CUBDataset(transform = transform)
loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size, 
        sampler=SequentialSampler(dataset), collate_fn = collate_pad,
        num_workers=config.training.num_workers, 
        pin_memory=True,)
retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                      config.RAM.scaling, config.gpu)
feature_extractor = ResNet18_module(blocks=4, maxpool = False)
memory = bnlstm
model = RAM_baseline(config.name, config.RAM.std, retina, 
                     feature_extractor, memory, config.gpu).cuda()

namestring = "retina-RAM-25-LR-PE-{}"
steps = 5
model.set_timesteps(steps)
config.name = namestring.format(steps)

model.eval()
loader.dataset.test()
num = len(loader.dataset.test_data)

# Sample a random batch
#i = randint(1, num//config.training.batch_size)
i = 10
it = iter(loader)
for _ in range(i):
    x, y, y_locs = it.next()
y = y.cuda()

if model.require_locs: x = (x, y_locs)
log_probas, locs, log_pi, baselines = model(x)
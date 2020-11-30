#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 02:20:57 2020

Retina annealing for CUB RAM experiments.

@author: piotr
"""

from retinavision.ssnn import SSNN
from retinavision.rf_generation import rf_ozimek
from retinavision.cortex_generation import LRsplit, cort_map, cort_prepare
from retinavision.utils import writePickle, loadPickle
import os

from CUB_loader import CUBDataset, collate_pad
from utils import showArray, showTensor
import torch
from torch.utils.data.sampler import RandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize

from retinavision.retina import Retina
from retinavision.cortex import Cortex
import numpy as np

# Settings
seed = 2
gpu = True

# Anneal tessellation
N = 16384 #372x372
#tessellation = SSNN(N)
tess_path = os.path.join(os.getcwd(), 'retina_data', 'tess_16384.pkl')
#writePickle(tess_path, tessellation)
tessellation = loadPickle(tess_path)

# Pre-load dataset image
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
transform = Compose([ToTensor()])
dataset = CUBDataset(transform = transform)
collate = collate_pad
loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, 
        sampler=RandomSampler(dataset), collate_fn = collate,
        num_workers=0, 
        pin_memory=True,)
it = iter(loader)
x, y, _ = it.next()
x = x.cuda()

# Produce receptive field configs
sigma_base, mean_rf = 0.5, 1
#rf_loc, rf_coeff, fov_dist_5 = rf_ozimek(tessellation, 3, sigma_base, 1, mean_rf)
pathstring = os.path.join(os.getcwd(), 'retina_data', 'ret16k_{}_{}_{}.pkl')
#writePickle(pathstring.format(sigma_base, mean_rf, 'loc'), rf_loc)
#writePickle(pathstring.format(sigma_base, mean_rf, 'coeff'), rf_coeff)

# Load and prepare retina
Ret = Retina(gpu=gpu)
Ret.loadLoc(pathstring.format(sigma_base, mean_rf, 'loc'))
Ret.loadCoeff(pathstring.format(sigma_base, mean_rf, 'coeff'))
fixation = (x.shape[-1]//2, x.shape[-2]//2)
Ret.prepare(x.shape[-2:], fixation)

# Test against image
img = np.moveaxis(x[7].cpu().numpy(), 0, -1)
img = np.uint8(img*255)
V = Ret.sample(img, fixation)
backproj = Ret.backproject_last()
print("\n Retina width {}".format(Ret.width))
print("sigma_base {}, mean_rf {}".format(sigma_base, mean_rf))
print("V.max() - {}".format(V.max()))
showArray(img, size=(10,10))
showArray(backproj, size=(10,10))

# Produce a cortex
c_sigma_base, target_d5 = 1.0, 1.0
rf_loc = Ret.loc
#L, R = LRsplit(rf_loc)
#L_loc, R_loc = cort_map(L, R, target_d5 = target_d5)
#L_loc, R_loc, L_coeff, R_coeff, cort_size = cort_prepare(
#        L_loc, R_loc, sigma_base = c_sigma_base)
pathstring2 = os.path.join(os.getcwd(), 'retina_data', 'cort16k_{}_{}_{}.pkl')

Llocpath = pathstring2.format(c_sigma_base, target_d5, 'Lloc')
Rlocpath = pathstring2.format(c_sigma_base, target_d5, 'Rloc')
Lcoeffpath = pathstring2.format(c_sigma_base, target_d5, 'Lcoeff')
Rcoeffpath = pathstring2.format(c_sigma_base, target_d5, 'Rcoeff')

#writePickle(Llocpath, L_loc)
#writePickle(Rlocpath, R_loc)
#writePickle(Lcoeffpath, L_coeff)
#writePickle(Rcoeffpath, R_coeff)

#L_loc = loadPickle(Llocpath)
#R_loc = loadPickle(Rlocpath)
#L_coeff = loadPickle(Lcoeffpath)
#R_coeff = loadPickle(Rcoeffpath)

# Test against image
C = Cortex(gpu=gpu)
C.loadLocs(Llocpath, Rlocpath)
C.loadCoeffs(Lcoeffpath, Rcoeffpath)
c_img = C.cort_img(V)

showArray(c_img, size=(10,10))
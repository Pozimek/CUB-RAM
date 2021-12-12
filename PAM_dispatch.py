#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:11:43 2021

@author: piotr
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.optim as optim
 
from model import PAM
from modules import crude_retina, V, Encoder, Decoder
from modules import WW_module, WW_LSTM, classification_network_short, WhereMix
from modules import WWMix, Mix_ResBlock, ResBlock, ResBlockNew, WW_BN, Reshape, WW_V
from modules import VAR1Encoder, VAR2Encoder, VAR1Decoder, VAR2Decoder, ResNetEncoder, VAR4Decoder
from PAM_trainer import Trainer
from CUB_loader import CUBDataset, collate_pad, seed_worker
from utils import get_ymlconfig, set_seed

def main(config):
    set_seed(config.seed, gpu=config.gpu)
    transform = Compose([ToTensor(),
                             Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5])]) 
    AE_variant = config.AE.variant
    
    config.name = "PAM-WWLSTMv10"
    config.name += "-s{}".format(config.seed)
    config.name += "-AE{}".format(AE_variant)
    config.name += "VAR{}".format(config.variant)
    
    dataset = CUBDataset(transform = transform, shuffle=True)
    generator = torch.Generator()
    generator.manual_seed(303) # separate seed for more control
    loader = torch.utils.data.DataLoader(
        dataset, batch_size = config.batch_size, 
        sampler = RandomSampler(dataset, generator = generator), 
        collate_fn = collate_pad, num_workers = config.num_workers, 
        pin_memory = config.gpu, worker_init_fn = seed_worker)
    
    retina = crude_retina(config.retina.foveal_size, config.retina.n_patches, 
                              config.retina.scaling, config.gpu, clamp = False)
    # Autoencoder
    AE_ckpts = ["PAM-spatialAEv0-s9001-VAR1phase1_best.pth.tar",
                "PAM-spatialAEv0-s9001-VAR2phase1_best.pth.tar",
                "PAM-CLASSv11alt-s9001phase1_best.pth.tar",
                None]
    AE_fname = AE_ckpts[AE_variant-1]
    
    E_variants = [VAR1Encoder(retina.out_shape, BN = config.AE.BN, 
                              activation = nn.PReLU()),
                    VAR2Encoder(retina.out_shape, BN = config.AE.BN, 
                                activation = nn.PReLU()),
                    Encoder(retina.out_shape, BN = config.AE.BN, 
                            activation = nn.PReLU()),
                    ResNetEncoder(retina.out_shape, pretrained=False)]
    encoder = E_variants[AE_variant-1]
    
    D_variants = [VAR1Decoder(encoder.out_shape, BN = config.AE.BN,
                              activation = nn.PReLU()),
                    VAR2Decoder(encoder.out_shape, BN = config.AE.BN, 
                                activation = nn.PReLU()),
                    Decoder(encoder.out_shape, BN = config.AE.BN, 
                            activation = nn.PReLU()),
                    VAR4Decoder(encoder.out_shape, BN = config.AE.BN, 
                            activation = nn.PReLU())]
    decoder = D_variants[AE_variant-1]
    
    bottleneck = WW_V(encoder.out_shape) if config.AE.VAE else nn.Identity()
    if config.variant == 2: 
        bottleneck = nn.Sequential(WW_module(encoder.out_shape, 16),
                                   nn.ReLU(inplace = True),
                                   WWMix(torch.Size((64, 16)), torch.Size((64, 16))),
                                   nn.ReLU(inplace = True),
                                   Reshape((-1,)+encoder.out_shape))
    # Classification stream
    where_dim = 10
    what_dim = encoder.out_shape[0]
#    aggr_shape = torch.Size((encoder.out_shape[0], encoder.out_shape[1], 12))
    mem_in = torch.Size((256, where_dim))
    
    FE_variants = [nn.Sequential(WW_module(encoder.out_shape, where_dim),
                                  nn.ReLU(inplace = True),
                                  WWMix(torch.Size((what_dim, where_dim)), mem_in),
                                  nn.ReLU(inplace = True))]
    FE = FE_variants[0]
    
    memory = WW_LSTM(mem_in, mem_in[0], mem_in[1],
                     gate_op=WhereMix, in_op=WhereMix)
    classifier = classification_network_short(mem_in.numel(), 200)
    if config.variant == 1:
        classifier = nn.Sequential(Reshape((-1,) + mem_in),
                                   WWMix(mem_in, mem_in),
                                   nn.ReLU(inplace = True),
                                   nn.Flatten(),
                                   classification_network_short(mem_in.numel(), 200))
    # PAM
    model = PAM(config.name, retina, encoder, decoder, bottleneck, FE, memory, 
                classifier, config.gpu)
    trainer = Trainer(config, loader, model)
    
    #enable WD
    trainer.P3.optimizer = optim.SGD(trainer.P3.params, lr=config.P3.init_lr, 
                                         momentum=0.9, weight_decay = 5e-3)
    trainer.P3.lr_scheduler = ReduceLROnPlateau(trainer.P3.optimizer, factor=0.5,
                                                    patience = 10, threshold = 0.1)
    if config.variant == 2 or config.variant == 3:
        trainer.phase1() #train autoencoder
        AE_fname = None
#    trainer.preload_AE("")
#    trainer.visAE()
    trainer.phase3(preload_filename = AE_fname) #train classifier 
#    trainer.aggregation_tests(strategy = config.strat)
    
    # Visualize classification sequence
#    trainer.load_checkpoint(3)
#    x, masked, concat, r_concat, y_p, z = trainer.valid_classifier(0, True)
#    
#    for Bid in range(len(x)):
#        # Visualize input image
#        im_x = (x[Bid] - x[Bid].min()).numpy()
#        im_x = np.moveaxis(im_x, 0, -1)/im_x.max()
#        plt.figure(figsize=(8,8))
#        plt.axis('off')
#        plt.imshow(im_x)
#        plt.show()
#        
#        # Visualize masked image
#        im_masked = (masked[Bid] - masked[Bid].min()).numpy()
#        im_masked = np.moveaxis(im_masked, 0, -1)/im_masked.max()
#        plt.figure(figsize=(8,8))
#        plt.axis('off')
#        plt.imshow(im_masked)
#        plt.show()
#        
#        # Visualize image patches
#        im_concat = (concat[Bid] - concat[Bid].min()).numpy()
#        im_concat = np.moveaxis(im_concat, 0, -1)/im_concat.max()
#        plt.figure(figsize=(5*2,4))
#        plt.axis('off')
#        plt.imshow(im_concat)
#        plt.show()
#        
#        # Visualize reconstructions
#        im_rconcat = (r_concat[Bid] - r_concat[Bid].min()).numpy()
#        im_rconcat = np.moveaxis(im_rconcat, 0, -1)/im_rconcat.max()
#        plt.figure(figsize=(5*2,4))
#        plt.axis('off')
#        plt.imshow(im_rconcat)
#        plt.show()
#        
#    print(torch.max(y_p, 1)[1])

if __name__ == '__main__':
    for seed in [9001]:
        for variant in [1,2,3]:
            config = get_ymlconfig('./PAM_dispatch.yml')
            config.seed = seed
            config.AE.VAE = variant==3
            config.AE.variant = 3 if variant==1 else 1
            config.variant = variant
        
#            config.training.resume = True
            main(config)
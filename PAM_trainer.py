#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:12:14 2021

Training and testing functions for the PAM model.
Implements the three phase training approach.

@author: piotr
"""
import os
import sys
from os.path import join, exists
import yaml
import time
import shutil
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils import AverageMeter, ir, showTensor, showFixedTensor
from model import Guide

class Trainer(object):
    def __init__(self, C, data_loader, model):
        # Store config file
        self.C = C
        
        # data
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.num_train = len(self.data_loader.dataset)
        self.num_valid = len(self.data_loader.dataset.test_data)
        self.x_size = torch.tensor((500,500), device='cuda')
        self.zero = torch.tensor([0], device='cuda')
        self.batch_size = C.batch_size
        
        # model
        self.model = model
        if C.gpu: self.model.cuda()
        
        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        
        # training objects and parameter groups for each phase
        self.P1 = lambda: None #namespace dummies
        self.P2 = lambda: None
        self.P3 = lambda: None
        
        self.P1.params = [{'params': self.model.fov_encoder.parameters()},
                           {'params': self.model.fov_decoder.parameters()},
                           {'params': self.model.bottleneck.parameters()}]
        self.P2.params = [{'params': self.model.per_encoder.parameters()},
                           {'params': self.model.SR.parameters()}]
        self.P3.params = [{'params': self.model.FE.parameters()},
                           {'params': self.model.memory.parameters()},
                           {'params': self.model.classifier.parameters()}]
#                           {'params': self.model.fov_encoder.parameters()}, #XXX
#                           {'params': self.model.fov_decoder.parameters()}, #XXX
#                           {'params': self.model.bottleneck.parameters()}] #XXX
        
        self.P1.optimizer = optim.Adam(self.P1.params, lr=C.P1.init_lr)
#        self.P2.optimizer = 
        self.P3.optimizer = optim.SGD(self.P3.params, lr=C.P3.init_lr, 
                                      momentum=0.9)
#        self.predAtt_optimizer = 
        self.P1.lr_scheduler = StepLR(self.P1.optimizer, step_size=10, gamma=0.5)
#        self.P2.lr_scheduler = StepLR(self.P2.optimizer, step_size=10, gamma=0.5)
#        self.P3.lr_scheduler = StepLR(self.P3.optimizer, step_size=10, gamma=0.5)
        self.P3.lr_scheduler = ReduceLROnPlateau(self.P3.optimizer, factor=0.5,
                                                 patience = 10, threshold = 0.1)
        with torch.random.fork_rng():
            torch.manual_seed(303)
            self.P2.data_rng = torch.get_rng_state()
            self.P3.data_rng = torch.get_rng_state()
        
        # set up logging
        if C.tensorboard:
            tensorboard_dir = C.log_dir + self.model.name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not exists(tensorboard_dir): os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir, comment=C.comment)
            
            #dump a (buggy) copy of config to log directory just in case
            cpath = join(C.log_dir, C.name, C.name+'_config.yml')
            with open(cpath, 'w') as file:
                yaml.dump(C.__D__, file)
                
    def train_ae(self, epoch):
        """ One training epoch"""
        self.model.train()
        self.data_loader.dataset.train()
        
        loss_meter = AverageMeter()
        tic = time.time()
        W = self.model.retina.g
        unfold = nn.Unfold(W, stride = W) #no overlap
        
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y, y_locs) in enumerate(self.data_loader):
                #divide the images into patches
                patches = self.model.retina.foveal_grid(x, unfold).flatten(0,1)
                b_size = -( -len(patches) // 2)
                Rloss, Lloss, KLDloss = [],[],[] #sub-batch storage for logs
                
                #process in smaller sub-batches to fit in vram
                for p_batch in torch.split(patches, b_size):
                    recon = self.model.AE(p_batch)
                    
                    recon_loss = F.mse_loss(recon, p_batch, reduction='mean')
                    KLD = self.model.bottleneck.KLD
                    loss = recon_loss + KLD
                    loss.backward()
                    self.P1.optimizer.step()
                    self.P1.optimizer.zero_grad()
                    loss_meter.update(loss.item(), self.batch_size)
                    
                    Rloss.append(recon_loss)
                    Lloss.append(loss)
                    KLDloss.append(KLD)
                
                if self.C.tensorboard:
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalars("AE Loss (Detailed)", {
                            "Recon_loss": torch.tensor(Rloss).mean(),
                            "KLD_loss": torch.tensor(KLDloss).mean(),
                            "Total_loss": torch.tensor(Lloss).mean()}, iteration)
                
                # update status bar
                toc = time.time()
                pbar.set_description(
                    ("{:.1f}s - loss: {:.2f}".format(
                            (toc-tic), loss.item())))
                pbar.update(self.batch_size)
        
        print("\nTrain epoch {} - avg loss: {:.3f}".format(
                epoch, loss_meter.avg))
        return loss_meter.avg
        
    def valid_ae(self, epoch):
        """ One validation epoch"""
        self.model.eval()
        self.data_loader.dataset.test()
        
        loss_meter = AverageMeter()
        W = self.model.retina.g
        unfold = nn.Unfold(W, stride = W) #no overlap
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            patches = self.model.retina.foveal_grid(x, unfold).flatten(0,1)
            b_size = -( -len(patches) // 2)
            Rloss, Lloss, KLDloss = [],[],[] #sub-batch storage for logs
            
            #process in smaller sub-batches to fit in vram
            for p_batch in torch.split(patches, b_size):
                recon = self.model.AE(p_batch)
                    
                recon_loss = F.mse_loss(recon, p_batch, reduction='mean')
                KLD = self.model.bottleneck.KLD
                loss = recon_loss + KLD
                loss_meter.update(loss.item(), self.batch_size)
                
                Rloss.append(recon_loss)
                Lloss.append(loss)
                KLDloss.append(KLD)
            
            if self.C.tensorboard:
                iteration = epoch * len(self.data_loader) + i
                self.writer.add_scalars("AE Loss (Detailed)", {
                        "Recon_loss_val": torch.tensor(Rloss).mean(),
                        "KLD_loss_val": torch.tensor(KLDloss).mean(),
                        "Total_loss_val": torch.tensor(Lloss).mean()}, iteration)
            
        print("\n Val epoch {} - avg loss: {:.3f}".format(
                epoch, loss_meter.avg))
        
        return loss_meter.avg
        
    def visAE(self, filename=None):
        """Visualize autoencoder behaviour."""
        if filename is None: self.load_checkpoint(1, best=True)
        else: 
            self.preload_AE(filename)
        self.model.eval()
        self.data_loader.dataset.test()
        
        W = self.model.retina.g
        unfold = nn.Unfold(W, stride = W)
        fold = nn.Fold((500,500), W, stride = W)
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            patches = self.model.retina.foveal_grid(x, unfold)
            
            batch = patches.flatten(0,1) #combine into a single batch
            z = self.model.bottleneck(self.model.fov_encoder(batch))
            r_patches = self.model.fov_decoder(self.model.bottleneck(z))
            MSE = F.mse_loss(r_patches, batch, reduction='mean')
            
            #reconstruct full image for vis
            w_patch = int(patches.shape[1]**0.5) #output width in patches
            fold = nn.Fold((w_patch*W, w_patch*W), W, stride = W)
            recon = r_patches.reshape(patches.shape).flatten(2,4).permute(0,2,1)
            recon = fold(recon)
            target = fold(unfold(x))
            
            for j in range(len(x)):
                print(j)
                showFixedTensor(target[j])
                showFixedTensor(recon[j])
                print("x.min: {}, x.max: {}".format(x[j].min(), x[j].max()))
                print("recon.min: {}, recon.max: {}".format(
                        recon[j].min(), recon[j].max()))
            print("MSE: {}".format(MSE))
            break
    
    def visFAE(self, filename=None):
        """Visualize full fov autoencoder behaviour."""
        if filename is None: self.load_checkpoint(1, best=True)
        else: 
            self.preload_AE(filename)
        self.model.eval()
        self.data_loader.dataset.test()
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            x = x.cuda()
            z = self.model.bottleneck(self.model.fov_encoder(x))
            recon = self.model.fov_decoder(self.model.bottleneck(z))
            MSE = F.mse_loss(recon, x, reduction='mean')
            
            for j in range(len(x)):
                print(j)
                showFixedTensor(x[j])
                showFixedTensor(recon[j])
                print("x.min: {}, x.max: {}".format(x[j].min(), x[j].max()))
                print("recon.min: {}, recon.max: {}".format(
                        recon[j].min(), recon[j].max()))
            print("MSE: {}".format(MSE))
            break
    
    def phase1(self):
        """Train foveal autoencoder."""
        # training params 
        self.start_epoch = 0
        self.best_valid_loss = None
        self.es_loss = None
        self.counter = 0
        
        print("\n Phase 1 of " + self.C.name)
        if self.C.P1.resume:
            print("\n Resuming training from checkpoint.")
            self.load_checkpoint(1, best=False)
            
        # reopen writer if running immediately after a previous phase
        if self.C.tensorboard and self.writer.all_writers is None:
            tensorboard_dir = self.C.log_dir + self.model.name
            self.writer = SummaryWriter(tensorboard_dir, comment=self.C.comment)
            
        for epoch in range(self.start_epoch, self.C.P1.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch+1, 
                  self.C.P1.epochs, self.P1.optimizer.param_groups[0]['lr']))
            
            train_loss = self.train_ae(epoch)
            valid_loss = self.valid_ae(epoch)
            
            self.P1.lr_scheduler.step()
            
            # Log to tensorboard
            if self.C.tensorboard:
                self.writer.add_scalars("Smoothed Results/Losses", {
                        "AE_train_loss":train_loss,
                        "AE_valid_loss":valid_loss}, epoch)
            
            # Save results
            if self.best_valid_loss is None: self.best_valid_loss = valid_loss
            is_best = valid_loss < self.best_valid_loss
            self.best_valid_loss = min(valid_loss, self.best_valid_loss)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.P1.optimizer.state_dict(),
                 'scheduler_state': self.P1.lr_scheduler.state_dict(),
                 'best_valid_loss': self.best_valid_loss,
                 }, is_best, 1)
    
            # Early stopping
            if self.es_loss is None: self.es_loss = valid_loss
            ES_best = valid_loss < (self.es_loss - self.C.P1.delta)
            if ES_best: 
                self.es_loss = valid_loss
                self.counter = 0
            else: self.counter += 1
            
            if self.C.P1.es and (self.counter > self.C.P1.patience):
                print("[!] No improvement in a while, early stopping.")
                break
    
        print("Training has ended, best val loss was {:.6f}.".format(
                self.best_valid_loss))
        self.writer.close()
        
    def train_classifier(self, epoch, aggr_strat = None):
        """ One training epoch"""
        model = self.model #readability
        model.train()
        self.data_loader.dataset.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        tic = time.time()
        
        with tqdm(total=self.num_train, file=sys.stdout) as pbar:
            for i, (x, y, y_locs) in enumerate(self.data_loader):
                if self.C.gpu: y = y.cuda()
                if aggr_strat: y_p = self.aggr_strat(x, y_locs, aggr_strat)
                else:
                    # regular forward pass
                    _ = model.reset(B=x.shape[0])
                    model.retina.reset()
                    guide = Guide(5, model.training, "MHL3", model.retina.width, 
                                  random_mix = True, rng_state = self.P3.data_rng)
                    l_t_prev = model.retina.to_exocentric_action(self.x_size,
                                                               guide(y_locs, 0))
                    for t in range(5):
                        fov = model.retina.foveate_ego(x, l_t_prev)[:,0,...]
                        z = model.bottleneck(model.fov_encoder(fov))
                        features = model.FE(z)
                        h_t = model.memory(features)
                        l_t_prev = model.retina.to_egocentric_action(
                            guide(y_locs, t+1) - model.retina.fixation)
                        
                    y_p = model.classifier(h_t.flatten(1))
                
                # loss, backprop, meters, logs
                loss = F.nll_loss(y_p, y)
                loss.backward()
                self.P3.optimizer.step()
                self.P3.optimizer.zero_grad()
                
                correct = torch.max(y_p, 1)[1].detach() == y
                acc = 100 * (correct.sum().item() / self.batch_size)
                
                loss_meter.update(loss.item(), self.batch_size)
                acc_meter.update(acc, self.batch_size)
                
                if self.C.tensorboard:
                    suffix = str(aggr_strat) if aggr_strat else ""
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalars('CLASS Loss (Detailed)/Training', {
                        "Train_loss"+suffix:loss_meter.avg}, iteration)
                    self.writer.add_scalars('Accuracy (Detailed)/Training', {
                        "Train_acc"+suffix:acc_meter.avg}, iteration)
                
                # update status bar
                toc = time.time()
                pbar.set_description(
                    ("{:.1f}s - loss: {:.2f}".format(
                            (toc-tic), loss.item())))
                pbar.update(self.batch_size)
        
        print("Train epoch {} - avg acc: {:.2f} | avg loss: {:.3f}".format(
                epoch, acc_meter.avg, loss_meter.avg))
        return loss_meter.avg, acc_meter.avg
    
    def update_masks(self, masks, coords, size):
        """Uncovers a square patch of given size at given fixation coordinates
        by setting the mask values to 1. Corresponds to a single fixation."""
        from_x, from_y = coords[:, 1] - (size // 2), coords[:, 0] - (size // 2)
        to_x, to_y = from_x + size, from_y + size
        
        from_x = torch.max(self.zero, from_x)
        from_y = torch.max(self.zero, from_y)
        to_y = torch.min(self.x_size[0], to_y)
        to_x = torch.min(self.x_size[1], to_x)
        
        for b in range(masks.shape[0]):
            masks[b, :, from_y[b]:to_y[b], from_x[b]:to_x[b]] = 1
        
        return masks
    
    def valid_classifier(self, epoch, v = False, aggr_strat = None):
        """ One validation epoch.
        v - whether to visualize a single batch."""
        model = self.model #readability
        model.eval()
        self.data_loader.dataset.test()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            if self.C.gpu: y = y.cuda()
            if aggr_strat: y_p = self.aggr_strat(x, y_locs, aggr_strat)
            else:
                _ = model.reset(B=x.shape[0])
                model.retina.reset()
                guide = Guide(5, model.training, "MHL3", model.retina.width, 
                              random_mix = True, rng_state = self.P3.data_rng)
                l_t_prev = model.retina.to_exocentric_action(self.x_size,
                                                           guide(y_locs, 0))
                patches, recons, codes = [], [], []
                masks = torch.zeros_like(x)
                
                for t in range(5):
                    if v: #visualisation
                        if model.retina.fixation is None:
                            coords = model.retina.to_exocentric(
                                    self.x_size, l_t_prev)
                        else: 
                            coords = model.retina.to_egocentric(
                                    self.x_size, l_t_prev)
                        masks = self.update_masks(masks, coords, model.retina.g)
                    # forward pass
                    fov = model.retina.foveate_ego(x, l_t_prev)[:,0,...]
                    z = model.bottleneck(model.fov_encoder(fov))
                    if v:
                        patches.append(fov)
                        recons.append(model.fov_decoder(z).detach())
                        codes.append(z.detach())
                    features = model.FE(z)
                    h_t = model.memory(features)
                    
                    # next fixation
                    l_t_prev = model.retina.to_egocentric_action(
                        guide(y_locs, t+1) - model.retina.fixation)
                    
                y_p = model.classifier(h_t.flatten(1))
            
            # loss, backprop, accuracy
            loss = F.nll_loss(y_p, y)
            
            correct = torch.max(y_p, 1)[1].detach() == y
            acc = 100 * (correct.sum().item() / self.batch_size)
            
            loss_meter.update(loss.item(), self.batch_size)
            acc_meter.update(acc, self.batch_size)
            
            if v: 
                concat = torch.cat(patches, dim=3).squeeze().cpu()
                r_concat = torch.cat(recons, dim=3).squeeze().cpu()
                masked = x.cpu() * masks
                return x, masked, concat, r_concat, y_p, z
            
            if self.C.tensorboard:
                suffix = str(aggr_strat) if aggr_strat else ""
                iteration = epoch * len(self.data_loader) + i
                self.writer.add_scalars('CLASS Loss (Detailed)/Training', {
                    "Val_loss"+suffix:loss_meter.avg}, iteration)
                self.writer.add_scalars('Accuracy (Detailed)/Training', {
                    "Val_acc"+suffix:acc_meter.avg}, iteration)
                
        print("\nVal epoch {} - avg acc: {:.2f} | avg loss: {:.3f}".format(
                epoch, acc_meter.avg, loss_meter.avg))
        return loss_meter.avg, acc_meter.avg

    def preload_AE(self, filename):
        #TODO: include bottleneck if VAE ever used, add a str 'module' param to generalize away from just AE
        pretrained = torch.load(
                os.path.join(self.C.ckpt_dir, filename))['model_state']
        
        self_dict = self.model.state_dict()
        AE_dict = {k:v for k, v in self_dict.items() if k.startswith('AE')}
        update = {k:v for k, v in pretrained.items() if k in AE_dict}
        
        self_dict.update(update)
        self.model.load_state_dict(self_dict)
        
        print("\nLoaded a pre-trained autoencoder from file: " + filename)
    
    def phase3(self, preload_filename = "PAM-CLASSv11alt-s9001phase1_best.pth.tar"):
        """Train classifier (and later location module)"""
        if type(preload_filename) is str:
            self.preload_AE(preload_filename)
        
        # reopen writer if running immediately after a previous phase
        if self.C.tensorboard and self.writer.all_writers is None:
            tensorboard_dir = self.C.log_dir + self.model.name
            self.writer = SummaryWriter(tensorboard_dir, comment=self.C.comment)
        
        # training params 
        self.start_epoch = 0
        self.best_valid_acc = None
        self.es_acc = None
        self.counter = 0        
        
        print("\n Phase 3 of " + self.C.name)
        if self.C.P3.resume:
            print("\n Resuming training from checkpoint.")
            self.load_checkpoint(3, best=False)
        
        for epoch in range(self.start_epoch, self.C.P3.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch+1, 
                  self.C.P3.epochs, self.P3.optimizer.param_groups[0]['lr']))
            
            train_loss, train_acc = self.train_classifier(epoch)
            valid_loss, valid_acc = self.valid_classifier(epoch)
            
            self.P3.lr_scheduler.step(valid_acc)
            
            # Log to tensorboard
            if self.C.tensorboard:
                self.writer.add_scalars("Smoothed Results/Losses", {
                        "CLASS_train_loss":train_loss,
                        "CLASS_valid_loss":valid_loss}, epoch)
                self.writer.add_scalars("Smoothed Results/Accuracies", {
                        "Train_accuracy":train_acc,
                        "Valid_accuracy":valid_acc}, epoch)
            
            # Save results
            if self.best_valid_acc is None: self.best_valid_acc = valid_acc
            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.P3.optimizer.state_dict(),
                 'scheduler_state': self.P3.lr_scheduler.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 'rng_state': self.P3.data_rng
                 }, is_best, 3)
    
            # Early stopping
            if self.es_acc is None: self.es_acc = valid_acc
            ES_best = valid_acc > (self.es_acc + self.C.P3.delta)
            if ES_best: 
                self.es_acc = valid_acc
                self.counter = 0
            else: self.counter += 1
            
            if self.C.P3.es and (self.counter > self.C.P3.patience):
                print("[!] No improvement in a while, early stopping.")
                break
    
        print("Training has ended, best val accuracy was {:.6f}.".format(
                self.best_valid_acc))
        self.writer.close()
    
    def aggr_strat(self, x, y_locs, strat):
        """Forward passes of different feedforward aggregation strategies."""
        model = self.model
        _ = model.reset(B=x.shape[0])
        model.retina.reset()
        guide = Guide(5, model.training, "MHL3", model.retina.width, 
                      random_mix = True, rng_state = self.P3.data_rng)
        l_t_prev = model.retina.to_exocentric_action(self.x_size,
                                                   guide(y_locs, 0))
        if strat == 1:
            """Imagespace Concatenation"""
            patches = []
            for t in range(5):
                patches.append(
                        model.retina.foveate_ego(x, l_t_prev)[:,0,...])
                l_t_prev = model.retina.to_egocentric_action(
                        guide(y_locs, t+1) - model.retina.fixation) 
            
            concat = torch.cat(patches, dim=3)
            z = model.bottleneck(model.fov_encoder(concat))
            features = model.FE(z)
            y_p = model.classifier(features.flatten(1))
            
        elif strat == 2:
            """Featurespace Averaging"""
            featuremaps = []
            for t in range(5):
                fov = model.retina.foveate_ego(x, l_t_prev)[:,0,...]
                z = model.bottleneck(model.fov_encoder(fov))
                featuremaps.append(z)
                l_t_prev = model.retina.to_egocentric_action(
                        guide(y_locs, t+1) - model.retina.fixation) 
                
            z_aggr = torch.mean(torch.stack(featuremaps, dim=1), dim=1)
            features = model.FE(z_aggr)
            y_p = model.classifier(features.flatten(1))
            
        elif strat == 3:
            """Pre-activation Output Averaging."""
            log_probas = []
            for t in range(5):
                fov = model.retina.foveate_ego(x, l_t_prev)[:,0,...]
                z = model.bottleneck(model.fov_encoder(fov))
                features = model.FE(z).flatten(1)
                log_probas.append(model.classifier.fc(features))
                l_t_prev = model.retina.to_egocentric_action(
                        guide(y_locs, t+1) - model.retina.fixation) 
            
            #only final timestep counts
            log_probas = torch.stack(log_probas)
            y_p = F.log_softmax(torch.mean(log_probas, dim=0), dim=1)
        
        return y_p

    def aggregation_tests(self, strategy = 1, preload_filename = "PAM-CLASSv11alt-s9001phase1_best.pth.tar"):
        """
        Trains and tests 3 different aggregation strategies that replace
        recurrent memory. Requires PAM's classifier to match FE's output shape.
        Strategy 1 - imagespace aggregation.
        Strategy 2 - featurespace aggregastion.
        Strategy 3 - output aggregation via pre-softmax averaging.
        """
        if type(preload_filename) is str:
            self.preload_AE(preload_filename)
            
        init_classifier = deepcopy(self.model.classifier.state_dict())
        init_FE = deepcopy(self.model.FE.state_dict())
        print("\n Aggregation strategy {}".format(strategy))
        # reset classifier, FE, optim and lr_scheduler
        self.model.classifier.load_state_dict(deepcopy(init_classifier))
        self.model.FE.load_state_dict(deepcopy(init_FE))
        self.P3.optimizer = optim.SGD(self.P3.params, lr=self.C.P3.init_lr, 
                                  momentum=0.9, weight_decay=5e-3)
        self.P3.lr_scheduler = ReduceLROnPlateau(self.P3.optimizer, factor=0.5,
                                             patience = 10, threshold = 0.1)
        # training params 
        self.start_epoch = 0
        self.best_valid_acc = None
        self.es_acc = None
        self.counter = 0      
        
        # reopen writer if running immediately after a previous phase
        if self.C.tensorboard and self.writer.all_writers is None:
            tensorboard_dir = self.C.log_dir + self.model.name
            self.writer = SummaryWriter(tensorboard_dir, 
                                        comment=self.C.comment)
        if self.C.P3.resume:
            print("\n Resuming training from checkpoint.")
            self.load_checkpoint(3+strategy, best=False)
        
        for epoch in range(self.start_epoch, self.C.P3.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch+1, 
                  self.C.P3.epochs, self.P3.optimizer.param_groups[0]['lr']))
            
            train_loss, train_acc = self.train_classifier(epoch, aggr_strat = strategy)
            valid_loss, valid_acc = self.valid_classifier(epoch, aggr_strat = strategy)
            
            self.P3.lr_scheduler.step(valid_acc)
            
            # Log to tensorboard
            if self.C.tensorboard:
                suffix = str(strategy)
                self.writer.add_scalars("Smoothed Results/Losses", {
                        "CLASS_train_loss"+suffix:train_loss,
                        "CLASS_valid_loss"+suffix:valid_loss}, epoch)
                self.writer.add_scalars("Smoothed Results/Accuracies", {
                        "Train_accuracy"+suffix:train_acc,
                        "Valid_accuracy"+suffix:valid_acc}, epoch)
            
            # Save results
            if self.best_valid_acc is None: self.best_valid_acc = valid_acc
            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.P3.optimizer.state_dict(),
                 'scheduler_state': self.P3.lr_scheduler.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 'rng_state': self.P3.data_rng
                 }, is_best, 3+strategy)
    
            # Early stopping
            if self.es_acc is None: self.es_acc = valid_acc
            ES_best = valid_acc > (self.es_acc + self.C.P3.delta)
            if ES_best: 
                self.es_acc = valid_acc
                self.counter = 0
            else: self.counter += 1
            
            if self.C.P3.es and (self.counter > self.C.P3.patience):
                print("[!] No improvement in a while, early stopping.")
                break
    
        print("Strategy {} training has ended, best val accuracy was {:.6f}.".format(
                strategy, self.best_valid_acc))
        self.writer.close()
                
    def save_checkpoint(self, state, is_best, phase):
        """
        If is_best, a second file with the suffix `_best` is created.
        """
        name = self.C.name + "phase{}".format(phase)
        filename = name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.C.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = name + '_best.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.C.ckpt_dir, filename))
        
    def load_checkpoint(self, phase, best=False):
        """
        If best load the best model, otherwise load the most recent one.
        """
        print("[*] Loading model from {}".format(self.C.ckpt_dir))
        name = self.C.name + "phase{}".format(phase)
        filename = name + '_ckpt.pth.tar'
        if best: filename = name + '_best.pth.tar'
        
        ckpt = torch.load(os.path.join(self.C.ckpt_dir, filename))

        # load variables and states from checkpoint
        if phase == 1:
            optimizer = self.P1.optimizer
            scheduler = self.P1.lr_scheduler
            self.best_valid_loss = ckpt['best_valid_loss']
        elif phase == 2:
            optimizer = self.P2.optimizer
            scheduler = self.P2.lr_scheduler
            self.P2.data_rng = ckpt['rng_state']
        elif phase >= 3:
            optimizer = self.P3.optimizer
            scheduler = self.P3.lr_scheduler
            self.P3.data_rng = ckpt['rng_state']
            self.best_valid_acc = ckpt['best_valid_acc']
            
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        self.start_epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model_state'])
        
        if best:
            print("[*] Loaded {} checkpoint @ epoch {} "
                "with best valid loss of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_loss']))
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch']))
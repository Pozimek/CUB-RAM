#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 02:10:54 2022

Trainer and tester functions for chapter 3 experiments.

@author: piotr
"""
import os
from os.path import join
import yaml
import time
import shutil
from tqdm import tqdm
from utils import AverageMeter

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

class Trainer(object):
    def __init__(self, C, data_loader, model):
        self.C = C
        
        # dataset params
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.num_train = len(self.data_loader.dataset)
        self.num_valid = len(self.data_loader.dataset.test_data)
        
        # model
        self.model = model
        if C.gpu: self.model.cuda()
        params = self.model.parameters()
        
        print('Model Name: {}'.format(C.name))
        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        
        # misc params
        self.batch_size = self.C.batch_size
        self.start_epoch = 0
        self.best_valid_acc = 0.
        self.es_acc = 0.
        self.counter = 0
        self.lr = C.training.init_lr
        self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, 
                                   weight_decay=5e-3)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                step_size=10, gamma=0.5)
        # tensorboard logging
        if C.tensorboard:
            tensorboard_dir = C.log_dir + C.name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir, comment=C.comment)
            #dump a copy of config to log directory
            cpath = join(C.log_dir, C.name, C.name+'_config.yml')
            with open(cpath, 'w') as file:
                yaml.dump(C.__D__, file)
                
    def train(self):
        if self.C.training.resume:
            self.load_checkpoint(best=False)
        
        print("\n[*] Train on {} images, validate on {} images".format(
                self.num_train, self.num_valid))
        
        tic = time.time()
        for epoch in range(self.start_epoch, self.C.training.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.C.training.epochs, 
                    self.optimizer.param_groups[0]['lr']))
            
            # Train and validate
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            
            #compute real time elapsed in full minutes
            toc = time.time()
            minutes = (toc - tic)/60
            
            # Update lr
            self.lr_scheduler.step()
            
            # Log per epoch metrics
            if self.C.tensorboard:
                self.writer.add_scalars("Smoothed Results/Losses", {
                        "Train_loss":train_loss,
                        "Valid_loss":valid_loss}, epoch)
                self.writer.add_scalars("Smoothed Results/Accuracies", {
                        "Train_accuracy":train_acc,
                        "Valid_accuracy":valid_acc}, epoch)
                self.writer.add_scalars("Smoothed Results/LR", {
                        "Learning_rate": self.lr_scheduler.get_last_lr()[0]
                        }, epoch)
                self.writer.add_scalars("Smoothed Results/Time", {
                        "Time_elapsed": minutes
                        }, epoch)
            
            # Save results
            print('\n\n')
            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best)
            
            # Early stopping
            ES_best = valid_acc > (self.es_acc + self.C.training.delta)
            if ES_best: 
                self.es_acc = valid_acc
                self.counter = 0
            else: self.counter += 1
            
            if self.C.training.es and (self.counter > self.C.training.patience):
                print("[!] No improvement in a while, early stopping.")
                break
            
        print("Training has ended, best val accuracy was {:.6f}.".format(
                self.best_valid_acc))
        self.writer.close()
        
        return self.best_valid_acc
        
    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader.dataset.train()
        
        # Averages for logs
        loss_t = AverageMeter()
        accs = AverageMeter()
        R = AverageMeter() 
        B = AverageMeter()
        loss_base = AverageMeter()
        loss_class = AverageMeter()
        
        tic = time.time()
        
        #loader: img, label, parts
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y, y_locs) in enumerate(self.data_loader):
                if self.C.gpu:
                    y = y.cuda()
#                x, y = Variable(x), Variable(y) #TODO: deprecated, check version
                
                # classify image
                if self.model.hardcoded_locs: x = (x, y_locs)
                log_probas, locs, log_pi, baselines = self.model(x)
                
                # compute losses, gradients and update
                losses = self.model.loss(log_probas, log_pi, baselines, y, 
                                         locs, y_locs)

                # classification loss
                total_loss = sum(losses[:3]) if type(losses) is tuple else losses
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # compute accuracy
                correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
                acc = 100 * (correct.sum().item() / self.batch_size)
                
                # update meters
                if type(losses) is tuple:
                    loss_class.update(losses[0].item(), self.batch_size)
                    R.update(losses[3].item(), self.batch_size) 
                    B.update(losses[4].item(), self.batch_size) 
                    loss_base.update(losses[2].item(), self.batch_size)
                loss_t.update(total_loss.item(), self.batch_size)
                accs.update(acc, self.batch_size)
                
                # log to tensorboard
                if self.C.tensorboard:
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalars('Loss (Detailed)/Training', {
                        "Train_loss":loss_t.val}, iteration)
                    self.writer.add_scalars('Accuracy (Detailed)/Training', {
                        "Train_acc":accs.val}, iteration)
                    if type(losses) is tuple:
                        self.writer.add_scalars('Partial Losses/Training',{
                                "Reward_train" : R.val,
                                "Baseline_train" : B.val,
                                "Base_Loss_train" : loss_base.val,
                                "Class_Loss_train" : loss_class.val}, iteration)
                    
                # update status bar
                toc = time.time()
                pbar.set_description(
                    ("{:.1f}s - loss: {:.2f} - acc: {:.1f}".format(
                            (toc-tic), total_loss.item(), acc)))
                pbar.update(self.batch_size)
            
            print("\nTrain epoch {} - avg acc: {:.2f} | avg loss: {:.3f}".format(
                epoch, accs.avg, loss_t.avg))
            return loss_t.avg, accs.avg
                
    def validate(self, epoch):
        self.model.eval()
        self.data_loader.dataset.test()
        
        # Averages for logs
        loss_t = AverageMeter()
        accs = AverageMeter()
        R = AverageMeter()
        B = AverageMeter()
        loss_base = AverageMeter()
        loss_class = AverageMeter()
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            with torch.no_grad():
                if self.C.gpu:
                    y = y.cuda()
#                x, y = Variable(x), Variable(y)
                if self.model.hardcoded_locs: x = (x, y_locs)
                
                # classify batch
                log_probas, locs, log_pi, baselines = self.model(x)
                
                losses = self.model.loss(log_probas, log_pi, baselines, y, locs, y_locs)
                
                # classification loss                    
                total_loss = sum(losses[:3]) if type(losses) is tuple else losses
                
                # compute accuracy
                correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
                acc = 100 * (correct.sum().item() / self.batch_size)
                
                # update meters
                if type(losses) is tuple:
                    loss_class.update(losses[0].item(), self.batch_size)
                    R.update(losses[3].item(), self.batch_size) 
                    B.update(losses[4].item(), self.batch_size) 
                    loss_base.update(losses[2].item(), self.batch_size)
                loss_t.update(total_loss.item(), self.batch_size)
                accs.update(acc, self.batch_size)
                    
                # log to tensorboard
                if self.C.tensorboard:
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalars('Loss (Detailed)/Training', {
                        "Val_loss":loss_t.val}, iteration)
                    self.writer.add_scalars('Accuracy (Detailed)/Training', {
                        "Val_acc":accs.val}, iteration)
                    if type(losses) is tuple:
                        self.writer.add_scalars('Partial Losses/Training',{
                                "Reward_val" : R.val,
                                "Baseline_val" : B.val,
                                "Base_Loss_val" : loss_base.val,
                                "Class_Loss_val" : loss_class.val}, iteration)

        print("Val epoch {} - avg acc: {:.2f} | avg loss: {:.3f}".format(
                epoch, accs.avg, loss_t.avg))
        return loss_t.avg, accs.avg
        
    def save_checkpoint(self, state, is_best):
        """
        If is_best, a second file with the suffix `_best` is created.
        """
        filename = self.C.name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.C.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.C.name + '_best.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.C.ckpt_dir, filename))
            
    def load_checkpoint(self, name = None, best=False):
        """
        Args:
        - best: if True loads the best model, otherwise the most recent one.
        """
        print("[*] Loading model from {}".format(self.C.ckpt_dir))
        
        if name == None: name = self.C.name
        
        filename = name + '_ckpt.pth.tar'
        if best: filename = name + '_best.pth.tar'
        
        ckpt_path = os.path.join(self.C.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print("[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc']))
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch']))
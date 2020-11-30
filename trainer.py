#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:39:12 2020

CUB-RAM trainer

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
        
        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        
        # misc params
        self.batch_size = self.C.training.batch_size
        self.start_epoch = 0
        self.best_valid_acc = 0.
        self.es_acc = 0.
        self.counter = 0
        self.lr = C.training.init_lr
        self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                step_size=10, gamma=0.5)
        #TODO scheduler hyperparams to config file
        
        
        # set up logging
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
        for epoch in range(self.start_epoch, self.C.training.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.C.training.epochs, 
                    self.optimizer.param_groups[0]['lr']))
            
            # Train and validate
            train_loss, train_acc = self.train_one_epoch(epoch)
            valid_loss, valid_acc = self.validate(epoch)
            
            # Update lr
            self.lr_scheduler.step()
            
            # Log to tensorboard
            if self.C.tensorboard:
                self.writer.add_scalars("Smoothed Results/Losses", {
                        "Train_loss":train_loss,
                        "Valid_loss":valid_loss}, epoch)
                
                self.writer.add_scalars("Smoothed Results/Accuracies", {
                        "Train_accuracy":train_acc,
                        "Valid_accuracy":valid_acc}, epoch)
            
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
            self.es_acc = max(valid_acc + self.C.training.delta, self.es_acc)
            self.counter = self.counter + 1 if not ES_best else 0
            if self.C.training.es and (self.counter > self.C.training.patience):
                print("[!] No improvement in a while, early stopping.")
                break
            
        print("Training has ended, best val accuracy was {:.6f}.".format(
                self.best_valid_acc))
        self.writer.close()
        
    def train_one_epoch(self, epoch):
        self.model.train()
        self.data_loader.dataset.train()
        
        loss_t = AverageMeter()
        accs = AverageMeter()
        loss_act = AverageMeter() 
        loss_base = AverageMeter()
        
        tic = time.time()
        
        #loader: img, label, parts
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y, y_locs) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                if self.C.gpu:
                    y = y.cuda()
                x, y = Variable(x), Variable(y) #TODO: deprecated, check version
                if self.model.require_locs: x = (x, y_locs)
                
                # classify image
                log_probas, locs, log_pi, baselines = self.model(x)
                
#                # extract prediction
#                prediction = torch.max(log_probas[-1], 1)[1].detach()
#                
#                # compute reward
#                baselines = baselines.squeeze()
#                R = (prediction == y).float()
#                R = R.unsqueeze(1).repeat(1, self.model.timesteps)
#                adjusted_R = R - baselines.detach()
#
#                # intermediate classification supervision
#                Y = y.repeat(1, self.model.timesteps)
#                loss_classify = F.nll_loss(log_probas, Y)
#                
#                loss_reinforce = torch.sum(-log_pi*adjusted_R, dim=1) #sum timesteps
#                loss_reinforce = torch.mean(loss_reinforce, dim=0) #avg batch
#                loss_baseline = F.mse_loss(baselines, R)
#                total_loss = loss_classify + loss_reinforce + loss_baseline
                
                # compute losses, gradients and update
                losses = self.model.loss(log_probas, log_pi, baselines, y)
                total_loss = sum(losses) if type(losses) is tuple else losses
                total_loss.backward()
                self.optimizer.step()
                
                # compute accuracy
                correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
                acc = 100 * (correct.sum().item() / self.batch_size)
                
                # update meters
                if type(losses) is tuple:
                    loss_act.update(losses[0].item(), self.batch_size) 
                    loss_base.update(losses[2].item(), self.batch_size)
                loss_t.update(total_loss.item(), self.batch_size)
                accs.update(acc, self.batch_size)
                
                # log to tensorboard
                if self.C.tensorboard:
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalar('Loss (Detailed)/Training', 
                                           loss_t.avg, iteration)
                    self.writer.add_scalar('Accuracy (Detailed)/Training',
                                           accs.avg, iteration)
                    self.writer.add_scalars('Partial Losses/Training',{
                            "Action_Loss" : loss_act.avg,
                            "Base_Loss" : loss_base.avg}, iteration) 
                
                # update status bar
                toc = time.time()
                pbar.set_description(
                    ("{:.1f}s - loss: {:.2f} - acc: {:.1f}".format(
                            (toc-tic), total_loss.item(), acc)))
                pbar.update(self.C.training.batch_size)
            
            print("\nTrain epoch {} - avg acc: {:.2f} | avg loss: {:.3f}".format(
                epoch, accs.avg, loss_t.avg))
            return loss_t.avg, accs.avg
                
    def validate(self, epoch):
        self.model.eval()
        self.data_loader.dataset.test()
        
        loss_t = AverageMeter()
        accs = AverageMeter()
        loss_act = AverageMeter()
        loss_base = AverageMeter()
        
        for i, (x, y, y_locs) in enumerate(self.data_loader):
            with torch.no_grad():
                if self.C.gpu:
                    y = y.cuda()
                x, y = Variable(x), Variable(y)
                if self.model.require_locs: x = (x, y_locs)
                
                # M sample duplication to account for stochasticity
                for j in range(self.C.training.M):
                
                    # classify batch
                    log_probas, locs, log_pi, baselines = self.model(x)
                    
                    losses = self.model.loss(log_probas, log_pi, baselines, y)
                    total_loss = sum(losses) if type(losses) is tuple else losses
                    
                    # compute accuracy
                    correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
                    acc = 100 * (correct.sum().item() / self.batch_size)
                    
                    # update meters
                    #These are averaged, so no need to change anything
                    if type(losses) is tuple:
                        loss_act.update(losses[0].item(), self.batch_size) 
                        loss_base.update(losses[2].item(), self.batch_size)
                    loss_t.update(total_loss.item(), self.batch_size)
                    accs.update(acc, self.batch_size)
                    
                # log to tensorboard
                if self.C.tensorboard:
                    iteration = epoch * len(self.data_loader) + i
                    self.writer.add_scalar('Loss (Detailed)/Training', 
                                           loss_t.avg, iteration)
                    self.writer.add_scalar('Accuracy (Detailed)/Training',
                                           accs.avg, iteration)
                    self.writer.add_scalars('Partial Losses/Training',{
                            "Action_Loss" : loss_act.avg,
                            "Base_Loss" : loss_base.avg}, iteration)
        
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
            
    def load_checkpoint(self, best=False):
        """
        Args:
        - best: if True loads the best model, otherwise the most recent one.
        """
        print("[*] Loading model from {}".format(self.C.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best: filename = self.model_name + '_best.pth.tar'
        
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
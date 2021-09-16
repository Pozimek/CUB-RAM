#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:29:27 2021

A Tester class for evaluating the performance and behaviours of RAM models.

@author: piotr

Canonical: 56.9@1, 25.3@2, 35.2@3, 43.7@4, 20.5@5
MHL5: 43.7, 35.2, 56.9, 20.5, 25.3
MHL3: 45, 56-62, 10-25
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable
from random import randint
import torch.nn.functional as F


from CUB_loader import CUBDataset, collate_pad
import matplotlib.patches as patches
from utils import showArray, get_ymlconfig, AverageMeter
from modules import RAM_sensor, crude_retina, ResNet18_module, lstm_, laRNN
from model import Guide, WW_RAM, RAM_baseline, FF_GlimpseModel
from trainer import Trainer

def main(config):
    predictive_mem_test(config)
#    val(config)
#    features(config)
#    vis(config)

def predictive_mem_test(config):
    """Hypothesis: foveal memory moves in a specific direction determined by the
    class observed and the peripheral predictive stream guesses that direction
    from peripheral data.
    
    Tests the cosine distance between:
        - successive foveal memory state deltas
        - final memory states at different samples of the same output class (how?)
    """
    # Seeds, transform and config
    config.tensorboard = False
#    torch.manual_seed(config.seed)
#    np.random.seed(config.seed)
#    os.environ['PYTHONHASHSEED'] = str(config.seed)
    kwargs = {}
    if config.gpu: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
#        torch.cuda.manual_seed(config.seed)
        kwargs = {'num_workers': config.training.num_workers, 
                  'pin_memory': True}
    transform = Compose([ToTensor(), 
                             Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])])
    dataset = CUBDataset(transform = transform)
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                              config.RAM.scaling, config.gpu, clamp=False)
    
    modelstrings = ["retina-WWRAMfix-imaginationv26-{}"]
    fixation_sets = ["MHL3"]
    
    for model_T in [3]:
        for M in modelstrings:
            config.name = M.format(model_T)
            print(" ")
            print("Validating ", config.name, "...")
            feature_extractor = ResNet18_module(blocks=4, maxpool=False, stride=True)
            
            for fixation_set in fixation_sets:
                print("         ... with fixation set", fixation_set)
                model = WW_RAM(config.name, config.RAM.std, retina, feature_extractor, 
                            config.gpu, fixation_set = fixation_set)
                model.set_timesteps(model_T)
                filename = config.name + '_best.pth.tar'
                ckpt_path = os.path.join(config.ckpt_dir, filename)
                ckpt = torch.load(ckpt_path) 
                model.load_state_dict(ckpt['model_state'])
                
                data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=config.training.batch_size, 
                    sampler=RandomSampler(dataset), collate_fn = collate_pad,
                    num_workers=config.training.num_workers, 
                    pin_memory=kwargs['pin_memory'],)
                
                ## Data loop
#                batch_size = config.training.batch_size
                model.eval()
                if config.gpu: model.cuda()
                data_loader.dataset.test()
                cos = nn.CosineSimilarity(dim=2)
                
                for i, (x, y, y_locs) in enumerate(data_loader):
                    with torch.no_grad():
                        if config.gpu:
                            y = y.cuda()
                        x, y = Variable(x), Variable(y)
                        if model.require_locs: x = (x, y_locs)
                        
                        # classify batch
                        log_probas, locs, log_pi, baselines = model(x)
                        
                        # compute cosine distances
                        fov_h_deltas = model.fov_h[:,1:,...].flatten(2) - model.fov_h[:,:-1,...].flatten(2)
#                        fov_ph_deltas = model.fov_h[:,1:,...].flatten(2) - model.fov_ph[:,:-1,...].flatten(2) #XXX HERE WAS A BUG
                        fov_ph_deltas = model.fov_ph[:,:-1,...].flatten(2) - model.fov_h[:,:-1,...].flatten(2) #here it is fixed
                        
                        imaginary_cos = 1 - cos(fov_h_deltas, fov_ph_deltas)
                        print("i_cos:", imaginary_cos.mean())
                        
                        successive_delta_cos = 1 - cos(fov_h_deltas[:,1:,...], fov_h_deltas[:,:-1,...])
                        print("sd_cos:", successive_delta_cos.mean())
                        
                        successive_cos = 1 - cos(model.fov_h[:,1:,...].flatten(2), model.fov_h[:,:-1,...].flatten(2))
                        print("s_cos:", successive_cos.mean())
                        
                        sid_cos = 1 - cos(fov_ph_deltas[:,1:,...], fov_ph_deltas[:,:-1,...])
                        print("sid_cos:", sid_cos.mean())
                        
                        sph_cos = 1 - cos(model.fov_ph[:,1:,...].flatten(2), model.fov_ph[:,:-1,...].flatten(2))
                        print("sph_cos:", sph_cos.mean())
                        
                        spout_cos = 1 - cos(model.p_out[:,1:,...].flatten(2), model.p_out[:,:-1,...].flatten(2))
                        print("spout_cos:", spout_cos.mean())
                        
                        sWWfov_cos = 1 - cos(model.WWfov[:,1:,...].flatten(2), model.WWfov[:,:-1,...].flatten(2))
                        print("sWWfov_cos:", sWWfov_cos.mean())
                        
#                        p_target = model.WWfov[:,1:,...]
                        p_target = model.L3[:,1:,...]
                        loss_cos = 1 - cos(p_target.flatten(2), model.p_out[:,:-1,...].flatten(2))
                        print("loss_cos:", loss_cos.mean())
                        
                        short_cos = 1 - cos(p_target[:,:,:,0], model.p_out[:,:-1,:,0])
                        print("short_cos:", short_cos.mean())
                        
                        FE_cos = 1 - cos(model.WWfov[:,1:,...].flatten(2), model.WWfov[:,:-1,...].flatten(2))
                        print("FE_cos:", FE_cos.mean())
                        
                        FE_short_cos = 1 - cos(model.WWfov[:,1:,:,0], model.WWfov[:,:-1,:,0])
                        print("FE_short_cos:", FE_short_cos.mean())
                        
#                        short_i_cos = 1 - cos(fov_h_deltas[:,:,:,0], fov_ph_deltas[:,:,:,0])
#                        print("short_i_cos:", torch.mean(torch.sum(short_i_cos, dim=1)))
                        print(" ")
                        if i == 4: return                        

def features(config):
    """Compare feature maps across time using cosine similarity and MAE"""
    # Seeds, transform and config
    config.tensorboard = False
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
    model_T = 3
    modelstring = "TEST-{}"
    fixation_set = "MHL3" #MHL3 is [3,0,4]
    
    config.name = modelstring.format(model_T)
    feature_extractor = ResNet18_module(blocks=4, maxpool=False, stride=True)
    
    model = WW_RAM(config.name, config.RAM.std, retina, feature_extractor, 
                   config.gpu, fixation_set = fixation_set)
    model.set_timesteps(model_T)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size, 
        sampler=RandomSampler(dataset), collate_fn = collate_pad,
        num_workers=config.training.num_workers, 
        pin_memory=kwargs['pin_memory'],)
    
    tester = Tester(config, loader, model)
    cos = nn.CosineSimilarity(dim=1)
    mae = nn.L1Loss(reduction='none')
    g = tester.get_feature_vectors()
    g_flat = g.flatten(2)
    cos_dist, mae_dist = [], []
    
    for t in range(model_T-1):
        g_now = g_flat[:,t,:]
        g_next = g_flat[:,t+1,:]
        cos_dist.append(1 - cos(g_now, g_next))
        mae_dist.append(torch.mean(mae(g_now, g_next), dim=1))
    
    print("Cosine distances:")
    print(torch.stack(cos_dist).T)
    print("MAE distances:")
    print(torch.stack(mae_dist).T)

def vis(config):
    """Visualize full image, masked out fixations and extracted patches, fovea-only"""
    # Seeds, transform and config
    config.tensorboard = False
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
    model_T = 3
    modelstring = "TEST-{}"
    fixation_set = "MHL3" #MHL3 is [3,0,4]
    
    config.name = modelstring.format(model_T)
    feature_extractor = ResNet18_module(blocks=4, maxpool=False, stride=True)
    
    model = FF_GlimpseModel(config.name, retina, feature_extractor, 1,
                            config.gpu, fixation_set = fixation_set)
    model.set_timesteps(model_T)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size, 
        sampler=RandomSampler(dataset), collate_fn = collate_pad,
        num_workers=config.training.num_workers, 
        pin_memory=kwargs['pin_memory'],)
    
    print("Visualizing fixation mode", fixation_set)
    tester = Tester(config, loader, model)
    x, masked, concat = tester.visualize_sensor(eval_set = False)
    
    for Bid in range(len(x)):
        # Visualize input image
        im_x = (x[Bid] - x[Bid].min()).numpy()
        im_x = np.moveaxis(im_x, 0, -1)/im_x.max()
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(im_x)
        plt.show()
        
        # Visualize masked image
        im_masked = (masked[Bid] - masked[Bid].min()).numpy()
        im_masked = np.moveaxis(im_masked, 0, -1)/im_masked.max()
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(im_masked)
        plt.show()
        
        # Visualize image patches
        im_concat = (concat[Bid] - concat[Bid].min()).numpy()
        im_concat = np.moveaxis(im_concat, 0, -1)/im_concat.max()
        plt.figure(figsize=(model_T*2,4))
        plt.axis('off')
        plt.imshow(im_concat)
        plt.show()
    
def val(config):
    """old main() that validated checkpoints on custom fixation orders"""
    # Seeds, transform and config
    config.tensorboard = False
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
                              config.RAM.scaling, config.gpu, clamp=False)
#    memory = lstm_
    
#    model_T = 4
    modelstrings = ["retina-WWRAMfix-la2streamv18-{}"]
    fixation_sets = ["MHL3"]
    
#    for model_T in range(1,6):
    for model_T in [3]:
        for M in modelstrings:
            config.name = M.format(model_T)
            print(" ")
            print("Validating ", config.name, "...")
            feature_extractor = ResNet18_module(blocks=4, maxpool=False, stride=True)
            
            for fixation_set in fixation_sets:
                print("         ... with fixation set", fixation_set)
                model = WW_RAM(config.name, config.RAM.std, retina, feature_extractor, 
                            config.gpu, fixation_set = fixation_set)
#                model = RAM_baseline(config.name, config.RAM.std, retina, 
#                                     feature_extractor, memory, 0, config.gpu, 
#                                     fixation_set = fixation_set)
                model.set_timesteps(model_T)
                filename = config.name + '_best.pth.tar'
                ckpt_path = os.path.join(config.ckpt_dir, filename)
                ckpt = torch.load(ckpt_path) 
                model.load_state_dict(ckpt['model_state'])
                
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=config.training.batch_size, 
                    sampler=RandomSampler(dataset), collate_fn = collate_pad,
                    num_workers=config.training.num_workers, 
                    pin_memory=kwargs['pin_memory'],)
                
#                tester = Tester(config, loader, model)
#                tester.validate(1)
                trainer = Trainer(config, loader, model)
                trainer.validate(1)

class Tester(object):
    def __init__(self, C, data_loader, model):
        self.C = C
        
        # data loader params
        self.data_loader = data_loader
        self.num_classes = self.data_loader.dataset.num_classes
        self.num_train = len(self.data_loader.dataset)
        self.num_valid = len(self.data_loader.dataset.test_data)
        self.batch_size = self.C.training.batch_size
        
        # model
        self.model = model
        if C.gpu: self.model.cuda()
        
        self.zero = torch.tensor([0], device='cuda')
        
    def validate(self, epoch):
            """An extended version of Trainer's validate fn"""
            self.model.eval()
            self.data_loader.dataset.test()
            
            # Averages for logs
            loss_t = AverageMeter()
            accs = [AverageMeter() for i in range(self.model.timesteps)]
            
            for i, (x, y, y_locs) in enumerate(self.data_loader):
                with torch.no_grad():
                    if self.C.gpu:
                        y = y.cuda()
                    x, y = Variable(x), Variable(y)
                    if self.model.require_locs: x = (x, y_locs)
                    
                    # classify batch
                    log_probas, locs, log_pi, baselines = self.model(x)
                    
                    losses = self.model.loss(log_probas, log_pi, baselines, y, locs, y_locs)
                    total_loss = sum(losses) if type(losses) is tuple else losses
                    
                    # compute accuracies
                    correct = torch.max(log_probas, 2)[1].detach() == y.unsqueeze(1)
                    acc = 100 * (correct.sum(0).float() / self.batch_size)
                    
                    # update avgmeters
                    loss_t.update(total_loss.item(), self.batch_size)
                    for i in range(self.model.timesteps):
                        accs[i].update(acc[i].item(), self.batch_size)
                        
            print("Final accuracy: {:.2f} | Loss: {:.3f}".format(
                    accs[-1].avg, loss_t.avg))
            print("Intermediate accs:", " ".join(["{:.2f}".format(a.avg) for a in accs[:-1]]))
            return loss_t.avg, accs[-1].avg
    
    def get_random_batch(self, eval_set):
        # Prepare model and choose data split
        if eval_set: 
            self.model.eval()
            self.data_loader.dataset.test()
            num = self.num_valid
        else: 
            self.data_loader.dataset.train()
            num = self.num_train
        
        # Sample a random batch
        i = randint(1, num//self.batch_size)
        it = iter(self.data_loader)
        for _ in range(i):
            x, y, y_locs = it.next()
        y = y.cuda()
        
        return x, y, y_locs

    def get_feature_vectors(self, eval_set = True):
        """
        Return g_t from each timestep. Only works on RAM and WWRAM models.
        """
        x, y, y_locs = self.get_random_batch(eval_set)
        model = self.model
        T = model.T
        assert hasattr(model, "glimpse_module")
        guide = Guide(model.timesteps, model.training, None, model.fixation_set)
        g = torch.zeros(((x.shape[0], model.timesteps) + model.glimpse_module.out_shape))
        
        l_t_prev = model.retina.to_exocentric_action(T, guide(y_locs, 0))
        for t in range(model.timesteps):
            g[:,t,...] = model.glimpse_module(x, l_t_prev)[0]
        l_t_prev = model.retina.to_egocentric_action(guide(y_locs, t+1) - model.retina.fixation) 
        
        return g

    def visualize_sensor(self, eval_set = True):
        """
        Visualize sensor outputs: full image, masked image, image patches.
        Fovea only.
        """
        x, y, y_locs = self.get_random_batch(eval_set)
        B, C, H, W = x.shape
        model = self.model
        R = model.retina
        size = 37
        R.reset()
        guide = Guide(model.timesteps, model.training, None, model.fixation_set)
        l_t_prev = R.to_exocentric_action(model.T, guide(y_locs, 0))
        
        # Produce masked image
        masks = torch.zeros_like(x)
        for t in range(model.timesteps):
            if R.fixation is None:
                coords = R.to_exocentric(model.T, l_t_prev)
            else: coords = R.to_egocentric(model.T, l_t_prev)
            R.fixation = coords
            
            from_x, from_y = coords[:, 1] - (size // 2), coords[:, 0] - (size // 2)
            to_x, to_y = from_x + size, from_y + size
            from_x = torch.max(self.zero, from_x)
            from_y = torch.max(self.zero, from_y)
            to_y = torch.min(model.T[0], to_y)
            to_x = torch.min(model.T[1], to_x)
            for b in range(B):
                masks[b, :, from_y[b]:to_y[b], from_x[b]:to_x[b]] = 1
            l_t_prev = R.to_egocentric_action(guide(y_locs, t+1) - R.fixation) 
        masked = x * masks
        
        # Extract and concatenate patches
        R.reset()
        guide = Guide(model.timesteps, model.training, None, model.fixation_set)
        l_t_prev = R.to_exocentric_action(model.T, guide(y_locs, 0))
        patches = []
        for t in range(model.timesteps):
            phi = R.foveate_ego(x, l_t_prev)[:,0,:,:,:]
            patches.append(phi)
            l_t_prev = R.to_egocentric_action(guide(y_locs, t+1) - R.fixation) 
        concat = torch.cat(patches, dim=3).squeeze().cpu()
        
        return x, masked, concat

    def occlusion_check(self, eval_set = False):
        """
        Tests for occluded parts in the dataset.
        """
        # Choose data split
        if eval_set: 
            self.data_loader.dataset.test()
            num = self.num_valid
        else: 
            self.data_loader.dataset.train()
            num = self.num_train
            
        # Iterate
        O = False
        it = iter(self.data_loader)
        for x, y, y_locs in it:
            if O is False: O = y_locs[:,:,-1]
            else: O = torch.cat((O,y_locs[:,:,-1]))
        
        return num-torch.sum(O,dim=0)
    
    def visualize_gaze(self, eval_set = False):
        """
        Visualizes model's gaze path on a random batch of training images.
        """
        # Sample a batch
        x, y, y_locs = self.get_random_batch(eval_set)
        if self.model.require_locs: x = (x, y_locs)
        
        # Feed through model
        log_probas, locs, log_pi, baselines = self.model(x)
        correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
        
        # Compute absolute gaze locations
        S = RAM_sensor()
        S.width = self.model.retina.width
        T = torch.tensor((500,500)).cuda()
        abs_locs = torch.zeros_like(locs[:,:-1])
        
        for timestep in range(locs.shape[1]-1):
            if timestep==0: abs_locs[:,timestep,:] = S.to_exocentric(T, locs[:,0,:])
            else:
                S.fixation = abs_locs[:,timestep-1,:]
                abs_locs[:,timestep,:] = S.to_egocentric(T, locs[:,timestep,:])
        
        # Retrieve to CPU
        if self.model.require_locs: x = x[0].cpu().numpy()
        else: x = x[0].cpu().numpy()
        abs_locs = abs_locs.detach().cpu().numpy()
        
        for ind, im in enumerate(x):
            # Normalize
            im = np.moveaxis(im, 0, -1) - im.min()
            im = im/im.max()
            
            # Visualize sensor fov
            if ind==0:
                fov_rect = bounding_box(250, 250, self.C.RAM.foveal_size)
                per_rect = bounding_box(250, 250, self.C.RAM.foveal_size * self.C.RAM.scaling)
                fig, ax = plt.subplots(1, figsize=(8,8))
                ax.axis('off')
                ax.imshow(im)
                ax.add_patch(fov_rect)
                ax.add_patch(per_rect)
                plt.show()
                
            im_locs = abs_locs[ind]
            print(locs[ind])
            print(im_locs)
            
            plt.figure(figsize=(8,8))
            plt.axis('off')
            plt.imshow(im)
            
            # Plot gaze path on top of images
            plt.plot(im_locs[:,1], im_locs[:,0], c='g')
            plt.scatter(im_locs[0,1], im_locs[0,0], c='r', s=60)
            plt.scatter(x=im_locs[1:,1], y=im_locs[1:,0], s=60)
            
            # Classification accuracy indicator
            colour = 'r' if not correct[ind] else 'g'
            marker = 'X' if not correct[ind] else 'P'
            plt.scatter(475, 25, s = 200, c = colour, marker = marker)
            plt.show()
            
    def visualize_ylocs(self, eval_set = False):
        """
        
        """
        # Sample a batch
        x, y, y_locs = self.get_random_batch(eval_set)
        if self.model.require_locs: x = (x, y_locs)
        
        # Feed through model
        log_probas, locs, log_pi, baselines = self.model(x)
#        correct = torch.max(log_probas[:,-1], 1)[1].detach() == y
        
        # Compute absolute gaze locations
        S = RAM_sensor()
        S.width = self.model.retina.width
        T = torch.tensor((500,500)).cuda()
#        abs_locs = torch.zeros_like(locs[:,:-1])
        
        locs2 = []
        l_t_prev = S.to_exocentric_action(T, guide(y_locs, 0))
        S.fixation = S.to_exocentric(T, l_t_prev)
        locs2.append(l_t_prev)
        
        for t in range(5):
            l_t_prev = S.to_egocentric_action(guide(y_locs, t+1) - S.fixation)
            S.fixation = S.to_egocentric(T, l_t_prev)
            locs2.append(l_t_prev)
        
        locs2 = torch.stack(locs2).transpose(1, 0)
        
        print(locs2.shape, locs.shape)
#        for i in range(len(locs2)):
        i = 0
        print("hand: ", locs2[i])
        print("model: ", locs[i])
        print("y_locs: ", y_locs[i])
        abs_locs = torch.zeros_like(locs)
        abs_locs[:,0,:] = S.to_exocentric(T, locs[:,0,:])
        print("abs_locs: ", abs_locs[i])
        
#        fix_locs = torch.zeros_like(locs)
#        G = guide(y_locs, 0)
#        
#        fix_locs[:,0,:] = S.to_exocentric_action(T,)

def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False)
    return rect

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
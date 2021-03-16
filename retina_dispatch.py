#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:40:09 2020

Retina training script, following curriculum learning from dispatch.py.

@author: piotr
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data.sampler import RandomSampler

from utils import get_ymlconfig, showTensor
from CUB_loader import CUBDataset, collate_pad
from trainer import Trainer
from tester import Tester
from model import CUBRAM_baseline, ff_r18, RAM_baseline
from modules import retinocortical_sensor, crude_retina, ResNet18_short, FC_RNN, lstm

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

    # 0 - Prepare modules
#    retina = retinocortical_sensor()
    retina = crude_retina(config.RAM.foveal_size, config.RAM.n_patches, 
                          config.RAM.scaling, config.gpu)
    feature_extractor = ResNet18_short(layers=4, maxpool=False)
    memory = lstm #retima-RAM-hardcoded-noavgnomaxLSTM
#    memory = FC_RNN
    
#    # 1 - Pretrain feature extractor
#    config.name = "curriculum-r18-retina"
#    r18 = ff_r18(retina=retina, pretrained=True)
#    r18_trainer = Trainer(config, loader, r18)
#    r18_trainer.train()
#    r18_weights = r18.resnet18.state_dict()
    
    # 1 - Load in pretrained feature extractor
#    w_path = os.path.join(config.ckpt_dir, config.name+"_best.pth.tar")
#    r18 = ff_r18()
#    r18.load_state_dict(torch.load(w_path)['model_state'])
#    r18_weights = r18.resnet18.state_dict()
   
    # 2 - Train RAM, iteratively increasing number of timesteps
    namestring = "retina-RAM-hardcoded-nocurriculumfix2-{}"
    config.name = namestring.format(0)
#    model = CUBRAM_baseline(config.name, config.RAM.std, retina, config.gpu)
#    model = RAM_baseline(config.name, config.RAM.std, retina, 
#                         feature_extractor, memory, config.gpu)
    
    #transfer pretrained weights
#    model.glimpse_module.feature_extractor.load_weights(r18_weights)
    
    #cleanup
#    del(r18)
    
    #train
    for steps in range(1,6):
#        if steps != 2:
#            best_path = os.path.join(config.ckpt_dir, config.name+"_best.pth.tar")
#            model.load_state_dict(torch.load(best_path)['model_state'])
        model = RAM_baseline(config.name, config.RAM.std, retina, 
                             feature_extractor, memory, config.gpu)
        print("Now training {}-step RAM.\n".format(steps))
        model.set_timesteps(steps)
        config.name = namestring.format(steps)
        trainer = Trainer(config, loader, model)
        trainer.train()
    #TODO config file restructure
    #TODO config class seems to triplicate dicts
    
##    #visualize gaze path for a random batch
#    timesteps = 5
#    config.name = namestring.format(timesteps)
#    best_path = os.path.join(config.ckpt_dir, config.name+"_best.pth.tar")
#    model.load_state_dict(torch.load(best_path)['model_state'])
#    model.set_timesteps(timesteps)
##    
##    from tester import Tester
#    tester = Tester(config, loader, model)
#    tester.visualize_ylocs()
#    tester.visualize_gaze()

if __name__ == '__main__':
    config = get_ymlconfig('./dispatch.yml')
    main(config)
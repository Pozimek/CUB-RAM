#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 19:35:13 2022

A script for parsing and visualizing tensorboard logs for chapter 3 experiments
(literature review and framework).

To be used with Python 3.7.

Resources:
https://www.codecademy.com/article/seaborn-design-i

@author: piotr
"""

from tbparse import SummaryReader
from utils import get_ymlconfig
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

## Parse logs

config = get_ymlconfig('./3i_dispatch.yml')
parsed = {}

for variant in [[True, True, False, True, False],   #1a. ego prop RL
                [False, True, False, True, False]#,  #1b. exo prop RL
#                [True, True, True, True, False],    #1c. ego prop HC 
#                [False, True, True, True, False],   #1d. exo prop HC
#                [True, False, False, False, False], #2a. ego wide RL
#                [True, False, False, True, False],  #2b. ego narrow RL
#                [True, False, True, False, False],  #2c. ego wide HC
#                [True, False, True, True, False],   #2d. ego narrow HC
#                [True, False, True, True, True]     #3. retina HC
                ]:
    V = {}
    name = "ch3"
    var = "EGO{}PROP{}HC{}NRW{}RET{}".format(*[int(i) for i in variant])
    
    for seed in [1]:
        name += "-s{}".format(seed)
        name += var
        print(var)
        
        log_dir = config.log_dir + name
        reader = SummaryReader(log_dir)
        S = {}
        for key in list(reader.children.keys())[:-2]: #log tags
            df = reader[key].scalars
#            print(key)
            y = df.value.to_numpy()
            x = df.step.to_numpy()
            S[key] = (x, y)
        
        V[seed] = S
    parsed[var] = V
    
        
## Renaming tables
TAGS = {'Accuracy (Detailed)_Training_Train_acc':       'Train acc (sharp)',
        'Accuracy (Detailed)_Training_Val_acc':         'Val acc (sharp)',
        'Loss (Detailed)_Training_Train_loss':          'Train loss (sharp)',
        'Loss (Detailed)_Training_Val_loss':            'Val loss (sharp)',
        'Partial Losses_Training_Base_Loss_train':      'Train base loss',
        'Partial Losses_Training_Base_Loss_val':        'Val base loss',
        'Partial Losses_Training_Baseline_train':       'Train baseline',
        'Partial Losses_Training_Baseline_val':         'Val baseline',
        'Partial Losses_Training_Class_Loss_train':     'Train class loss',
        'Partial Losses_Training_Class_Loss_val':       'Val class loss',
        'Partial Losses_Training_Reward_train':         'Train reward',
        'Partial Losses_Training_Reward_val':           'Val reward',
        'Smoothed Results_Accuracies_Train_accuracy':   'Train acc (smooth)',
        'Smoothed Results_Accuracies_Valid_accuracy':   'Val acc (smooth)',
        'Smoothed Results_LR_Learning_rate':            'LR',
        'Smoothed Results_Losses_Train_loss':           'Train loss (smooth)',
        'Smoothed Results_Losses_Valid_loss':           'Val loss (smooth)',
        'Smoothed Results_Time_Time_elapsed':           'Time elapsed'
        }

#apply table
for variant in parsed.keys():
    for seed in parsed[variant].keys():
        for T in TAGS.keys():
            parsed[variant][seed][TAGS[T]] = parsed[variant][seed].pop(T)

##prep
#d = {}
#for k in parsed.keys():
#    d[k] = ''
#
#VARIANTS = {} #TODO: paste in d
      
  
## Compute mean and std where applicable 

#(not time or lr)
NA = ['Time elapsed', 'LR']
tags = list(TAGS.values())
_ = [tags.remove(i) for i in NA]

STATS = {}
for variant in parsed.keys():
    V = {}
    for tag in tags:
        T = {}
        y = []
        for seed in parsed[variant].keys():
            y.append(parsed[variant][seed][tag][1])
        
        #match shapes by padding with numpy.nan
        y = [np.pad(j, (0, int(max([len(i) for i in y])) - len(j)), 'constant',
                    constant_values = np.nan) for j in y]
        y = np.stack(y)

        T['std'] = np.nanstd(y,0)
        T['u'] = np.nanmean(y,0)
        V[tag] = T
    STATS[variant] = V


## Visualize
def vis_compare(var_names, var_labels, metrics, metric_labels, title, 
                size = (8,8)):
    """ Visualize multiple variants' metrics on a single plot"""
    sns.set_style('whitegrid') #darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("notebook") #paper talk poster notebook
    #TODO: smoothing?
    #TODO: make title bigger
    #TODO: axis labels
    plt.figure(figsize=size)
    plt.title(title)
    
    for i, variant in enumerate(var_names):
        for j, metric in enumerate(metrics):
            u = STATS[variant][metric]['u']
            std = STATS[variant][metric]['std']
            x = np.arange(len(u))
            
            plt.plot(x, u, label=var_labels[i]+' - '+metric_labels[j])
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()
    
variants = ['EGO1PROP1HC0NRW1RET0', 'EGO0PROP1HC0NRW1RET0']
var_labels = ['EGO1','EGO0']
trainval = ['Training', 'Validation']
vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
            trainval,
            "Losses (smooth)", size=(8,5))

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            trainval,
            "Accuracies (smooth)", size=(8,5))
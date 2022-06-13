#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:57:15 2022

A script for parsing and visualizing tensorboard logs for ch 4 experiments
(Active Vision Memory), feedforward aggregation strategies evaluation.

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

config = get_ymlconfig('./4ii_dispatch.yml')
parsed = {}

for strategy in range(1,6):
    V = {}
    var = "STRAT{}".format(strategy)
    
    for seed in [1, 9, 919]:
        name = "ch4ii-s{}".format(seed)
        name += var
        print(name)
        
        log_dir = config.log_dir + name
        reader = SummaryReader(log_dir)
        S = {}
        
        for key in list(reader.children.keys())[:10]: #log tags
            df = reader[key].scalars
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
#        'Partial Losses_Training_Base_Loss_train':      'Train base loss',
#        'Partial Losses_Training_Base_Loss_val':        'Val base loss',
#        'Partial Losses_Training_Baseline_train':       'Train baseline',
#        'Partial Losses_Training_Baseline_val':         'Val baseline',
#        'Partial Losses_Training_Class_Loss_train':     'Train class loss',
#        'Partial Losses_Training_Class_Loss_val':       'Val class loss',
#        'Partial Losses_Training_Reward_train':         'Train reward',
#        'Partial Losses_Training_Reward_val':           'Val reward',
        'Smoothed Results_Accuracies_Train_accuracy':   'Train acc (smooth)',
        'Smoothed Results_Accuracies_Valid_accuracy':   'Val acc (smooth)',
        'Smoothed Results_LR_Learning_rate':            'LR',
        'Smoothed Results_Losses_Train_loss':           'Train loss (smooth)',
        'Smoothed Results_Losses_Valid_loss':           'Val loss (smooth)',
        'Smoothed Results_Time_Time_elapsed':           'Time elapsed'
        }

#apply TAGS table
for variant in parsed.keys():
    for seed in parsed[variant].keys():
        for T in TAGS.keys():
            parsed[variant][seed][TAGS[T]] = parsed[variant][seed].pop(T)

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

# Save
#fname = 'stats_ch4ii.npy'
#np.save(fname,STATS)
#STATS = np.load(fname, allow_pickle=True)[()]

                 ### WIP ###
                 # Visualize plots
                 # Get peak val accs

## Visualize
def vis_compare(var_names, var_labels, metrics, title, axes, 
                metric_labels = ['Training', 'Validation'], title_size=15, 
                size = (8,5), v=False, y_lim = None):
    """ Visualize multiple variants' metrics on a single plot"""
    sns.set_style('darkgrid') #darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("notebook") #paper talk poster notebook
    #TODO: smoothing?
    plt.figure(figsize=size)
    plt.title(title, fontsize = title_size, wrap=True)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    
    if y_lim is not None:
        plt.ylim([y_lim[0], y_lim[1]])
    
    for i, variant in enumerate(var_names):
        for j, metric in enumerate(metrics):
            u = STATS[variant][metric]['u']
            std = STATS[variant][metric]['std']
            x = np.arange(len(u))
            
            #test
            if v:
                print(len(std) == len(u))
                print(var_labels[i]+' - '+metric_labels[j])
                print(std.max(), std.mean(), std.min(), len(std))
            
            plt.plot(x, u, label=var_labels[i]+' - '+metric_labels[j])
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()
    

### 1 - All strategies, train/val all on one plot.
strats = ['STRAT{}'.format(i) for i in range(1,6)]
STRATS = ["Unmasking", "Spatial Concatenation", "Feature Averaging", 
          "Output (softmax) Averaging", "Output (pre-softmax) Averaging"]
loc_labels = STRATS
vis_compare(strats, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by a ResNet18 utilizing different \n aggregation strategies",
            ["Epoch","Loss"], y_lim=[-0.5,14])

vis_compare(strats, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by a ResNet18 utilizing different \n aggregation strategies",
            ["Epoch","Accuracy (%)"])

#max mean acc
for i, part in enumerate(strats):
    u = STATS[part]['Val acc (smooth)']['u']
    print(loc_labels[i], u.max())
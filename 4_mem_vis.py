#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:07:21 2022

A script for parsing and visualizing tensorboard logs for ch 4 experiments
(Active Vision Memory), memory variants evaluation.

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
#0:T123 (+ T3), 1:T123+curr, 2:T3+intermediate 3:T3+BN-LSTM 

#training: ch4methods-s919VAR0T1
#eval: ch4methods-s919VAR0T3evalT1

config = get_ymlconfig('./4_mem_dispatch.yml')

#parse logs
parsed = {}
for variant in range(4):
    V = {}
    var = "VAR{}".format(variant)
    
    for seed in [1,9,919]:
        S = {}

        for timestep in [1,2,3]:
            t = {}
            name = "ch4mem-s{}".format(seed) + var + "T{}".format(timestep)
            print(name)
            
            log_dir = config.log_dir + name
            reader = SummaryReader(log_dir)
            
            for key in list(reader.children.keys())[:-2]: #log tags
                df = reader[key].scalars
                y = df.value.to_numpy()
                x = df.step.to_numpy()
                t[key] = (x, y)
            S[timestep] = t
        V[seed] = S
    parsed[var] = V

## Renaming table
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

#apply TAGS table
for variant in parsed.keys():
    for seed in parsed[variant].keys():
        for timestep in [1,2,3]:
            for T in TAGS.keys():
                parsed[variant][seed][timestep][TAGS[T]] = parsed[variant][seed][timestep].pop(T)
          
            
## Compute mean and std where applicable 

#(not for time or lr)
NA = ['Time elapsed', 'LR']
tags = list(TAGS.values())
_ = [tags.remove(i) for i in NA]

STATS = {}
for variant in parsed.keys():
    V = {}
    for timestep in [1,2,3]:
        t = {}
        for tag in tags:
            TAG = {}
            y = []
            for seed in parsed[variant].keys():
                y.append(parsed[variant][seed][timestep][tag][1])
            
            #match shapes by padding with numpy.nan
            y = [np.pad(j, (0, int(max([len(i) for i in y])) - len(j)), 'constant',
                        constant_values = np.nan) for j in y]
            y = np.stack(y)
    
            TAG['std'] = np.nanstd(y,0)
            TAG['u'] = np.nanmean(y,0)
            t[tag] = TAG
        V[timestep] = t
    STATS[variant] = V


# Compute best T123 results (max acc, lowest loss). 
source = ['Val acc (smooth)', 'Val loss (smooth)', 'Train acc (smooth)', 'Train loss (smooth)']
target = ['Val acc (sharp)', 'Val loss (sharp)', 'Train acc (sharp)', 'Train loss (sharp)']
MHL_STATS = {}
# get values
for variant in parsed.keys():
    V = {}
    for i, tag in enumerate(source):
        TAG = {}
        U, STD = [], []
        for t in [1,2,3]:
            u = STATS[variant][t][tag]['u']
            std = STATS[variant][t][tag]['std']
            u = u.max() if i==0 or i == 2 else u.min()
            std = std.max() if i==0 or i == 2 else std.min()
            U.append(u)
            STD.append(std)
        
        TAG['u'] = np.array(U)
        TAG['std'] = np.array(STD)
        V[target[i]] = TAG
    MHL_STATS[variant] = V

# Save
#fname = 'stats_ch4methods.npy'
#np.save(fname,STATS)
#STATS = np.load(fname, allow_pickle=True)[()]

## Visualize
def vis_compare(STATS, var_names, timesteps, var_labels, metrics, title, axes, 
                metric_labels = ['Training', 'Validation'], title_size=15, 
                size = (8,5), v=False, y_lim = None, v_time = False):
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
        for t in timesteps[i]:
            for j, metric in enumerate(metrics):
                u = STATS[variant][t][metric]['u']
                std = STATS[variant][t][metric]['std']
                x = np.arange(len(u))
                
                #test
                if v:
                    print(len(std) == len(u))
                    print(var_labels[i]+' - '+metric_labels[j])
                    print(std.max(), std.mean(), std.min(), len(std))
                
                tstring = '@'+str(t) if v_time else ''
                plt.plot(x, u, 
                         label=var_labels[i]+tstring+' - '+metric_labels[j])
                plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()

def vis_MHL(STATS, var_names, var_labels, metrics, title, axes,
            metric_labels = None, title_size=15, 
            size = (8,5), v=False, y_lim = None, MHL=None, combined = None):
    """ Visualize multiple variants' MHL metrics on a single plot"""
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
            
            label = var_labels[i] 
            label += ' - ' + metric_labels[j] if metric_labels else ''
            plt.plot(x, u, marker='o', label=label)
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    if MHL: plt.plot(x, MHL, ':.', label="Individual Policy Values")
    if combined: plt.plot(x, combined, '--', 
                          label="Combined Value (Spatial Concatenation)")
    
    plt.xticks(np.array([0,1,2]), ["[medium]", "[medium, high]","[medium, high, low]"])
    plt.legend()
    plt.show()

### 1 - Eval: MHL plots for VAR0 (training val data), VAR0[T3], VAR1,2,3
variants = ['VAR{}'.format(i) for i in range(4)]
VARIANTS = ["WW-LSTM", "WW-RNN", "LSTM", "RNN"]
loc_labels = VARIANTS
MHL = [41,61,12]
combined = [71]*3

vis_MHL(MHL_STATS, variants, loc_labels,
            ['Val loss (sharp)', 'Train loss (sharp)'],
            "Lowest validation losses obtained by different memory variants",
            ["Policy sequence","Loss"], metric_labels = ['Validation', 'Training'])

vis_MHL(MHL_STATS, variants, loc_labels,
            ['Val acc (sharp)'],
            "Peak validation accuracies obtained by different memory variants",
            ["Policy sequence","Accuracy (%)"], MHL = MHL, 
            combined = combined)

### 2 - Training: Loss/Acc for T3 of all VARs (for your own use)
variants = ['VAR{}'.format(i) for i in range(4)]
timesteps = [[3], [3], [3], [3]]
VARIANTS = ["WW-LSTM", "WW-RNN", "LSTM", "RNN"]
loc_labels = VARIANTS

vis_compare(STATS, variants, timesteps, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by different memory variants \n with full policy sequence",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, timesteps, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by different memory variants \n with full policy sequence",
            ["Epoch","Accuracy (%)"], v_time = True)

##max mean acc
#for i, part in enumerate(variants):
#    u = STATS[part]['Val acc (smooth)']['u']
#    print(loc_labels[i], u.max())
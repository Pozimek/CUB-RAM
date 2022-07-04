#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:33:50 2022

A script for parsing and visualizing tensorboard logs for ch 4 experiments
(Active Vision Memory), training methods evaluation.

To be used with Python 3.7.

NOTES:
- unlike other vis scripts, this one has to deal with a non uniform log 
structure.
- All except 0 were ran through en eval script.
- have to parse both training loss/acc plots and eval script results
- eval script plotted in MHL

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

config = get_ymlconfig('./4_methods_dispatch.yml')

#training logs
parsed = {}
for variant in range(4):
    V = {}
    var = "VAR{}".format(variant)
    
    for seed in [919]:
        S = {}
        name = "ch4methods-s{}".format(seed) + var

        T = [1,2,3] if variant <= 1 else [3]
        for timestep in T:
            t = {}
            if variant == 1:
                name += "T{}".format(timestep)
            else:
                name = "ch4methods-s{}".format(seed) + var
                name += "T{}".format(timestep)
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

#eval logs
eval_parsed = {}
for variant in range(4):
    V = {}
    var = "VAR{}".format(variant)
    time = "T3" if variant != 1 else "T1T2T3"
    var += time
    
    for seed in [919]:
        S = {}
        name = "ch4methods-s{}".format(seed) + var

        for timestep in [1,2,3]:
            t = {}
            name = "ch4methods-s{}".format(seed) + var
            name += "evalT{}".format(timestep)
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
    eval_parsed[var] = V
    
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

e = np.array(list(TAGS.keys()))[[1,3,5,7,9,11]]
EVAL_TAGS = {k: TAGS[k] for k in e}

#apply TAGS table
for variant in parsed.keys():
    for seed in parsed[variant].keys():
        T = [1,2,3] if int(variant[-1]) <= 1 else [3]
        for timestep in T:
            for T in TAGS.keys():
                parsed[variant][seed][timestep][TAGS[T]] = parsed[variant][seed][timestep].pop(T)
            
for variant in eval_parsed.keys():
    for seed in eval_parsed[variant].keys():
        for timestep in [1,2,3]:
            for T in EVAL_TAGS.keys():
                eval_parsed[variant][seed][timestep][EVAL_TAGS[T]] = eval_parsed[variant][seed][timestep].pop(T)
          
            
## Compute mean and std where applicable 

#(not for time or lr)
NA = ['Time elapsed', 'LR']
tags = list(TAGS.values())
eval_tags = list(EVAL_TAGS.values())
_ = [tags.remove(i) for i in NA] #not present in EVAL_TAGS

STATS = {}
for variant in parsed.keys():
    V = {}
    steps = [1,2,3] if int(variant[-1]) <= 1 else [3]
    for timestep in steps:
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

#EVAL needs to compute MHL values from detailed losses here -.-
EVAL_STATS = {}
for variant in eval_parsed.keys():
    V = {}
    for tag in eval_tags:
        U, STD = [], []
        
        for timestep in [1,2,3]:
            y = []
            #collect y values across seeds
            for seed in eval_parsed[variant].keys():
                y.append(eval_parsed[variant][seed][timestep][tag][1])
            
            #match shapes by padding with numpy.nan
            y = [np.pad(j, (0, int(max([len(i) for i in y])) - len(j)), 'constant',
                        constant_values = np.nan) for j in y]
            y = np.stack(y)
            
            #compute epoch-wide mean and std
            means = np.nanmean(y, 1)
            STD.append(np.nanstd(means, 0))
            U.append(np.nanmean(means, 0))
        V[tag] = {'u': np.array(U), 'std': np.array(STD)}
    EVAL_STATS[variant] = V
            
# append missing T123 results (best, ie max). ouch :<
missing = ['Val acc (smooth)', 'Val loss (smooth)']
target = ['Val acc (sharp)', 'Val loss (sharp)']
# get values
V = {}
for i, tag in enumerate(missing):
    TAG = {}
    U, STD = [], []
    for t in [1,2,3]:
        u = STATS['VAR0'][t][tag]['u']
        std = STATS['VAR0'][t][tag]['std']
        u = u.max() if i==0 else u.min()
        std = std.max() if i==0 else std.min()
        U.append(u)
        STD.append(std)
    
    TAG['u'] = np.array(U)
    TAG['std'] = np.array(STD)
    V[target[i]] = TAG
EVAL_STATS['VAR0'] = V

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
            metric_labels = [''], title_size=15, 
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
            
            plt.plot(x, u, marker='o', 
                     label=var_labels[i])
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    if MHL: plt.plot(x, MHL, ':.', label="Individual Policy Values")
    if combined: plt.plot(x, combined, '--', 
                          label="Combined Value (Spatial Concatenation)")
    
    plt.xticks(np.array([0,1,2]), ["[medium]", "[medium, high]","[medium, high, low]"])
    plt.legend()
    plt.show()

### 1 - Eval: MHL plots for VAR0 (training val data), VAR0[T3], VAR1,2,3
variants = ['VAR0T3', 'VAR1T1T2T3', 'VAR2T3', 'VAR3T3', 'VAR0']
VARIANTS = ["Single Instance", "Curriculum Learning", "Intermediate Supervision", 
            "BN-LSTM", "Multiple Instances"]
loc_labels = VARIANTS
MHL = [41,61,12]
combined = [71]*3

vis_MHL(EVAL_STATS, variants, loc_labels,
            ['Val loss (sharp)'],
            "Lowest validation losses obtained by an LSTM with \n different approaches",
            ["Policy sequence","Loss"])

vis_MHL(EVAL_STATS, variants, loc_labels,
            ['Val acc (sharp)'],
            "Peak validation accuracies obtained by an LSTM with \n different approaches",
            ["Policy sequence","Accuracy (%)"], MHL = MHL, 
            combined = combined)
    
### 2 - Training: Loss/Acc for T3 of all VARs (for your own use)
variants = ['VAR{}'.format(i) for i in range(4)]
timesteps = [[3], [3], [3], [3]]
VARIANTS = ["Single Instance", "Curriculum", "Intermediate", "BN-LSTM", 
            "Multiple Instances"]
loc_labels = VARIANTS
    
vis_compare(STATS, variants, timesteps, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by an LSTM with different approaches",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, timesteps, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by an LSTM with different approaches",
            ["Epoch","Accuracy (%)"], v_time = True)

##max mean acc
#for i, part in enumerate(variants):
#    u = STATS[part]['Val acc (smooth)']['u']
#    print(loc_labels[i], u.max())
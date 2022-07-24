#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:54:33 2022

Chapter 5 (active vision attention, ch4 in latex) visualisation.

@author: piotr
"""
from tbparse import SummaryReader
from utils import get_ymlconfig
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

## Parse logs
#Mem: 0:WW_LSTM, 1:SpaCat
#Curriculum: 0: off, 1: on

config = get_ymlconfig('./5_curr_dispatch.yml')

#parse logs
parsed = {}
for mem in [0,1]:
    M = {}
    memory = "mem{}".format(mem)
    for curr in [0,1]:
        C = {}
        curriculum = "curr{}".format(curr)
        for seed in [1,9,919]:
            S = {}
            name = "ch5curr-s{}".format(seed) + memory + curriculum
            print(name)
            
            log_dir = config.log_dir + name
            reader = SummaryReader(log_dir)
            
            for key in list(reader.children.keys())[:-2]:
                df = reader[key].scalars
                y = df.value.to_numpy()
                x = df.step.to_numpy()
                S[key] = (x, y)
            C[seed] = S
        M[curr] = C
    parsed[mem] = M
        
    
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
for mem in parsed.keys():
    for curr in parsed[mem].keys():
        for seed in [1,9,919]:
            for T in TAGS.keys():
                parsed[mem][curr][seed][TAGS[T]] = parsed[mem][curr][seed].pop(T)


## Compute mean and std where applicable 

#(not for time or lr)
NA = ['Time elapsed', 'LR']
tags = list(TAGS.values())
_ = [tags.remove(i) for i in NA]

STATS = {}
for mem in parsed.keys():
    M = {}
    for curr in [0,1]:
        C = {}
        for tag in tags:
            TAG = {}
            y = []
            for seed in parsed[mem][curr].keys():
                y.append(parsed[mem][curr][seed][tag][1])
            
            #match shapes by padding with numpy.nan
            y = [np.pad(j, (0, int(max([len(i) for i in y])) - len(j)), 
                        'constant', constant_values = np.nan) for j in y]
            y = np.stack(y)
            
            TAG['std'] = np.nanstd(y,0)
            TAG['u'] = np.nanmean(y,0)
            C[tag] = TAG
        M[curr] = C
    STATS[mem] = M
    
# Save
#fname = 'stats_ch5curriculum.npy'
#np.save(fname,STATS)
##STATS = np.load(fname, allow_pickle=True)[()]
    
## Visualize
def vis_compare(STATS, variants, var_labels, metrics, title, axes, 
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
    
    for i, var in enumerate(variants):
        for j, metric in enumerate(metrics):
            u = STATS[var[0]][var[1]][metric]['u']
            std = STATS[var[0]][var[1]][metric]['std']
            x = np.arange(len(u))
            
            #test
            if v:
                print(len(std) == len(u))
                print(var_labels[i]+' - '+metric_labels[j])
                print(std.max(), std.mean(), std.min(), len(std))
            
            plt.plot(x, u, 
                     label=var_labels[i]+' - '+metric_labels[j])
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()
    

#variants = []
#for mem in [0,1]:
#    for curr in [0,1]:
#        variants.append("mem{}curr{}".format(mem, curr))
    

## 1 - WW-LSTM with and without curr
variants = [[0,0], [0,1]]
loc_labels = ["WW-LSTM", "WW-LSTM + curriculum"]

vis_compare(STATS, variants, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by a WW-LSTM with and without \n attention curriculum",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by a WW-LSTM with and without \n attention curriculum",
            ["Epoch","Accuracy (%)"], v_time = True)

## 2 - SpaCat with and without curr
variants = [[1,0], [1,1]]
loc_labels = ["SpaCat", "SpaCat + curriculum"]

vis_compare(STATS, variants, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by a spatial concatenation model \n with and without attention curriculum",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by a spatial concatenation model \n with and without attention curriculum",
            ["Epoch","Accuracy (%)"], v_time = True)

"""
>WW-LSTM
- curriculum appears slightly better in val losses but it's weird, identical in
val accs.
- curiously enough each curriculum stage is clearly indicated in the training 
loss but not in the val loss, where only the last one is visible
- WEIRDLY enough the last curriculum stage decreased val loss
- FE, mem etc could be tailored to final stage sensor params.
- LR scheduling is complicated for this kind of curriculum, might be confounding.
- Results suggest potential benefit, but val loss increases for both approaches
suggesting that this isn't enough to overcome the issues with RL.

>SpaCat
- curriculum appears better: slightly better acc and significantly better loss.
- Val loss can also be seen to drop with final stage and a bit with prior 
stages
- unlike in WW-LSTM val loss doesn't rise so much with curriculum - it stays 
more or less flat. Much less overfitting.
- w/o curr val loss rises

- Overall exploring curriculum is recommended: these results suggest it has 
potential across different memory mechanisms in RL. They don't suggest its
generalizability to non-RL methods for attention, but any hypothetical reasons
as to why it wouldn't generalize are not immediately obvious.
"""
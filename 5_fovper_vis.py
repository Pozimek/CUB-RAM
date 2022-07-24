#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 19:56:19 2022

Chapter 5 (active vision attention, ch4 in latex) visualisation.

Foveal vs foveal+peripheral experiment.

@author: piotr
"""
from tbparse import SummaryReader
from utils import get_ymlconfig
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

## Parse logs
#Mem: 0:WW_LSTM, 1:SpaCat
#Fovper: 0:fovonly, 1:fovper

config = get_ymlconfig('./5_fovper_dispatch.yml')

#    config.name = "ch5fovper"
#    config.name += "-s{}".format(config.seed)
#    config.name += "mem{}".format(config.vars.mem)
#    config.name += "fovper{}".format(config.vars.fovper)

#parse logs
parsed = {}
for mem in [0,1]:
    M = {}
    memory = "mem{}".format(mem)
    for fovper in [0,1]:
        C = {}
        sensor = "fovper{}".format(fovper)
        for seed in [1,9,919]:
            S = {}
            name = "ch5fovper-s{}".format(seed) + memory + sensor
            print(name)
            
            log_dir = config.log_dir + name
            reader = SummaryReader(log_dir)
            
            for key in list(reader.children.keys())[:-2]:
                df = reader[key].scalars
                y = df.value.to_numpy()
                x = df.step.to_numpy()
                S[key] = (x, y)
            C[seed] = S
        M[fovper] = C
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
    for fovper in parsed[mem].keys():
        for seed in [1,9,919]:
            for T in TAGS.keys():
                parsed[mem][fovper][seed][TAGS[T]] = parsed[mem][fovper][seed].pop(T)


## Compute mean and std where applicable 

#(not for time or lr)
NA = ['Time elapsed', 'LR']
tags = list(TAGS.values())
_ = [tags.remove(i) for i in NA]

STATS = {}
for mem in parsed.keys():
    M = {}
    for fovper in [0,1]:
        C = {}
        for tag in tags:
            TAG = {}
            y = []
            for seed in parsed[mem][fovper].keys():
                y.append(parsed[mem][fovper][seed][tag][1])
            
            #match shapes by padding with numpy.nan
            y = [np.pad(j, (0, int(max([len(i) for i in y])) - len(j)), 
                        'constant', constant_values = np.nan) for j in y]
            y = np.stack(y)
            
            TAG['std'] = np.nanstd(y,0)
            TAG['u'] = np.nanmean(y,0)
            C[tag] = TAG
        M[fovper] = C
    STATS[mem] = M
    
# Save
#fname = 'stats_ch5fovper.npy'
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
loc_labels = ["WW-LSTM foveal-only", "WW-LSTM foveal+peripheral"]

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
loc_labels = ["SpaCat foveal-only", "SpaCat foveal+peripheral"]

vis_compare(STATS, variants, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by a spatial concatenation model \n with and without the peripheral patch",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by a spatial concatenation model \n with and without the peripheral patch",
            ["Epoch","Accuracy (%)"], v_time = True)

## 3 - All
variants = [[0,0], [0,1], [1,0], [1,1]]
loc_labels = ["WW-LSTM, foveal-only", "WW-LSTM, foveal+peripheral",
              "SpaCat, foveal-only", "SpaCat, foveal+peripheral"]

vis_compare(STATS, variants, loc_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'],
            "Losses obtained by different models with and without \n the peripheral patch",
            ["Epoch","Loss"], v_time = True)

vis_compare(STATS, variants, loc_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'],
            "Accuracies obtained by different models with and without \n the peripheral patch",
            ["Epoch","Accuracy (%)"], v_time = True)

"""
- The peripheral patch doesn't appear to have provided value to the 
classification task in any case. 
- This calls into question using per information in this stream.

- All models have comparable performance on the validation set aside from the
fovper WW-LSTM that performed significantly worse.
- The performance gap between WW-LSTM and SpaCat appears to be closed when
WW-LSTM is made fovonly.
- This suggests that per itself is not the issue, but the way WW-LSTM integrates
it is, ie merging via concatenating after feature extraction.
- Some possible explanations for fovper WW-LSTM's poor perf:
    - FE performs worse due to the equal processing two patches with different 
    scales and effectively different visual distributions. In SpaCat the per
    patch is in a different position relative to the fov patch, opening the 
    possibility of the net learning to ignore it. As a result the FE is forced
    to fit to low-res features that lead to overfitting. Supported by the model
    reaching about training 100% acc the fastest out of all of them.
    
TODO: check if this matches prior spacat results, where it outperformed everything.
"""
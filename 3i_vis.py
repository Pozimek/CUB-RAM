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
                [False, True, False, True, False],  #1b. exo prop RL
                [True, True, True, True, False],    #1c. ego prop HC 
                [False, True, True, True, False],   #1d. exo prop HC
                [True, False, False, False, False], #2a. ego wide RL
                [True, False, False, True, False],  #2b. ego narrow RL
                [True, False, True, False, False],  #2c. ego wide HC
                [True, False, True, True, False],   #2d. ego narrow HC
                [True, False, True, True, True]     #3. retina HC
                ]:
    V = {}
    var = "EGO{}PROP{}HC{}NRW{}RET{}".format(*[int(i) for i in variant])
    
    for seed in [1, 9, 919]:
        #retina ran only for one seed
        if (variant == [True, False, True, True, True] and seed != 1): 
            continue
        name = "ch3-s{}".format(seed)
        name += var
        print(name)
        
        log_dir = config.log_dir + name
        reader = SummaryReader(log_dir)
        S = {}
        
        for key in list(reader.children.keys())[:18]: #log tags
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


#np.save('stats.npy',STATS)

s2 = np.load('stats.npy', allow_pickle=True)[()]

## Visualize
def vis_compare(var_names, var_labels, metrics, title, axes, 
                metric_labels = ['Training', 'Validation'], title_size=15, 
                size = (8,5), v=False):
    """ Visualize multiple variants' metrics on a single plot"""
    sns.set_style('darkgrid') #darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("notebook") #paper talk poster notebook
    #TODO: smoothing?
    plt.figure(figsize=size)
    plt.title(title, fontsize = title_size, wrap=True)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    
    for i, variant in enumerate(var_names):
        for j, metric in enumerate(metrics):
            u = STATS[variant][metric]['u']
            std = STATS[variant][metric]['std']
            x = np.arange(len(u))
            
            #test
            if v:
                print(all(u == s2[variant][metric]['u']) and
                      all(std == s2[variant][metric]['std']))
                print(len(std) == len(u))
                print(var_labels[i]+' - '+metric_labels[j])
                print(std.max(), std.mean(), std.min(), len(std))
            
            plt.plot(x, u, label=var_labels[i]+' - '+metric_labels[j])
            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()
    
    
## 1 - Egocentric vs exocentric in RL
variants = ['EGO1PROP1HC0NRW1RET0', 
            'EGO0PROP1HC0NRW1RET0']
var_labels = ['Egocentric (#8)','Exocentric (#1)']
vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
             "Total losses of models using egocentric and exocentric \n coordinate frames", 
             ["Epoch","Loss"])

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            "Accuracies of models using egocentric and exocentric \n coordinate frames", 
            ["Epoch","Accuracy (%)"])
print("PROP1 HC0 NRW1")

"""
Egocentric frame outperforms exocentric. Exocentric overfits drastically,
so the network essentially cheats; it memorizes fixation coordinates based
on what it sees, boosting perf during training in a way that does not
generalize to val.

Egocentric has high loss variance; later this is found to be caused by 
proprioception, but you didn't test why it didn't happen with exocentric.
Likely due to not running enough seeds.
"""

## 2 - Egocentric vs exocentric in HC
variants = ['EGO1PROP1HC1NRW1RET0', 
            'EGO0PROP1HC1NRW1RET0', 
            'EGO1PROP0HC1NRW1RET0']
var_labels = ['Egocentric (#9)','Exocentric (#2)', 'No proprioception (#6)']

vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
             "Total losses of models utilizing hardcoded fixations", 
             ["Epoch","Loss"])

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            'Accuracies of models utilizing hardcoded fixations', 
            ["Epoch","Accuracy (%)"])
print("HC1 NRW1")

"""
With ablated attention, the two coordinate frames have effectively the same
performance, meaning that the format of data input into proprioception does
not make a difference. 

Removing proprioception shows improved performance on both train and val. Shows
that prop is reducing quality of FE likely by introducing noise.
"""

## 3 - No proprioception vs prop in RL
# Consider running no vision and prop-only variants to see if it can cheat
variants = ['EGO1PROP0HC0NRW1RET0', 
            'EGO1PROP1HC0NRW1RET0']
var_labels = ['No proprioception (#4)','Proprioception (#8)']

vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
             "Total losses of egocentric models with and without \n proprioception", 
             ["Epoch","Loss"])

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            'Accuracies of egocentric models with and without \n proprioception', 
            ["Epoch","Accuracy (%)"])
print("HC0 NRW1")

""" 
With RL enabled proprioception introduces a lot of variance in performance 
across seeds. Models without proprioception perform slightly better on the 
validation set, but do worse on the training set suggesting that the same 
effect as in #1 is taking place - proprioceptive input is enabling the network 
to 'cheat' just as the exocentric coordinate frame was enabling the attention 
module to 'cheat' (exploiting bias in centering the object of interest in 
photography), albeit to a lesser extent. The result could be largely due to 
the noise introduced by the prop modules (confounding effect).
"""

## 4 - Narrow vs wide in RL
variants = ['EGO1PROP0HC0NRW1RET0', 
            'EGO1PROP0HC0NRW0RET0']
var_labels = ['Foveated Patches (#4)', 'Large Patch (#3)']

vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
             "Total losses of egocentric models without proprioception, \n with varying fields of view", 
             ["Epoch","Loss"])

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            "Accuracies of egocentric models without proprioception, \n with varying fields of view",
            ["Epoch","Accuracy"])
print("EGO1 PROP0 HC0")

""" 
The wide FOV model significantly outperforms the narrow FOV model in accuracies,
even though their losses are more similar. This could mean that the wide FOV 
model is overconfident and not well-calibrated.
"""

## 5 - Narrow vs wide in HC
# if needed run narrow vs wide w/ random fixations
variants = ['EGO1PROP0HC1NRW1RET0', 
            'EGO1PROP0HC1NRW0RET0', 
            'EGO1PROP0HC1NRW1RET1']
var_labels = ['Foveated Patches (#6)', 'Large Patch (#5)', 'Software Retina (#7)']

vis_compare(variants, var_labels,
            ['Train loss (smooth)', 'Val loss (smooth)'], 
             "Total losses of egocentric models without proprioception, \n with hardcoded fixations and varying sensors", 
             ["Epoch","Loss"])

vis_compare(variants, var_labels,
            ['Train acc (smooth)', 'Val acc (smooth)'], 
            "Accuracies of egocentric models without proprioception, \n with hardcoded fixations and varying sensors",
            ["Epoch","Accuracy"])
print("EGO1 PROP0 HC1")

"""
With hardcoded fixations the losses of models with wide and narrow FOVs are
no longer similar, with wide fov having half the val loss of narrow. Wide FOV
is almost 80% accurate on the val set, with narrow being around 50% accurate.
This large gap shows that there is a lot of work to be done on enabling the AV
architecture to utilize information from multiple observations in a way that
is as effective as wide fov or passive vision.

The software retina performed worse than either of the patch sensors in spite
of having the widest fov. This result combined with its slow performance 
(13h42m28s for 88 epochs) is further reason to drop it.
"""
    
"""
- sw retina - nodes: 16384 , fov:372, out shape: (3,273,500)
- single patch sensor width: 224 (DT-RAM)
- foveated patches sizes and params: 37, 370 -> 37 (!!!!)
- resnet: layer4 conv1 and downsample have strides=1, preload imagenet,
output of layer4 fed into avgpool
- patches are concatenated in chan dim after FE before prop_merge
- Proprioception layers, sizes, activations: one layer (2, FE_outshape).
Outshape of FE depends on sensor (1024,2,2) for narrow, (512,7,7) for wide, 
(512,9,12) for sw retina. relu activation
- Avgpool happens after prop_merge
- lstm: input 1024, hidden 1024 (always cos of avgpool)
- classifier module: 1024,200, log softmax
- locator module: 1024, 512, 2, relu tanh
- Baseliner: 1024, 512, 1, relus
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 02:46:50 2022

Chapter 5 (active vision attention, ch4 in latex) visualisation.

HAS experiment.

@author: piotr
"""
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

fname = 'stats_ch5HAS.npy'
STATS = np.load(fname, allow_pickle=True)[()]

#D[seed][dist_log | loss_log][values]

def vis_HAS(STATS, metric, title, axes, title_size=15, size = (8,5), 
            v=False, y_lim = None):
    sns.set_style('darkgrid') #darkgrid, whitegrid, dark, white, and ticks
    sns.set_context("notebook") #paper talk poster notebook
    plt.figure(figsize=size)
    plt.title(title, fontsize = title_size, wrap=True)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    
    if y_lim is not None:
        plt.ylim([y_lim[0], y_lim[1]])
    
    for i, seed in enumerate(STATS.keys()):
        y = STATS[seed][metric]
        x = np.arange(len(y))
        
        plt.plot(x, y, label="Seed {}".format(seed))
        
#        for j, metric in enumerate(metrics):
#            u = STATS[var[0]][var[1]][metric]['u']
#            std = STATS[var[0]][var[1]][metric]['std']
#            x = np.arange(len(u))
#            
#            #test
#            if v:
#                print(len(std) == len(u))
#                print(seed+' - '+metric_labels[j])
#                print(std.max(), std.mean(), std.min(), len(std))
#            
#            plt.plot(x, u, 
#                     label=var_labels[i]+' - '+metric_labels[j])
#            plt.fill_between(x, u-std, u+std, alpha=0.2)
    
    plt.legend()
    plt.show()
    
vis_HAS(STATS, "dist_log", 
        "Distance from the fixation location to the blob \n throughout optimisation",
        ["Iteration", "Distance to blob"])
vis_HAS(STATS, "loss_log", 
        "Change in loss throughout optimisation",
        ["Iteration", "Loss Change"])
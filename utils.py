#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:14:20 2020

CUB-RAM utility functions.

@author: piotr
"""
import yaml
#import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# converts a nested dict to a nested object for ease of access
class conf:
    def __init__(self, D):
        self.__D__ = D
        for entry in D:
            if type(D[entry]) is dict:
                self.__dict__.update({entry: conf(D[entry])})
            else:
                self.__dict__.update({entry:D[entry]})
                
    def __repr__(self):
        return str(self.__dict__)
    
    def __str__(self):
        return str(self.__dict__)

def get_ymlconfig(path): #loads a .yaml config for non-RAM models
    with open(path) as file:
        return conf(yaml.load(file))
   
class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#Get GPU memory stats
def memshow():
    total = torch.cuda.get_device_properties(0).total_memory
    GB = 1000000000
    a = torch.cuda.memory_allocated()
    c = torch.cuda.memory_cached()
    r = total-torch.cuda.memory_allocated()
    print("Allocated:", a/GB)
    print("Cached:", c/GB)
    print("Remaining:", r/GB, "(", r/total,")")
    print("Total:", total/GB)
 
    
#pyplot visualisation
def showArray(array, size=(8,8), cmap = None):
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(array, interpolation='none', cmap=cmap)
    plt.show()

def showPIL(PIL, size=(8,8), cmap=None):
    array = np.array(PIL)[...,:3]
    showArray(array, size, cmap)
    
def showTensor(tensor, size=(8,8), cmap=None):
    if tensor.device.type == 'cuda': tensor = tensor.cpu()
#    array = np.moveaxis(tensor.numpy(), 0, -1) #why did you even do this?
    array = tensor.numpy()
    showArray(array, size, cmap)
 
    
#OH BOY PYTHON 3 SURELY HURTS
def normal_round(n):
    if n - np.floor(np.abs(n)) < 0.5:
        return np.floor(n)
    return np.ceil(n)

#i = int, r = round.
def ir(val):
    return int(normal_round(val))

#compute the primary output dims of a 2d conv layer. Lazy/incomplete formula
def conv_outshape(in_shape, conv_layer):
    out_shape = []
    for dim, val in enumerate(in_shape):
        out_shape.append((val-(conv_layer.kernel_size[dim]-1)-1) // 
                         conv_layer.stride[dim] + 1)
    return tuple(out_shape)

def out_shape(module, in_shape):
    """
    Computes an nn.Module's output shape given an input shape. 
    Skips batch dimension.
    """
    device = next(module.parameters()).device
    T = torch.zeros((2,)+in_shape).to(device)
    with torch.no_grad():
        O = module(T)
    return O.shape[1:]

#Gauss(sigma,x,y) function, 1D
def gauss(sigma,x,y,mean=0):
    d = np.linalg.norm(np.array([x,y]))
    return np.exp(-(d-mean)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def gausskernel(sigma, channels):
    width = ir(2*sigma)
    k = np.zeros((width, width))    
    
    shift = (width-1)/2.0

    for x in range(width):
        for y in range(width):
            k[y,x] = gauss(sigma,(x-shift),(y-shift))
    
    k = np.expand_dims(k/np.sum(k), 0) #in-channel/groups
    k = torch.from_numpy(np.stack(channels*[k], 0)).float() #group dim (dumb)
    
    return k
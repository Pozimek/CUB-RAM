#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAS Layer
Hardcoded Attentional Spotlight

@author: piotr
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import showTensor, set_seed, validate_values, print_patch, ir, d

class HAS(nn.Module):
    """
    Hardcoded Attention Spotlight layer. Takes as input (x,y) coordinates, 
    returns an attention map with a singular spotlight centered at the 
    specified coordinates. The spotlight consists of two generalized Gaussian 
    distributions: a spotlight distribution and a support distribution. Allows 
    backpropagation through the input so that a prior layer can learn to guide 
    the spotlight elsewhere and amplify the desired features.
    
    Only tested with odd-sized spotlights.
    
    The support Gaussian is necessary to maintain non-zero values (and thus 
    gradients) at distant corners away from the spotlight, ie. to maintain an 
    *effective* infinite support region. Its params have been selected with the
    assumption that the peripheral patch is 10x as large as the foveal patch. 
    
    Args
    - out_shape: the 2D shape of the output tensor.
    - a: peak height of the spotlight Gaussian. Default: 0.99
    - c: standard deviation of the spotlight Gaussian. Default: 0.1. 
    - f: power term determining the 'squareness' of the Gaussian (ie how fast
    it rolls off and how flat the peak is). Default: 80
    - width: the approximate width of the attentional spotlight. Default: 37
    """
    def __init__(self, out_shape, width = 37, a = 0.98):
        super(HAS, self).__init__()
        assert width%2==1
        assert width!=1
        self.out_shape = out_shape
        self.a = a
        self.width = width
        
        # coordinate scaling factor for controlling spotlight width
        self.div = torch.sqrt(torch.tensor(2))*(width/2 - 0.5)
        
        # coordinate grid to be used later
        self.B = torch.arange(out_shape[0]).repeat(out_shape[1],1)
        
        #DEBUG LOGS
        self.xd = None
        self.d = None
        
    def modGauss(self, tensor, a, f, c):
        """Generalized Gaussian"""
        return a * torch.exp(-(tensor**f)/(2*c**2))
        
    def forward(self, coords):
        # clamps necessary to prevent NaN gradients
        xd = (self.B - coords[0])/self.div
        yd = (self.B - coords[1])/self.div
        xd.retain_grad()
        self.xd = xd
        d = torch.sqrt((xd**2 + yd.T**2).clamp(min=1e-6))
        d.retain_grad()
        self.d = d
#        spotlight = self.modGauss(d, self.a, self.f, self.c)
        support = self.modGauss(d, 1-self.a, 2, 10)
        
        # add spotlight
        from_y, to_y = ir(coords[1].item())-self.width//2, ir(coords[1].item()) + 1 + self.width//2
        from_x, to_x = ir(coords[0].item())-self.width//2, ir(coords[0].item()) + 1 + self.width//2
##        support[from_y:to_y, from_x:to_x] += self.modGauss(d[from_y:to_y, from_x:to_x].detach(), self.a, self.f, self.c)
#        support[from_y:to_y, from_x:to_x] += self.a
        return support

""" 
TEST: can the spotlight learn to fixate on a hot spot?
Goal: maximize the intensity of pixels captured by the spotlight.

NOTES:
- Local minimas exist and are an occasional problem to the toy example. Would they also 
pose a problem in deployment?
- You can actually plot out the entire gradient map over a grid of x,y values
as a flow map.
"""

def main(seed):
    size = (5,5) #matplotlib img size
    v = 0 #whether to visualize progress with images
    
    # Experiment params
    set_seed(seed)
    LR = 0.06
    coords = nn.Parameter(torch.tensor([36.0,36.0])) #starting location
    shape = (37,37)
    tolerance = shape[0]/40 # max tolerated distance from blob 
    layer = HAS(shape, width=3)
    scene = torch.randn(shape)
#    scene = torch.rand(shape)
    blob = [1,1] #expecting backprop to shift xy to these values
    scene[blob[1]-1:blob[1]+2, blob[0]-1:blob[0]+2] = 40
    max_steps = 50000
    
    # set up the parameters and optimizer
    opt = optim.SGD([coords], lr=LR)
    last_loss = 0
    
    # logging
    dist_log = []
    first_loss = None
    loss_log = []
    
    if v:
        showTensor(layer(coords).detach(), size=size)
        showTensor(scene, size=size)
    print("target xy: ", blob)
    print("starting xy: ", coords.tolist())
    for i in range(max_steps):
        # forward pass
        attention = layer(coords)
        attention.retain_grad()
        output = attention * scene
        
        # compute loss and update parameters
        loss = -output.sum()
        if first_loss is None: first_loss = loss
        loss.backward(retain_graph=True)
        opt.step()
        if i%5000 == 0: 
            print("T{}, loss: {}, xy: ({:.2f}, {:.2f})".format(
                i, loss, coords[0].item(), coords[1].item()))
            if v: showTensor(output.detach(), size=size)
        
        # validate for nans and infs
        if not (validate_values(attention) and validate_values(attention.grad) and
                validate_values(output) and validate_values(coords) and 
                validate_values(coords.grad)):
            print("Invalid values detected.")
            break
        distance = d(torch.tensor(blob) - coords)
        # end early if close enough or if stuck in a local minima
        if distance < tolerance or last_loss == loss:
            print("T{}, loss: {}, xy: ({:.2f}, {:.2f})".format(
                i, loss, coords[0].item(), coords[1].item()))
            print("Early stopping at T{}".format(i))
            break
        
        opt.zero_grad()
        if i != max_steps-1: last_loss = loss
        
        # LOGS
        #distance to blob coordinates
        dist_log.append(distance.item())
        #loss change, ie loss - initial loss
        loss_log.append((loss-first_loss).detach().item())
    
    print("Final distance: {}".format(d(torch.tensor(blob) - coords)))
    if last_loss == loss: print("**LOCAL MINIMA**")
    if distance < tolerance: print("**CLOSE ENOUGH**")
    else: print("**FAILED**")
    if v:
        showTensor(scene.detach(), size=size)
        showTensor(attention.detach(), size=size)
        showTensor(output.detach(), size=size)
    #py, px = 2, 2
    #print_patch(attention, py, px, w=5)
    #print_patch(output, py, px, w=5)
    #print_patch(layer.d, py, px, w=5)
    #print_patch(layer.d.grad, py, px, w=5)
    #print_patch(scene, py, px, w=5)
    
    return dist_log, loss_log
    
if __name__ == '__main__':
    D = {}
    for seed in [1,3,6,9,919,9001,12345,42,1337,1984]:
        dist_log, loss_log = main(seed)
        D[seed] = {"dist_log":dist_log, 
                   "loss_log":loss_log}
    
    # Save
    fname = 'stats_ch5HAS.npy'
    np.save(fname,D)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:38:41 2020

Modules used in CUB-RAM investigations.

@author: piotr
"""
#TODO: proper gpu checks, atm most code wouldn't run on cpu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.utils import load_state_dict_from_url

from torch.autograd import Variable
from torch.distributions import Normal

from kornia import translate

import numpy as np
from utils import gausskernel, conv_outshape, out_shape, ir

from retinavision.retina import Retina
from retinavision.cortex import Cortex
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class RAM_sensor(object):
    """ 
    A superclass for RAM sensor modules.
    """
    def __init__(self):
        """Required variables"""
        self.k = None #num of patches
        self.width = None #assuming sensor width=height
        self.out_shape = None
        self.fixation = None
        
    def reset(self):
        self.fixation = None
    
    def _foveate(self, x, coords, v=0):
        """Sensor-specific sampling function, uses absolute coordinates.
        
        Args
        ----
        - x: batch of input images. (B, C, H, W)
        - l: Normalized coordinates in the image range. (B, 2)
        - v: visualisation level (0-2) to be stored in variable. 
             0: None 
             1: Input, (x,y) and foveation 
             2: Input, (x,y), pre-processed and post-processed foveation
        
        Returns
        -------
        - phi: The foveated image glimpse. (B, k, C, g, g)"""
        raise NotImplementedError
        
    def foveate_exo(self, x, l, v=0):
        """Sample using exocentric coordinates"""
        B, C, H, W = x.shape
        T = torch.tensor([H,W]).float().cuda()
        coords = self.to_exocentric(T, l)
        self.fixation = coords
        
        return self._foveate(x, coords, v=v)
        
    def foveate_ego(self, x, l, v=0):
        """Sample using egocentric coordinates"""
        if self.fixation is None: #first fixation has to be exocentric
            return self.foveate_exo(x, l, v=v)
        B, C, H, W = x.shape
        T = torch.tensor([H,W]).float().cuda()
        coords = self.to_egocentric(T, l)
        self.fixation = coords
        
        return self._foveate(x, coords, v=v)
    
    def to_egocentric(self, T, coords):
        """
        Convert coordinates from [-1,1] range to egocentric coordinates,
        ie fixation + offset from fixation location.
        """
        #compute and add offset to current fixation
        loc = self.fixation + (0.5*self.width*coords)
        
        #clip to image boundaries
        torch.clamp_(loc[:,0], min=1, max=T[0])
        torch.clamp_(loc[:,1], min=1, max=T[1])
        
        return loc.long()
   
    def to_exocentric(self, T, coords):
        """
        Convert coordinates in the range [-1, 1] to range [0, T] 
        where T is the (H, W) size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()
    
    def to_egocentric_action(self, loc):
        """
        Convert egocentric coordinates, ie offset from fixation location,
        to range [-1, 1] (normalized by sensor width).
        """
        action = 2 * (loc / self.width)
        
        #clip to [-1, 1]
        torch.clamp_(action[:,0], min=-1, max=1)
        torch.clamp_(action[:,1], min=-1, max=1)
        
        return action
    
    def to_exocentric_action(self, T, coords):
        """
        Convert exocentric coordinates in the range [0, T] to range [-1, 1]
        where T is the (H, W) size of the image.
        """
        return ((coords / T) * 2.0 - 1.0)
        
class retinocortical_sensor(RAM_sensor):
    """
    First software retina wrapper, hardcoded 16k retina. Very inefficient in 
    batch processing, lots of typecasting, assumes gpu=True. Just horrible.
    """
    def __init__(self, gpu=True):
        super(retinocortical_sensor, self).__init__()
        self.k = 1 #one "patch"
        self.sigma_base = 0.5
        self.mean_rf = 1
        self.c_sigma_base = 1.0
        self.target_d5 = 1.0
        self.out_shape = (3,273,500)
        
        retpath = os.path.join(os.getcwd(), 'retina_data', 'ret16k_{}_{}_{}.pkl')
        cortpath = os.path.join(os.getcwd(), 'retina_data', 'cort16k_{}_{}_{}.pkl')
        Llocpath = cortpath.format(self.c_sigma_base, self.target_d5, 'Lloc')
        Rlocpath = cortpath.format(self.c_sigma_base, self.target_d5, 'Rloc')
        Lcoeffpath = cortpath.format(self.c_sigma_base, self.target_d5, 'Lcoeff')
        Rcoeffpath = cortpath.format(self.c_sigma_base, self.target_d5, 'Rcoeff')
        
        self.R = Retina(gpu=gpu)
        self.R.loadLoc(retpath.format(self.sigma_base, self.mean_rf, 'loc'))
        self.R.loadCoeff(retpath.format(self.sigma_base, self.mean_rf, 'coeff'))
#        if gpu: self.R._cudaRetina.set_samplingfields(self.R.loc, self.R.coeff) #prepare
        self.R.prepare((500,500), (250,250))
        self.width = self.R.width
        
        self.C = Cortex(gpu=gpu)
        self.C.loadLocs(Llocpath, Rlocpath)
        self.C.loadCoeffs(Lcoeffpath, Rcoeffpath)
        
    def _foveate(self, x, coords, v=0):
        #TODO: visualisation code
        phi = []
        B, C, H, W = x.shape
        
        for i, sample in enumerate(x):
            fixation = (coords[i,0].item(), coords[i,1].item())
            #gpu retina bug workaround
            if fixation[0] == 0: 
                fixation = (fixation[0] + 1, fixation[1])
            if fixation[1] == 0:
                fixation = (fixation[0], fixation[1] + 1)
            if fixation[0] == H:
                fixation = (fixation[0] - 1, fixation[1])
            if fixation[1] == W:
                fixation = (fixation[0], fixation[1] - 1)
            
            img = np.moveaxis(sample.numpy(), 0, -1)
            img = np.uint8(img*255) #XXX retina only works with uint8 right now
            
            V = self.R.sample(img, fixation)
            c_img = torch.tensor(self.C.cort_img(V)).permute(2,0,1)
            phi.append(c_img.unsqueeze(0).type(torch.float32))
        
        #rebuild batch, add glimpse dimension
        phi = torch.cat(phi).unsqueeze(1).cuda()
        
        return phi

class crude_retina(RAM_sensor):
    """
    A retina that extracts a foveated glimpse phi around location l. 
    Phi consists of multi-resolution overlapping square image patches.

    Args
    ----
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of successive patches.
    """
    def __init__(self, g, k, s, gpu):
        super(crude_retina, self).__init__()
        self.g = g
        self.k = k
        self.s = s
        self.width = g*(s**(k-1))
        self.gpu = gpu
        self.sigma = np.pi*s
        self.lowpass = False #TODO: fix sigma formula. Lowpass off until then
        self.gauss = None
        self.vis_data = None #variable to store visualisation data
        self.out_shape = (3,g,g)
        
    def _foveate(self, x, l, v=0):
        phi = []
        size = self.g
        if self.gauss is None: 
            self.gauss = gausskernel(self.sigma, x.shape[1]).cuda()
            
        if self.gpu: x = x.cuda() #XXX

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
        
        # store preprocessed patches
        preblur = []
        if v:
            for i in range(len(phi)):
                preblur.append(phi[i].clone())
        
        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            #lowpass filter
            if self.lowpass: phi[i] = F.conv2d(phi[i], self.gauss, 
               padding=self.gauss.shape[-1]//2, groups=x.shape[1]) 
            phi[i] = F.avg_pool2d(phi[i], k) #XXX

        # add an empty glimpse dimension (k), concatenate along it
        for i in range(len(phi)):
            phi[i] = torch.unsqueeze(phi[i], 1)
        phi = torch.cat(phi, 1)
        
        # update visualisation variable
        if v == 1:
            self.vis_data = x.clone(), self.fixation, phi.clone()
        elif v == 2:
            self.vis_data = x.clone(), self.fixation, preblur, phi.clone()
        
        return phi

    def extract_patch(self, x, coords, size):
        """
        Extract a single patch (B, C, size, size) for each image in 
        the minibatch x (B, C, H, W) at specified locations coords (B, 2).
        """
        B, C, H, W = x.shape
        T = torch.tensor([H,W]).float().cuda()

        # compute top left corner of patch
        patch_x = coords[:, 1] - (size // 2)
        patch_y = coords[:, 0] - (size // 2)

        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)

            # compute slice indices, cast to ints
            from_x, to_x = patch_x[i].item(), (patch_x[i] + size).item()
            from_y, to_y = patch_y[i].item(), (patch_y[i] + size).item()

            # pad tensor in case exceeds
            if self.exceeds(from_x, to_x, from_y, to_y, T):
                pad_dims = (
                    size//2+1, size//2+1,
                    size//2+1, size//2+1,
                    0, 0,
                    0, 0,)
                im = F.pad(im, pad_dims, "constant", 0)

                # add correction factor
                from_x += (size//2+1)
                to_x += (size//2+1)
                from_y += (size//2+1)
                to_y += (size//2+1)

            # and finally extract
            patch.append(im[:, :, from_y:to_y, from_x:to_x])

        # concatenate into a single tensor
        patch = torch.cat(patch)
        return patch

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """
        Check if the extracted patch fits within the image of size T.
        """
        if ((from_x < 0) or (from_y < 0) or (to_x > T[1]) or (to_y > T[0])):
            return True
        return False

class ResNet18_short(ResNet):
    #Resnet18 with ablated avgpool, flatten, fc and a selected number of layers
    def __init__(self, layers = 4, pretrained = True, cleanup=True, maxpool=True):
        super(ResNet18_short, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.num_layers = layers
        self.remove_maxpool = not maxpool
        
        if pretrained:
            self.load_state_dict(resnet18(pretrained=True).state_dict())
            
        if cleanup: self.cleanup()
        if self.remove_maxpool: self.maxpool = nn.Identity()
        
    def cleanup(self):
        #delete unused layers
        for i in range(self.num_layers+1, 5):
            setattr(self, "layer{}".format(i), None)
        
        #correct maxpool?
        if self.remove_maxpool: self.maxpool = nn.Identity()
            
    def load_weights(self, weights):
        # First load weights, then delete unused layers
        self.__init__(layers=self.num_layers, cleanup=False, 
                      maxpool= not self.remove_maxpool)
        self.load_state_dict(weights)
        self.cleanup()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(1, self.num_layers+1):
            x = getattr(self, "layer{}".format(i))(x)
        
        return x

class glimpse_module(nn.Module):
    def __init__(self, retina, feature_extractor):
        super(glimpse_module, self).__init__()
        self.retina = retina
        self.feature_extractor = feature_extractor
        
        self.out_shape = out_shape(self.feature_extractor, retina.out_shape)
        self.n_patches = retina.k if type(retina) is crude_retina else 1
        
    def forward(self, x, l_t_prev):
        phi = self.retina.foveate_ego(x, l_t_prev)
        
        #split up phi into patches, process individually
        out = []
        for i in range(self.n_patches):
            x = phi[:,i,:,:,:] #select patch
            out.append(self.feature_extractor(x))

        return out #n_patches of g_t's, each g_t has shape of self.out_shape
    
    def reset(self):
        self.retina.reset()

class sensor_resnet18(nn.Module):
    def __init__(self, retina, pretrained=True):
        super(sensor_resnet18, self).__init__()
        self.retina = retina
        
        self.resnet18 = ResNet18_short()
        if pretrained:
            self.resnet18.load_state_dict(resnet18(pretrained=True).state_dict())
        
        self.out_shape = out_shape(self.resnet18, retina.out_shape)
        self.n_patches = retina.k if type(retina) is crude_retina else 1
        
    def forward(self, x, l_t_prev):
        phi = self.retina.foveate_ego(x, l_t_prev)
        #phi = self.retina.foveate_exo(x, l_t_prev)
        
        #split up phi into patches, process individually
        out = []
        for i in range(self.n_patches):
            x = phi[:,i,:,:,:] #select patch
            out.append(self.resnet18(x))

        return out #n_patches of g_t's, each g_t has shape of self.out_shape
    
    def reset(self):
        self.retina.reset()
    
class glimpse_network(nn.Module):
    """
    Unused code.
    Args
    ----
    - x: a 4D Tensor of shape (B, C, H, W). The minibatch of images.
    - l_t_prev: a 2D tensor of shape (B, 2). The glimpse coordinates [x, y] 
      for the previous timestep `t-1`.

    Returns
    -------
    - out: a list of g_t tensors (B, conv_out), one for each image patch. 
      g_t is a glimpse representation.
    """
    def __init__(self, retina):
        super(glimpse_network, self).__init__()
        self.retina = retina
        self.n_patches = retina.k if type(retina) is crude_retina else 1
        
        # Visual stack, IN: 50x50
        self.conv1 = nn.Conv2d(3, 64, (5,5), stride=2)  #OUT= 23x23 x64
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride=1) #OUT= 21x21 x64
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1) #OUT= 19x19 x64
        self.conv4 = nn.Conv2d(64, 32, (1,1), stride=1) #OUT= 19x19 x32
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        
        #automatically compute conv output size
        hw = (self.retina.g, self.retina.g)
        for c in self.convs:
            hw = conv_outshape(hw, c)
        self.out_shape = hw[0] * hw[1] * self.convs[-1].out_channels #19x19x32
        
    def forward(self, x, l_t_prev):
        #generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        #split up phi into patches, process individually
        out = []
        for i in range(self.n_patches):
            x = phi[:,i,:,:,:]
            
            #Visual stack
            for conv in self.convs:
                x = F.relu(conv(x))
            
            out.append(x)

        return out #n_patches of g_t's, each g_t has shape of self.conv_out


##ConvLSTM Code adapted from https://github.com/automan000/Convolution_LSTM_PyTorch
class ConvLSTMCell(nn.Module):
    """
    An LSTM cell with all convolutional gates. Expects spatially encoded input 
    (ie. either an image or a convolutional activation map).
    
    Args
    ----
    - input_channels: num of channels in the input tensor x
    - hidden_channels: num of channels in the hidden state and memory tensors
    - kernel_size: the size of the convolutional kernels within all gates
    
    Returns
    -------
    - ch, cc: the 'hidden state' vector and the 'memory' vectors (aka c)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        
        # input gate
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1,
                             self.padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, 
                             self.padding, bias=False)
        
        # forget gate
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, 
                             self.padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1,
                             self.padding, bias=False)
        
        # input selection
        self.Wxg = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, 
                             self.padding, bias=True)
        self.Whg = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1,
                             self.padding, bias=False)
        
        # output gate
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, 
                             self.padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1,
                             self.padding, bias=False)
        
    def forward(self, x, h, c):
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h)) #forget gate
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h)) #input gate
        cc = cf * c + ci * torch.tanh(self.Wxg(x) + self.Whg(h)) #new memory
        co = torch.sigmoid(self.Wxo(x) + self.Who(h)) #output gate
        ch = co * torch.tanh(cc) #new state
        
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class ConvLSTM(nn.Module):
    """
    A wrapper class for managing ConvLSTMCell objects. Originally intended
    for sequential processing (cell1's input would be cell0's output), but now
    repurposed for parallel processing (each cell's input is different).
    
    Args
    ----
    - input_channels: list of scalars denoting each cell's input channels, the 
    length of the list equals number of retinal patches
    - hidden_channels: list of scalars (n_patches) denoting each cell's num of 
    channels in the hidden state and memory tensors.
    - kernel_size: scalar denoting the size of the convolutional 
    kernels within each cell's gates.
    - x: a list of inputs. (n_patches)
    
    Returns
    -------
    ch, cc - the 'hidden state' vectors and the 'memory' vectors.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_cells = len(hidden_channels)
        self.internal_state = [] #list of (h, c) tensors for each cell
        
        #Instantiate ConvLSTM Cells
        for i in range(self.num_cells):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i],
                                self.kernel_size)
            setattr(self, name, cell)
            
    def reset(self, B=1):
        """
        Resets the hidden state and memory tensors, called after every sequence.
        """
        self.internal_state = [None for cell in range(self.num_cells)]

    def forward(self, x):
        """
        Compute a single forward step for each cell and return new hidden and 
        memory tensors.
        """
        #Iterate over all cells
        for i in range(self.num_cells):
            patch = x[i]
            cell = getattr(self,'cell{}'.format(i))
            
            #initialize memory if beginning of sequence
            if self.internal_state[i] == None:
                B, _, H, W = patch.size()
                h, c = cell.init_hidden(B, self.hidden_channels[i], (H, W))
                self.internal_state[i] = (h, c)

            #forward pass
            h, c = self.internal_state[i]
            new_h, new_c = cell(patch, h, c)
            self.internal_state[i] = (new_h, new_c)
        
        h_out = [hi for hi, _ in self.internal_state]
        return h_out
##END OF ConvLSTM Code


######
#Active ConvLSTM
######
#TODO comment documentation
"""
500x500 scene -> 16x16 latent space
100x100 patch -> 4x4 latent patch
"""
#TODO: write a *good* description of what's happening in this fn.
def remap(x, l_t_prev, mem_shape, scaling):
    # magnitude scaling factor correction
    x_mem = x*scaling 
    
    # true memory space fixation coordinates (y,x)
    mem_coords = (0.5 * (l_t_prev + 1.0) * torch.tensor(mem_shape).cuda())
    
    # pad x on bottom-right to match mem_shape
    pad_right = mem_shape[-1] - x.shape[-1]
    pad_bottom = mem_shape[-2] - x.shape[-2]
    x_mem = F.pad(x_mem, (0, pad_right, 0, pad_bottom))

    # true top-left pixel location in memory space
    topleft = mem_coords - (torch.tensor(x.shape[-2:], device=x.device)-1)/2
        
    # translate patches into correct positions
    x_mem = translate(x_mem, topleft.flip(1))
    #TODO: flip probably redundant in latest kornia
    return x_mem

class ActiveConvLSTMCell(ConvLSTMCell):
    def __init__(self, input_shape, mem_shape, kernel_size, scaling):
        super(ActiveConvLSTMCell, self).__init__(
                input_shape[0], mem_shape[0], kernel_size)
        self.scaling = scaling
        self.mem_size = mem_shape[1:]
        
    def forward(self, x, h, c, l_t_prev):
        cf = torch.sigmoid(
                remap(self.Wxf(x), l_t_prev, self.mem_size, self.scaling) 
                + self.Whf(h)) #forget gate
        ci = torch.sigmoid(
                remap(self.Wxi(x), l_t_prev, self.mem_size, self.scaling)
                + self.Whi(h)) #input gate
        cc = cf * c + ci * torch.tanh(
                remap(self.Wxg(x), l_t_prev, self.mem_size, self.scaling)
                + self.Whg(h)) #new memory
        co = torch.sigmoid(
                remap(self.Wxo(x), l_t_prev, self.mem_size, self.scaling)
                + self.Who(h)) #output gate
        ch = co * torch.tanh(cc) #new state
        
        return ch, cc

class ActiveConvLSTM(nn.Module):
    def __init__(self, input_shapes, mem_shapes, kernel_size, scaling):
        super(ActiveConvLSTM, self).__init__()
        self.input_shapes = input_shapes
        self.mem_shapes = mem_shapes
        self.num_cells = len(mem_shapes)
        self.internal_state = [] #list of (h, c) tensors for each cell
        
        #Instantiate ConvLSTM Cells
        for i in range(self.num_cells):
            name = 'cell{}'.format(i)
            cell = ActiveConvLSTMCell(input_shapes[i], mem_shapes[i], 
                                      kernel_size, scaling[i])
            setattr(self, name, cell)
    
    def reset(self, B=1):
        self.internal_state = [None for cell in range(self.num_cells)]
    
    def forward(self, x, l_t_prev):
        #Iterate over all cells
        for i in range(self.num_cells):
            patch = x[i]
            cell = getattr(self,'cell{}'.format(i))
            
            #initialize memory if beginning of sequence
            if self.internal_state[i] == None:
                B, _, H, W = patch.size()
                h, c = cell.init_hidden(B, self.mem_shapes[i][0], 
                                        self.mem_shapes[i][-2:])
                self.internal_state[i] = (h, c)

            #forward pass
            h, c = self.internal_state[i]
            new_h, new_c = cell(patch, h, c, l_t_prev)
            self.internal_state[i] = (new_h, new_c)
        
        h_out = [hi for hi, _ in self.internal_state]
        return h_out
######
#ENDOF Active ConvLSTM
######

class FC_RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FC_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = None

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_h = nn.Linear(hidden_size, hidden_size)
        
    def reset(self):
        self.hidden_state = None

    def forward(self, x):
        B, L = x.shape
        if self.hidden_state == None:
            h_t_prev = Variable(torch.zeros(B, self.hidden_size)).cuda()
            
        h1 = self.fc_in(x)
        h2 = self.fc_h(h_t_prev)
        h_t = F.relu(h1 + h2)
        h_t_prev = h_t        
        
        return h_t

#TODO: RAM_memory(nn.Module) superclass for CONVlstm, ACONVlstm, RC_RNN and lstm
#TODO: refactor lstm class to support multiple cells just like CONVlstm

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm, self).__init__()
        self.core = nn.LSTM(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.hidden_state = None
        self.cell_state = None
    
    def reset(self):
        self.hidden_state = None
        self.cell_state = None
    
    def forward(self, x):
        x = x.unsqueeze(0) #add sequence_len dimension
        
        if self.hidden_state is None:
            _, chcc = self.core(x)
        else:
            _, chcc = self.core(x, (self.hidden_state, self.cell_state))
            
        self.hidden_state = chcc[0]
        self.cell_state = chcc[1]
        
        return chcc[0].squeeze(0)

class location_network(nn.Module):
    """
    Uses the internal state h_t of the recurrent net to produce the location 
    coordinates l_t for the next time step.

    Feeds h_t through a hidden fc layer followed by a tanh to clamp the output 
    between [-1, 1]. This produces a 2D vector of means used to parametrize a 
    two-component Gaussian with a fixed variance from which the location 
    coords `l_t` for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically from a distribution 
    conditioned on an affine transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer, ie the shape of the h_t vector.
    - hidden_size: size of the hidden fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the current hidden state vector of the core network.
    
    Returns
    -------
    - log_pi: a vector of length (B,) that the gradient can flow through.
    - l_t: a 2D vector of shape (B, 2) containing glimpse locations.
    """
    def __init__(self, input_size, hidden_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # compute mean
        #TODO consistency in flattening in/out of model.py
        mu = F.relu(self.fc1(x.flatten(1,-1).detach())) 
        mu = torch.tanh(self.fc2(mu))

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise
        
        # Log of probability density at l_t. Maximized using REINFORCE.
        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)
        
        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t

class classification_network(nn.Module):
    """
    Uses the RNN internal state h_t to classify the image.

    Args
    ----
    - input_size: input size of the fc layer, ie the shape of the h_t vector.
    - hidden_size: size of the hidden fc layer.
    - output_size: number of class labels.
    - h_t: the current hidden state vector of the core network.

    Returns
    -------
    - y_p: output probability vector over class labels.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(classification_network, self).__init__()
        self.fc1_verb = nn.Linear(input_size, hidden_size)
        self.fc2_verb = nn.Linear(hidden_size, output_size)

    def forward(self, h_t):
        y_p = F.relu(self.fc1_verb(h_t))
        y_p = F.log_softmax(self.fc2_verb(y_p), dim=1)
        
        return y_p

class classification_network_short(nn.Module):
    def __init__(self, input_size, output_size):
        super(classification_network_short, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        y_p = F.log_softmax(self.fc(h_t), dim=1)
        return y_p

class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function to reduce the variance of the
    location network's gradient update.

    Args
    ----
    - input_size: input size of the fc layer, ie the shape of the h_t vector.
    - hidden_size: size of the hidden fc layer.
    - h_t: the current hidden state vector of the core network.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The current timestep's baseline.
    """
    def __init__(self, input_size, hidden_size,):
        super(baseline_network, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, h_t):
        b_t = F.relu(self.fc1(h_t.detach()))
        b_t = F.relu(self.fc2(b_t))
        
        return b_t
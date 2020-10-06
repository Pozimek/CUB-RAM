#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUB-200-2011 dataset object

@author: piotr

Notes:
    - In the OG dataset files class and part labels start at 1, not 0.
    - Although the dataset authors provided a training/test split of the 
    dataset, researchers have used the test set as a validation set to select
    the best performing model during training. For a 'fair' comparison with the
    literature this dataloader follows this practice.
"""

from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch
from os.path import join, isfile
from torchvision.datasets.folder import default_loader
from utils import ir
from torch.nn.functional import pad


def getSpeciesName(label):
    if type(label) is torch.Tensor:
        if label.device.type == 'cuda': label = label.cpu()
        
    root = './CUB_200_2011/'
    classes = pd.read_csv(join(root, 'classes.txt'), sep=' ', 
                          names=['class_id', 'class_name'],
                          index_col ="class_id")
    return classes.iloc[label]
    
def getPartName(label):
    root = './CUB_200_2011/'
    parts = pd.read_csv(join(root, 'parts/parts.txt'), sep=' ', 
                        names=['part_id', 'part_name'], index_col = "part_id")
    return parts.iloc[label][1]

def pad_to(tensor, target_shape=(500,500)):
    Hp1 = (500-tensor.shape[-2])//2
    Hp2 = Hp1 + tensor.shape[-2]%2
    Wp1 = (500-tensor.shape[-1])//2
    Wp2 = Wp1 + tensor.shape[-1]%2
    return pad(tensor, (Wp1, Wp2, Hp1, Hp2))

def collate_pad(batch):
    data = torch.stack([pad_to(item[0]) for item in batch])
    t1 = torch.LongTensor([item[1] for item in batch])
    t2 = torch.stack([torch.Tensor(item[2]) for item in batch])
    
    return [data, t1, t2]
    
class CUBDataset(Dataset):
    """
    Caltech Uni Birds Dataset class.
    
    __getitem__ returns: image, species label, part labels
    """
    images_path = './CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    
    def __init__(self, transform=None):
        self.root = './CUB_200_2011/'
        self.loader = default_loader
        self.transform = transform
        self.training = True
        self.num_classes = 200
        
        # Metadata
        images = pd.read_csv(join(self.root,'images.txt'), sep=' ', 
                                  names=['img_id', 'filepath'], index_col ="img_id")
        labels = pd.read_csv(join(self.root, 'image_class_labels.txt'),
                                  sep=' ', names=['img_id', 'label'], index_col ="img_id")
        split = pd.read_csv(join(self.root, 'train_test_split.txt'),
                                 sep=' ', names=['img_id', 'is_training_img'], index_col ="img_id")
        
        #image_id, part_id, x, y, visible
        self.part_locs = pd.read_csv(join(self.root, 'parts/part_locs.txt'), 
                                sep=' ', names=['image_id', 'part_id', 'x', 'y', 'visible'])
        
        #id, path, label, is_training
        data = images.merge(labels, on='img_id')
        self.all_data = data.merge(split, on='img_id')
        self.train_data = self.all_data[self.all_data.is_training_img == 1]
        self.test_data = self.all_data[self.all_data.is_training_img == 0]
        
        #the data to be iterated over
        self.data = self.train_data if self.training == True else self.test_data
        
        if not self._check_integrity(): print("Integrity check failed.")
        
    def _check_integrity(self):
            for index, row in self.data.iterrows():
                filepath = join(self.root, 'images', row.filepath)
                if not isfile(filepath):
                    print(filepath)
                    return False
            return True
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = join(self.root, 'images', sample.filepath)
        label = sample.label - 1 #label ids start at 1
        parts = self.part_locs[self.part_locs.image_id == sample.name].values[:,2:]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, parts
    
    def train(self):
        self.training = True
        self.data = self.train_data
        
    def test(self):
        self.training = False
        self.data = self.test_data
    
    def subset(self, size):
        """
        Returns a randomly subsampled subset of the dataset. Only the training 
        data is altered. The test data remains the same.
        
        - size: a fraction representing the size of the subset to be returned.
        """
        
        #gather all training data entries
        A = np.stack((self.train_data.index.values,
                      self.train_data.label.values), -1)
        subset_ids = np.array([])
        
        #for each class label
        for label in range(1, 201):
            #gather all label entries
            L = A[A[:,1] == label]
            
            #shuffle using torch seeded rng
            L = L[torch.randperm(len(L)).numpy()]
            
            #select img_ids to append to subset index list
            ids = L[:ir(size*len(L)), 0]
            subset_ids = np.concatenate((subset_ids, ids))
            
        #pass to subset creator
        return CUBSubset(subset_ids, size, transform=self.transform)
    
    
class CUBSubset(CUBDataset):
    """
    A sub-class of CUBDataset. Provides training data subsetting utility.
    Does not alter (shrink or expand) the test data; it only shrinks the
    training data while maintaining class proportionality.
    """
    def __init__(self, indices, size, transform=None):
        super().__init__(transform=transform)
        
        #update training data
        self.train_data = self.train_data.loc[indices]
        self.data = self.train_data if self.training == True else self.test_data
        self.size = size
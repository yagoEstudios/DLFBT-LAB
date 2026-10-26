import copy
import time 
from pathlib import Path

import numpy as np
import  torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import get_device, set_seed, sync

def to_tensor(array,dtype):
    if isinstance(array,torch.Tensor):
        return array.to(dtype)
    return torch.as_tensor(np.asarray(array),dtype=dtype)

def make_loader(X,y,batch_size=64, shuffle=False, seed = None):
    dataset = TensorDataset(to_tensor(X,torch.float32), to_tensor(y,torch.long))
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,generator=generator)

def train_one_epoch(model,loader,criterion,optimizer,device,grad_clip=None):
    pass



def fit(*args, **kwargs):
    raise NotImplementedError("fit() has not been implemented yet.")

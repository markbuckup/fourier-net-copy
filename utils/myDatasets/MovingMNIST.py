import os
import sys
import PIL
import time
import scipy
import torch
import random
import pickle
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

import sys
sys.path.append('../../')

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MovingMNIST(Dataset):
    data = None
    
    @classmethod
    def init_data(cls, path):
        data = np.load(os.path.join(path, 'mnist_test_seq.npy'))[:,:4000,:,:] # 20,10000,64,64
        if not torch.isfinite(torch.tensor(data)).all():
            asdf
        cls.data = torch.from_numpy(data).permute((1,0,2,3)).float()/255. # 10000,20,64,64
    
    @classmethod
    def access_data(cls, i):
        return cls.data[i,:,:,:]

    @classmethod
    def access_full_data(cls):
        return cls.data

    @classmethod
    def data_length(cls):
        return cls.data.shape[0]

    @classmethod
    def num_frames(cls):
        return cls.data.shape[1]


    def __init__(self, path, train = True, train_split = 0.8, norm = True, encoding = False):
        super(MovingMNIST, self).__init__()
        self.path = path
        self.train = train
        self.train_split = train_split
        self.norm = norm
        self.encoding = encoding
        MovingMNIST.init_data(self.path)
        
        train_len = int(train_split*MovingMNIST.data_length())
        if self.train:
            self.indices = np.arange(train_len)
        else:
            self.indices = np.arange(train_len, MovingMNIST.data_length())
        
        self.mean = MovingMNIST.access_full_data().mean()
        self.std = MovingMNIST.access_full_data().std()
        self.encod_mean = 0
        self.encod_std = 1


    def __getitem__(self, i):
        # Batch,20,64,64
        data = MovingMNIST.access_data(self.indices[i]).unsqueeze(1)
        loop = 1
        a1 = torch.flip(data[0:-1], [0])
        ans = torch.cat((data, a1, data[1:,:,:,:]),0)
        return i, ans
        
    def __len__(self):
        return len(self.indices)

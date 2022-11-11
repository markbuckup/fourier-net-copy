import os
import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
import PIL
import time
import scipy
import torch
import random
import pickle
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter as median_filter_func
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
# from utils.functions import get_window_mask
from utils.functions import get_coil_mask, get_golden_bars

EPS = 1e-10

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ToyVideoSet(Dataset):
    def __init__(self, path, train = True, train_split = 0.8, norm = None):
        super(ToyVideoSet, self).__init__()
        self.path = path
        dic = torch.load(os.path.join(self.path, 'data/data_tensor.pth'))
        self.data = dic['data']
        self.ft_data = dic['ft_data']
        self.window_mask = dic['window_mask']
        self.num_videos = dic['num_videos'] 
        self.video_start_ends = dic['video_start_ends'] 
        self.chain_locations = dic['chain_locations']
        self.window_locations = dic['window_locations']
        self.window_size = dic['window_size']
        self.train = train
        self.train_split = train_split
        self.norm = norm

        tot = len(self.window_locations)
        train_len = int(tot*self.train_split)
        if self.train:
            self.window_locations = self.window_locations[:train_len]
        else:
            self.window_locations = self.window_locations[train_len:]

        self.ft_mean = 1.0020166635513306
        self.ft_std = 2.2110860347747803
        self.image_mean = 0.08318176120519638
        self.image_std = 0.22408060729503632


    def __getitem__(self, i):
        win_start = self.window_locations[i]
        # return data of the form Nc, Nw, Nx, Ny, 2
        # return data of the form Nc, Nw, Nx, Ny
        half = (self.window_size//2) - 1
        ft = self.ft_data[win_start:win_start+self.window_size,:,:,:].unsqueeze(0)
        ft = torch.complex(ft[:,:,:,:,0], ft[:,:,:,:,1]).log()
        ft = torch.stack((ft.real, ft.imag), -1)
        mid = self.data[win_start+half,:,:].unsqueeze(0)
        if self.norm is None:
            ft_masked = ft*self.window_mask.unsqueeze(0).unsqueeze(4)
            return i, ft, ft_masked, mid
        else:
            ft = (ft - self.ft_mean) / self.ft_std
            ft_masked = ft*self.window_mask.unsqueeze(0).unsqueeze(4)
            mid = (mid - self.image_mean) / self.image_std
            return i, ft, ft_masked, mid
        
    def __len__(self):
        return len(self.window_locations)

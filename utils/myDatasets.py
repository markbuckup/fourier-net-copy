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
from utils.MDCNN import MDCNN

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


class ACDC(Dataset):
    def __init__(self, path, train = True, train_split = 0.8, norm = True, resolution = 128, window_size = 7, ft_num_radial_views = 14, predict_mode = 'middle', num_coils = 8, device = torch.device('cpu'), rank = 0):
        super(ACDC, self).__init__()
        self.path = path
        self.train = train
        self.train_split = train_split
        self.resolution = resolution
        self.window_size = window_size
        self.ft_num_radial_views = ft_num_radial_views
        self.norm = norm
        self.device = torch.device('cpu')
        # self.device = device
        self.num_coils = num_coils
        assert(self.window_size % 2 != 0)
        self.predict_mode = predict_mode
        assert(self.predict_mode in ['middle', 'last'])
        if self.predict_mode == 'middle':
            self.target_frame = (self.window_size-1)//2
        else:
            self.target_frame = self.window_size-1
        self.golden_bars = get_golden_bars(resolution = self.resolution)
        self.num_golden_cycle = self.golden_bars.shape[0]
        self.coil_mask = get_coil_mask(n_coils = self.num_coils, resolution = self.resolution)
        dic = torch.load(os.path.join(self.path, 'processed/processed_data_{}.pth'.format(self.resolution)))
        self.data = dic['data']
        (self.mu, self.std, self.ft_mu_r, self.ft_mu_i, self.ft_std_r, self.ft_std_i) = dic['normalisation_constants']
        tot = len(self.data)
        self.train_len = int(tot*self.train_split)
        if self.train:
            self.data = self.data[:self.train_len]
        else:
            self.data = self.data[self.train_len:]
        self.data_fft = [torch.fft.fftshift(torch.fft.fft2(x.float()/255.),dim = (-2,-1)) for x in self.data]
        self.data_fft = [torch.stack((x.real, x.imag), -1) for x in self.data_fft]

        self.num_patients = len(self.data)
        self.num_vids_per_patient = np.array([x.shape[0] for x in self.data])
        self.frames_per_vid_per_patient = np.array([x.shape[1] for x in self.data])
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        
        self.ft_data = [None for i in range(self.num_patients)]
        self.num_mem = 0
        self.rank = rank

        self.num_videos = int((self.num_vids_per_patient*self.frames_per_vid_per_patient).sum())

    def index_to_location(self, i):
        p_num = (self.vid_frame_cumsum <= i).sum()
        if p_num == 0:
            index_num = i
        else:
            index_num = i - self.vid_frame_cumsum[p_num-1]
        v_num = index_num // self.frames_per_vid_per_patient[p_num]
        f_num = index_num % self.frames_per_vid_per_patient[p_num]
            
        return p_num, v_num, f_num

    def __getitem__(self, i):
        # FT data - num_coils,num_windows, 256, 256
        p_num, v_num, f_num = self.index_to_location(i)
        if self.ft_data[p_num] is None:
            # vnum, fnum, 1, r, c
            indata = ((self.data[p_num].type(torch.float64)/255.).expand(-1,-1,self.num_coils, -1,-1)*self.coil_mask.unsqueeze(0).unsqueeze(0))
            self.ft_data[p_num] = torch.fft.fftshift(torch.fft.fft2(indata) ,dim = (-2,-1)).log()
            self.ft_data[p_num] = torch.stack((self.ft_data[p_num].real, self.ft_data[p_num].imag), -1)
            if self.rank == 0:
                if self.train:
                    print('Memoising for patient {}'.format(p_num), flush = True)
                else:
                    print('Memoising for patient {}'.format(p_num+self.train_len), flush = True)
                self.num_mem += 1 
                if self.num_mem == len(self.data):
                    print('#########################')
                    print('Memoised all patients!')
                    print('#########################', flush = True)

        indexed_ft_data = self.ft_data[p_num][v_num,:,:,:,:].to(self.device)
        indices = torch.arange(f_num,f_num + self.window_size).to(self.device)%self.frames_per_vid_per_patient[p_num]
        r_ft_data = indexed_ft_data[indices,:,:,:].permute(1,0,2,3,4).to(self.device)
        target = self.data[p_num][v_num, (f_num+self.target_frame)%self.frames_per_vid_per_patient[p_num],:,:,:].type(torch.float64).to(self.device)
        target = target/255.
        if self.norm:
            target = (target-self.mu)/self.std
            r_ft_data[:,:,:,:,0] = (r_ft_data[:,:,:,:,0]-self.ft_mu_r)/self.ft_std_r
            r_ft_data[:,:,:,:,1] = (r_ft_data[:,:,:,:,1]-self.ft_mu_i)/self.ft_std_i
        
        golden_bars_indices = torch.arange(i,i+self.ft_num_radial_views*self.window_size).to(self.device)%self.num_golden_cycle
        selection = self.golden_bars.to(self.device)[golden_bars_indices,:,:].reshape(self.window_size, self.ft_num_radial_views, self.resolution, self.resolution)
        current_window_mask = selection.sum(1).sign().float()
        
        ft_masked = r_ft_data * current_window_mask.unsqueeze(0).unsqueeze(-1)
        target_ft = self.data_fft[p_num][v_num, (f_num+self.target_frame)%self.frames_per_vid_per_patient[p_num],:,:,:].to(self.device)
        
        return i, r_ft_data.float().cpu(), ft_masked.float().cpu(), target.float().cpu(), target_ft.cpu()

        
    def __len__(self):
        return self.num_videos

# a = ACDC('../datasets/ACDC/', resolution = 256, norm = False)
# x1, ft, ft_masked, targ, target_ft = a[0]
# for i in range(8):
#     tft = torch.complex(ft[i,0,:,:,0], ft[i,0,:,:,1])
#     for j in range(7):
#         plt.imsave('dir/input_ft_coil_{}_win_{}.jpg'.format(i,j), (ft_masked[i,j,:,:,:]**2).sum(2)**0.5, cmap = 'gray')
#     plt.imsave('dir/window_0_coil_{}.jpg'.format(i), torch.fft.ifft2(torch.fft.ifftshift(tft.exp(), dim = (-2,-1))).real, cmap = 'gray')

# m = MDCNN(8, 7).to(torch.device('cuda:1'))
# # print(m(ft_masked.unsqueeze(0).to(torch.device('cuda:1'))).shape, flush = True)
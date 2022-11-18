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
from utils.models.MDCNN import MDCNN

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

class ACDC(Dataset):
    train_data = []
    test_data = []
    train_mem_ft_data = []
    test_mem_ft_data = []
    train_mem_ft_data_filled = []
    test_mem_ft_data_filled = []
    train_data_fft = []
    test_data_fft = []
    train_num_mem = None
    test_num_mem = None
    data_init_done = False
    mu, std, ft_mu_r, ft_mu_i, ft_std_r, ft_std_i = 0,0,0,0,0,0

    @classmethod
    def set_mem_ft(cls, train, i,val):
        if train:
            cls.train_mem_ft_data[i] += val
            cls.train_mem_ft_data_filled[i] += 1
        else:
            cls.test_mem_ft_data[i] += val
            cls.test_mem_ft_data_filled[i] += 1
    @classmethod
    def get_mem_filled(cls, train, i):
        if train:
            return cls.train_mem_ft_data_filled[i].item()
        else:
            return cls.test_mem_ft_data_filled[i].item()
    @classmethod
    def inc_num_mem(cls, train,i):
        if train:
            if cls.train_num_mem[i] == 0:
                cls.train_num_mem[i] += 1
        else:
            if cls.test_num_mem[i] == 0:
                cls.test_num_mem[i] += 1
    @classmethod
    def get_num_mem(cls, train):
        if train:
            return cls.train_num_mem.sum()
        else:
            return cls.test_num_mem.sum()

    @classmethod
    def data_init(cls, path, resolution, train_split, num_coils, memoise_disable):
        if cls.data_init_done:
            return
        cls.data_init_done = True
        dic = torch.load(os.path.join(path, 'processed/processed_data_{}.pth'.format(resolution)))
        (cls.mu, cls.std, cls.ft_mu_r, cls.ft_mu_i, cls.ft_std_r, cls.ft_std_i) =  dic['normalisation_constants']
        data = dic['data']
        tot = len(data)
        train_len = int(tot*train_split)
        cls.train_data = data[:train_len]
        cls.test_data = data[train_len:]
        num_train_patients = len(cls.train_data)
        num_test_patients = len(cls.test_data)
        if memoise_disable:
            cls.train_mem_ft_data = []
            cls.test_mem_ft_data = []
        else:
            cls.train_mem_ft_data = [torch.zeros(x.shape[0],x.shape[1],num_coils,x.shape[3],x.shape[4],2) for x in cls.train_data]
            cls.test_mem_ft_data = [torch.zeros(x.shape[0],x.shape[1],num_coils,x.shape[3],x.shape[4],2) for x in cls.test_data]
        cls.train_mem_ft_data_filled = [torch.zeros(1,) for x in range(num_train_patients)]
        cls.test_mem_ft_data_filled = [torch.zeros(1,) for x in range(num_test_patients)]
        cls.train_data_fft = [torch.fft.fftshift(torch.fft.fft2(x.float()/255.),dim = (-2,-1)) for x in cls.train_data]
        cls.train_data_fft = [torch.stack((x.real, x.imag), -1) for x in cls.train_data_fft]
        cls.test_data_fft = [torch.fft.fftshift(torch.fft.fft2(x.float()/255.),dim = (-2,-1)) for x in cls.test_data]
        cls.test_data_fft = [torch.stack((x.real, x.imag), -1) for x in cls.test_data_fft]

        cls.train_num_mem = torch.zeros((len(cls.train_data)))
        cls.test_num_mem = torch.zeros((len(cls.test_data)))

        [x.share_memory_() for x in cls.train_data]
        [x.share_memory_() for x in cls.test_data]
        [x.share_memory_() for x in cls.train_data_fft]
        [x.share_memory_() for x in cls.train_data_fft]
        [x.share_memory_() for x in cls.test_data_fft]
        [x.share_memory_() for x in cls.test_data_fft]
        [x.share_memory_() for x in cls.train_mem_ft_data]
        [x.share_memory_() for x in cls.test_mem_ft_data]
        [x.share_memory_() for x in cls.train_mem_ft_data_filled]
        [x.share_memory_() for x in cls.test_mem_ft_data_filled]
        cls.train_num_mem.share_memory_()
        cls.test_num_mem.share_memory_()
    
    @classmethod
    def get_shared_lists(cls):
        ans = []
        ans.append(cls.train_data)
        ans.append(cls.test_data)
        ans.append(cls.train_data_fft)
        ans.append(cls.train_data_fft)
        ans.append(cls.test_data_fft)
        ans.append(cls.test_data_fft)
        ans.append(cls.train_mem_ft_data)
        ans.append(cls.test_mem_ft_data)
        ans.append(cls.train_mem_ft_data_filled)
        ans.append(cls.test_mem_ft_data_filled)
        ans.append((cls.mu, cls.std, cls.ft_mu_r, cls.ft_mu_i, cls.ft_std_r, cls.ft_std_i))
        ans.append(cls.train_num_mem)
        ans.append(cls.test_num_mem)
        return ans
    @classmethod
    def set_shared_lists(cls, data):
        cls.train_data = data[0]
        cls.test_data = data[1]
        cls.train_data_fft = data[2]
        cls.train_data_fft = data[3]
        cls.test_data_fft = data[4]
        cls.test_data_fft = data[5]
        cls.train_mem_ft_data = data[6]
        cls.test_mem_ft_data = data[7]
        cls.train_mem_ft_data_filled = data[8]
        cls.test_mem_ft_data_filled = data[9]
        cls.mu, cls.std, cls.ft_mu_r, cls.ft_mu_i, cls.ft_std_r, cls.ft_std_i = data[10]
        cls.train_num_mem = data[11]
        cls.test_num_mem = data[12]

    def __init__(self, path, parameters, train = True, blank = False, window_mode = True):
        super(ACDC, self).__init__()
        self.path = path
        self.train = train
        self.blank = blank
        self.parameters = parameters
        self.train_split = parameters['train_test_split']
        self.resolution = parameters['image_resolution']
        self.num_coils = parameters['num_coils']
        self.window_size = parameters['window_size']
        self.ft_num_radial_views = parameters['FT_radial_sampling']
        self.norm = parameters['normalisation']
        self.memoise_disable = parameters['memoise_disable']
        self.window_mode = window_mode
        assert(self.window_size % 2 != 0)
        self.predict_mode = parameters['predicted_frame']
        assert(self.predict_mode in ['middle', 'last'])
        if self.predict_mode == 'middle':
            self.target_frame = (self.window_size-1)//2
        else:
            self.target_frame = self.window_size-1
        self.loop_videos = parameters['loop_videos']
        self.shm_loop = parameters['SHM_looping']
        if not self.blank:
            ACDC.data_init(self.path, self.resolution, self.train_split, self.num_coils, self.memoise_disable)
        
    def rest_init(self):
        if self.train:
            self.data = ACDC.train_data
            self.data_fft = ACDC.train_data_fft
            self.ft_data = ACDC.train_mem_ft_data
            self.num_mem = ACDC.train_num_mem
        else:
            self.data = ACDC.test_data
            self.data_fft = ACDC.test_data_fft
            self.ft_data = ACDC.test_mem_ft_data
            self.num_mem = ACDC.test_num_mem
        (self.mu, self.std, self.ft_mu_r, self.ft_mu_i, self.ft_std_r, self.ft_std_i) = (ACDC.mu, ACDC.std, ACDC.ft_mu_r, ACDC.ft_mu_i, ACDC.ft_std_r, ACDC.ft_std_i)
        
        self.golden_bars = get_golden_bars(resolution = self.resolution)
        self.num_golden_cycle = self.golden_bars.shape[0]
        self.coil_mask = get_coil_mask(n_coils = self.num_coils, resolution = self.resolution)

        self.num_patients = len(self.data)
        self.num_vids_per_patient = np.array([x.shape[0] for x in self.data])
        self.actual_frames_per_vid_per_patient = np.array([x.shape[1] for x in self.data])
        self.frames_per_vid_per_patient = np.array([x.shape[1] for x in self.data])
        if self.loop_videos != -1:
            self.frames_per_vid_per_patient *=0
            self.frames_per_vid_per_patient += self.loop_videos
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        self.vid_cumsum = np.cumsum(self.num_vids_per_patient)


        if self.window_mode:
            self.num_videos = int((self.num_vids_per_patient*self.frames_per_vid_per_patient).sum())
        else:
            self.num_videos = int(self.num_vids_per_patient.sum())

    def index_to_location(self, i):
        if self.window_mode:
            p_num = (self.vid_frame_cumsum <= i).sum()
            if p_num == 0:
                index_num = i
            else:
                index_num = i - self.vid_frame_cumsum[p_num-1]
            v_num = index_num // self.frames_per_vid_per_patient[p_num]
            f_num = index_num % self.frames_per_vid_per_patient[p_num]
        else:
            p_num = (self.vid_cumsum <= i).sum()
            if p_num == 0:
                v_num = i
            else:
                v_num = i - self.vid_cumsum[p_num-1]
            f_num = None

        if self.loop_videos:
            if self.shm_loop:
                if f_num is not None:
                    k = f_num
                    n = self.actual_frames_per_vid_per_patient[p_num]
                    if k >= n:
                        ni = k-n
                        inter = ni//(n-1)
                        if inter%2 == 0:
                            f_num = (n-2-ni%(n-1))
                        else:
                            f_num = 1+ni%(n-1)

            else:
                if f_num is not None:
                    f_num = f_num%self.actual_frames_per_vid_per_patient[p_num]

        return p_num, v_num, f_num

    def __getitem__(self, i):
        # FT data - num_coils,num_windows, 256, 256
        p_num, v_num, f_num = self.index_to_location(i)
        if ACDC.get_mem_filled(self.train, p_num) == 0:
            # vnum, fnum, 1, r, c
            indata = ((self.data[p_num].type(torch.float64)/255.).expand(-1,-1,self.num_coils, -1,-1)*self.coil_mask.unsqueeze(0).unsqueeze(0))
            temp = (torch.fft.fftshift(torch.fft.fft2(indata) ,dim = (-2,-1)) + EPS).log()
            if not self.memoise_disable:
                ACDC.set_mem_ft(self.train, p_num, torch.stack((temp.real, temp.imag), -1))
            # if self.train:
            #     print('Memoising for patient {}'.format(p_num), flush = True)
            # else:
            #     print('Memoising for patient {}'.format(p_num+len(ACDC.train_data)), flush = True)
            if not self.memoise_disable:
                ACDC.inc_num_mem(self.train, p_num)
            if ACDC.get_num_mem(self.train) == len(self.data):
                print('#########################')
                print('Memoised all patients!')
                print('#########################', flush = True)

        if self.memoise_disable:
            indexed_ft_data = torch.stack((temp.real, temp.imag), -1)[v_num,:,:,:,:]
        else:
            indexed_ft_data = self.ft_data[p_num][v_num,:,:,:,:]
        if self.window_mode:
            indices = torch.arange(f_num,f_num + self.window_size)%self.frames_per_vid_per_patient[p_num]
        else:
            limit = self.frames_per_vid_per_patient[p_num]
            max_avail = self.actual_frames_per_vid_per_patient[p_num]
            indices = []
            iter = 0
            while 1:
                iter += 1
                if limit == 0:
                    break
                if iter == 1:
                    choose = min(limit, max_avail)
                    indices += range(choose)
                else:
                    choose = min(limit, max_avail-1)
                    if self.shm_loop:
                        if iter % 2 == 0:
                            indices += range(max_avail-2,max_avail-2-choose,-1)
                        else:
                            indices += range(1,1+choose)
                    else:
                        indices += range(choose)
                limit -= choose
            indices = torch.tensor(indices)
        
        r_ft_data = indexed_ft_data[indices,:,:,:].permute(1,0,2,3,4)
        if self.window_mode:
            target = self.data[p_num][v_num, (f_num+self.target_frame)%self.frames_per_vid_per_patient[p_num],:,:,:].type(torch.float64)
            target = target/255.
        else:
            target = self.data[p_num][v_num, indices,:,:,:].type(torch.float64)
            target = target/255.
        if self.norm:
            target = (target-self.mu)/self.std
            r_ft_data[:,:,:,:,0] = (r_ft_data[:,:,:,:,0]-self.ft_mu_r)/self.ft_std_r
            r_ft_data[:,:,:,:,1] = (r_ft_data[:,:,:,:,1]-self.ft_mu_i)/self.ft_std_i
        
        if self.window_mode:
            golden_bars_indices = torch.arange(i,i+self.ft_num_radial_views*self.window_size)%self.num_golden_cycle
        else:
            if p_num > 0:
                done_till_now = self.vid_frame_cumsum[p_num-1]
            else:
                done_till_now = 0
            if v_num > 0:
                done_till_now += self.frames_per_vid_per_patient[p_num]*(v_num-1)

            golden_bars_indices = torch.arange(done_till_now,done_till_now+self.ft_num_radial_views*self.frames_per_vid_per_patient[p_num])%self.num_golden_cycle
        selection = self.golden_bars[golden_bars_indices,:,:].reshape(-1, self.ft_num_radial_views, self.resolution, self.resolution)
        current_window_mask = selection.sum(1).sign().float()
        
        ft_masked = r_ft_data * current_window_mask.unsqueeze(0).unsqueeze(-1)
        
        if self.window_mode:
            target_ft = self.data_fft[p_num][v_num, (f_num+self.target_frame)%self.frames_per_vid_per_patient[p_num],:,:,:]
        else:
            target_ft = self.data_fft[p_num][v_num, indices,:,:,:]
            
        if f_num is None:
            f_num = -1
            r_ft_data = r_ft_data.permute(1,0,2,3,4)
            ft_masked = ft_masked.permute(1,0,2,3,4)
            target = target.permute(1,0,2,3).squeeze()
            target_ft = target_ft.permute(1,0,2,3,4).squeeze()
        return torch.tensor([p_num, v_num, f_num]), r_ft_data.float().cpu(), ft_masked.float().cpu(), target.float().cpu(), target_ft.cpu()

        
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
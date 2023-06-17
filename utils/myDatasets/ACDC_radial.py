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
import torchkbnufft as tkbn
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
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS)).exp()

def complex_log(ct):
    indices = ct.abs() < 1e-10
    ct[indices] = CEPS
    return ct.log()

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Manual CROP
class ACDC_radial(Dataset):

    def __init__(self, path, parameters, device, train = True):
        super(ACDC_radial, self).__init__()
        
        self.path = path
        self.train = train
        self.parameters = parameters
        self.device = device
        self.train_split = parameters['train_test_split']
        
        self.final_resolution = parameters['image_resolution']
        self.resolution = 256
        assert(self.resolution == 256)
        assert(self.final_resolution <= 256)
        
        self.num_coils = parameters['num_coils']
        assert(self.num_coils == 8)

        self.ft_num_radial_views = parameters['FT_radial_sampling']

        nufft_numpoints = self.parameters['NUFFT_numpoints']
        nufft_kbwidth = self.parameters['NUFFT_kbwidth']

        self.loop_videos = parameters['loop_videos']
        self.shm_loop = parameters['SHM_looping']

        self.memoise_path = os.path.join(self.path, 'radial_memoised/manual_crop_views{}_res{}_nufftneighbors{}_nufftkbwidth{}_loop{}_shmloop{}'.format(self.ft_num_radial_views, self.final_resolution, nufft_numpoints, int(nufft_kbwidth*100), self.loop_videos, int(self.shm_loop)))

        self.norm = parameters['normalisation']
        self.shuffle_coils = parameters['shuffle_coils']
        self.memoise = parameters['memoise']
        assert(not self.norm)

        
        # Read metadata
        metadic = torch.load(os.path.join(self.path, 'radial/metadata.pth'))
        self.num_patients = metadic['num_patients']
        
        self.omega = metadic['omega']
        self.omega = self.omega.reshape(2,377,-1)
        

        self.num_vids_per_patient = np.array(metadic['num_vids_per_patient'])
        self.actual_frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        self.frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        
        self.adjkb_ob = tkbn.KbInterpAdjoint(im_size=(self.resolution*3, self.resolution*3), grid_size = (self.resolution*3, self.resolution*3), numpoints = nufft_numpoints, kbwidth = nufft_kbwidth, device = self.device)

        if self.train:
            self.offset = 0
            self.num_patients = int(self.train_split*self.num_patients)
        else:
            self.offset = int(self.train_split*self.num_patients)
            self.num_patients = self.num_patients - int(self.train_split*self.num_patients)
        self.num_vids_per_patient = self.num_vids_per_patient[self.offset:self.offset+self.num_patients]
        self.frames_per_vid_per_patient = self.frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.actual_frames_per_vid_per_patient = self.actual_frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
            
        if self.loop_videos != -1:
            self.frames_per_vid_per_patient *=0
            self.frames_per_vid_per_patient += self.loop_videos
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        self.vid_cumsum = np.cumsum(self.num_vids_per_patient)

        self.num_videos = int(self.num_vids_per_patient.sum())

        self.total_frames = self.num_coils*(self.frames_per_vid_per_patient*self.num_vids_per_patient).sum()

        self.draw_radial_views_indices = torch.zeros(self.frames_per_vid_per_patient.max(), 377)
        for i in range(self.draw_radial_views_indices.shape[0]):
            for j in range(i*self.ft_num_radial_views,(i+1)*self.ft_num_radial_views):
                self.draw_radial_views_indices[i,j%377] = 1
        self.draw_radial_views_indices = self.draw_radial_views_indices == 1

    def index_to_location(self, i):
        p_num = (self.vid_cumsum <= i).sum()
        if p_num == 0:
            v_num = i
        else:
            v_num = i - self.vid_cumsum[p_num-1]

        return p_num, v_num

    def __getitem__(self, i):
        p_num, v_num = self.index_to_location(i)
        actual_pnum = self.offset + p_num

        # Loop all videos
        limit = self.frames_per_vid_per_patient[p_num]
        max_avail = self.actual_frames_per_vid_per_patient[p_num]
        loop_indices = []
        iter = 0
        while 1:
            iter += 1
            if limit == 0:
                break
            if iter == 1:
                choose = min(limit, max_avail)
                loop_indices += range(choose)
            else:
                choose = min(limit, max_avail-1)
                if self.shm_loop:
                    if iter % 2 == 0:
                        loop_indices += range(max_avail-2,max_avail-2-choose,-1)
                    else:
                        loop_indices += range(1,1+choose)
                else:
                    loop_indices += range(choose)
            limit -= choose
        loop_indices = torch.tensor(loop_indices)


        datadic = torch.load(os.path.join(self.path, 'radial/patient_{}/vid_{}.pth'.format(actual_pnum+1, v_num)))
        og_video = datadic['data']      # Nf, RES, RES
        og_video = (og_video.float()/255.)
        og_video = og_video[loop_indices,:,:] # Nf2, RES, RES
        coil = datadic['coil']          # Nc, RES, RES
        Nf = og_video[loop_indices,:,:].shape[0]
        if os.path.exists(os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(actual_pnum, v_num))):
            load_dic = torch.load(os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(actual_pnum, v_num)))

            grid_data = load_dic['grid_data']
        else:
            ft_data = datadic['ft_radial']      # Nf, Nc, 377, 3*RES+1
            ft_data = ft_data[loop_indices,:,:,:] # Nf2, Nc, 377, 3*RES+1
            # undersampled lines
            istart = i%ft_data.shape[2]

            temp_draw_radial_views_indices = self.draw_radial_views_indices[:Nf,:]
            temp_draw_radial_views_indices = torch.roll(temp_draw_radial_views_indices, i, 1).unsqueeze(1).unsqueeze(3)
            undersampled_lines = ft_data[temp_draw_radial_views_indices.expand(*ft_data.shape)].reshape(Nf*self.num_coils, 1, -1).to(self.device)
            loop_omegas = self.omega.unsqueeze(0).repeat(Nf, 1, 1, 1)
            loop_omegas = loop_omegas[temp_draw_radial_views_indices.expand(*loop_omegas.shape)].reshape(Nf, 1, 2, -1).expand(Nf,self.num_coils,2,-1).reshape(Nf*self.num_coils, 2, -1).to(self.device)

            nufft_numpoints = self.parameters['NUFFT_numpoints']
            nufft_kbwidth = self.parameters['NUFFT_kbwidth']
            dcomp = tkbn.calc_density_compensation_function(ktraj=loop_omegas, im_size=(self.resolution*3,self.resolution*3), grid_size = (self.resolution*3,self.resolution*3), numpoints = nufft_numpoints, kbwidth = nufft_kbwidth).to(self.device)
            grid_data = self.adjkb_ob(dcomp*undersampled_lines, loop_omegas).squeeze().reshape(-1, self.num_coils, self.resolution*3, self.resolution*3)

            grid_data = torch.permute(grid_data, (1,0,2,3)) # Nc, Nf2, 3*RES, 3*RES
            undersampled_lines = undersampled_lines.cpu()
            loop_omegas = loop_omegas.cpu()
            dcomp = dcomp.cpu()

            start_ind = (self.resolution - self.final_resolution)//2

            grid_data = torch.fft.ifft2(grid_data, dim = (-2, -1))[:,:,self.resolution:2*self.resolution,self.resolution:2*self.resolution] # Nc, Nf2, RES*3, RES*3
            grid_data = grid_data[:,:,start_ind:start_ind+self.final_resolution,start_ind:start_ind+self.final_resolution]
            grid_data = torch.fft.fftshift(torch.fft.fft2(grid_data), dim = (-2, -1)) # Nc, Nf2, RES, RES
            grid_data = grid_data.cpu() # Nc, Nf, RES, RES

            if self.memoise:
                load_dic = {}

                load_dic['grid_data'] = grid_data
                if not os.path.isdir(os.path.join(self.memoise_path, 'patient_{}'.format(actual_pnum))):
                    os.makedirs(os.path.join(self.memoise_path, 'patient_{}'.format(actual_pnum)))
                torch.save(load_dic, os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(actual_pnum, v_num)))

        start_ind = (self.resolution - self.final_resolution)//2
        og_video = og_video[:,start_ind:start_ind+self.final_resolution,start_ind:start_ind+self.final_resolution]
        coil = coil[:,start_ind:start_ind+self.final_resolution,start_ind:start_ind+self.final_resolution]
        # grid_data = torch.fft.fftshift(grid_data, dim = (-2, -1)) # Nc, Nf2, RES, RES
        # grid_data = grid_data[:,:,::3,::3]

        if self.shuffle_coils:
            coil_perm = torch.randperm(coil.shape[0])
            coil = coil[coil_perm,:,:]
            grid_data = grid_data[coil_perm,:,:,:]

        og_video_coils = og_video.unsqueeze(0)*coil.unsqueeze(1) # Nc, Nf2, RES, RES
        og_coiled_fft = torch.fft.fftshift(torch.fft.fft2(og_video_coils), dim = (-2, -1)) # Nc, Nf, RES, RES
        

        # All data Nf, Nc, RES, RES
        grid_data = torch.permute(grid_data, (1,0,2,3))
        og_video_coils = torch.permute(og_video_coils, (1,0,2,3))
        og_coiled_fft = torch.permute(og_coiled_fft, (1,0,2,3))

        return torch.tensor([actual_pnum, v_num]), grid_data, og_coiled_fft, og_video_coils, og_video, Nf

        
    def __len__(self):
        return self.num_videos


# # AUTO CROP
# class ACDC_radial(Dataset):

#     def __init__(self, path, parameters, device, train = True):
#         super(ACDC_radial, self).__init__()
        
#         self.path = path
#         self.train = train
#         self.parameters = parameters
#         self.device = device
#         self.train_split = parameters['train_test_split']
        
#         self.final_resolution = parameters['image_resolution']
#         self.resolution = 256
#         assert(self.resolution == 256)
#         assert(self.final_resolution <= 256)
        
#         self.num_coils = parameters['num_coils']
#         assert(self.num_coils == 8)

#         self.ft_num_radial_views = parameters['FT_radial_sampling']

#         nufft_numpoints = self.parameters['NUFFT_numpoints']
#         nufft_kbwidth = self.parameters['NUFFT_kbwidth']

#         self.loop_videos = parameters['loop_videos']
#         self.shm_loop = parameters['SHM_looping']

#         self.memoise_path = os.path.join(self.path, 'radial_memoised/views{}_res{}_nufftneighbors{}_nufftkbwidth{}_loop{}_shmloop{}'.format(self.ft_num_radial_views, self.final_resolution, nufft_numpoints, int(nufft_kbwidth*100), self.loop_videos, int(self.shm_loop)))

#         self.norm = parameters['normalisation']
#         self.shuffle_coils = parameters['shuffle_coils']
#         self.memoise = parameters['memoise']
#         assert(not self.norm)

        
#         # Read metadata
#         metadic = torch.load(os.path.join(self.path, 'radial/metadata.pth'))
#         self.num_patients = metadic['num_patients']
        
#         self.omega = metadic['omega']
#         self.omega = self.omega.reshape(2,377,-1)
        

#         self.num_vids_per_patient = np.array(metadic['num_vids_per_patient'])
#         self.actual_frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
#         self.frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        
#         self.adjkb_ob = tkbn.KbInterpAdjoint(im_size=(self.final_resolution,self.final_resolution), grid_size = (self.final_resolution,self.final_resolution), numpoints = nufft_numpoints, kbwidth = nufft_kbwidth, device = self.device)

#         if self.train:
#             self.offset = 0
#             self.num_patients = int(self.train_split*self.num_patients)
#         else:
#             self.offset = int(self.train_split*self.num_patients)
#             self.num_patients = self.num_patients - int(self.train_split*self.num_patients)
#         self.num_vids_per_patient = self.num_vids_per_patient[self.offset:self.offset+self.num_patients]
#         self.frames_per_vid_per_patient = self.frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
#         self.actual_frames_per_vid_per_patient = self.actual_frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
            
#         if self.loop_videos != -1:
#             self.frames_per_vid_per_patient *=0
#             self.frames_per_vid_per_patient += self.loop_videos
#         self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
#         self.vid_cumsum = np.cumsum(self.num_vids_per_patient)

#         self.num_videos = int(self.num_vids_per_patient.sum())

#         self.total_frames = self.num_coils*(self.frames_per_vid_per_patient*self.num_vids_per_patient).sum()

#         self.draw_radial_views_indices = torch.zeros(self.frames_per_vid_per_patient.max(), 377)
#         for i in range(self.draw_radial_views_indices.shape[0]):
#             self.draw_radial_views_indices[i,i:i+self.ft_num_radial_views] = 1
#         self.draw_radial_views_indices = self.draw_radial_views_indices == 1

#     def index_to_location(self, i):
#         p_num = (self.vid_cumsum <= i).sum()
#         if p_num == 0:
#             v_num = i
#         else:
#             v_num = i - self.vid_cumsum[p_num-1]

#         return p_num, v_num

#     def __getitem__(self, i):
#         p_num, v_num = self.index_to_location(i)
#         Nf = self.actual_frames_per_vid_per_patient[p_num]

#         # Loop all videos
#         limit = self.frames_per_vid_per_patient[p_num]
#         max_avail = self.actual_frames_per_vid_per_patient[p_num]
#         loop_indices = []
#         iter = 0
#         while 1:
#             iter += 1
#             if limit == 0:
#                 break
#             if iter == 1:
#                 choose = min(limit, max_avail)
#                 loop_indices += range(choose)
#             else:
#                 choose = min(limit, max_avail-1)
#                 if self.shm_loop:
#                     if iter % 2 == 0:
#                         loop_indices += range(max_avail-2,max_avail-2-choose,-1)
#                     else:
#                         loop_indices += range(1,1+choose)
#                 else:
#                     loop_indices += range(choose)
#             limit -= choose
#         loop_indices = torch.tensor(loop_indices)

#         datadic = torch.load(os.path.join(self.path, 'radial/patient_{}/vid_{}.pth'.format(p_num+1+self.offset, v_num)))
#         og_video = datadic['data']      # Nf, RES, RES
#         coil = datadic['coil']          # Nc, RES, RES

#         og_video = og_video[loop_indices,:,:] # Nf2, RES, RES
        

#         if os.path.exists(os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(p_num, v_num))):
#             load_dic = torch.load(os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(p_num, v_num)))

#             grid_data = load_dic['grid_data']
#         else:

#             ft_data = datadic['ft_radial']      # Nf, Nc, 377, 3*RES+1
#             ft_data = ft_data[loop_indices,:,:,:] # Nf2, Nc, 377, 3*RES+1

#             # undersampled lines
#             istart = i%ft_data.shape[2]
#             Nf = og_video.shape[0]

#             temp_draw_radial_views_indices = self.draw_radial_views_indices[:Nf,:]
#             temp_draw_radial_views_indices = torch.roll(temp_draw_radial_views_indices, i, 1).unsqueeze(1).unsqueeze(3)

#             undersampled_lines = ft_data[temp_draw_radial_views_indices.expand(*ft_data.shape)].reshape(Nf*self.num_coils, 1, -1).to(self.device)
#             loop_omegas = self.omega.unsqueeze(0).repeat(Nf, 1, 1, 1)
#             loop_omegas = loop_omegas[temp_draw_radial_views_indices.expand(*loop_omegas.shape)].reshape(Nf, 1, 2, -1).expand(Nf,self.num_coils,2,-1).reshape(Nf*self.num_coils, 2, -1).to(self.device)

#             nufft_numpoints = self.parameters['NUFFT_numpoints']
#             nufft_kbwidth = self.parameters['NUFFT_kbwidth']
#             dcomp = tkbn.calc_density_compensation_function(ktraj=loop_omegas, im_size=(self.final_resolution,self.final_resolution), grid_size = (self.final_resolution,self.final_resolution), numpoints = nufft_numpoints, kbwidth = nufft_kbwidth).to(self.device)
#             grid_data = self.adjkb_ob(dcomp*undersampled_lines, loop_omegas).squeeze().reshape(-1, self.num_coils, self.final_resolution, self.final_resolution)

#             grid_data = torch.permute(grid_data, (1,0,2,3)) # Nc, Nf2, 3*RES, 3*RES
#             undersampled_lines = undersampled_lines.cpu()
#             loop_omegas = loop_omegas.cpu()
#             dcomp = dcomp.cpu()
#             grid_data = torch.fft.fftshift(grid_data, dim = (-2, -1)).cpu() # Nc, Nf2, RES, RES

#             if self.memoise:
#                 load_dic = {}

#                 load_dic['grid_data'] = grid_data
#                 if not os.path.isdir(os.path.join(self.memoise_path, 'patient_{}'.format(p_num))):
#                     os.makedirs(os.path.join(self.memoise_path, 'patient_{}'.format(p_num)))
#                 torch.save(load_dic, os.path.join(self.memoise_path, 'patient_{}/location_{}.pth'.format(p_num, v_num)))

#         start_ind = (self.resolution - self.final_resolution)//2

#         og_video = og_video[:,start_ind:start_ind+self.final_resolution,start_ind:start_ind+self.final_resolution]
#         coil = coil[:,start_ind:start_ind+self.final_resolution,start_ind:start_ind+self.final_resolution]

#         if self.shuffle_coils:
#             coil_perm = torch.randperm(coil.shape[0])
#             coil = coil[coil_perm,:,:]
#             grid_data = grid_data[coil_perm,:,:,:]

#         og_video_coils = og_video.unsqueeze(0)*coil.unsqueeze(1) # Nc, Nf2, RES, RES
#         og_coiled_fft = torch.fft.fftshift(torch.fft.fft2(og_video_coils), dim = (-2, -1)) # Nc, Nf, RES, RES
        
#         # All data Nf, Nc, RESm RES
#         grid_data = torch.permute(grid_data, (1,0,2,3))
#         og_video_coils = torch.permute(og_video_coils, (1,0,2,3))
#         og_coiled_fft = torch.permute(og_coiled_fft, (1,0,2,3))

#         return torch.tensor([p_num, v_num]), grid_data, og_coiled_fft, og_video_coils, og_video, Nf

        
#     def __len__(self):
#         return self.num_videos


# parameters = {}
# parameters['image_resolution'] = 64
# parameters['train_batch_size'] = 64
# parameters['test_batch_size'] = 164
# parameters['lr_kspace'] = 3e-4
# parameters['lr_ispace'] = 1e-5
# parameters['num_epochs_ispace'] = 0
# parameters['num_epochs_kspace'] = 1000
# parameters['kspace_architecture'] = 'KLSTM1'
# parameters['ispace_architecture'] = 'ILSTM1'
# parameters['history_length'] = 0
# parameters['loop_videos'] = 30
# parameters['dataset'] = 'acdc'
# parameters['train_test_split'] = 0.8
# parameters['normalisation'] = False
# parameters['window_size'] = -1
# parameters['init_skip_frames'] = 0
# parameters['SHM_looping'] = True
# parameters['FT_radial_sampling'] = 20
# parameters['num_coils'] = 8
# parameters['scale_input_fft'] = False
# parameters['dataloader_num_workers'] = 0
# parameters['optimizer'] = 'Adam'
# parameters['scheduler'] = 'StepLR'
# parameters['optimizer_params'] = (0.9, 0.999)
# parameters['scheduler_params'] = {
#     'base_lr': 3e-4,
#     'max_lr': 1e-3,
#     'step_size_up': 10,
#     'mode': 'triangular',
#     'step_size': parameters['num_epochs_kspace']//3,
#     'gamma': 0.5,
#     'verbose': True
# }
# parameters['Automatic_Mixed_Precision'] = False
# parameters['predicted_frame'] = 'last'
# parameters['loss_params'] = {
#     'SSIM_window': 11,
#     'alpha_phase': 1,
#     'alpha_amp': 1,
#     'grayscale': True,
#     'deterministic': False,
#     'watson_pretrained': True,
# }
# parameters['NUFFT_numpoints'] = 8
# parameters['NUFFT_kbwidth'] = 0.84
# parameters['shuffle_coils'] = True
# parameters['memoise'] = False


# dd = ACDC_radial('../../datasets/ACDC/', parameters, torch.device('cuda:1'), train = True)
# import time
# t1 = time.time()
# a = dd[0]
# print('Time taken = {}'.format(time.time()-t1))
# grid_data = a[1]
# for i in range(8):
#     tt = torch.fft.ifft2(torch.fft.ifftshift(grid_data[0,i,:,:])) 
#     plt.imsave('aim2_coil{}.png'.format(i),tt.real, cmap = 'gray')
# plt.imsave('aim.png',(grid_data[0,0,:,:].abs()+1).log(), cmap = 'gray')
# plt.imsave('aim_orig.png',a[-2][0,:,:], cmap = 'gray')

# for i in range(8):
#     tt = torch.fft.ifft2(torch.fft.ifftshift(grid_data[1,i,:,:])) 
#     plt.imsave('bim2_coil{}.png'.format(i),tt.real, cmap = 'gray')
# plt.imsave('bim.png',(grid_data[1,0,:,:].abs()+1).log(), cmap = 'gray')
# plt.imsave('bim_orig.png',a[-2][1,:,:], cmap = 'gray')

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

    def __init__(self, path, parameters, device, train = True, visualise_only = False):
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

        if visualise_only:
            self.loop_videos = 120
        else:
            self.loop_videos = parameters['loop_videos']
        self.shm_loop = parameters['SHM_looping']
        assert(not self.shm_loop)

        self.data_path = os.path.join(self.path, 'radial_faster/{}_resolution_{}_spokes'.format(self.final_resolution, self.ft_num_radial_views))

        self.norm = parameters['normalisation']
        self.shuffle_coils = parameters['shuffle_coils']
        assert(not self.shuffle_coils)
        self.memoise = parameters['memoise']
        assert(not self.norm)

        
        # Read metadata
        metadic = torch.load(os.path.join(self.data_path, 'metadata.pth'))
        
        self.num_patients = metadic['num_patients']
        self.omegas = metadic['omegas']
        self.num_vids_per_patient = metadic['num_vids_per_patient']
        self.coil_masks = metadic['coil_masks']
        self.coils_per_patient_per_video = metadic['coils_per_patient_per_video']
        self.GAs_per_patient_per_video = metadic['GAs_per_patient_per_video']
        self.masks = metadic['masks']

        self.actual_frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        self.frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        
        if self.train:
            self.offset = 0
            self.num_patients = int(self.train_split*self.num_patients)
        else:
            self.offset = int(self.train_split*self.num_patients)
            self.num_patients = self.num_patients - int(self.train_split*self.num_patients)
        
        self.num_vids_per_patient = self.num_vids_per_patient[self.offset:self.offset+self.num_patients]
        self.frames_per_vid_per_patient = self.frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.actual_frames_per_vid_per_patient = self.actual_frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.GAs_per_patient_per_video = self.GAs_per_patient_per_video[self.offset:self.offset+self.num_patients]
        self.coils_per_patient_per_video = self.coils_per_patient_per_video[self.offset:self.offset+self.num_patients]

        if self.loop_videos != -1:
            self.frames_per_vid_per_patient *=0
            self.frames_per_vid_per_patient += self.loop_videos
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        self.vid_cumsum = np.cumsum(self.num_vids_per_patient)

        self.num_videos = int(sum(self.num_vids_per_patient))

        self.RAM_memoised = [None for k in range(self.num_videos)]

        self.total_unskipped_frames = self.num_coils*((self.frames_per_vid_per_patient-self.parameters['init_skip_frames'])*self.num_vids_per_patient).sum()


    def index_to_location(self, i):
        p_num = (self.vid_cumsum <= i).sum()
        if p_num == 0:
            v_num = i
        else:
            v_num = i - self.vid_cumsum[p_num-1]

        return p_num, v_num

    def __getitem__(self, i):

        # actual_pnum = 1
        # v_num = 1
        # grid_data = torch.zeros(30,8,256,256)
        # grid_data = torch.complex(grid_data, grid_data)
        # og_coiled_fft = grid_data
        # og_video_coils = torch.zeros(30,8,256,256)
        # og_video = torch.zeros(30,256,256)
        # Nf = 30

        # return torch.tensor([actual_pnum, v_num]), grid_data, og_coiled_fft, og_video_coils, og_video, Nf

        # if self.parameters['memoise_RAM']:
        #     if self.RAM_memoised[i] is not None:
        #         ind, grid_data, og_coiled_fft, og_video_coils, og_video, Nf = self.RAM_memoised[i]
        #         return ind, grid_data, og_coiled_fft, og_video_coils, og_video, Nf

        # start = time.time()
        p_num, v_num = self.index_to_location(i)
        actual_pnum = self.offset + p_num
        coils_used = self.coils_per_patient_per_video[p_num][v_num]
        coils_used = self.coil_masks[coils_used,:,:].unsqueeze(0)
        GAs_used = self.GAs_per_patient_per_video[p_num][v_num][:self.loop_videos,:]
        masks_applicable = (self.masks[GAs_used.reshape(-1),:,:]).reshape(self.loop_videos,-1,self.final_resolution,self.final_resolution)
        masks_applicable = torch.from_numpy(masks_applicable.any(1)).unsqueeze(1).type(torch.int32)

        dic_path = os.path.join(self.data_path, 'patient_{}/vid_{}.pth'.format(actual_pnum+1, v_num))

        dic = torch.load(dic_path, map_location = torch.device('cpu'))
        og_video = ((dic['targ_video']/255.)[:self.loop_videos,:,:,:])
        # og_video_coils = og_video*coils_used
        Nf = self.actual_frames_per_vid_per_patient[p_num]
        coilwise_input = (dic['coilwise_input']/255.)[:self.loop_videos,:,:,:]
        # grid_data = torch.fft.fftshift(torch.fft.fft2(coilwise_input), dim = (-2,-1))
        # og_coiled_fft = torch.fft.fftshift(torch.fft.fft2(og_video_coils), dim = (-2,-1))

        # grid_data, masks_applicable, og_coiled_fft, og_video_coils, og_video = grid_data.cpu(), masks_applicable.cpu(), og_coiled_fft.cpu(), og_video_coils.cpu(), og_video.cpu()

        # print('Time = ', time.time() - start)
        # del dic
        # del coilwise_input
        # del coils_used

        # if self.parameters['memoise_RAM']:
        #     if self.RAM_memoised[i] is None:
        #         self.RAM_memoised[i] = [torch.tensor([actual_pnum, v_num]), grid_data, og_coiled_fft, og_video_coils, og_video, Nf]


        # return torch.tensor([actual_pnum, v_num]), grid_data, masks_applicable, og_coiled_fft, og_video_coils, og_video, Nf
        return torch.tensor([actual_pnum, v_num]), masks_applicable, og_video, coilwise_input, coils_used, Nf

        
    def __len__(self):
        return self.num_videos


if __name__ == '__main__':
    parameters = {}
    parameters['image_resolution'] = 256
    parameters['train_batch_size'] = 256
    parameters['test_batch_size'] = 2
    parameters['lr_kspace'] = 3e-4
    parameters['lr_ispace'] = 1e-5
    parameters['num_epochs_ispace'] = 0
    parameters['num_epochs_kspace'] = 1000
    parameters['kspace_architecture'] = 'KLSTM1'
    parameters['ispace_architecture'] = 'ILSTM1'
    parameters['history_length'] = 0
    parameters['loop_videos'] = 30
    parameters['dataset'] = 'acdc'
    parameters['train_test_split'] = 0.8
    parameters['normalisation'] = False
    parameters['window_size'] = -1
    parameters['init_skip_frames'] = 0
    parameters['SHM_looping'] = False
    parameters['FT_radial_sampling'] = 10
    parameters['memoise_RAM'] = False
    parameters['num_coils'] = 8
    parameters['scale_input_fft'] = False
    parameters['dataloader_num_workers'] = 0
    parameters['optimizer'] = 'Adam'
    parameters['scheduler'] = 'StepLR'
    parameters['optimizer_params'] = (0.9, 0.999)
    parameters['scheduler_params'] = {
        'base_lr': 3e-4,
        'max_lr': 1e-3,
        'step_size_up': 10,
        'mode': 'triangular',
        'step_size': parameters['num_epochs_kspace']//3,
        'gamma': 0.5,
        'verbose': True
    }
    parameters['Automatic_Mixed_Precision'] = False
    parameters['predicted_frame'] = 'last'
    parameters['loss_params'] = {
        'SSIM_window': 11,
        'alpha_phase': 1,
        'alpha_amp': 1,
        'grayscale': True,
        'deterministic': False,
        'watson_pretrained': True,
    }
    parameters['NUFFT_numpoints'] = 8
    parameters['NUFFT_kbwidth'] = 0.84
    parameters['shuffle_coils'] = False
    parameters['memoise'] = False


    dd = ACDC_radial('../../datasets/ACDC/', parameters, torch.device('cuda:1'), train = True)
    id, masks, og_video, coilwise_input, coils_used, Nf = dd[0]
    grid_data = torch.fft.fftshift(torch.fft.fft2(coilwise_input), dim = (-2,-1))
    og_coiled_vids = og_video * coils_used
    og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
    if not os.path.isdir('trial'):
        os.mkdir('trial')
    if not os.path.isdir('trial/grid_data'):
        os.mkdir('trial/grid_data')
    if not os.path.isdir('trial/og_coiled_fft'):
        os.mkdir('trial/og_coiled_fft')
    if not os.path.isdir('trial/og_video_coils'):
        os.mkdir('trial/og_video_coils')
    if not os.path.isdir('trial/og_video'):
        os.mkdir('trial/og_video')
        
    # for i in tqdm(range(len(dd))):
    #     a = dd[i]

    # for fi in tqdm(range(Nf), desc = 'grid_data'):
    #     fig = plt.figure(figsize = (32,16))
    #     loop_mask = masks[fi,0,:,:]
    #     for ci in range(grid_data.shape[1]):
    #         loop_grid_image = grid_data[fi,ci,:,:].cpu()
    #         inverse = torch.fft.ifft2(torch.fft.ifftshift(loop_grid_image, dim = (-2, -1))).real.cpu()
    #         masked_grid_image = loop_mask * loop_grid_image
    #         inverse_masked = torch.fft.ifft2(torch.fft.ifftshift(masked_grid_image, dim = (-2, -1))).real.cpu()
    #         plt.subplot(4,8,ci+1)
    #         plt.axis('off')
    #         plt.imshow((1+loop_grid_image.abs()).log(), cmap = 'gray')
    #         plt.subplot(4,8,ci+9)
    #         plt.axis('off')
    #         plt.imshow(inverse, cmap = 'gray')
    #         plt.subplot(4,8,ci+17)
    #         plt.axis('off')
    #         plt.imshow((1+(masked_grid_image).abs()).log(), cmap = 'gray')
    #         plt.subplot(4,8,ci+25)
    #         plt.axis('off')
    #         plt.imshow(inverse_masked, cmap = 'gray')
    #     plt.savefig('trial/grid_data/{}.jpg'.format(fi))

    # for fi in tqdm(range(Nf), desc = 'og_coiled_fft'):
    #     fig = plt.figure(figsize = (32,8))
    #     plt.axis('off')
    #     for ci in range(og_coiled_fft.shape[1]):
    #         loop_og_coiled_image = og_coiled_fft[fi,ci,:,:].cpu()
    #         inverse = torch.fft.ifft2(torch.fft.ifftshift(loop_og_coiled_image, dim = (-2, -1))).real.cpu()
    #         plt.subplot(2,8,ci+1)
    #         plt.axis('off')
    #         plt.imshow((1+loop_og_coiled_image.abs()).log(), cmap = 'gray')
    #         plt.subplot(2,8,ci+9)
    #         plt.axis('off')
    #         plt.imshow(inverse, cmap = 'gray')
    #     plt.savefig('trial/og_coiled_fft/{}.jpg'.format(fi))

    # for fi in tqdm(range(Nf), desc = 'og_video_coils'):
    #     fig = plt.figure(figsize = (16,8))
    #     for ci in range(og_video_coils.shape[1]):
    #         loop_og_video_coils = og_video_coils[fi,ci,:,:].cpu()
    #         plt.subplot(2,4,ci+1)
    #         plt.axis('off')
    #         plt.imshow(loop_og_video_coils, cmap = 'gray')
    #     plt.savefig('trial/og_video_coils/{}.jpg'.format(fi))

    # for fi in tqdm(range(Nf), desc = 'og_video'):
    #     fig = plt.figure(figsize = (4,4))
    #     loop_og_video = og_video[fi,0,:,:].cpu()
    #     plt.subplot(1,1,1)
    #     plt.axis('off')
    #     plt.imshow(loop_og_video, cmap = 'gray')
    #     plt.savefig('trial/og_video/{}.jpg'.format(fi))



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

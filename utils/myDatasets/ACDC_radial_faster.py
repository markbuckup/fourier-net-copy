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

from scipy.ndimage import median_filter as median_filter_func
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

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

class ACDC_radial_ispace(Dataset):
    
    mem_inputs = []
    mem_inputs_min = []
    mem_inputs_max = []
    mem_gts = []
    mem_stored_bool = []
    data_init_done = False
    
    @classmethod
    def set_data(cls, input, gt, pat_id, vid_id):
        # print('setting', pat_id, vid_id)
        
        cls.mem_inputs_max[pat_id][vid_id] = input.max(-1)[0].max(-1)[0]
        cls.mem_inputs_min[pat_id][vid_id] = input.min(-1)[0].min(-1)[0]

        input_scaled = input - input.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
        input_scaled = input / (1e-10 + input.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

        cls.mem_inputs[pat_id][vid_id] = (input_scaled*255).type(torch.uint8)
        cls.mem_gts[pat_id][vid_id] = (gt*255).type(torch.uint8)
        cls.mem_stored_bool[pat_id][vid_id] = 1

    @classmethod
    def get_data(cls, pat_id, vid_id):
        # print('loading', pat_id, vid_id)

        maxs = (cls.mem_inputs_max[pat_id][vid_id]).unsqueeze(-1).unsqueeze(-1)
        mins = (cls.mem_inputs_min[pat_id][vid_id]).unsqueeze(-1).unsqueeze(-1)

        assert(cls.mem_stored_bool[pat_id][vid_id] == 1)
        ret_inp = (cls.mem_inputs[pat_id][vid_id]).float()/255. 
        ret_inp = ret_inp * (maxs-mins)
        ret_inp = ret_inp + mins
        ret_gt = (cls.mem_gts[pat_id][vid_id]).float()/255.
        return ret_inp, ret_gt

    @classmethod
    def check_data(cls, pat_id, vid_id):
        # print('checking', pat_id, vid_id)
        return cls.mem_stored_bool[pat_id][vid_id] == 1

    @classmethod
    def bulk_set_data(cls, actual_pnums, vnums, inputs, gts):
        assert(len(inputs.shape) == 5)
        assert(len(gts.shape) == 5)
        for i,(pnum, vnum) in enumerate(zip(actual_pnums, vnums)):
            # input shape = B, 120, Chan, 256, 256
            # GT shape = B, 120, 1, 256, 256
            cls.set_data(inputs.cpu()[i], gts.cpu()[i], pnum, vnum)

    @classmethod
    def data_init(cls, whole_num_vids_per_patient, parameters):
        if cls.data_init_done:
            return
        for n_vids in whole_num_vids_per_patient:
            cls.mem_inputs.append([])
            cls.mem_inputs_max.append([])
            cls.mem_inputs_min.append([])
            cls.mem_gts.append([])
            cls.mem_stored_bool.append([])
            for vi in range(n_vids):
                if parameters['memoise_ispace']:
                    if parameters['coil_combine'] == 'SOS':
                        cls.mem_inputs[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1,parameters['image_resolution'],parameters['image_resolution']).type(torch.uint8))
                        cls.mem_inputs_max[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1))
                        cls.mem_inputs_min[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1))
                    else:
                        cls.mem_inputs[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils'],parameters['image_resolution'],parameters['image_resolution']).type(torch.uint8))
                        cls.mem_inputs_max[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils']))
                        cls.mem_inputs_min[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],parameters['num_coils']))
                    cls.mem_gts[-1].append(torch.zeros(cls.max_frames - parameters['init_skip_frames'],1,parameters['image_resolution'],parameters['image_resolution']))
                cls.mem_stored_bool[-1].append(torch.zeros(1))

            [x.share_memory_() for x in cls.mem_inputs[-1]]
            [x.share_memory_() for x in cls.mem_gts[-1]]
            [x.share_memory_() for x in cls.mem_stored_bool[-1]]

        cls.data_init_done = True



    def __init__(self, path, parameters, device, train = True, visualise_only = False):
        super(ACDC_radial_ispace, self).__init__()
        ACDC_radial_ispace.max_frames = 120
        self.train = train
        self.parameters = parameters.copy()
        self.parameters['loop_videos'] = ACDC_radial_ispace.max_frames
        self.parameters['init_skip_frames'] = 90
        self.orig_dataset = ACDC_radial(path, self.parameters, device, train = train, visualise_only = visualise_only)
        self.total_unskipped_frames = self.parameters['num_coils']*((ACDC_radial_ispace.max_frames - self.parameters['init_skip_frames'])*self.orig_dataset.num_vids_per_patient).sum()
        # ACDC_radial_ispace.data_init(self.orig_dataset.whole_num_vids_per_patient, self.parameters)


    def __getitem__(self, i):
        if not ACDC_radial_ispace.data_init_done:
            ACDC_radial_ispace.data_init(self.orig_dataset.whole_num_vids_per_patient, self.parameters)
        pat_id, vid_id = self.orig_dataset.index_to_location(i)
        pat_id += self.orig_dataset.offset
        if self.train:
            assert(pat_id < 120)
        else:
            assert(pat_id >= 120)

        if ACDC_radial_ispace.check_data(pat_id, vid_id):
            mem = torch.tensor(1)
            data, gt = ACDC_radial_ispace.get_data(pat_id, vid_id)
            return mem, data, gt
        else:
            mem = torch.tensor(0)
            return mem, *self.orig_dataset[i]

    def __len__(self):
        return len(self.orig_dataset)

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
        
        self.num_coils = parameters['num_coils']
        assert(self.num_coils == 8)

        self.ft_num_radial_views = parameters['kspace_num_spokes']

        if visualise_only:
            self.loop_videos = 120
        else:
            self.loop_videos = parameters['loop_videos']
        
        self.data_path = os.path.join(self.path, 'radial_faster/{}_resolution_{}_spokes'.format(self.final_resolution, self.ft_num_radial_views))

        # Read metadata
        metadic = torch.load(os.path.join(self.data_path, 'metadata.pth'))
        
        self.num_patients = metadic['num_patients']
        self.num_vids_per_patient = np.array(metadic['num_vids_per_patient'])
        self.coil_masks = metadic['coil_masks']
        self.coil_variant_per_patient_per_video = metadic['coil_variant_per_patient_per_video']
        self.GAs_per_patient_per_video = metadic['GAs_per_patient_per_video']

        self.whole_num_patients = self.num_patients
        self.whole_num_vids_per_patient = self.num_vids_per_patient.copy()

        self.actual_frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        self.frames_per_vid_per_patient = np.array(metadic['frames_per_vid_per_patient'])
        
        if self.train:
            self.offset = 0
            self.num_patients = int(self.train_split*self.num_patients)
        else:
            self.offset = int(self.train_split*self.num_patients)
            self.num_patients = self.num_patients - int(self.train_split*self.num_patients)
        
        # # DEBUG
        if self.parameters['acdc_debug_mini']:
            self.num_vids_per_patient *= 0
            self.num_vids_per_patient += 1
        # # DEBUG

        self.num_vids_per_patient = self.num_vids_per_patient[self.offset:self.offset+self.num_patients]
        self.frames_per_vid_per_patient = self.frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.actual_frames_per_vid_per_patient = self.actual_frames_per_vid_per_patient[self.offset:self.offset+self.num_patients]
        self.GAs_per_patient_per_video = self.GAs_per_patient_per_video[self.offset:self.offset+self.num_patients]
        self.coil_variant_per_patient_per_video = self.coil_variant_per_patient_per_video[self.offset:self.offset+self.num_patients]

        if self.loop_videos != -1:
            self.frames_per_vid_per_patient *= 0
            self.frames_per_vid_per_patient += self.loop_videos
        self.vid_frame_cumsum = np.cumsum(self.num_vids_per_patient*self.frames_per_vid_per_patient)
        self.vid_cumsum = np.cumsum(self.num_vids_per_patient)

        self.num_videos = int(sum(self.num_vids_per_patient))

        self.total_unskipped_frames = self.parameters['num_coils']*((self.frames_per_vid_per_patient-self.parameters['init_skip_frames'])*self.num_vids_per_patient).sum()


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

        p_num, v_num = self.index_to_location(i)
        actual_pnum = self.offset + p_num
        index_coils_used = self.coil_variant_per_patient_per_video[p_num][v_num]
        coils_used = torch.flip(self.coil_masks[index_coils_used].unsqueeze(0), dims = (-2,-1))

        GAs_used = self.GAs_per_patient_per_video[p_num][v_num][:self.loop_videos,:]


        # for i in range(8):
        #     plt.imsave('{}_{}_coil_{}.jpg'.format(p_num, v_num, i), (coils_used[0,i,:,:]), cmap = 'gray')

        dic_path = os.path.join(self.data_path, 'patient_{}/vid_{}.pth'.format(actual_pnum+1, v_num))


        dic = torch.load(dic_path, map_location = torch.device('cpu'))
        masks_applicable = dic['spoke_mask'].type(torch.float32)[:self.loop_videos,:,:,:]
        og_video = ((dic['targ_video']/255.)[:self.loop_videos,:,:,:])
        undersampled_fts = dic['coilwise_input'][:self.loop_videos,:,:,:]
        if 'coilwise_targets' not in dic:
            coilwise_targets = og_video * coils_used
            temp = ((coilwise_targets**2).sum(2)**0.5).max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            coilwise_targets = coilwise_targets / (1e-10 + temp)
        else:
            coilwise_targets = (dic['coilwise_targets']/255.)[:self.loop_videos,:,:,:]

        # for i in range(8):
        #     plt.imsave('{}_{}_undersampled_fts_{}.jpg'.format(p_num, v_num, i), torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts[0,i,:,:])).abs(), cmap = 'gray')
        # asdf

        Nf = self.actual_frames_per_vid_per_patient[p_num]

        return torch.tensor([actual_pnum, v_num]), masks_applicable, og_video, coilwise_targets, undersampled_fts, coils_used, Nf

        
    def __len__(self):
        return self.num_videos
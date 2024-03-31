import torch
import torchvision
import os
import random
import PIL
import argparse
import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
from tqdm import tqdm
import kornia
import pickle
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
import numpy as np
import nibabel as nib
from skimage.exposure import match_histograms

import torchkbnufft as tkbn

from utils.functions import get_window_mask, get_coil_mask

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(0)

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type = int, default = 256)
parser.add_argument('--gpu', type = int, default = -1)
parser.add_argument('--pat_start', type = int, default = 1)
parser.add_argument('--pat_end', type = int, default = 150)
parser.add_argument('--metadata_only', action = 'store_true')
args = parser.parse_args()

if args.gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.gpu))

histogram_target = nib.load('raw/patient001/patient001_4d.nii.gz').get_fdata()[:,:,0,7]

def read_nib_preprocess(path, heq = True):
    img = nib.load(path).get_fdata()
    drop_videos = []
    flag = False
    tot = 0
    for vi in range(img.shape[2]):
        if (not np.isfinite(img[:,:,vi,:]).all()) or (img[:,:,vi,:].sum(0).sum(0) == 0).any():
            # drop video vi
            drop_videos.append(vi)
            flag = True
    for di in reversed(sorted(drop_videos)):
        tot += 1
        img = np.concatenate((img[:,:,:di,:], img[:,:,di+1:,:]), 2)
    if heq:
        for i in range(img.shape[2]):
            for j in range(img.shape[3]):
                img[:,:,i,j] = match_histograms(img[:,:,i,j], histogram_target)
                img[:,:,i,j] = img[:,:,i,j] - img[:,:,i,j].min()
                img[:,:,i,j] = img[:,:,i,j] / img[:,:,i,j].max()
    return img, flag, tot

def num_to_str(num):
    x = str(num)
    while len(x) < 3:
        x = '0' + x
    return './raw/patient{}/patient{}_4d.nii.gz'.format(x,x)

GR = (1 + (5**0.5))/2
GA = np.pi/GR

NUM_PATIENTS = 150
RES=args.resolution
NUM_SPOKES = 10
N_COILS_VARIANTS = 100
N_COILS = 8
LOOP_FRAMES = 120

kb_ob = tkbn.KbNufft(im_size=(512,512), grid_size = (1024,1024), numpoints = 4, kbwidth = 2.34, device = device)
kbinterp = tkbn.KbInterpAdjoint(im_size=(256,256), grid_size = (256,256), numpoints = 3, kbwidth = 2.34, device = device)
kbinterp2 = tkbn.KbInterpAdjoint(im_size=(256,256), grid_size = (1024,1024), numpoints = 8, kbwidth = 0.84, device = device)

def get_omega2d(thetas, rad_max = 128):
    
    omega1 = torch.zeros(2,thetas.shape[0],2*rad_max+1).to(device)

    for itheta, theta_rad in enumerate(thetas):

        rads = np.arange(-rad_max,rad_max+1) # ex) - 256 to 256, this is the grid in the k-space i think ?
        xs = rads*np.cos(theta_rad) # ? - ex) xs = [-256,-255,...255,256]
        # and multiple with the grid size (=rads)
        ys = rads*np.sin(theta_rad) # ? - ex) ys = [0,0,...,0,0]
        
        omega1[0,itheta,:] = torch.from_numpy(np.pi*(xs/rad_max)) # ?, in 0 axis omega, filled with xs, rad_max=256, every omega value will be between 0 and 1
        omega1[1,itheta,:] = torch.from_numpy(np.pi*(ys/rad_max)) # ?, in 1 axis omega, filled with ys
        # omega1 = torch.Size([2, 65, 513]) for the 65 spoke, first nufft
    return omega1


transform = torch.nn.Sequential(
    transforms.Resize((256,256), antialias = True),
    transforms.CenterCrop(RES),
)

metapath = 'radial_faster/{}_resolution_{}_spokes/metadata.pth'.format(RES, NUM_SPOKES)
if os.path.isfile(metapath):
    metadic = torch.load(metapath)
    print("Meta Data Loaded")
else:
    os.makedirs('radial_faster/{}_resolution_{}_spokes'.format(RES, NUM_SPOKES), exist_ok = True)
    metadic = {}
    metadic['num_patients'] = NUM_PATIENTS
    metadic['coil_masks'] = torch.zeros((N_COILS_VARIANTS,N_COILS, RES, RES))
    for i in range(N_COILS_VARIANTS):
        metadic['coil_masks'][i] = get_coil_mask(theta_init = (np.pi/4)*i*(1/N_COILS_VARIANTS), n_coils = N_COILS, resolution = RES)

    metadic['coil_variant_per_patient_per_video'] = [[] for i in range(NUM_PATIENTS)]

    metadic['GAs_per_patient_per_video'] = [[] for i in range(NUM_PATIENTS)]
    metadic['num_vids_per_patient'] = [0 for i in range(NUM_PATIENTS)]
    metadic['frames_per_vid_per_patient'] = [0 for i in range(NUM_PATIENTS)]
    

    # metadic['spoke_masks'] = torch.zeros(377,RES,RES) == 0
    # for angle_index in tqdm(range(377), desc = 'Writing Metadata spoke_masks'):
    #     angles = GA*(np.arange(1)+angle_index)
    #     omega1 = get_omega2d(angles, rad_max = 256).to(device).reshape(2,-1)
    #     dcomp_full = tkbn.calc_density_compensation_function(ktraj=omega1, im_size=(256,256), grid_size = (1024,1024), numpoints = 8, kbwidth = 0.34)
    #     metadic['spoke_masks'][angle_index] = torch.fft.fftshift(kbinterp(dcomp_full, omega1)[0,:].abs().squeeze(), dim = (-2,-1)).cpu() > 0
    #     mask = torch.fft.fftshift(kbinterp(dcomp_full, omega1)[0,:].abs().squeeze(), dim = (-2,-1)).cpu()
    #     print(mask.sum())
    #     plt.imsave('mask_8.jpg', mask > 0)
    #     asdf


    for p_num in tqdm(range(NUM_PATIENTS), desc = 'Writing Metadata patient data'):
        path = num_to_str(p_num+1)
        d, flag, dvid = read_nib_preprocess(path)
        if flag:
            print("Patient {} flagged, {} video(s) dropped".format(i, dvid))
        d = torch.permute(torch.FloatTensor(d).unsqueeze(4), (2,3,4,0,1))
        num_vids, num_frames, n_coils, n_row, n_col = d.shape

        metadic['num_vids_per_patient'][p_num] = num_vids
        metadic['frames_per_vid_per_patient'][p_num] = num_frames

        for vi in range(num_vids):
            ga_random_start = np.random.randint(377)
            metadic['GAs_per_patient_per_video'][p_num].append((ga_random_start+np.arange(LOOP_FRAMES*NUM_SPOKES).reshape(LOOP_FRAMES,NUM_SPOKES))%377)
            if not metadic['coil_variant_per_patient_per_video'][p_num]:
                metadic['coil_variant_per_patient_per_video'][p_num] = [0 for i in range(num_vids)]
            metadic['coil_variant_per_patient_per_video'][p_num][vi] = np.random.randint(N_COILS_VARIANTS)

    torch.save(metadic, metapath)
    print("Meta Data Generated")


if not args.metadata_only:
    for p_num in range(args.pat_start-1, args.pat_end):
        folder_name = 'radial_faster/{}_resolution_{}_spokes/patient_{}'.format(RES, NUM_SPOKES,p_num+1)
        os.makedirs(folder_name, exist_ok = True)

        path = num_to_str(p_num+1)
        d_inp, flag, dvid = read_nib_preprocess(path)
        if flag:
            print("Patient {} flagged, {} video(s) dropped".format(p_num, dvid))
        d_inp = torch.permute(torch.FloatTensor(d_inp).unsqueeze(4), (2,3,4,0,1))
        num_vids, num_frames, n_coils, n_row, n_col = d_inp.shape
        d_padded = torch.zeros(num_vids, num_frames, n_coils, max(n_row, n_col),max(n_row, n_col))
        row_pad = (max(n_row, n_col) - n_row)//2
        col_pad = (max(n_row, n_col) - n_col)//2

        d_padded[:,:,:,row_pad:row_pad + n_row, col_pad:col_pad+n_col] = d_inp
        del d_inp

        d = transform(d_padded.reshape(num_vids*num_frames, 1, max(n_row,n_col), max(n_row,n_col))).reshape(num_vids, num_frames, n_coils, RES, RES)
        d = d - d.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
        d = d / (1e-10 + d.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1))


        for vi in tqdm(range(num_vids), desc = 'Running for Patient {}'.format(p_num+1)):
            dic = {}
            dic['targ_video'] = d[vi,np.arange(LOOP_FRAMES)%num_frames] # FRAMES, 1, RES, RES
            dic['spoke_mask'] = torch.zeros((LOOP_FRAMES, 1, RES, RES)).type(torch.uint8) # FRAMES, COILS, RES, RES
            # dic['coilwise_input_mag'] = torch.zeros((LOOP_FRAMES, N_COILS, RES, RES)).type(torch.uint8) # FRAMES, COILS, RES, RES
            dic['coilwise_input'] = torch.zeros((LOOP_FRAMES, N_COILS, RES, RES)) # FRAMES, COILS, RES, RES
            dic['coilwise_input'] = torch.complex(dic['coilwise_input'],dic['coilwise_input'])

            for fi in range(LOOP_FRAMES):
                coil_variant = metadic['coil_variant_per_patient_per_video'][p_num][vi]
                curr_coil = metadic['coil_masks'][coil_variant]

                input_frame = dic['targ_video'][fi,:,:,:] * curr_coil
                angle_indices = metadic['GAs_per_patient_per_video'][p_num][vi][fi]
                angles = GA*angle_indices

                omega1 = get_omega2d(angles, rad_max = 256).to(device).reshape(2,-1)

                padded_frame = torch.zeros(1, N_COILS, 2*input_frame.shape[1],2*input_frame.shape[2])
                padded_frame[0,:,(RES//2):(RES//2)+RES,(RES//2):(RES//2)+RES] = input_frame

                y = kb_ob(torch.complex(padded_frame,padded_frame*0).to(device),omega1)

                dcomp_full = tkbn.calc_density_compensation_function(ktraj=omega1, im_size=(256,256), grid_size = (1024,1024), numpoints = 4, kbwidth = 2.34).to(device)
                dic['spoke_mask'][fi,0,:,:] = torch.fft.fftshift(kbinterp(dcomp_full, omega1)[0,:].abs().squeeze(), dim = (-2,-1)).cpu() > 0

                myfft_interp = torch.fft.fftshift(kbinterp2(dcomp_full*y, omega1)[0,:].cpu())
                myfft_interp = myfft_interp[:,::4,::4]

                dic['coilwise_input'][fi] = myfft_interp


                # inverse_interp = torch.fft.ifft2(torch.fft.ifftshift(myfft_interp))
                # for coil in range(8):
                #     plt.imsave('coil_{}_inverse_interp.jpg'.format(coil), inverse_interp.abs()[coil], cmap = 'gray')
                #     plt.imsave('coil_{}_mag_of_log_interp.jpg'.format(coil), (myfft_interp + torch.complex(torch.tensor([1e-10]),torch.tensor([1e-10])).exp()).log().abs()[coil], cmap = 'gray')
                #     plt.imsave('coil_{}_phase_interp.jpg'.format(coil), torch.atan2(myfft_interp.imag, myfft_interp.real)[coil], cmap = 'gray')
                #     plt.imsave('coil_{}_mask.jpg'.format(coil), dic['spoke_mask'][fi,0], cmap = 'gray')
                #     plt.imsave('coil_{}.jpg'.format(coil), curr_coil[coil], cmap = 'gray')
                # asdf

                # if fi == 1:
                #     asdf

            dic['targ_video'] = (dic['targ_video']*255).type(torch.uint8)
            torch.save(dic, os.path.join(folder_name, 'vid_{}.pth'.format(vi)))

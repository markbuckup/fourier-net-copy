import torch
import torchvision
import os
import PIL
import sys
sys.path.append('../../')
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
import numpy as np
import nibabel as nib
from utils.utils import get_window_mask, get_coil_mask
from skimage.exposure import match_histograms

parser = argparse.ArgumentParser()
parser.add_argument('--resolution', type = int, default = 128)

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
                img[:,:,i,j] = match_histograms(img[:,:,i,j], histogram_target, channel_axis=None)
                img[:,:,i,j] = img[:,:,i,j] - img[:,:,i,j].min()
                img[:,:,i,j] = img[:,:,i,j] / img[:,:,i,j].max()
    return img, flag, tot

def num_to_str(num):
    x = str(num)
    while len(x) < 3:
        x = '0' + x
    return './raw/patient{}/patient{}_4d.nii.gz'.format(x,x)

NUM_PATIENTS = 150
RES=args.resolution
data = []

transform = torchvision.transforms.Resize((RES,RES))

if os.path.isfile('processed/processed_data_{}.pth'.format(RES)):
    dic = torch.load('processed/processed_data_{}.pth'.format(RES))
    data = dic['data']
    (mu, std) = dic['normalisation_constants']
    patient_num_videos = dic['patient_num_videos']
else:
    squared_sum = 0
    ft_squared_sum = [0,0]
    sum = 0
    ft_sum = [0,0]
    n_samples = 0
    ft_n_samples = 0
    patient_num_videos = {}
    NUM_COILS = 8
    coil_mask = get_coil_mask(n_coils = NUM_COILS, resolution = RES)

    for i in tqdm(range(1,NUM_PATIENTS+1)):
        path = num_to_str(i)
        d, flag, dvid = read_nib_preprocess(path)
        if flag:
            print("Patient {} flagged, {} video(s) dropped".format(i, dvid))
        r,c,d1,d2 = d.shape
        d = torch.permute(torch.FloatTensor(d).unsqueeze(4), (2,3,4,0,1)) # d1, d2, 1, r, c
        data.append(((transform(d.reshape(d1*d2, 1, r, c)).reshape(d1, d2, 1, RES, RES))*255).type(torch.uint8))
        patient_num_videos[i-1] = (data[-1].shape[0], data[-1].shape[1])
        
        indata = ((data[-1].type(torch.float64)/255.).expand(-1,-1,NUM_COILS, -1,-1)*coil_mask.unsqueeze(0).unsqueeze(0))
        ft_data = torch.fft.fftshift(torch.fft.fft2(indata) ,dim = (-2,-1)).log()
        ft_sum[0] += ft_data.real.sum().item()
        ft_sum[1] += ft_data.imag.sum().item()
        ft_squared_sum[0] += (ft_data.real**2).sum().item()
        ft_squared_sum[1] += (ft_data.imag**2).sum().item()
        ft_n_samples += ft_data.numel()
        sum += (data[-1].type(torch.float64)/255.).sum().item()
        squared_sum += ((data[-1].type(torch.float64)/255.)**2).sum().item()
        n_samples += data[-1].numel()
        


    dic = {}
    ft_mu_r = ft_sum[0]/ft_n_samples
    ft_mu_i = ft_sum[1]/ft_n_samples
    ft_std_r = ((ft_squared_sum[0]/ft_n_samples) - (ft_mu_r **2)) ** 0.5
    ft_std_i = ((ft_squared_sum[1]/ft_n_samples) - (ft_mu_i **2)) ** 0.5
    mu = sum/n_samples
    std = ((squared_sum/n_samples) - (mu **2)) ** 0.5
    dic['data'] = data
    dic['normalisation_constants'] = (mu, std, ft_mu_r, ft_mu_i, ft_std_r, ft_std_i)
    print(mu, std, ft_mu_r, ft_mu_i, ft_std_r, ft_std_i)
    dic['patient_num_videos'] = patient_num_videos
    torch.save(dic, 'processed/processed_data_{}.pth'.format(RES))

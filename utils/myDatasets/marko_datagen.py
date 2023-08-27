import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
import argparse
import scipy.io
import nibabel
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets

import torchkbnufft as tkbn

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = int, default = -1)
args = parser.parse_args()

if args.gpu == -1:
	device = torch.device('cpu')
else:
	device = torch.device('cuda:{}'.format(args.gpu))

data_path = '/root/Cardiac-MRI-Reconstrucion/datasets/ACDC/processed'

data_in = torch.load(os.path.join(data_path, 'processed_data_256.pth'))

EPS = 1e-10
GR = (1 + (5**0.5))/2
GA = np.pi/GR

num_patients = len(data_in['data'])
per_patient_vids_frames_dic = data_in['patient_num_videos']
vids_per_patients = [x.shape[0] for x in data_in['data']]

possible_shots = [10*(x+1) for x in range(20)]
DATASET_LENGTH = len(possible_shots)*1000
remaining_data = DATASET_LENGTH
store_data = [[] for i in possible_shots]

if not os.path.isdir('dataset'):
	os.mkdir('dataset')

def get_nufft_data(image, omega, kbwidth1 = 2.34, grid_size1 = None, numpoints1 = 8, device = torch.device('cpu')):
	B,Chan,R,C = image.shape
	image2 = torch.zeros(B,Chan,3*R,3*C).to(device)
	image2[:,:,R:R*2,C:C*2] = image

	_, n_thetas, n_rads = omega.shape
	omega = omega.reshape(2,-1)

	if grid_size1 is None:
		grid_size1 = (6*R, 6*C)

	kb_ob = tkbn.KbNufft(im_size=(3*R,3*C), grid_size = grid_size1, numpoints = numpoints1, kbwidth = kbwidth1, device = device)
	data = kb_ob(torch.complex(image2,image2*0),omega)
	data = data.reshape(-1,n_thetas, n_rads)
	return data

def nufft_to_grid(data, omega, kbwidth2 = 0.84, grid_size2 = None, numpoints2 = 8, device = torch.device('cpu')):
	B,n_thetas, n_rads = data.shape
	data = data.reshape(B,-1)
	omega = omega.reshape(2,-1)

	if grid_size2 is None:
		grid_size2 = (3*256, 3*256)

	adjkb_ob = tkbn.KbInterpAdjoint(im_size=(256,256), grid_size = grid_size2, numpoints = numpoints2, kbwidth = kbwidth2, device = device)

	dcomp_full = tkbn.calc_density_compensation_function(ktraj=omega, im_size=(256,256), grid_size = grid_size2, numpoints = numpoints2, kbwidth = kbwidth2).to(device)

	recon_ft_full = adjkb_ob(dcomp_full*data, omega).squeeze()
	recon_ft_full = recon_ft_full[::3,::3]

	shifted_reconft_full = torch.fft.ifftshift(recon_ft_full)
	shifted_reconft_full = recon_ft_full
	
	inverse_full = torch.fft.ifftn(recon_ft_full).real
	inverse_full = inverse_full - inverse_full.min()
	inverse_full = inverse_full / (1e-10 + inverse_full.max())

	return inverse_full

def get_omega2d(thetas, rad_max = 128, device = torch.device('cpu')):
	omega = torch.zeros(2,thetas.shape[0],2*rad_max+1).to(device)

	for itheta, theta in enumerate(thetas):
		theta_rad = np.pi*(theta/180)
		for ir, rad in enumerate(range(-rad_max,rad_max+1)):
			x,y = rad*np.cos(theta_rad), rad*np.sin(theta_rad)
			omega[0,itheta,ir] = np.pi*(x/rad_max)
			omega[1,itheta,ir] = np.pi*(y/rad_max)
	return omega

pbar = tqdm(total=remaining_data)
for pat_id in range(num_patients):
	if remaining_data == 0:
		break
	for vid_i in range(vids_per_patients[pat_id]):
		if remaining_data == 0:
			break
		for frame_i in range(data_in['data'][pat_id].shape[1]):
			if remaining_data == 0:
				break
			data = data_in['data'][pat_id][vid_i,frame_i,:,:,:]
			data = data.reshape(-1,1,256,256)
			if data.shape[0]*1 < remaining_data:
				data = data[:remaining_data//len(possible_shots)]
			remaining_data -= data.shape[0]
			pbar.update(DATASET_LENGTH - remaining_data)
			
			angles = GA*np.arange(max(possible_shots))
			omegas = get_omega2d(angles, device = device)
			nufft_data = get_nufft_data(data, omegas, device = device)

			for li, lines_curr in enumerate(possible_shots):
				temp_omegas = omegas[:,:lines_curr,:]
				temp_nufft = nufft_data[:,:lines_curr,:]

				inverses = nufft_to_grid(temp_nufft, temp_omegas, device = device)
				inverses = np.array(inverses.cpu())
				inverses = inverses - inverses.min()
				inverses = inverses / (inverses.max()+EPS)
				inverses = (inverses * 255.).astype(np.uint8)
				store_data[li].append(inverses)
pbar.close()

store_data = [np.stack(x,0) for x in store_data]

torch.save({"data":store_data, "num_shots":possible_shots}, 'dataset/dataset.pth')
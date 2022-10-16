import torch
import torchvision
import os
import PIL
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
import numpy as np
import nibabel as nib
from skimage.exposure import match_histograms

histogram_target = nib.load('raw/patient001/patient001_4d.nii.gz').get_fdata()[:,:,0,7]

def read_nib_preprocess(path, heq = True):
	img = nib.load(path).get_fdata()
	if heq:
		for i in range(img.shape[2]):
			for j in range(img.shape[3]):
				img[:,:,i,j] = match_histograms(img[:,:,i,j], histogram_target, channel_axis=None)
	return img

def num_to_str(num):
	x = str(num)
	while len(x) < 3:
		x = '0' + x
	return './raw/patient{}/patient{}_4d.nii.gz'.format(x,x)

NUM_PATIENTS = 150

data = torch.zeros(NUM_PATIENTS, 10, 30, 1, 256, 256)

transform = torchvision.transforms.Resize((256,256))

squared_sum = 0
sum = 0
n_samples = 0

for i in tqdm(range(1,NUM_PATIENTS+1)):
	path = num_to_str(i)
	d = read_nib_preprocess(path)
	r,c,d1,d2 = d.shape
	d = torch.permute(torch.FloatTensor(d).unsqueeze(4), (2,3,4,0,1)) # d1, d2, 1, r, c
	data[i] = transform(d.reshape(d1*d2, 1, r, c)).reshape(d1, d2, 1, 256, 256)
	sum += data[i].sum()
	squared_sum += (data[i]**2).sum()
	n_samples += d1*d2*256*256

dic = {}
mu = sum/n_samples
std = ((squared_sum/n_samples) - (mu **2)) ** 0.5
dic['data'] = data
dic['normalisation_constants'] = (mu, std)
torch.save('processed/processed_data.pth', dic)
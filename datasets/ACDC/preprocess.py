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
RES=256
data = []

transform = torchvision.transforms.Resize((RES,RES))

# if os.path.isfile('processed/processed_data_{}.pth'.format(RES)):
# 	dic = torch.load('processed/processed_data_{}.pth'.format(RES))
# 	data = dic['data']
# 	(mu, std) = dic['normalisation_constants']
# 	patient_num_videos = dic['patient_num_videos']
# else:
squared_sum = 0
sum = 0
n_samples = 0
patient_num_videos = {}

for i in tqdm(range(1,NUM_PATIENTS+1)):
	path = num_to_str(i)
	d, flag, dvid = read_nib_preprocess(path)
	if flag:
		print("Patient {} flagged, {} video(s) dropped".format(i, dvid))
	r,c,d1,d2 = d.shape
	d = torch.permute(torch.FloatTensor(d).unsqueeze(4), (2,3,4,0,1)) # d1, d2, 1, r, c
	data.append(((transform(d.reshape(d1*d2, 1, r, c)).reshape(d1, d2, 1, RES, RES))*255).type(torch.uint8))
	patient_num_videos[i] = (data[-1].shape[0], data[-1].shape[1])
	sum += data[-1].sum()
	squared_sum += (data[-1]**2).sum()
	n_samples += d1*d2*RES*RES

dic = {}
mu = sum/n_samples
std = ((squared_sum/n_samples) - (mu **2)) ** 0.5
dic['data'] = data
dic['normalisation_constants'] = (mu, std)
dic['patient_num_videos'] = patient_num_videos
torch.save(dic, 'processed/processed_data_{}.pth'.format(RES))

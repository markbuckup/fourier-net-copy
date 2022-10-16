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
import numpy as np
from kornia.utils import draw_line

if os.path.isfile('data/data_tensor.pth'):
	data = torch.load('data/data_tensor.pth')['data']
	num_videos = torch.load('data/data_tensor.pth')['num_videos']
	video_start_ends = torch.load('data/data_tensor.pth')['video_start_ends']
	chain_locations = torch.load('data/data_tensor.pth')['chain_locations']
	window_locations = torch.load('data/data_tensor.pth')['window_locations']
	window_size = torch.load('data/data_tensor.pth')['window_size']
else:
	data = None
	iter = 0
	num_videos = 0
	video_start_ends = {}
	transform = transforms.Compose([transforms.ToTensor()])
	for foldername in sorted(os.listdir('raw_data')):
		total_images = len(os.listdir(os.path.join('raw_data/',foldername)))
		ims = torch.zeros(total_images, 128,128)
		num_videos += 1
		vstart = iter
		for i, image in enumerate(tqdm(sorted(os.listdir(os.path.join('raw_data/',foldername))), desc = 'Video: {}'.format(num_videos))):
			comp_path = os.path.join('raw_data/',foldername)
			comp_path = os.path.join(comp_path,image)
			im = PIL.Image.open(comp_path).convert("L").resize((128,128))
			ims[i] = transform(im).squeeze()
			iter += 1
		vend = iter
		video_start_ends[num_videos-1] = (vstart, vend)
		if data is None:
			data = ims
		else:
			data = torch.cat((data, ims), 0)

	window_size = 7
	chain_locations = []
	window_locations = []
	for vi in range(num_videos):
		vstart, vend = video_start_ends[vi]
		prev_frame = None
		chain_start = vstart
		for fi in range(vstart, vend):
			current_frame = data[fi,:,:]
			if prev_frame is None:
				prev_frame = current_frame
				continue
			else:
				sim = ssim(prev_frame.unsqueeze(0).unsqueeze(0), current_frame.unsqueeze(0).unsqueeze(0), 5).mean().item()
				if sim > 0.95:
					hypothetic_start = fi - window_size + 1
					if hypothetic_start >= chain_start:
						window_locations.append(hypothetic_start)
				else:
					if fi - chain_start - 1 >= window_size:
						chain_locations.append(chain_start)
					chain_start = fi
				prev_frame = current_frame

	torch.save({
		'data':data, 
		'num_videos':num_videos, 
		'video_start_ends':video_start_ends, 
		'chain_locations': chain_locations,
		'window_locations': window_locations,
		'window_size': window_size
		}, 
			'data/data_tensor.pth')

def mask_theta(theta, size):
	r = size[-2]
	c = size[-1]
	minval = float('inf')
	for tx in range(-c+1,c):
		costheta = (tx) / ((r-1-0)**2 + (0-tx)**2)**0.5
		if np.abs(costheta - np.cos((theta*np.pi)/180)) < minval:
			startx = 0
			starty = r-1
			endx = tx
			endy = 0
			minval = np.abs(costheta - np.cos((theta*np.pi)/180))
	if endx < 0:
		endx += c-1
		startx += c-1
	for ty in range(2*r-1):
		if ty < r:
			costheta = (c-1) / ((r-1-ty)**2 + (0-c-1)**2)**0.5
		else:
			costheta = -(c-1) / ((r-1-ty)**2 + (0-c-1)**2)**0.5
		if np.abs(costheta - np.cos((theta*np.pi)/180)) < minval:
			startx = 0
			starty = r-1
			endx = c-1
			endy = ty
			minval = np.abs(costheta - np.cos((theta*np.pi)/180))
	if endy >= r:
		endy -= r-1
		starty -= r-1
	
	distfromtop = min(starty, endy)
	distfrombottom = min(r-1-starty, r-1-endy)
	distfromleft = min(startx, endx)
	distfromright = min(c-1-startx, c-1-endx)
	moveup = distfromtop // 2
	moveright = distfromright // 2
	moveup -= distfrombottom // 2
	moveright -= distfromleft // 2

	starty -= moveup
	endy -= moveup
	startx += moveright
	endx += moveright
	return draw_line(torch.zeros(size), torch.tensor([startx, starty]), torch.tensor([endx, endy]), torch.tensor([1.]))

NUMBER_OF_RADIAL_VIEWS = 14
window_thetas = []
window_mask = torch.zeros(window_size, 128, 128)
for i in range(14):
	window_thetas.append(i*(180/NUMBER_OF_RADIAL_VIEWS))
theta_translations = []
for i in range(window_size):
	theta_translations.append((i*180)/(window_size*NUMBER_OF_RADIAL_VIEWS))

for wi in range(window_size):
	for thetai in window_thetas:
		m = mask_theta(thetai + theta_translations[wi] , (1,128,128))
		window_mask[wi,:,:] += m.squeeze()
window_mask = torch.sign(window_mask)
ft_data = torch.fft.fftshift(torch.fft.fft2(data), dim = (-1,-2))
ft_data = torch.stack((ft_data.real, ft_data.imag), -1)

torch.save({
		'data':data,
		'ft_data':ft_data,
		'window_mask': window_mask,
		'num_videos':num_videos, 
		'video_start_ends':video_start_ends, 
		'chain_locations': chain_locations,
		'window_locations': window_locations,
		'window_size': window_size,
		}, 
			'data/data_tensor.pth')
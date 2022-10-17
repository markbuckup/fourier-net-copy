import torch
import torchvision
import os
import PIL
import numpy as np
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line

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

def get_window_mask(window_size = 7, number_of_radial_views = 14, resolution = 128):
    window_thetas = []
    window_mask = torch.zeros(window_size, resolution, resolution)
    for i in range(number_of_radial_views):
        window_thetas.append(i*(180/number_of_radial_views))
    theta_translations = []
    for i in range(window_size):
        theta_translations.append((i*180)/(window_size*number_of_radial_views))

    for wi in range(window_size):
        for thetai in window_thetas:
            m = mask_theta(thetai + theta_translations[wi] , (1,resolution,resolution))
            window_mask[wi,:,:] += m.squeeze()
    window_mask = torch.sign(window_mask)
    return window_mask

def get_coil_mask(n_coils = 8, resolution = 128):
    temp_mask = get_window_mask(window_size = 1, number_of_radial_views = n_coils, resolution = resolution)[0,:,:]
    centres = []
    ri = 0
    for ci in range(resolution):
        if temp_mask[ri,ci] == 1 and (ri,ci) not in centres:
            centres.append((ri,ci))
    ci = resolution-1
    for ri in range(resolution):
        if temp_mask[ri,ci] == 1 and (ri,ci) not in centres:
            centres.append((ri,ci))
    ri = resolution-1
    for ci in reversed(range(resolution)):
        if temp_mask[ri,ci] == 1 and (ri,ci) not in centres:
            centres.append((ri,ci))
    ci = 0
    for ri in reversed(range(resolution)):
        if temp_mask[ri,ci] == 1 and (ri,ci) not in centres:
            centres.append((ri,ci))

    centres = centres[::2]

    assert(len(centres) == n_coils)

    ans = torch.zeros(n_coils, resolution, resolution)
    ar,ac = resolution//2, resolution//2
    halfline = resolution//2
    for i,(rc, cc) in enumerate(centres):
        scale = (((rc-ar)**2 + (cc - ac)**2)**0.5 / halfline)**0.5
        sigma = (resolution/3) * scale
        m1,m2 = torch.meshgrid(torch.arange(resolution),torch.arange(resolution), indexing='ij')
        dists = ((m1-rc)**2 + (m2-cc)**2)**0.5
        ans[i,:,:] = np.exp(-((dists**2)/(2*(sigma**2))))
    return ans
    
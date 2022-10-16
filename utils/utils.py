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

def get_window_mask(window_size = 7, number_of_radial_views = 14)
    window_thetas = []
    window_mask = torch.zeros(window_size, 128, 128)
    for i in range(14):
        window_thetas.append(i*(180/number_of_radial_views))
    theta_translations = []
    for i in range(window_size):
        theta_translations.append((i*180)/(window_size*number_of_radial_views))

    for wi in range(window_size):
        for thetai in window_thetas:
            m = mask_theta(thetai + theta_translations[wi] , (1,128,128))
            window_mask[wi,:,:] += m.squeeze()
    window_mask = torch.sign(window_mask)
    return window_mask
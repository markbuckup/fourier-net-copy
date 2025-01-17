import torch
import torchvision
import os
import PIL
import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
import numpy as np
from tqdm import tqdm
import kornia
from kornia.metrics import ssim
from torchvision import transforms
import matplotlib.pyplot as plt
import kornia
from kornia.utils import draw_line
from torch import nn, optim
import utils
from utils.watsonLoss.watson_fft import WatsonDistanceFft
from utils.watsonLoss.shift_wrapper import ShiftWrapper
from utils.watsonLoss.color_wrapper import ColorWrapper
from utils.models.complexCNNs.polar_transforms import (
    convert_cylindrical_to_polar,
    convert_polar_to_cylindrical,
)

EPS = 1e-10
GR = (1 + (5**0.5))/2
GA = np.pi/GR

def get_golden_bars(num_bars = 376, resolution = 128):
    ans = mask_theta(90, (1,resolution, resolution))
    angle = 90
    for i in range(1,num_bars):
        angle = (angle + (360*(GA))/(2*np.pi)) 
        ans = torch.cat((ans,mask_theta(angle, (1,resolution, resolution))))
    return ans


def fetch_loss_function(loss_str, device, loss_params):
    def load_state_dict(filename):
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, 'watsonLoss/weights', filename)
        return torch.load(path, map_location='cpu')
    if loss_str == 'None':
        return None
    elif loss_str == 'Cosine':
        return lambda x,y: torch.acos(torch.cos(x - y)*(1 - EPS*10**3)).mean()
    elif loss_str == 'L1':
        return nn.L1Loss().to(device)
    elif loss_str == 'L2':
        return nn.MSELoss().to(device)
    elif loss_str == 'Cosine-Watson':
        reduction = 'mean'
        if loss_params['grayscale']:
            if loss_params['deterministic']:
                loss = WatsonDistanceFft(reduction=reduction).to(device)
                if loss_params['watson_pretrained']: 
                    loss.load_state_dict(load_state_dict('gray_watson_fft_trial0.pth'))
            else:
                loss = ShiftWrapper(WatsonDistanceFft, (), {'reduction': reduction}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.loss.load_state_dict(load_state_dict('gray_watson_fft_trial0.pth'))
        else:
            if loss_params['deterministic']:
                loss = ColorWrapper(WatsonDistanceFft, (), {'reduction': reduction}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.load_state_dict(load_state_dict('rgb_watson_fft_trial0.pth'))
            else:
                loss = ShiftWrapper(ColorWrapper, (WatsonDistanceFft, (), {'reduction': reduction}), {}).to(device)
                if loss_params['watson_pretrained']: 
                    loss.loss.load_state_dict(load_state_dict('rgb_watson_fft_trial0.pth'))
        if loss_params['watson_pretrained']: 
            for param in loss.parameters():
                param.requires_grad = False
        return loss
    elif loss_str == 'SSIM':
        return kornia.losses.SSIMLoss(loss_params['SSIM_window']).to(device)
    elif loss_str == 'MS_SSIM':
        return kornia.losses.MS_SSIMLoss().to(device)
    elif loss_str == 'Cosine-L1':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L1', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-L2':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('L2', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )
    elif loss_str == 'Cosine-MS_SSIM':
        # expects a shifted fourier transform
        return dual_fft_loss(
                    fetch_loss_function('Cosine', device, loss_params), 
                    fetch_loss_function('MS_SSIM', device, loss_params), 
                    alpha_phase = loss_params['alpha_phase'], 
                    alpha_amp = loss_params['alpha_amp']
                        )

def dual_fft_loss(phase_loss, amp_loss, alpha_phase = 1, alpha_amp = 1):
    def apply_func(x1,x2):
        real1, imag1 = torch.unbind(x1, -1)
        real2, imag2 = torch.unbind(x2, -1)
        amp1,phase1 = convert_cylindrical_to_polar(real1, imag1)
        amp2,phase2 = convert_cylindrical_to_polar(real2, imag2)
        return phase_loss(phase1, phase2)*alpha_phase + amp_loss(amp1, amp2)*alpha_amp
    return apply_func

def mask_theta(theta, size):
    r = size[-2]
    c = size[-1]
    theta = theta%180
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

def get_window_mask(theta_init = 0, window_size = 7, number_of_radial_views = 14, resolution = 128):
    window_thetas = []
    window_mask = torch.zeros(window_size, resolution, resolution)
    for i in range(number_of_radial_views):
        window_thetas.append(i*(180/number_of_radial_views))
    theta_translations = []
    for i in range(window_size):
        theta_translations.append((i*180)/(window_size*number_of_radial_views))

    for wi in range(window_size):
        for thetai in window_thetas:
            m = mask_theta(theta_init + thetai + theta_translations[wi] , (1,resolution,resolution))
            window_mask[wi,:,:] += m.squeeze()
    window_mask = torch.sign(window_mask)
    return window_mask

def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return torch.FloatTensor(g/g.sum())

def get_coil_mask(theta_init = 0, n_coils = 8, resolution = 128):
    # assert(len(centres) == n_coils)
    centrex, centrey = 2*resolution, 2*resolution
    sigma = resolution/(2.5)
    radius = resolution//2
    
    filter = fspecial_gauss(2*resolution, sigma)
    ans = torch.zeros(n_coils, resolution, resolution)

    for i in range(n_coils):
        theta = theta_init + ((i/n_coils)*((2*np.pi)))
        x_delta = int(radius*np.cos(theta))
        y_delta = int(radius*np.sin(theta))
        centre_g_x = centrex + x_delta
        centre_g_y = centrey + y_delta
        big_ans = torch.zeros((4*resolution,4*resolution))
        big_ans[centre_g_x-resolution:centre_g_x+resolution, centre_g_y-resolution:centre_g_y+resolution] = filter
        ans[i,:,:] = big_ans[centrex-radius:centrex+radius, centrey-radius:centrey+radius]
        ans[i,:,:] = ans[i,:,:] - (ans[i,:,:].min())
        ans[i,:,:] = ans[i,:,:] / (ans[i,:,:].max()+1e-8)


    return ans    

def save_coils(theta_init, n_coils = 4):
    coils = get_coil_mask(theta_init = theta_init, n_coils = n_coils)
    for ci in range(coils.shape[0]):
        plt.imsave('Coil_{}.png'.format(ci), coils[ci,:,:], cmap = 'gray')

# x1 = torch.randn(10,3,128,128)
# fx1 = torch.randn(10,1,128,128,2)
# mydic = {
#      'SSIM_window': 11,
#      'alpha_phase': 1,
#      'alpha_amp': 1,
#      'grayscale': False,
#      'deterministic': False,
#      'watson_pretrained': True,
# } 
# print(fetch_loss_function('None', torch.device('cpu'), mydic))
# print(fetch_loss_function('Cosine', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('L1', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('L2', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('SSIM', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('MS_SSIM', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('Cosine-Watson', torch.device('cpu'), mydic)(x1,x1))
# print(fetch_loss_function('Cosine-L1', torch.device('cpu'), mydic)(fx1,fx1))
# print(fetch_loss_function('Cosine-L2', torch.device('cpu'), mydic)(fx1,fx1))
# print(fetch_loss_function('Cosine-SSIM', torch.device('cpu'), mydic)(fx1,fx1))
# print(fetch_loss_function('Cosine-MS_SSIM', torch.device('cpu'), mydic)(fx1,fx1))

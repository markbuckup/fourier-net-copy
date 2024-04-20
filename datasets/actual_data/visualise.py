import os
import nibabel as nib
import gc
import sys
import PIL
import time
import scipy
import torch
import random
import pickle
import kornia
import argparse
import numpy as np
import torchvision
import torchkbnufft as tkbn
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from pytorch_histogram_matching import Histogram_Matching
import h5py

from sklearn.decomposition import PCA

def complex_PCA(data, n_components, inverse=False):
    """
    Perform PCA on complex-valued data without splitting into real and imaginary parts,
    and optionally transform back adding the mean.

    Parameters:
    data (torch.Tensor): A complex tensor of shape (N, D) where N is the number of samples
                         and D is the number of features.
    n_components (int): The number of principal components to return.
    inverse (bool): If True, perform the inverse transformation to reconstruct the data.

    Returns:
    torch.Tensor: The transformed data or reconstructed data depending on the 'inverse' flag.
    """

    # Center the data by subtracting the mean
    mean = data.mean(0)
    data_centered = data - mean

    # Compute the covariance matrix using the conjugate transpose
    covariance_matrix = torch.mm(data_centered.t().conj(), data_centered) / (data_centered.shape[0] - 1)

    # Perform eigen decomposition on the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    principal_components = eigenvectors[:, sorted_indices[:n_components]]

    # Project the centered data onto the principal components
    transformed_data = torch.mm(data, principal_components)

    if inverse:
        # Reconstruct the data from the principal components
        reconstructed_data = torch.mm(transformed_data, principal_components.conj().t()) + mean
        return reconstructed_data
    else:
        return transformed_data


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type = int, default = -1)
args = parser.parse_args()

if args.gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.gpu))

file_name = 'meas_MID00045_FID10428_CV_6kSpokes_SingleSlice.mat'


with h5py.File(file_name, 'r') as f:
    data = f['image_data'][:]

# real_data = torch.complex(torch.from_numpy(data['real']), torch.from_numpy(data['imag']))
real_data = data['real']
imag_data = data['imag']


NUM_FRAMES_LIMIT = 100

reduced_real_data = np.transpose(real_data, (0, 2, 1)).reshape(-1, real_data.shape[1])
reduced_imag_data = np.transpose(imag_data, (0, 2, 1)).reshape(-1, imag_data.shape[1])

NUM_COILS = 8

complex_data = torch.complex(torch.from_numpy(reduced_real_data), torch.from_numpy(reduced_imag_data))
complex_data = complex_PCA(complex_data, n_components = NUM_COILS).reshape(real_data.shape[0], real_data.shape[2], NUM_COILS)
complex_data = complex_data.permute((2, 0, 1))

# PCA_real = PCA(n_components=NUM_COILS)
# PCA_imag = PCA(n_components=NUM_COILS)

# means_real = reduced_real_data.mean(0).reshape(1,15)
# means_imag = reduced_imag_data.mean(0).reshape(1,15)

# # stds_real = reduced_real_data.std(0).reshape(1,15)
# # stds_imag = reduced_imag_data.std(0).reshape(1,15)

# reduced_real_data = reduced_real_data - means_real
# reduced_imag_data = reduced_imag_data - means_imag

# # reduced_real_data = reduced_real_data / stds_real
# # reduced_imag_data = reduced_imag_data / stds_imag

# PCA_real.fit(reduced_real_data)
# PCA_imag.fit(reduced_imag_data)

# reduced_real_data = PCA_real.transform(reduced_real_data).reshape(real_data.shape[0], real_data.shape[2], NUM_COILS)
# reduced_imag_data = PCA_imag.transform(reduced_imag_data).reshape(imag_data.shape[0], imag_data.shape[2], NUM_COILS)

# transformed_means_real = PCA_real.transform(means_real).reshape(1,1,NUM_COILS)
# transformed_means_imag = PCA_imag.transform(means_imag).reshape(1,1,NUM_COILS)

# reduced_real_data = reduced_real_data + transformed_means_real
# reduced_imag_data = reduced_imag_data + transformed_means_imag

# breakpoint()

# reduced_real_data = reduced_real_data.reshape(real_data.shape[0], real_data.shape[2], NUM_COILS)
# reduced_imag_data = reduced_imag_data.reshape(imag_data.shape[0], imag_data.shape[2], NUM_COILS)

# reduced_real_data = np.transpose(reduced_real_data, (2, 0, 1))
# reduced_imag_data = np.transpose(reduced_imag_data, (2, 0, 1))

# complex_data = torch.complex(torch.from_numpy(reduced_real_data), torch.from_numpy(reduced_imag_data))


GR = (1 + (5**0.5))/2
GA = np.pi/GR


def get_omega2d(thetas, diameter = 128):
    
    omega1 = torch.zeros(2,thetas.shape[0],diameter).to(device)

    for itheta, theta_rad in enumerate(thetas):

        shift = 0
        rads = np.arange(-(diameter//2)+shift,-(diameter//2)+ diameter +shift) # ex) - 256 to 256, this is the grid in the k-space i think ?
        # rads = np.arange(diameter) - ((diameter-1)/2)
        # breakpoint()
        # print(rads)
        xs = rads*np.cos(theta_rad) # ? - ex) xs = [-256,-255,...255,256]
        # and multiple with the grid size (=rads)
        ys = rads*np.sin(theta_rad) # ? - ex) ys = [0,0,...,0,0]
        
        omega1[0,itheta,:] = torch.from_numpy(np.pi*(xs/np.abs(rads).max()))
        omega1[1,itheta,:] = torch.from_numpy(np.pi*(ys/np.abs(rads).max()))
        # omega1 = torch.Size([2, 65, 513]) for the 65 spoke, first nufft
    return omega1

def gridder(kspace_data, angle_start = 0):
    # kspace_data = N_spokes, N_points_per_spoke, N_coils

    N_coils, N_spokes, N_points_per_spoke = kspace_data.shape
    omegas = get_omega2d(np.arange(angle_start,angle_start+N_spokes)*GA, diameter = N_points_per_spoke).to(device)

    # plt.figure()
    # plt.scatter(omegas.reshape(2,-1)[0,:],omegas.reshape(2,-1)[1,:])
    # plt.savefig('omegas.jpg')

    dcomp_full = tkbn.calc_density_compensation_function(ktraj=omegas.reshape(2,-1), im_size=(128,128), grid_size = (256,256), numpoints = 6, kbwidth = 2.34).to(device)
    kbinterp2 = tkbn.KbInterpAdjoint(im_size=(128,128), grid_size = (256,256), numpoints = 6, kbwidth = 2.34, device = device)

    interp_result = torch.fft.fftshift(kbinterp2(dcomp_full*kspace_data.to(device).reshape(kspace_data.shape[0],1,-1), omegas.reshape(2,-1))[:,0], dim = (-2,-1))
    interp_result = interp_result[:,::2,::2]
    myfft_interp = torch.zeros(N_coils, 256,256, device = interp_result.device)
    myfft_interp = torch.complex(myfft_interp,myfft_interp)
    # DC component should match - 
    myfft_interp[:,64:64+128,64:64+128] = interp_result
    
    zero_mask = myfft_interp == 0

    inverse_mag = torch.fft.ifft2(torch.fft.ifftshift(myfft_interp, dim = (-2,-1)))
    inverse_phase = inverse_mag / (1e-10 + inverse_mag.abs())
    inverse_mag = inverse_mag.abs()

    inverse_mag = inverse_mag / (inverse_mag.max(-1)[0].max(-1)[0]).unsqueeze(-1).unsqueeze(-1)
    inverse = inverse_phase * inverse_mag
    myfft_interp = torch.fft.fftshift(torch.fft.fft2(inverse))
    myfft_interp[zero_mask] *= 0

    # for ci in range(N_coils):
    #     plt.imsave('myfft{}.jpg'.format(ci), (1e-10+ myfft_interp[0].abs().cpu()).log(), cmap = 'gray')
    #     print(inverse.shape)
    #     plt.imsave('inverse{}.jpg'.format(ci), inverse[ci].abs().cpu(), cmap = 'gray')
    return myfft_interp


# complex_data = complex_data[[2,3,4,8,9,10,11,13],:,:]
N_coils, N_spokes, N_points_per_spoke = complex_data.shape
SPOKES_PER_FRAME = 10
final_undersampled = torch.zeros((min(N_spokes//SPOKES_PER_FRAME, NUM_FRAMES_LIMIT), N_coils, 256, 256))
final_undersampled = torch.complex(final_undersampled,final_undersampled)
# vals = torch.zeros(NUM_COILS,complex_data.shape[2],complex_data.shape[2])
for i in tqdm(range(final_undersampled.shape[0])):
    final_undersampled[i,:,:,:] = gridder(complex_data[:,i*SPOKES_PER_FRAME:(i+1)*SPOKES_PER_FRAME,:], angle_start = i*SPOKES_PER_FRAME)
    # break

# a = torch.fft.ifft2(final_undersampled[0,0,:,:]).abs()
# plt.imsave('sanity.jpg', a)

torch.save({'undersampled_data':final_undersampled}, 'data1.pth')

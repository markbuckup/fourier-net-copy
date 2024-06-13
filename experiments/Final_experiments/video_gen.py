import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.io import write_video
def images2video(images, out_path, fps = 30, video_codec = 'libx264', repeat = 1):
    write_video(out_path, images.repeat(repeat, 1, 1, 1), fps= fps, video_codec = video_codec)

sys.path.append('../../')
os.environ['display'] = 'localhost:14.0'

parser = argparse.ArgumentParser()
parser.add_argument('--fouriernet_id', type = str, required = True)
parser.add_argument('--mdcnn_id', type = str, required = True)
parser.add_argument('--train', action = 'store_true')
args = parser.parse_args()

sys.path.append('./{}/'.format(args.fouriernet_id))

from params import parameters
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
 


temp = os.getcwd().split('/')
temp = temp[temp.index('experiments'):]
raw_path = os.path.join(parameters['save_folder'], '/'.join(temp))
mdcnn_raw_path = os.path.join(raw_path, args.mdcnn_id)
fouriernet_raw_path = os.path.join(raw_path, args.fouriernet_id)
video_path = os.path.join(parameters['save_folder'], '/'.join(temp))
video_path = os.path.join(video_path, args.fouriernet_id)
if args.train:
    mdcnn_raw_path = os.path.join(mdcnn_raw_path, 'images/raw/train/patient_120/by_location_number/location_0')
    fouriernet_raw_path = os.path.join(fouriernet_raw_path, 'images/raw/train/patient_120/by_location_number/location_0')
    video_path = os.path.join(video_path, 'images/videos/train')
else:
    mdcnn_raw_path = os.path.join(mdcnn_raw_path, 'images/raw/test/patient_120/by_location_number/location_0')
    fouriernet_raw_path = os.path.join(fouriernet_raw_path, 'images/raw/test/patient_120/by_location_number/location_0')
    video_path = os.path.join(video_path, 'images/videos/test')

os.makedirs(video_path, exist_ok=True)

kspace_preds = []
ispace_preds = []

def read_gray(path):
    a = plt.imread(path)
    if len(a.shape) == 2:
        return a
    else:
        return a
        return a.mean(2).reshape(1, *a.shape[:2])

for fnum in tqdm(range(120)):
    curr_path = os.path.join(fouriernet_raw_path, 'frame_{}'.format(fnum))
    mdcnn_path = os.path.join(mdcnn_raw_path, 'frame_{}'.format(fnum))
    

    coilwise_gt = np.flip(read_gray(os.path.join(curr_path, 'coiled_gt_coil_0.jpg'))/255., 0)
    undersampled_ft = read_gray(os.path.join(curr_path, 'undersampled_ft_coil_0.jpg'))/255.
    ft_pred = read_gray(os.path.join(curr_path, 'pred_kspace_ft_coil_0.jpg'))/255.
    pred_kspace_rnn_ifft = np.flip(read_gray(os.path.join(curr_path, 'pred_kspace_coil_0.jpg'))/255., 0)
    pred_ilstm_ifft = np.flip(read_gray(os.path.join(curr_path, 'pred_image_lstm_coil_0.jpg'))/255., 0)
    ispace_pred = np.flip(read_gray(os.path.join(curr_path, 'ispace_pred.jpg'))/255.,0)
    mdcnn_pred = np.flip(read_gray(os.path.join(mdcnn_path, 'ispace_pred.jpg'))/255.,0)
    ispace_gt = np.flip(read_gray(os.path.join(curr_path, 'ground_truth.jpg'))/255.,0)

    # Assuming you have the variables coilwise_gt, undersampled_ft, ft_pred, pred_kspace_rnn_ifft, pred_ilstm_ifft containing the images

    # Create a figure and a GridSpec layout
    fig = plt.figure(figsize=(15, 16))
    gs = gridspec.GridSpec(3, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1, 0.1])

    # Plotting the images in the specified layout
    # First row
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.imshow(coilwise_gt, cmap='gray')
    ax0.set_title("Ground Truth", fontsize=20)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.imshow(undersampled_ft, cmap='gray')
    ax1.set_title("Undersampled K-space", fontsize=20)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 4:6])
    ax2.imshow(ft_pred, cmap='gray')
    ax2.set_title("Recurrent Subnetwork Output", fontsize=20)
    ax2.axis('off')

    # Second row
    ax3 = fig.add_subplot(gs[1, 0:3])
    ax3.imshow(pred_kspace_rnn_ifft, cmap='gray')
    ax3.set_title("K-Space RNN output", fontsize=20)
    ax3.axis('off')

    diff_pred_kspace_rnn = np.abs(pred_kspace_rnn_ifft - coilwise_gt).mean(2)
    ax4 = fig.add_subplot(gs[1, 3:6])
    im = ax4.imshow(diff_pred_kspace_rnn, cmap = 'plasma')
    ax4.set_title("Difference Frame", fontsize=20)
    ax4.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[1, 6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
    im.set_clim(vmin=0, vmax=1)

    # Third row
    ax5 = fig.add_subplot(gs[2, 0:3])
    ax5.imshow(pred_ilstm_ifft, cmap='gray')
    ax5.set_title("Image LSTM output", fontsize=20)
    ax5.axis('off')

    diff_pred_ilstm = np.abs(pred_ilstm_ifft - coilwise_gt).mean(2)
    ax6 = fig.add_subplot(gs[2, 3:6])
    im = ax6.imshow(diff_pred_ilstm, cmap = 'plasma')
    ax6.set_title("Difference Frame", fontsize=20)
    ax6.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[2, 6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks
    im.set_clim(vmin=0, vmax=1)

    # Global super title
    fig.suptitle("Frame {}\n".format(fnum), fontsize=32)

    plt.tight_layout()
    plt.savefig('kspace_temp.jpg')
    plt.close('all')
    kspace_preds.append(plt.imread('kspace_temp.jpg'))

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

    ax0 = fig.add_subplot(gs[0:2, 0])
    ax0.imshow(ispace_gt, cmap='gray')
    ax0.set_title("Ground Truth", fontsize=18)
    ax0.axis('off')

    ax0 = fig.add_subplot(gs[0, 1])
    ax0.imshow(ispace_pred, cmap='gray')
    ax0.set_title("FOURIER-Net Output", fontsize=18)
    ax0.axis('off')

    diff_pred_ispace = np.abs(ispace_pred - ispace_gt).mean(2)
    ax0 = fig.add_subplot(gs[0, 2])
    im = ax0.imshow(diff_pred_ispace, cmap = 'plasma')
    ax0.set_title("Difference Frame", fontsize=18)
    ax0.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_yticklabels(np.arange(11)/10)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks

    im.set_clim(vmin=0, vmax=1)


    ax0 = fig.add_subplot(gs[1, 1])
    ax0.imshow(mdcnn_pred, cmap='gray')
    ax0.set_title("MD-CNN Output", fontsize=18)
    ax0.axis('off')

    diff_pred_mdcnn = np.abs(mdcnn_pred - ispace_gt).mean(2)
    ax0 = fig.add_subplot(gs[1, 2])
    im = ax0.imshow(diff_pred_mdcnn, cmap = 'plasma')
    ax0.set_title("Difference Frame", fontsize=18)
    ax0.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[1, 3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_yticklabels(np.arange(11)/10)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks

    im.set_clim(vmin=0, vmax=1)

    # Global super title
    fig.suptitle("Frame {}\n".format(fnum), fontsize=24)

    plt.tight_layout()
    plt.savefig('ispace_temp.jpg')
    plt.close('all')
    ispace_preds.append(plt.imread('ispace_temp.jpg'))

    os.system('rm ispace_temp.jpg')
    os.system('rm kspace_temp.jpg')
    

#### save lag plot
fig = plt.figure(figsize=(20, 19))
gs = gridspec.GridSpec(7, 10, figure=fig, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05], height_ratios=[1,0.05,1,1,0.05,1,1])
# gs.update(hspace=0.4)  # Add spacing between the rows


for ii,fnum in enumerate(range(112,117)):
    curr_path = os.path.join(fouriernet_raw_path, 'frame_{}'.format(fnum))

    ispace_pred = np.flip(read_gray(os.path.join(curr_path, 'ispace_pred.jpg'))/255.,0)
    
    ispace_gt = np.flip(read_gray(os.path.join(curr_path, 'ground_truth.jpg'))/255.,0)

    ax0 = fig.add_subplot(gs[0, 2*ii])
    ax0.imshow(ispace_gt, cmap='gray')
    if ii == 2:
        ax0.set_title("Ground Truths\nFrame {}".format(100+ii), fontsize=24, pad=10)
    else:
        ax0.set_title("Frame {}".format(100+ii), fontsize=24, pad=10)
    ax0.axis('off')

    ax0 = fig.add_subplot(gs[2, 2*ii])
    ax0.imshow(ispace_pred, cmap='gray')
    if ii == 2:
        ax0.set_title("FOURIER-Net Predictions \nFrame {}".format(100+ii), fontsize=24, pad=10)
    else:
        ax0.set_title("Frame {}".format(100+ii), fontsize=24, pad=10)
    ax0.axis('off')

    diff_pred_ispace = np.abs(ispace_pred - ispace_gt).mean(2)
    ax0 = fig.add_subplot(gs[3, 2*ii])
    im = ax0.imshow(diff_pred_ispace, cmap = 'plasma')
    # if ii == 2:
    #     ax0.set_title("Difference Frames", fontsize=24, pad=10)
    ax0.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[3, 2*ii + 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_yticklabels(np.arange(11)/10)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks

    im.set_clim(vmin=0, vmax=1)


for ii,fnum in enumerate(range(112,117)):
    curr_path = os.path.join(fouriernet_raw_path, 'frame_{}'.format(fnum))
    lag_path = os.path.join(fouriernet_raw_path, 'frame_{}'.format(fnum+3))

    ispace_pred = np.flip(read_gray(os.path.join(lag_path, 'ispace_pred.jpg'))/255.,0)
    
    ispace_gt = np.flip(read_gray(os.path.join(curr_path, 'ground_truth.jpg'))/255.,0)

    ax0 = fig.add_subplot(gs[5, 2*ii])
    ax0.imshow(ispace_pred, cmap='gray')
    
    if ii == 2:
        ax0.set_title("FOURIER-Net Predictions Shifted by 3 Frames\nFrame {}".format(100+ii), fontsize=24, pad=10)
    else:
        ax0.set_title("Frame {}".format(100+ii+3), fontsize=24, pad=10)
    ax0.axis('off')

    diff_pred_ispace = np.abs(ispace_pred - ispace_gt).mean(2)
    ax0 = fig.add_subplot(gs[6, 2*ii])
    im = ax0.imshow(diff_pred_ispace, cmap = 'plasma')
    # if ii == 2:
    #     ax0.set_title("Difference Frames of Shifted Outputs", fontsize=24)
    ax0.axis('off')

    # Color bar for Difference Frame
    cbar_ax = fig.add_subplot(gs[6, 2*ii + 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_yticklabels(np.arange(11)/10)
    cbar.ax.tick_params(labelsize=14)  # Adjust the font size of the colorbar ticks

    im.set_clim(vmin=0, vmax=1)


# Global super title
# fig.suptitle("Lag Analysis", fontsize=24)
plt.tight_layout()
plt.savefig(os.path.join(video_path,'lag_analysis.jpg'))
plt.close('all')


kspace_preds = (torch.FloatTensor(np.array(kspace_preds)))
ispace_preds = (torch.FloatTensor(np.array(ispace_preds)))

images2video(kspace_preds, os.path.join(video_path, 'kspace_preds.mp4'), fps = 8, repeat = 1)
images2video(ispace_preds, os.path.join(video_path, 'ispace_preds.mp4'), fps = 8, repeat = 1)
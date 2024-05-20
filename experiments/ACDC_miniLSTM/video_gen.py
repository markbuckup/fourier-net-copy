import os
import gc
import sys
import PIL
import time
import cv2
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import neptune as neptune
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from neptune.types import File
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.io import write_video
def images2video(images, out_path, fps = 30, video_codec = 'libx264', repeat = 1):
    write_video(out_path, images.repeat(repeat, 1, 1, 1), fps= fps, video_codec = video_codec)

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'
from utils.DDP_paradigms_LSTM_nufft import train_paradigm, test_paradigm

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type = str, required = True)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
parser.add_argument('--train', action = 'store_true')
args = parser.parse_args()

sys.path.append('/root/Cardiac-MRI-Reconstrucion/experiments/ACDC_miniLSTM/{}/'.format(args.run_id))

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
 

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")

temp = os.getcwd().split('/')
temp = temp[temp.index('experiments'):]
raw_path = os.path.join(parameters['save_folder'], '/'.join(temp))
raw_path = os.path.join(raw_path, args.run_id)
video_path = os.path.join(parameters['save_folder'], '/'.join(temp))
video_path = os.path.join(video_path, args.run_id)
if args.train:
    raw_path = os.path.join(raw_path, 'images/raw/train/patient_120/by_location_number/location_0')
    video_path = os.path.join(video_path, 'images/videos/train')
else:
    raw_path = os.path.join(raw_path, 'images/raw/test/patient_120/by_location_number/location_0')
    video_path = os.path.join(video_path, 'images/videos/test')

os.makedirs(video_path, exist_ok=True)

ispace_preds = []

def read_gray(path):
    a = plt.imread(path)
    if len(a.shape) == 2:
        return a
    else:
        return a
        return a.mean(2).reshape(1, *a.shape[:2])

for fnum in range(120):
    curr_path = os.path.join(raw_path, 'frame_{}'.format(fnum))
    ispace_preds.append(read_gray(os.path.join(curr_path, 'ispace_pred.jpg')))


ispace_preds = (torch.FloatTensor(np.array(ispace_preds)))

images2video(ispace_preds, os.path.join(video_path, 'ispace_preds.mp4'), fps = 8, repeat = 1)
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

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'
from utils.DDP_paradigms_MDCNN import train_paradigm, test_paradigm

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--time_analysis', action = 'store_true')
parser.add_argument('--resume_kspace', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--test_only', action = 'store_true')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--numbers_only', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--run_id', type = str, required = True)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
parser.add_argument('--neptune_log', action = 'store_true')
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()

sys.path.append('/root/Cardiac-MRI-Reconstrucion/experiments/ACDC_MDCNN/{}/'.format(args.run_id))

from params import parameters
if parameters['dataset'] == 'acdc':
    from utils.myDatasets.ACDC_radial import ACDC_radial as dataset
    args.dataset_path = '../../datasets/ACDC'

# sys.path.append('/root/Cardiac-MRI-Reconstrucion/experiments/ACDC/{}/'.format(args.run_id))

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)
 
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['scheduler']) in ['StepLR', 'None', 'CyclicLR']
assert(parameters['scheduler_params']['mode']) in ['triangular', 'triangular2', 'exp_range']

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

checkpoint_path = os.path.join(args.run_id, './checkpoints/')
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir(os.path.join(args.run_id, './images/')):
    os.mkdir(os.path.join(args.run_id, './images/'))
if not os.path.isdir(os.path.join(args.run_id, './images/train')):
    os.mkdir(os.path.join(args.run_id, './images/train'))
if not os.path.isdir(os.path.join(args.run_id, './images/test')):
    os.mkdir(os.path.join(args.run_id, './images/test'))
if not os.path.isdir(os.path.join(args.run_id, './images/input')):
    os.mkdir(os.path.join(args.run_id, './images/input'))

if __name__ == '__main__':
    world_size = len(args.gpu) 
    if args.eval:
        mp.spawn(
            test_paradigm,
            args=[world_size, args, parameters],
            nprocs=world_size
        )
        os._exit(0)
    mp.spawn(
        train_paradigm,
        args=[world_size, args, parameters],
        nprocs=world_size
    )
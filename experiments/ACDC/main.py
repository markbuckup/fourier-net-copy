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
import neptune.new as neptune
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from neptune.new.types import File
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'
from utils.DDP_paradigms import train_paradigm, test_paradigm

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--test_only', action = 'store_true')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--run_id', type = str, required = True)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
parser.add_argument('--neptune_log', action = 'store_true')
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()

sys.path.append('/root/Cardiac-MRI-Reconstrucion/experiments/ACDC/{}/'.format(args.run_id))

from params import parameters
if parameters['dataset'] == 'acdc':
    from utils.myDatasets.ACDC import ACDC as dataset
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
 
assert(parameters['predicted_frame']) in ['last', 'middle']
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['scheduler']) in ['StepLR', 'None', 'CyclicLR']
assert(parameters['scheduler_params']['mode']) in ['triangular', 'triangular2', 'exp_range']
assert(parameters['loss_recon']) in ['L1', 'L2', 'SSIM', 'MS_SSIM']
assert(parameters['loss_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None']
assert(parameters['loss_reconstructed_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None', 'Cosine-Watson']

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

checkpoint_path = os.path.join(args.run_id, './checkpoints/')
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir(os.path.join(args.run_id, './results/')):
    os.mkdir(os.path.join(args.run_id, './results/'))
if not os.path.isdir(os.path.join(args.run_id, './results/train')):
    os.mkdir(os.path.join(args.run_id, './results/train'))
if not os.path.isdir(os.path.join(args.run_id, './results/test')):
    os.mkdir(os.path.join(args.run_id, './results/test'))

if __name__ == '__main__':
    world_size = len(args.gpu) 
    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        train = True, 
                        blank = False,
                    )
    shared_data = trainset.get_shared_lists()
    if args.eval:
        mp.spawn(
            test_paradigm,
            args=[world_size, shared_data, args, parameters],
            nprocs=world_size
        )
        os._exit(0)
    mp.spawn(
        train_paradigm,
        args=[world_size, shared_data, args, parameters],
        nprocs=world_size
    )
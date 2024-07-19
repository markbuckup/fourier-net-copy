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

# AERS: This codes creates the master/parents process and then creates subprocesses (children/spawn)

sys.path.append('../../')
os.environ['display'] = 'localhost:14.0'
from utils.DDP_paradigms_LSTM_nufft import train_paradigm, test_paradigm    # AERS: Load train and test paradigm functions 

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true', help = 'Resumes the code from the latest checkpoint')
parser.add_argument('--time_analysis', action = 'store_true', help = 'Computes and prints the time taken per frame')
parser.add_argument('--resume_kspace', action = 'store_true', help = 'Resumes the Neural Network after the first 100 epochs, that is, before the image lstm starts training')
parser.add_argument('--eval', action = 'store_true', help = 'Evaluates the architecture - prints the SSIM and saves the predicted images')
parser.add_argument('--eval_on_real', action = 'store_true', help = '[Under development] Runs the evaluate script on real data')
parser.add_argument('--write_csv', action = 'store_true', help = '[Not Updated since October 2023] Stores the l1/l2/ssim scores for each frame into a csv')
parser.add_argument('--test_only', action = 'store_true', help = '')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--motion_mask', action = 'store_true')
parser.add_argument('--numbers_only', action = 'store_true')           # AERS: Same as eval, but does not save the images, only prints SSIM, L1 loss and L2 loss.
parser.add_argument('--numbers_crop', action = 'store_true')
parser.add_argument('--ispace_visual_only', action = 'store_true')     # AERS: Saves the data in image space, after finishing training in image space. Used when you don't want to visualize k-space.
parser.add_argument('--raw_visual_only', action = 'store_true')
parser.add_argument('--train_ispace', action = 'store_true')
parser.add_argument('--dual_training', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)                 # AERS: Used to ensure that the code is reproducible. If trained multiple times, it will give the same results every time.
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--run_id', type = str, required = True)           # AERS: Only parameter that is required. Name of the folder inside experiments where results are saved.
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])    # AERS: If this is not passed, it will crash. Multiple GPUS can be passed aka 0, 1, 2 and 3 in ajax.
parser.add_argument('--neptune_log', action = 'store_true')            # AERS: Logs everything to Neptune
parser.add_argument('--actual_data_path', default = '../../datasets/actual_data/data1.pth')
parser.add_argument('--dataset_path', default = '/Data/ExtraDrive1/niraj/datasets/ACDC/')
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()


assert(not (args.raw_visual_only and args.ispace_visual_only))

sys.path.append('/root/Cardiac-MRI-Reconstrucion/experiments/ACDC_fouriernet/{}/'.format(args.run_id))

from params import parameters
if parameters['dataset'] == 'acdc':
    from utils.myDatasets.ACDC_radial import ACDC_radial as dataset
    
if args.numbers_only or args.visualise_only:
    parameters['init_skip_frames'] = 90

if args.write_csv:
    assert(args.eval)

if args.time_analysis or args.visualise_only:
    parameters['train_batch_size'] = 1
    parameters['test_batch_size'] = 1

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
 
 # AERS: Asserts. Although mode checks may be missing as of 20240613. Comments may be in some params.py.
assert(parameters['optimizer']) in ['Adam', 'SGD']                       # AERS: Make sure optimizer, scheduler, and scheduler_params are valid options
assert(parameters['scheduler']) in ['StepLR', 'None', 'CyclicLR']
assert(parameters['scheduler_params']['mode']) in ['triangular', 'triangular2', 'exp_range']
assert(not (args.visualise_only and args.numbers_only))                  # AERS: You can't visualize and generate numbers at the same time

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")

# AERS: As per Niraj, these are hacks to make things faster. 
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)       # AERS: Set set_detect_anomaly to True if there are NaNs in the code/data/training results.
torch.autograd.set_detect_anomaly(False)        # AERS: This is faster. We may want to add this as an argument rather than commenting/uncommenting.
torch.autograd.profiler.profile(False)  
torch.autograd.profiler.emit_nvtx(False)

if __name__ == '__main__':
    world_size = len(args.gpu) 
    if args.eval or args.eval_on_real:   
        # AERS: mp is the multiprocessing library, creates a child/spawn process
        mp.spawn(  # AERS: This is how a spawn process is created
            test_paradigm,
            args=[world_size, args, parameters], 
            nprocs=world_size # AERS: world size is the number of spawn that the master process will have
        )
        os._exit(0)
    mp.spawn(
        train_paradigm,
        args=[world_size, args, parameters],
        nprocs=world_size
    )

    # AERS: test_paradigm and train_paradigm are instructions a child/spawn process should perform when summoned